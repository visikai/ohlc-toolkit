"""Tests for fetching a release's assets and refusing anything unverified.

The contract under test is narrow and absolute: bytes reach their final
path only after their SHA-256 matches what the manifest declared. Every
test here is about a way that can fail -- a wrong digest, a short body, a
body that outgrows its declared size, an asset the release does not
serve, a manifest that describes a different release -- and about what is
left on disk afterwards.

Nothing here opens a socket. The transport is the seam, and it is filled
with an in-process map from URL to bytes.
"""

from pathlib import Path

import pytest

from ohlc_toolkit.snapshot.errors import (
    SnapshotIntegrityError,
    SnapshotManifestError,
)
from ohlc_toolkit.snapshot.fetcher import (
    ExistingAssetPolicy,
    SnapshotFetchResult,
    fetch_snapshot,
)
from ohlc_toolkit.snapshot.manifest import MANIFEST_ASSET_NAME
from ohlc_toolkit.temporal import ConfigError, DataValidationError
from tests.test_snapshot.factories import (
    HISTORY_ASSET,
    PARQUET_ASSET,
    PROVENANCE_ASSET,
    ReleaseFixture,
    build_release_fixture,
    sha256_hex,
)

_EXPECTED_ASSET_COUNT = 3
_TEMP_GLOB = "*.part"
_ONLY_THE_MANIFEST = 1


def _fetch(
    fixture: ReleaseFixture,
    directory: Path,
    *,
    payloads: dict[str, bytes] | None = None,
    existing: ExistingAssetPolicy = ExistingAssetPolicy.REFUSE,
) -> tuple[SnapshotFetchResult, list[str]]:
    """Fetch a fixture release, returning the result and the URLs requested."""
    transport = fixture.transport(payloads)
    result = fetch_snapshot(
        fixture.release, directory, transport=transport, existing=existing
    )
    return result, transport.requested


def _leftovers(directory: Path) -> list[Path]:
    """Return every temporary download file still present."""
    return sorted(directory.glob(_TEMP_GLOB)) + sorted(directory.glob(f".{_TEMP_GLOB}"))


def test_every_declared_asset_lands_with_the_bytes_the_manifest_declared(
    tmp_path: Path,
) -> None:
    """The point of the whole subpackage: the right bytes, in the right place."""
    fixture = build_release_fixture()

    result, _ = _fetch(fixture, tmp_path)

    assert len(result.assets) == _EXPECTED_ASSET_COUNT
    for name, payload in fixture.assets.items():
        asset = result.assets[name]
        assert asset.path.read_bytes() == payload
        assert asset.sha256 == sha256_hex(payload)
        assert asset.size_bytes == len(payload)
        assert asset.was_downloaded is True


def test_the_fetch_result_records_the_manifest_digest(tmp_path: Path) -> None:
    """The snapshot's identity is the SHA-256 over the manifest bytes.

    The manifest carries no digest of itself, so the consumer computes
    one: this is the single value that changes if the manifest -- and
    with it the whole self-consistent asset set it describes -- is
    swapped wholesale. Per-asset digests cannot see that swap, because a
    swapped manifest describes the swapped assets correctly.
    """
    fixture = build_release_fixture()

    result, _ = _fetch(fixture, tmp_path)

    assert result.manifest_sha256 == sha256_hex(fixture.manifest_bytes)


def test_a_mismatched_expected_manifest_digest_is_refused(tmp_path: Path) -> None:
    """A caller who knows the snapshot identity can demand exactly it."""
    fixture = build_release_fixture()

    with pytest.raises(SnapshotIntegrityError, match="manifest"):
        fetch_snapshot(
            fixture.release,
            tmp_path,
            transport=fixture.transport(),
            expected_manifest_sha256="0" * 64,
        )


def test_the_manifest_is_written_beside_the_assets(tmp_path: Path) -> None:
    """The statement the assets were checked against is kept with them."""
    fixture = build_release_fixture()

    result, _ = _fetch(fixture, tmp_path)

    assert result.manifest_path == tmp_path / MANIFEST_ASSET_NAME
    assert result.manifest_path.read_bytes() == fixture.manifest_bytes
    assert result.manifest.tag == fixture.release.tag
    assert result.directory == tmp_path
    assert result.release == fixture.release


def test_no_temporary_files_survive_a_successful_fetch(tmp_path: Path) -> None:
    """Every temp name is renamed away or removed; none is left to be read."""
    fixture = build_release_fixture()

    _fetch(fixture, tmp_path)

    assert _leftovers(tmp_path) == []


def test_no_temporary_file_survives_an_interrupt_mid_download(
    tmp_path: Path,
) -> None:
    """Cleanup is a ``finally``, so even Ctrl-C leaves no partial body.

    The failure-path tests all raise ordinary exceptions, which an
    ``except Exception`` would also clean up after -- so they cannot
    tell a ``finally`` from the weaker shape. ``KeyboardInterrupt`` is
    not an ``Exception``, and a partly-written body left behind by an
    interrupted fetch is exactly the file a later reader must never
    find.
    """
    fixture = build_release_fixture()
    transport = fixture.transport()
    original_download = transport.download

    def interrupted_download(url: str, destination: Path, *, max_bytes: int) -> None:
        original_download(url, destination, max_bytes=max_bytes)
        if url.endswith(HISTORY_ASSET):
            raise KeyboardInterrupt

    transport.download = interrupted_download  # type: ignore[method-assign]

    with pytest.raises(KeyboardInterrupt):
        fetch_snapshot(fixture.release, tmp_path, transport=transport)

    assert _leftovers(tmp_path) == []


def test_a_missing_directory_is_created(tmp_path: Path) -> None:
    """A caller names a directory; it does not have to exist yet."""
    directory = tmp_path / "nested" / "snapshot"
    fixture = build_release_fixture()

    result, _ = _fetch(fixture, directory)

    assert result.directory.is_dir()
    assert (directory / HISTORY_ASSET).is_file()


def test_a_destination_occupied_by_a_file_is_refused(tmp_path: Path) -> None:
    """A directory that is really a file is the caller's mistake, not the data's."""
    occupied = tmp_path / "snapshot"
    occupied.write_text("in the way")
    fixture = build_release_fixture()

    with pytest.raises(ConfigError, match="directory"):
        _fetch(fixture, occupied)


def test_an_asset_path_occupied_by_a_directory_is_refused(tmp_path: Path) -> None:
    """An asset's final path being a directory is likewise a caller problem."""
    (tmp_path / HISTORY_ASSET).mkdir()
    fixture = build_release_fixture()

    with pytest.raises(ConfigError, match=HISTORY_ASSET):
        _fetch(fixture, tmp_path)


def test_an_asset_whose_digest_does_not_match_never_reaches_its_final_path(
    tmp_path: Path,
) -> None:
    """The refusal this whole design exists for: wrong bytes, same length."""
    fixture = build_release_fixture()
    tampered = bytes(len(fixture.assets[HISTORY_ASSET]))

    with pytest.raises(SnapshotIntegrityError, match="sha256"):
        _fetch(fixture, tmp_path, payloads=fixture.serving(HISTORY_ASSET, tampered))

    assert not (tmp_path / HISTORY_ASSET).exists()
    assert _leftovers(tmp_path) == []


def test_an_asset_shorter_than_its_declared_size_is_refused(tmp_path: Path) -> None:
    """A truncated download is caught by size before it is caught by digest."""
    fixture = build_release_fixture()
    truncated = fixture.assets[HISTORY_ASSET][:-5]

    with pytest.raises(SnapshotIntegrityError, match="bytes"):
        _fetch(fixture, tmp_path, payloads=fixture.serving(HISTORY_ASSET, truncated))

    assert not (tmp_path / HISTORY_ASSET).exists()
    assert _leftovers(tmp_path) == []


def test_an_asset_larger_than_its_declared_size_is_refused(tmp_path: Path) -> None:
    """The declared size is the download cap, so an overlong body never lands."""
    fixture = build_release_fixture()
    overlong = fixture.assets[HISTORY_ASSET] + b"trailing"

    with pytest.raises(SnapshotIntegrityError, match="cap"):
        _fetch(fixture, tmp_path, payloads=fixture.serving(HISTORY_ASSET, overlong))

    assert not (tmp_path / HISTORY_ASSET).exists()
    assert _leftovers(tmp_path) == []


def test_an_asset_the_release_does_not_serve_is_refused(tmp_path: Path) -> None:
    """A manifest that declares an asset the release lacks is a broken publish."""
    fixture = build_release_fixture()

    with pytest.raises(SnapshotIntegrityError):
        _fetch(fixture, tmp_path, payloads=fixture.serving(PROVENANCE_ASSET, None))

    assert not (tmp_path / PROVENANCE_ASSET).exists()
    assert _leftovers(tmp_path) == []


def test_assets_verified_before_a_failure_are_kept_on_purpose(
    tmp_path: Path,
) -> None:
    """A fetch is not transactional, and says so.

    An asset that already matched its declared digest is not partial
    state: it is finished work. Deleting it on a later asset's failure
    would throw away a verified download and force it over the wire
    again. What must never survive is a temporary file, and none does.
    """
    fixture = build_release_fixture()

    with pytest.raises(SnapshotIntegrityError):
        _fetch(fixture, tmp_path, payloads=fixture.serving(PROVENANCE_ASSET, None))

    assert (tmp_path / HISTORY_ASSET).read_bytes() == fixture.assets[HISTORY_ASSET]
    assert (tmp_path / PARQUET_ASSET).read_bytes() == fixture.assets[PARQUET_ASSET]
    assert _leftovers(tmp_path) == []


def test_a_kept_asset_is_not_downloaded_again_by_the_next_fetch(
    tmp_path: Path,
) -> None:
    """Retrying after a failure asks only for what is still missing."""
    fixture = build_release_fixture()
    with pytest.raises(SnapshotIntegrityError):
        _fetch(fixture, tmp_path, payloads=fixture.serving(PROVENANCE_ASSET, None))

    result, requested = _fetch(fixture, tmp_path)

    assert result.assets[HISTORY_ASSET].was_downloaded is False
    assert result.assets[PARQUET_ASSET].was_downloaded is False
    assert result.assets[PROVENANCE_ASSET].was_downloaded is True
    assert fixture.url_for(HISTORY_ASSET) not in requested
    assert fixture.url_for(PROVENANCE_ASSET) in requested


def test_a_second_fetch_downloads_nothing_but_the_manifest(tmp_path: Path) -> None:
    """Re-fetching a complete, valid directory is a no-op over the wire."""
    fixture = build_release_fixture()
    _fetch(fixture, tmp_path)

    result, requested = _fetch(fixture, tmp_path)

    assert requested == [fixture.url_for(MANIFEST_ASSET_NAME)]
    assert len(requested) == _ONLY_THE_MANIFEST
    assert all(not asset.was_downloaded for asset in result.assets.values())


def test_the_manifest_is_always_refetched(tmp_path: Path) -> None:
    """The manifest cannot be checked against a digest, so it is never trusted stale.

    It is the authority every other check reads from, and it is small; a
    cached copy could describe a release that has since been re-cut.
    """
    fixture = build_release_fixture()
    _fetch(fixture, tmp_path)

    _, requested = _fetch(fixture, tmp_path)

    assert fixture.url_for(MANIFEST_ASSET_NAME) in requested


def test_a_malformed_manifest_is_refused_before_any_asset_is_requested(
    tmp_path: Path,
) -> None:
    """Nothing is downloaded on the word of a manifest that could not be read."""
    fixture = build_release_fixture(manifest_bytes=b"{ not json at all")

    with pytest.raises(SnapshotManifestError):
        _fetch(fixture, tmp_path)

    assert not (tmp_path / MANIFEST_ASSET_NAME).exists()
    assert list(tmp_path.iterdir()) == []


def test_a_manifest_declaring_another_tag_is_refused(tmp_path: Path) -> None:
    """A release serving someone else's manifest is refused, not merged."""
    fixture = build_release_fixture(manifest_tag="example-1m-2026-07")

    with pytest.raises(SnapshotManifestError, match="tag"):
        _fetch(fixture, tmp_path)

    assert not (tmp_path / MANIFEST_ASSET_NAME).exists()
    assert list(tmp_path.iterdir()) == []


def test_an_existing_asset_with_the_wrong_digest_is_refused_by_default(
    tmp_path: Path,
) -> None:
    """The default posture is to stop, not to silently overwrite the caller's file."""
    fixture = build_release_fixture()
    stale = tmp_path / HISTORY_ASSET
    stale.write_bytes(b"stale contents from some other run")

    with pytest.raises(SnapshotIntegrityError, match="already present"):
        _fetch(fixture, tmp_path)


def test_a_refused_existing_asset_is_left_exactly_as_it_was(tmp_path: Path) -> None:
    """Refusing does not damage what was there; the caller decides what to do."""
    fixture = build_release_fixture()
    stale = tmp_path / HISTORY_ASSET
    stale_bytes = b"stale contents from some other run"
    stale.write_bytes(stale_bytes)

    with pytest.raises(SnapshotIntegrityError):
        _fetch(fixture, tmp_path)

    assert stale.read_bytes() == stale_bytes
    assert _leftovers(tmp_path) == []


def test_replacing_is_opt_in_and_overwrites_an_invalid_existing_asset(
    tmp_path: Path,
) -> None:
    """Replacement happens only when the caller asks for it by name."""
    fixture = build_release_fixture()
    (tmp_path / HISTORY_ASSET).write_bytes(b"stale contents from some other run")

    result, requested = _fetch(fixture, tmp_path, existing=ExistingAssetPolicy.REPLACE)

    assert result.assets[HISTORY_ASSET].was_downloaded is True
    assert (tmp_path / HISTORY_ASSET).read_bytes() == fixture.assets[HISTORY_ASSET]
    assert fixture.url_for(HISTORY_ASSET) in requested


def test_replacing_still_refuses_bytes_that_fail_their_digest(
    tmp_path: Path,
) -> None:
    """Opting into replacement does not opt out of verification.

    The invalid file the caller already had is left exactly where it was:
    the replacement never verified, so there is nothing to put in its
    place, and destroying the original would lose data to no purpose.
    """
    fixture = build_release_fixture()
    stale = tmp_path / HISTORY_ASSET
    stale_bytes = b"stale contents from some other run"
    stale.write_bytes(stale_bytes)
    tampered = bytes(len(fixture.assets[HISTORY_ASSET]))

    with pytest.raises(SnapshotIntegrityError, match="sha256"):
        _fetch(
            fixture,
            tmp_path,
            payloads=fixture.serving(HISTORY_ASSET, tampered),
            existing=ExistingAssetPolicy.REPLACE,
        )

    assert stale.read_bytes() == stale_bytes
    assert _leftovers(tmp_path) == []


def test_the_result_is_frozen(tmp_path: Path) -> None:
    """A fetch result records what happened; it is not a mutable scratch pad."""
    fixture = build_release_fixture()

    result, _ = _fetch(fixture, tmp_path)

    with pytest.raises(AttributeError):
        result.directory = tmp_path  # type: ignore[misc]
    with pytest.raises(TypeError):
        result.assets["injected"] = result.assets[HISTORY_ASSET]  # type: ignore[index]


def test_integrity_failures_are_data_validation_errors(tmp_path: Path) -> None:
    """A refusal here is a statement about published data, catchable as such."""
    fixture = build_release_fixture()
    tampered = bytes(len(fixture.assets[HISTORY_ASSET]))

    with pytest.raises(DataValidationError):
        _fetch(fixture, tmp_path, payloads=fixture.serving(HISTORY_ASSET, tampered))


if __name__ == "__main__":
    pytest.main([__file__])
