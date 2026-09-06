"""Verifying a snapshot that is already on disk.

`fetch_snapshot` answers "did these bytes arrive intact".
`verify_snapshot_on_disk` answers "are these still the bytes", which is
what a caller reading a snapshot it downloaded last week actually needs.

Most of what is worth pinning here is what a single-asset fixture cannot
say. The releases this runs against publish three assets, so a loop
truncated to the first one verifies a third of the release and reports
success; every test below that touches an asset touches one that is not
the first.
"""

from dataclasses import replace
from pathlib import Path

import pytest

from ohlc_toolkit.snapshot import (
    MANIFEST_ASSET_NAME,
    SnapshotFetchResult,
    SnapshotIntegrityError,
    SnapshotManifestError,
    read_snapshot_frame,
    verify_snapshot_on_disk,
)
from ohlc_toolkit.snapshot import fetcher as fetcher_module
from ohlc_toolkit.snapshot.manifest import parse_manifest
from ohlc_toolkit.temporal import ConfigError
from tests.test_snapshot.factories import (
    HISTORY_ASSET,
    ReleaseFixture,
    build_default_assets,
    build_release_fixture,
    sha256_hex,
)

# The three-asset layout the real releases publish. The history asset is
# first in the manifest, so every tampering test below names one that is
# not, which is the only way to tell a loop from a single check.
_NOT_FIRST = "example_1min_provenance.csv"

_SHA256_HEX_DIGITS = 64


def _lay_out(fixture: ReleaseFixture, directory: Path) -> Path:
    """Write a fixture release's manifest and assets into a directory."""
    (directory / MANIFEST_ASSET_NAME).write_bytes(fixture.manifest_bytes)
    for name, payload in fixture.assets.items():
        (directory / name).write_bytes(payload)
    return directory


def _verify(directory: Path, fixture: ReleaseFixture) -> SnapshotFetchResult:
    """Verify a laid-out fixture, naming the release the caller knows.

    `repository` is required, so every call here states one. That is the
    point of it being required: a function that fetched nothing knows
    less about where the bytes came from than one that did, and a default
    would be an invention recorded as provenance.
    """
    return verify_snapshot_on_disk(
        directory,
        repository=fixture.release.repository,
        host=fixture.release.host,
    )


def test_every_declared_asset_is_verified_and_the_identity_is_returned(
    tmp_path: Path,
) -> None:
    """All three assets, and a result the frame reader consumes unchanged."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)

    result = _verify(tmp_path, fixture)

    assert set(result.assets) == set(fixture.assets)
    assert result.manifest_sha256 == sha256_hex(fixture.manifest_bytes)
    assert result.release.tag == fixture.release.tag
    # Nothing was fetched, and every record says so rather than implying it.
    assert not any(asset.was_downloaded for asset in result.assets.values())
    assert read_snapshot_frame(result, asset_name=HISTORY_ASSET).height > 0


@pytest.mark.parametrize("asset", sorted(build_default_assets()))
def test_tampering_with_any_declared_asset_is_refused(
    tmp_path: Path, asset: str
) -> None:
    """Every declared asset, not a chosen two of them.

    Position is what pins a loop body, and a count does not: a case on
    the first asset and a case on the last are both satisfied by an
    implementation that skips everything between. Measured on the version
    this replaces -- skipping the middle asset passed the entire suite,
    and the result object still reported it with its declared size,
    declared digest and `was_downloaded=False`, having read none of it.

    Parametrising over the fixture's own assets is what closes it for
    good rather than moving it: a fourth asset adds a fourth case here
    without anyone remembering to.
    """
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    swapped = bytearray(fixture.assets[asset])
    swapped[-1] ^= 0xFF
    (tmp_path / asset).write_bytes(bytes(swapped))

    with pytest.raises(SnapshotIntegrityError, match=asset):
        _verify(tmp_path, fixture)


def test_a_digest_differing_only_in_its_last_digits_is_refused(
    tmp_path: Path,
) -> None:
    """The whole digest is compared, not a prefix of it.

    A tampered FILE cannot say this: its digest differs from the first
    character, so a one-character comparison rejects it too. The manifest
    can, because the declared value is the test's to choose while the
    file's real digest is not. This one shares sixty hex digits with the
    truth.
    """
    fixture = build_release_fixture()
    real = sha256_hex(fixture.assets[_NOT_FIRST])
    near_miss = real[:60] + ("ffff" if real[60:] != "ffff" else "0000")
    payload = parse_manifest(fixture.manifest_bytes)
    edited = fixture.manifest_bytes.replace(real.encode(), near_miss.encode())
    assert edited != fixture.manifest_bytes
    assert payload.assets  # the fixture really does declare assets
    _lay_out(fixture, tmp_path)
    (tmp_path / MANIFEST_ASSET_NAME).write_bytes(edited)

    with pytest.raises(SnapshotIntegrityError) as caught:
        _verify(tmp_path, fixture)

    assert real in str(caught.value)
    assert near_miss in str(caught.value)
    assert len(real) == _SHA256_HEX_DIGITS


def test_an_asset_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    """Truncation is a different finding from substitution, and says so."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / _NOT_FIRST).write_bytes(fixture.assets[_NOT_FIRST][:-1])

    with pytest.raises(SnapshotIntegrityError, match="bytes, not the"):
        _verify(tmp_path, fixture)


def test_an_asset_the_manifest_declares_but_disk_lacks_is_refused(
    tmp_path: Path,
) -> None:
    """A manifest describing files that are not there is not a partial pass."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / _NOT_FIRST).unlink()

    with pytest.raises(SnapshotIntegrityError, match="which is not at"):
        _verify(tmp_path, fixture)


def test_an_asset_path_that_is_not_a_regular_file_is_refused(
    tmp_path: Path,
) -> None:
    """A directory at an asset path IS something, and it is not the asset."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / _NOT_FIRST).unlink()
    (tmp_path / _NOT_FIRST).mkdir()

    with pytest.raises(ConfigError, match="not a regular file"):
        _verify(tmp_path, fixture)


def test_a_missing_manifest_is_refused_rather_than_assumed(tmp_path: Path) -> None:
    """No manifest is nothing to verify against, which is a caller mistake."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / MANIFEST_ASSET_NAME).unlink()

    with pytest.raises(ConfigError, match="No snapshot manifest"):
        _verify(tmp_path, fixture)


def test_a_manifest_declaring_no_assets_is_refused_here(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refused by this function, not only by the parser it calls.

    `parse_manifest` refuses an empty asset table today. If that guard is
    ever relaxed, the verification loop runs zero times and reports
    success -- a verification that passes over nothing is worse than
    none, because it looks like one. The parser is stubbed so the guard
    is exercised through this function's own boundary.
    """
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    emptied = replace(parse_manifest(fixture.manifest_bytes), assets={})
    monkeypatch.setattr(fetcher_module, "parse_manifest", lambda _raw: emptied)

    with pytest.raises(SnapshotIntegrityError, match="declares no assets"):
        _verify(tmp_path, fixture)


def test_the_release_identity_comes_from_the_caller_and_the_manifest(
    tmp_path: Path,
) -> None:
    """Each field from the party that knows it, and none of them invented.

    `repository` and `host` are the caller's, because a manifest does not
    record where it was published. `tag` is the manifest's, because that
    is the one field it does. Measured on the version this replaces:
    defaulting the repository to this project's own and corrupting that
    default to "totally/wrong-repo" passed 1548 tests at 100% coverage --
    a default is run by every test and asserted by none.
    """
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)

    result = verify_snapshot_on_disk(
        tmp_path, repository="example/somewhere-else", host="https://example.invalid"
    )

    assert result.release.repository == "example/somewhere-else"
    assert result.release.host == "https://example.invalid"
    assert result.release.tag == fixture.release.tag
    # The consequence the field exists for: a URL that leads back to where
    # the caller says these bytes came from, not to a real release with no
    # relationship to them.
    assert result.release.asset_url(HISTORY_ASSET).startswith(
        "https://example.invalid/example/somewhere-else/"
    )


def test_the_repository_has_no_default(tmp_path: Path) -> None:
    """Stating it is the caller's job, and the signature says so.

    `fetch_snapshot` refuses to pick a directory on a caller's disk for
    the same reason. This function knows LESS about provenance than that
    one, having fetched nothing, so it is in no position to pick a
    repository either.
    """
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)

    with pytest.raises(TypeError, match="repository"):
        verify_snapshot_on_disk(tmp_path)  # type: ignore[call-arg]


def test_a_manifest_tag_the_release_grammar_refuses_is_a_manifest_problem(
    tmp_path: Path,
) -> None:
    """Not a caller problem, and the exception class has to say which.

    `parse_manifest` accepts any non-empty string as a tag;
    `SnapshotRelease` applies the stricter grammar. Left alone, a tag of
    `../../../etc/passwd` parsed, every asset verified, the success line
    logged, and THEN a `ConfigError` escaped -- the class this module
    documents as meaning the caller pointed at the wrong directory. The
    caller pointed at the right one.
    """
    fixture = build_release_fixture(tag="v1", manifest_tag="../../../etc/passwd")
    _lay_out(fixture, tmp_path)

    with pytest.raises(SnapshotManifestError, match="usable release tag"):
        verify_snapshot_on_disk(tmp_path, repository="example/dataset")


if __name__ == "__main__":
    pytest.main([__file__])
