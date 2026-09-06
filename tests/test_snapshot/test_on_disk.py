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
    SnapshotIntegrityError,
    read_snapshot_frame,
    verify_snapshot_on_disk,
)
from ohlc_toolkit.snapshot import fetcher as fetcher_module
from ohlc_toolkit.snapshot.manifest import parse_manifest
from ohlc_toolkit.temporal import ConfigError
from tests.test_snapshot.factories import (
    HISTORY_ASSET,
    ReleaseFixture,
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


def test_every_declared_asset_is_verified_and_the_identity_is_returned(
    tmp_path: Path,
) -> None:
    """All three assets, and a result the frame reader consumes unchanged."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)

    result = verify_snapshot_on_disk(tmp_path)

    assert set(result.assets) == set(fixture.assets)
    assert result.manifest_sha256 == sha256_hex(fixture.manifest_bytes)
    assert result.release.tag == fixture.release.tag
    # Nothing was fetched, and every record says so rather than implying it.
    assert not any(asset.was_downloaded for asset in result.assets.values())
    assert read_snapshot_frame(result, asset_name=HISTORY_ASSET).height > 0


def test_tampering_with_an_asset_that_is_not_the_first_is_refused(
    tmp_path: Path,
) -> None:
    """The finding a first-asset-only loop cannot make.

    Truncating the verification to the manifest's first asset leaves this
    file unread, and the run reports a verified snapshot with a
    real-looking identity.
    """
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    swapped = bytearray(fixture.assets[_NOT_FIRST])
    swapped[-1] ^= 0xFF
    (tmp_path / _NOT_FIRST).write_bytes(bytes(swapped))

    with pytest.raises(SnapshotIntegrityError, match=_NOT_FIRST):
        verify_snapshot_on_disk(tmp_path)


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
        verify_snapshot_on_disk(tmp_path)

    assert real in str(caught.value)
    assert near_miss in str(caught.value)
    assert len(real) == _SHA256_HEX_DIGITS


def test_an_asset_of_the_wrong_length_is_refused(tmp_path: Path) -> None:
    """Truncation is a different finding from substitution, and says so."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / _NOT_FIRST).write_bytes(fixture.assets[_NOT_FIRST][:-1])

    with pytest.raises(SnapshotIntegrityError, match="bytes, not the"):
        verify_snapshot_on_disk(tmp_path)


def test_an_asset_the_manifest_declares_but_disk_lacks_is_refused(
    tmp_path: Path,
) -> None:
    """A manifest describing files that are not there is not a partial pass."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / _NOT_FIRST).unlink()

    with pytest.raises(SnapshotIntegrityError, match="which is not at"):
        verify_snapshot_on_disk(tmp_path)


def test_an_asset_path_that_is_not_a_regular_file_is_refused(
    tmp_path: Path,
) -> None:
    """A directory at an asset path IS something, and it is not the asset."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / _NOT_FIRST).unlink()
    (tmp_path / _NOT_FIRST).mkdir()

    with pytest.raises(ConfigError, match="not a regular file"):
        verify_snapshot_on_disk(tmp_path)


def test_a_missing_manifest_is_refused_rather_than_assumed(tmp_path: Path) -> None:
    """No manifest is nothing to verify against, which is a caller mistake."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    (tmp_path / MANIFEST_ASSET_NAME).unlink()

    with pytest.raises(ConfigError, match="No snapshot manifest"):
        verify_snapshot_on_disk(tmp_path)


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
        verify_snapshot_on_disk(tmp_path)


def test_the_history_asset_is_verified_too(tmp_path: Path) -> None:
    """The first asset is not skipped either, which the others cannot say."""
    fixture = build_release_fixture()
    _lay_out(fixture, tmp_path)
    swapped = bytearray(fixture.assets[HISTORY_ASSET])
    swapped[-1] ^= 0xFF
    (tmp_path / HISTORY_ASSET).write_bytes(bytes(swapped))

    with pytest.raises(SnapshotIntegrityError, match=HISTORY_ASSET):
        verify_snapshot_on_disk(tmp_path)


if __name__ == "__main__":
    pytest.main([__file__])
