"""The one test that fetches the real published release over the network.

Every other test in this package builds its own bytes. This one does not:
it points the fetcher at the actual monthly full-history release and
holds it to the same contract, which is the only way to find out that the
real manifest is shaped the way the parser assumes.

It is deselected by default -- the marker is filtered out in
``pyproject.toml`` -- because a default test run must not depend on
GitHub being up, and because the assets are hundreds of megabytes. Run it
deliberately::

    uv run pytest -m network

The expected values below are read off the published release, not off
this package's own output, so the test can fail rather than merely agree
with itself. They pin one specific immutable tag; a later month is a new
release and would need its own numbers.
"""

from pathlib import Path

import pytest

from ohlc_toolkit.snapshot import (
    BITSTAMP_BTCUSD_1M_REPOSITORY,
    BITSTAMP_HISTORY_CSV_ASSET,
    SnapshotRelease,
    fetch_snapshot,
    read_snapshot_frame,
)

_TAG = "bitstamp-btcusd-1m-2026-08"
_PARQUET_ASSET = "btcusd_bitstamp_1min.parquet"
_PROVENANCE_ASSET = "btcusd_bitstamp_1min_provenance.csv"

_EXPECTED_ASSET_COUNT = 3
_EXPECTED_SCHEMA_VERSION = 1
_EXPECTED_ROW_COUNT = 7_714_079
_EXPECTED_FIRST_TIMESTAMP = 1_325_376_060
_EXPECTED_LAST_TIMESTAMP = 1_788_220_740
_EXPECTED_HISTORY_BYTES = 108_008_821
_CADENCE_SECONDS = 60
_TIMESTAMP_COLUMN = "timestamp"


@pytest.mark.network
def test_the_published_release_fetches_verifies_and_refetches(tmp_path: Path) -> None:
    """The real release passes every check this package makes of it.

    Three assertions here are about the world rather than about this
    code, and they are the reason the test exists:

    - The manifest declares 7,714,079 rows spanning 1325376060 to
      1788220740. Those bounds are 462,844,680 seconds apart, which is
      7,714,078 minutes, so the row count is exactly the span plus one:
      the published history is a complete minute grid with no gaps at
      all.
    - The history CSV really does hold that many rows, on that grid,
      sorted and unduplicated. Only reading all of it can show that.
    - Re-fetching into the same directory transfers nothing: every asset
      is already there with a matching digest.
    """
    release = SnapshotRelease(repository=BITSTAMP_BTCUSD_1M_REPOSITORY, tag=_TAG)

    result = fetch_snapshot(release, tmp_path)

    assert sorted(result.assets) == sorted(
        [BITSTAMP_HISTORY_CSV_ASSET, _PARQUET_ASSET, _PROVENANCE_ASSET]
    )
    assert len(result.assets) == _EXPECTED_ASSET_COUNT
    assert all(asset.was_downloaded for asset in result.assets.values())
    assert result.manifest.schema_version == _EXPECTED_SCHEMA_VERSION
    assert result.manifest.tag == _TAG
    assert result.manifest.row_count == _EXPECTED_ROW_COUNT
    assert result.manifest.first_timestamp == _EXPECTED_FIRST_TIMESTAMP
    assert result.manifest.last_timestamp == _EXPECTED_LAST_TIMESTAMP
    assert result.assets[BITSTAMP_HISTORY_CSV_ASSET].size_bytes == (
        _EXPECTED_HISTORY_BYTES
    )

    # The manifest's own three statements are mutually consistent, which
    # is what makes the history a complete grid rather than merely a
    # correctly-counted one.
    span = _EXPECTED_LAST_TIMESTAMP - _EXPECTED_FIRST_TIMESTAMP
    assert span % _CADENCE_SECONDS == 0
    assert span // _CADENCE_SECONDS + 1 == _EXPECTED_ROW_COUNT

    frame = read_snapshot_frame(result)

    assert frame.height == _EXPECTED_ROW_COUNT
    timestamps = frame.get_column(_TIMESTAMP_COLUMN)
    assert timestamps[0] == _EXPECTED_FIRST_TIMESTAMP
    assert timestamps[-1] == _EXPECTED_LAST_TIMESTAMP

    again = fetch_snapshot(release, tmp_path)

    assert not any(asset.was_downloaded for asset in again.assets.values())
    assert again.manifest == result.manifest


if __name__ == "__main__":
    pytest.main([__file__, "-m", "network"])
