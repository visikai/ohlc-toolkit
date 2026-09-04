"""Tests for checking a fetched history against what its manifest claims.

A matching SHA-256 says the bytes are the published bytes. It says
nothing about whether those bytes are the history the manifest describes.
These tests cover the second question: the row count, the first and last
timestamps, and a complete 60-second grid with no duplicates, no gaps,
and no rows out of order.

The grid half is not reimplemented here. It is
``ohlc_toolkit.source.validation`` run against the same profile the rest
of the package reads Bitstamp minute data with, so a snapshot is held to
exactly the standard a local file is.
"""

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from ohlc_toolkit.snapshot import continuity
from ohlc_toolkit.snapshot.continuity import (
    _MAX_ECHOED_ASSET_NAMES,
    ContinuityReport,
    SeamKind,
    SeamMismatch,
    SnapshotContinuityError,
    read_snapshot_frame,
    verify_snapshot_continuity,
)
from ohlc_toolkit.snapshot.fetcher import SnapshotFetchResult, fetch_snapshot
from ohlc_toolkit.snapshot.manifest import SnapshotManifest, parse_manifest
from ohlc_toolkit.snapshot.release import BITSTAMP_HISTORY_CSV_ASSET
from ohlc_toolkit.source import FindingKind
from ohlc_toolkit.temporal import MAX_ECHO_CHARS, ConfigError, DataValidationError
from tests.test_snapshot.factories import (
    CADENCE_SECONDS,
    FIXTURE_ROWS,
    FIXTURE_START,
    HISTORY_ASSET,
    PARQUET_ASSET,
    build_default_assets,
    build_history_csv,
    build_history_frame,
    build_manifest_payload,
    build_release_fixture,
    encode_manifest,
    gzip_bytes,
    history_timestamps,
)

_TIMESTAMP_COLUMN = "timestamp"
_EMPTY_ROWS = 0
_SHORTENED_ROWS = FIXTURE_ROWS - 1
_GAP_AT = 5
_MISSING_CANDLES_IN_GAP = 1


def _manifest(timestamps: list[int] | None = None) -> SnapshotManifest:
    """Build a manifest that truthfully describes the given history."""
    resolved = history_timestamps() if timestamps is None else timestamps
    return parse_manifest(
        encode_manifest(
            build_manifest_payload(assets=build_default_assets(), timestamps=resolved)
        )
    )


def _kinds(report: ContinuityReport) -> set[SeamKind]:
    """Return the seam kinds a report flagged."""
    return {mismatch.kind for mismatch in report.seam_mismatches}


def _raised(frame: pl.DataFrame, manifest: SnapshotManifest) -> ContinuityReport:
    """Run the check, assert it refused, and hand back the attached report."""
    with pytest.raises(SnapshotContinuityError) as caught:
        verify_snapshot_continuity(frame, manifest)
    return caught.value.report


def test_a_history_that_matches_its_manifest_passes() -> None:
    """The baseline: every statement holds and the grid is complete."""
    timestamps = history_timestamps()

    report = verify_snapshot_continuity(build_history_frame(timestamps), _manifest())

    assert report.passed
    assert report.rows_checked == FIXTURE_ROWS
    assert report.seam_mismatches == ()
    assert report.validation.passed


def test_the_report_is_frozen() -> None:
    """A continuity report records what was found; it is not editable after."""
    report = verify_snapshot_continuity(
        build_history_frame(history_timestamps()), _manifest()
    )

    with pytest.raises(AttributeError):
        report.rows_checked = 0  # type: ignore[misc]


def test_a_row_count_that_disagrees_with_the_manifest_is_refused() -> None:
    """A history missing a row the manifest counted is not the published one."""
    short = history_timestamps(rows=_SHORTENED_ROWS)

    report = _raised(build_history_frame(short), _manifest())

    assert SeamKind.ROW_COUNT in _kinds(report)
    assert (
        SeamMismatch(
            kind=SeamKind.ROW_COUNT, expected=FIXTURE_ROWS, observed=_SHORTENED_ROWS
        )
        in report.seam_mismatches
    )


def test_a_first_timestamp_that_disagrees_with_the_manifest_is_refused() -> None:
    """A history that starts somewhere else is a different history."""
    shifted = history_timestamps(start=FIXTURE_START + CADENCE_SECONDS)

    report = _raised(build_history_frame(shifted), _manifest())

    assert SeamKind.FIRST_TIMESTAMP in _kinds(report)


def test_a_last_timestamp_that_disagrees_with_the_manifest_is_refused() -> None:
    """A history that ends somewhere else is likewise a different history."""
    extended = history_timestamps(rows=FIXTURE_ROWS + 1)

    report = _raised(build_history_frame(extended), _manifest())

    assert SeamKind.LAST_TIMESTAMP in _kinds(report)


def test_a_manifest_that_contradicts_itself_is_reported() -> None:
    """The manifest's own three statements must agree before any row is read.

    A row count that does not match the count implied by the first and
    last timestamps at the source cadence is a broken manifest, and it is
    worth saying so rather than only reporting the frame that fails
    against it.
    """
    payload = build_manifest_payload(
        assets=build_default_assets(), timestamps=history_timestamps()
    )
    payload["row_count"] = _SHORTENED_ROWS
    manifest = parse_manifest(encode_manifest(payload))

    report = _raised(build_history_frame(history_timestamps()), manifest)

    assert SeamKind.MANIFEST_SPAN in _kinds(report)


def test_a_manifest_span_not_divisible_by_the_cadence_is_refused() -> None:
    """A span that does not divide by the cadence fits NO row count.

    The test above perturbs the row count, which the implied-count
    comparison alone would catch; here the count matches the floor of
    the implied count exactly, so ONLY the divisibility half of the
    check can object. Half a minute past the last full candle is not a
    place a 60-second grid can end.
    """
    payload = build_manifest_payload(
        assets=build_default_assets(), timestamps=history_timestamps()
    )
    payload["last_timestamp"] = payload["last_timestamp"] + CADENCE_SECONDS // 2
    manifest = parse_manifest(encode_manifest(payload))

    report = _raised(build_history_frame(history_timestamps()), manifest)

    assert SeamKind.MANIFEST_SPAN in _kinds(report)


def test_a_gap_in_the_grid_is_refused() -> None:
    """A missing minute is the failure the whole check exists for."""
    gapped = history_timestamps(omit=[_GAP_AT])

    report = _raised(build_history_frame(gapped), _manifest(gapped))

    gaps = [
        finding
        for finding in report.validation.findings
        if finding.kind is FindingKind.GAP
    ]
    assert len(gaps) == 1
    assert gaps[0].count == _MISSING_CANDLES_IN_GAP


def test_a_duplicated_timestamp_is_refused() -> None:
    """Two rows for one minute cannot both be that minute's candle."""
    timestamps = history_timestamps()
    duplicated = [*timestamps[:_GAP_AT], timestamps[_GAP_AT], *timestamps[_GAP_AT:]]

    report = _raised(build_history_frame(duplicated), _manifest())

    assert any(
        finding.kind is FindingKind.NON_INCREASING_TIMESTAMPS
        for finding in report.validation.findings
    )


def test_rows_out_of_order_are_refused() -> None:
    """A sorted grid is part of the claim, not something to fix on read."""
    timestamps = history_timestamps()
    swapped = list(timestamps)
    swapped[_GAP_AT], swapped[_GAP_AT + 1] = swapped[_GAP_AT + 1], swapped[_GAP_AT]

    report = _raised(build_history_frame(swapped), _manifest())

    assert any(
        finding.kind is FindingKind.NON_INCREASING_TIMESTAMPS
        for finding in report.validation.findings
    )


def test_an_empty_history_is_refused_without_reading_a_first_row() -> None:
    """Zero rows fails the row count, and the seam checks do not crash on it."""
    report = _raised(build_history_frame([]), _manifest())

    assert report.rows_checked == _EMPTY_ROWS
    assert SeamKind.ROW_COUNT in _kinds(report)
    assert SeamKind.FIRST_TIMESTAMP not in _kinds(report)
    assert SeamKind.LAST_TIMESTAMP not in _kinds(report)


def test_a_missing_timestamp_column_is_refused_without_a_crash() -> None:
    """Schema validation reports it; the seam checks step aside rather than throw."""
    frame = build_history_frame(history_timestamps()).drop(_TIMESTAMP_COLUMN)

    report = _raised(frame, _manifest())

    assert any(
        finding.kind is FindingKind.SCHEMA for finding in report.validation.findings
    )
    assert SeamKind.FIRST_TIMESTAMP not in _kinds(report)


def test_a_null_timestamp_stops_the_seam_checks_rather_than_being_read() -> None:
    """A null can hide a gap; reading it as a bound would launder the corruption."""
    timestamps = history_timestamps()
    frame = build_history_frame(timestamps).with_columns(
        pl.Series(
            _TIMESTAMP_COLUMN,
            [None, *timestamps[1:]],
            dtype=pl.Int64,
        )
    )

    report = _raised(frame, _manifest())

    assert any(
        finding.kind is FindingKind.NULL_VALUES
        for finding in report.validation.findings
    )
    assert SeamKind.FIRST_TIMESTAMP not in _kinds(report)
    assert SeamKind.LAST_TIMESTAMP not in _kinds(report)


def test_continuity_failures_are_data_validation_errors() -> None:
    """A history that contradicts its manifest is bad published data."""
    with pytest.raises(DataValidationError):
        verify_snapshot_continuity(
            build_history_frame(history_timestamps(rows=_SHORTENED_ROWS)), _manifest()
        )


# Far more assets than any real release declares, with deliberately short
# names so that bounding each name's length would change nothing here.
_CROWDED_ASSET_COUNT = 4000
# Room for the capped number of names, each at the echo bound, plus prose.
_MAX_REFUSAL_CHARS = 2 * _MAX_ECHOED_ASSET_NAMES * MAX_ECHO_CHARS


def _fetch(fixture_assets: dict[str, bytes], directory: Path) -> SnapshotFetchResult:
    """Fetch a fixture release built over the given asset bytes."""
    fixture = build_release_fixture(assets=fixture_assets)
    return fetch_snapshot(fixture.release, directory, transport=fixture.transport())


def test_reading_a_fetched_history_returns_the_verified_frame(
    tmp_path: Path,
) -> None:
    """The one path from a fetch result to a frame, and it checks continuity."""
    result = _fetch(build_default_assets(), tmp_path)

    frame = read_snapshot_frame(result, asset_name=HISTORY_ASSET)

    assert frame.height == FIXTURE_ROWS
    assert frame.get_column(_TIMESTAMP_COLUMN)[0] == FIXTURE_START


def test_rows_beyond_the_declared_count_are_never_materialized(
    tmp_path: Path,
) -> None:
    """The read materializes at most the declared row count, plus one.

    A file fifty rows longer than its manifest declares must be refused
    with an observed count of declared-plus-ONE: one extra row is enough
    to prove the file too long, and stopping there is what keeps a
    longer -- or adversarially compressed -- publish from occupying
    memory in proportion to ITS size rather than the manifest's. An
    observed count of declared-plus-fifty would mean the whole file was
    read first and compared after. (The bound is on materialization:
    polars still scans the remaining bytes, so this caps memory, not
    parse work.)
    """
    long_history = build_history_csv(history_timestamps(rows=FIXTURE_ROWS + 50))
    assets = build_default_assets()
    assets[HISTORY_ASSET] = gzip_bytes(long_history)
    result = _fetch(assets, tmp_path)

    with pytest.raises(SnapshotContinuityError) as caught:
        read_snapshot_frame(result, asset_name=HISTORY_ASSET)

    row_count_mismatches = [
        mismatch
        for mismatch in caught.value.report.seam_mismatches
        if mismatch.kind is SeamKind.ROW_COUNT
    ]
    assert len(row_count_mismatches) == 1
    assert row_count_mismatches[0].observed == FIXTURE_ROWS + 1


def test_reading_a_gapped_fetched_history_is_refused(tmp_path: Path) -> None:
    """Bytes can pass their digest and still not be the described history."""
    assets = build_default_assets()
    assets[HISTORY_ASSET] = gzip_bytes(
        build_history_csv(history_timestamps(omit=[_GAP_AT]))
    )
    result = _fetch(assets, tmp_path)

    with pytest.raises(SnapshotContinuityError) as caught:
        read_snapshot_frame(result, asset_name=HISTORY_ASSET)

    assert any(
        finding.kind is FindingKind.GAP
        for finding in caught.value.report.validation.findings
    )


def test_reading_an_asset_the_release_does_not_declare_is_refused(
    tmp_path: Path,
) -> None:
    """Asking for a name that is not in the result is a caller mistake."""
    result = _fetch(build_default_assets(), tmp_path)

    with pytest.raises(ConfigError, match="absent"):
        read_snapshot_frame(result, asset_name="not_in_this_release.csv.gz")


def test_an_absent_asset_does_not_quote_back_every_name_the_release_holds(
    tmp_path: Path,
) -> None:
    """A release declaring many assets must not echo all of them.

    Every name here is SHORT, which is the point: what is unbounded is
    how many assets a manifest may declare, and nothing caps that. A
    per-name length bound would leave this exactly as it was, so the
    cap has to be on the count.
    """
    result = _fetch(build_default_assets(), tmp_path)
    one = next(iter(result.assets.values()))
    crowded = replace(
        result,
        assets={
            f"a{index:05d}.csv.gz": replace(one, name=f"a{index:05d}.csv.gz")
            for index in range(_CROWDED_ASSET_COUNT)
        },
    )

    logged: list[str] = []
    sink_id = continuity.logger.add(logged.append, level="ERROR", format="{message}")
    try:
        with pytest.raises(ConfigError, match="absent") as raised:
            read_snapshot_frame(crowded, asset_name="not_in_this_release.csv.gz")
    finally:
        continuity.logger.remove(sink_id)

    message = str(raised.value)
    assert len(message) < _MAX_REFUSAL_CHARS
    # Bounded, but the reader is still told how many there really were.
    assert str(_CROWDED_ASSET_COUNT) in message

    assert logged, "the refusal logs before it raises; nothing was captured"
    for line in logged:
        assert len(line) < _MAX_REFUSAL_CHARS


def test_the_default_asset_is_the_published_bitstamp_history(
    tmp_path: Path,
) -> None:
    """Reading defaults to the real published history asset, and only that one.

    The fixture release publishes different names on purpose, so a call
    without an explicit name refuses. That is exactly how a caller finds
    out they pointed this at a release that is not this dataset, rather
    than at whichever asset happened to sort first.

    The Parquet and provenance assets are fetched and digest-verified
    alongside the history, but nothing here opens them: only the
    six-column CSV has a profile describing what a valid grid looks like.
    The published Parquet is not even dtype-compatible with that profile
    (its price and volume columns are strings), so this is a real gap,
    not a redundant check skipped for speed.
    """
    result = _fetch(build_default_assets(), tmp_path)

    assert PARQUET_ASSET in result.assets

    with pytest.raises(ConfigError, match=BITSTAMP_HISTORY_CSV_ASSET):
        read_snapshot_frame(result)


if __name__ == "__main__":
    pytest.main([__file__])
