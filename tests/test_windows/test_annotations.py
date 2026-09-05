"""The interval-annotation overlap transform, composed after the aggregator.

Every window frame here is genuinely engine-produced -- built with
:func:`~ohlc_toolkit.windows.engine.compute_windows` over the same
hand-written factories the rest of ``tests/test_windows`` uses -- so the
transform is exercised against the real output shape it composes after.
Annotation intervals are placed by arithmetic against the emitted window
bounds, and every expected overlap below is written as a literal worked
out by hand, never recomputed from the code under test.
"""

from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ohlc_toolkit.snapshot import BITSTAMP_BTCUSD_1M_REPOSITORY, SnapshotRelease
from ohlc_toolkit.snapshot.manifest import (
    MANIFEST_ASSET_NAME,
    MAX_MANIFEST_BYTES,
    parse_manifest,
)
from ohlc_toolkit.snapshot.transport import HttpAssetTransport
from ohlc_toolkit.temporal import MAX_ECHO_CHARS, ConfigError
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from ohlc_toolkit.windows import annotations as annotations_module
from ohlc_toolkit.windows.annotations import (
    AnnotationColumns,
    annotate_windows,
    read_annotations,
)
from tests.test_snapshot.factories import PROVENANCE_ASSET, build_default_assets
from tests.test_windows.factories import SourceRow, frame_from_rows, profile_for

# A one-minute source over half an hour, aggregated into five-minute windows
# emitted every minute: every window is [close_time - 300, close_time) on the
# minute grid, so an interval can be placed exactly on either bound.
_CADENCE_SECONDS = 60
_WINDOW = "5m"
_WINDOW_SECONDS = 300
_EMIT_EVERY = "1m"
_SOURCE_MINUTES = 30
# A real-looking Unix second ON THE MINUTE GRID (1_700_000_000 is not: it
# is 40s past a minute), so the epoch-anchored emit ticks line up with the
# candles and no bound is a small number that could appear inside an
# unrelated count.
_TIME_BASE = 1_700_000_040
_ONE_MINUTE = 60

# The first interval of the fixture sidecar in tests/test_snapshot/factories,
# which is also, to the second, the first row of the published Bitstamp
# sidecar: [1362229320, 1362233460), 69 minutes, suspected_outage. Against
# hour windows on the hour grid it starts 120s into the hour beginning at
# 1362229200 and ends 660s into the next, so the two hours it touches see
# 3480s and 660s of it and their neighbours see none.
_SIDECAR_START = 1_362_229_320
_SIDECAR_END = 1_362_233_460
_SIDECAR_FLAG = "suspected_outage"
_HOUR_SECONDS = 3600
_HOUR_GRID_BEFORE = 1_362_225_600
_HOUR_GRID_AFTER = 1_362_240_000
_EXPECTED_HOURLY_OVERLAP = [0, 3480, 660, 0]

# The published release the network test reads, and what its sidecar held
# when these numbers were written down. A later month is a new release and
# would need its own numbers.
_PUBLISHED_TAG = "bitstamp-btcusd-1m-2026-08"
_PUBLISHED_SIDECAR_ASSET = "btcusd_bitstamp_1min_provenance.csv"
_PUBLISHED_SIDECAR_ROWS = 71
_PUBLISHED_SIDECAR_FLAGS = [
    "confirmed_outage",
    "scheduled_maintenance",
    "suspected_outage",
]
_PUBLISHED_SIDECAR_COLUMNS = [
    "start_timestamp",
    "end_timestamp",
    "duration_minutes",
    "flag",
    "price_jump",
    "reference",
]

# Names far longer than any real one, for the refusals that quote a
# caller-chosen name or dtype; and the ceiling those refusals stay under,
# derived from the echo cap because every fix routes through it.
_ENORMOUS_NAME_CHARS = 200_000
_LOUD_NAME = "n" * _ENORMOUS_NAME_CHARS
_PATHOLOGICAL_STRUCT_FIELDS = 1000
_MAX_REFUSAL_CHARS = 6 * MAX_ECHO_CHARS
_DEEP_PATH_COMPONENTS = 14
_DEEP_COMPONENT_CHARS = 250


def _rows(*open_times: int) -> tuple[SourceRow, ...]:
    """Build source rows at the given open times, with distinct OHLCV values."""
    return tuple(
        (_TIME_BASE + open_time, 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i, float(i))
        for i, open_time in enumerate(open_times)
    )


def _windows() -> pl.DataFrame:
    """Five-minute windows every minute over a complete half-hour minute grid."""
    source = frame_from_rows(
        _rows(*range(0, _SOURCE_MINUTES * _CADENCE_SECONDS, _CADENCE_SECONDS))
    )
    return compute_windows(
        source,
        profile_for(_CADENCE_SECONDS),
        window=_WINDOW,
        emit_every=_EMIT_EVERY,
        materialization="skip_warmup",
    )


def _annotations(*intervals: tuple[int, int, str]) -> pl.DataFrame:
    """Build an annotation frame in the sidecar contract's own column names."""
    return pl.DataFrame(
        {
            "start_timestamp": [start for start, _, _ in intervals],
            "end_timestamp": [end for _, end, _ in intervals],
            "flag": [flag for _, _, flag in intervals],
        },
        schema={
            "start_timestamp": pl.Int64,
            "end_timestamp": pl.Int64,
            "flag": pl.String,
        },
    )


def _one_window(frame: pl.DataFrame) -> tuple[int, int]:
    """Return the bounds of a window well inside the frame, as plain ints."""
    row = frame.row(frame.height // 2, named=True)
    return int(row["open_time"]), int(row["close_time"])


def _overlap_of(annotated: pl.DataFrame, close_time: int) -> int:
    """Read one window's overlap seconds back by its close time."""
    matched = annotated.filter(pl.col("close_time") == close_time)
    assert matched.height == 1
    return int(matched.get_column("annotation_overlap_seconds")[0])


def _flags_of(annotated: pl.DataFrame, close_time: int) -> list[str]:
    """Read one window's flags back by its close time."""
    matched = annotated.filter(pl.col("close_time") == close_time)
    assert matched.height == 1
    return list(matched.get_column("annotation_flags")[0])


def test_the_window_grid_is_what_the_arithmetic_below_assumes() -> None:
    """Every emitted window is exactly five minutes on the minute grid."""
    frame = _windows()
    assert frame.height > 0
    assert frame.select(
        (pl.col("close_time") - pl.col("open_time") == _WINDOW_SECONDS).all()
    ).item()
    assert frame.select((pl.col("close_time") % _ONE_MINUTE == 0).all()).item()


@pytest.mark.parametrize(
    ("start_offset", "end_offset", "expected"),
    [
        pytest.param(-_ONE_MINUTE, 0, 0, id="ends exactly at open"),
        pytest.param(
            _WINDOW_SECONDS,
            _WINDOW_SECONDS + _ONE_MINUTE,
            0,
            id="starts exactly at close",
        ),
        pytest.param(0, _WINDOW_SECONDS, _WINDOW_SECONDS, id="exactly the window"),
        pytest.param(-_ONE_MINUTE, _ONE_MINUTE, _ONE_MINUTE, id="straddles open"),
        pytest.param(
            _WINDOW_SECONDS - _ONE_MINUTE,
            _WINDOW_SECONDS + _ONE_MINUTE,
            _ONE_MINUTE,
            id="straddles close",
        ),
        pytest.param(_ONE_MINUTE, 2 * _ONE_MINUTE, _ONE_MINUTE, id="strictly inside"),
        pytest.param(
            -_ONE_MINUTE,
            _WINDOW_SECONDS + _ONE_MINUTE,
            _WINDOW_SECONDS,
            id="contains the window",
        ),
    ],
)
def test_overlap_is_half_open_on_both_sides(
    start_offset: int, end_offset: int, expected: int
) -> None:
    """An interval touching only a bound touches nothing; anything inside counts."""
    frame = _windows()
    open_time, close_time = _one_window(frame)
    annotated = annotate_windows(
        frame, _annotations((open_time + start_offset, open_time + end_offset, "x"))
    )
    assert _overlap_of(annotated, close_time) == expected
    assert _flags_of(annotated, close_time) == (["x"] if expected else [])


def test_overlapping_intervals_count_their_shared_seconds_once() -> None:
    """The overlap column measures the union, and the flags list every distinct flag."""
    frame = _windows()
    open_time, close_time = _one_window(frame)
    annotated = annotate_windows(
        frame,
        _annotations(
            (open_time, open_time + 2 * _ONE_MINUTE, "b"),
            (open_time + _ONE_MINUTE, open_time + 3 * _ONE_MINUTE, "a"),
            (open_time + _ONE_MINUTE, open_time + 2 * _ONE_MINUTE, "a"),
        ),
    )
    assert _overlap_of(annotated, close_time) == 3 * _ONE_MINUTE
    assert _flags_of(annotated, close_time) == ["a", "b"]


def test_the_overlap_never_exceeds_the_window_length() -> None:
    """However the sidecar repeats itself, no window is more than fully covered."""
    frame = _windows()
    first_open = int(frame.select(pl.col("open_time").min()).item())
    last_close = int(frame.select(pl.col("close_time").max()).item())
    blanket = _annotations(
        (first_open - _ONE_MINUTE, last_close + _ONE_MINUTE, "a"),
        (first_open, last_close, "b"),
        (first_open + _ONE_MINUTE, last_close - _ONE_MINUTE, "a"),
    )
    annotated = annotate_windows(frame, blanket)
    assert annotated.select(
        (pl.col("annotation_overlap_seconds") == _WINDOW_SECONDS).all()
    ).item()
    assert annotated.select((pl.col("annotation_flags") == ["a", "b"]).all()).item()


def test_the_input_columns_come_back_byte_for_byte() -> None:
    """Only two columns are added; nothing the engine wrote is read or changed."""
    frame = _windows()
    open_time, _ = _one_window(frame)
    annotated = annotate_windows(
        frame, _annotations((open_time, open_time + _ONE_MINUTE, "x"))
    )
    assert annotated.columns == [
        *frame.columns,
        "annotation_flags",
        "annotation_overlap_seconds",
    ]
    assert_frame_equal(annotated.select(frame.columns), frame)
    assert annotated.schema["annotation_flags"] == pl.List(pl.String)
    assert annotated.schema["annotation_overlap_seconds"] == pl.Int64


def test_an_empty_sidecar_annotates_nothing_and_still_adds_both_columns() -> None:
    """Zero intervals is a legitimate sidecar: every window gets [] and 0."""
    frame = _windows()
    annotated = annotate_windows(frame, _annotations())
    assert annotated.schema["annotation_flags"] == pl.List(pl.String)
    assert annotated.schema["annotation_overlap_seconds"] == pl.Int64
    assert annotated.select(pl.col("annotation_flags").list.len().sum()).item() == 0
    assert annotated.select(pl.col("annotation_overlap_seconds").sum()).item() == 0
    assert_frame_equal(annotated.select(frame.columns), frame)


def test_the_prefix_and_the_column_names_are_the_callers() -> None:
    """Another layout and another prefix are honoured without renaming anything."""
    frame = _windows()
    open_time, close_time = _one_window(frame)
    sidecar = pl.DataFrame(
        {
            "from": [open_time],
            "to": [open_time + _ONE_MINUTE],
            "kind": ["x"],
            "note": ["kept but unread"],
        },
        schema={"from": pl.Int64, "to": pl.Int64, "kind": pl.String, "note": pl.String},
    )
    annotated = annotate_windows(
        frame,
        sidecar,
        columns=AnnotationColumns(start="from", end="to", flag="kind"),
        prefix="outage",
    )
    assert annotated.columns[-2:] == ["outage_flags", "outage_overlap_seconds"]
    matched = annotated.filter(pl.col("close_time") == close_time)
    assert int(matched.get_column("outage_overlap_seconds")[0]) == _ONE_MINUTE
    assert list(matched.get_column("outage_flags")[0]) == ["x"]


def _hour_windows() -> pl.DataFrame:
    """Hour windows on the hour grid across the fixture sidecar's first interval."""
    source = frame_from_rows(
        tuple(
            (open_time, 1.0, 1.0, 1.0, 1.0, 1.0)
            for open_time in range(
                _HOUR_GRID_BEFORE - _HOUR_SECONDS, _HOUR_GRID_AFTER, _CADENCE_SECONDS
            )
        )
    )
    return compute_windows(
        source,
        profile_for(_CADENCE_SECONDS),
        window="1h",
        emit_every="1h",
        materialization=ExplicitRange(
            start=_HOUR_GRID_BEFORE + _HOUR_SECONDS, end=_HOUR_GRID_AFTER + 1
        ),
    )


def test_the_fixture_sidecar_reads_and_annotates_hour_windows_as_worked_by_hand(
    tmp_path: Path,
) -> None:
    """The fixture sidecar carries the published file's first row and lands where arithmetic says."""
    path = tmp_path / PROVENANCE_ASSET
    path.write_bytes(build_default_assets()[PROVENANCE_ASSET])
    sidecar = read_annotations(path)
    assert sidecar.height == 1
    assert sidecar.row(0, named=True)["start_timestamp"] == _SIDECAR_START
    assert sidecar.row(0, named=True)["end_timestamp"] == _SIDECAR_END
    assert sidecar.row(0, named=True)["flag"] == _SIDECAR_FLAG
    assert "duration_minutes" in sidecar.columns
    annotated = annotate_windows(_hour_windows(), sidecar)
    assert annotated.get_column("close_time").to_list() == [
        _HOUR_GRID_BEFORE + (i + 1) * _HOUR_SECONDS for i in range(4)
    ]
    assert annotated.get_column("annotation_overlap_seconds").to_list() == (
        _EXPECTED_HOURLY_OVERLAP
    )
    assert annotated.get_column("annotation_flags").to_list() == [
        [] if seconds == 0 else [_SIDECAR_FLAG] for seconds in _EXPECTED_HOURLY_OVERLAP
    ]


# Refusals: what the transform will not do, and how loudly it says so.


def test_a_frame_without_window_bounds_is_refused() -> None:
    """Anything but an engine-shaped frame is refused before any join happens."""
    with pytest.raises(ConfigError, match="open_time"):
        annotate_windows(_windows().drop("open_time"), _annotations())


def test_a_null_interval_bound_is_refused() -> None:
    """A null start, end or flag is a broken sidecar, not an empty annotation."""
    sidecar = pl.DataFrame(
        {"start_timestamp": [None], "end_timestamp": [1], "flag": ["x"]},
        schema={
            "start_timestamp": pl.Int64,
            "end_timestamp": pl.Int64,
            "flag": pl.String,
        },
    )
    with pytest.raises(ConfigError, match="null"):
        annotate_windows(_windows(), sidecar)


@pytest.mark.parametrize(
    "end", [_TIME_BASE, _TIME_BASE - _ONE_MINUTE], ids=["empty", "inverted"]
)
def test_an_interval_whose_end_does_not_exceed_its_start_is_refused(end: int) -> None:
    """A half-open interval with nothing in it cannot mean anything, so it is refused."""
    with pytest.raises(ConfigError, match="start < end"):
        annotate_windows(_windows(), _annotations((_TIME_BASE, end, "x")))


def test_an_empty_prefix_is_refused() -> None:
    """The two appended columns need a stem to be named by."""
    with pytest.raises(ConfigError, match="prefix"):
        annotate_windows(_windows(), _annotations(), prefix="")


def test_an_empty_column_name_is_refused() -> None:
    """A layout cannot name a column with nothing."""
    with pytest.raises(ConfigError, match="empty"):
        AnnotationColumns(start="")


def _both_exits_bounded(trip: object, *, level: str = "WARNING") -> None:
    """Run ``trip`` expecting ConfigError; hold the message and the log under the ceiling."""
    logged: list[str] = []
    sink_id = annotations_module.logger.add(
        logged.append, level=level, format="{message}"
    )
    try:
        with pytest.raises(ConfigError) as raised:
            trip()  # type: ignore[operator]
    finally:
        annotations_module.logger.remove(sink_id)
    assert len(str(raised.value)) < _MAX_REFUSAL_CHARS
    assert logged, "the refusal logs before it raises; nothing was captured"
    assert len(logged[-1]) < _MAX_REFUSAL_CHARS


def _wide_struct(length: int) -> pl.Series:
    """Build a column whose dtype renders to thousands of characters."""
    return pl.Series(
        [{f"field_{i}": 0 for i in range(_PATHOLOGICAL_STRUCT_FIELDS)}] * length
    )


def test_a_missing_annotation_column_is_named_bounded_on_both_exits() -> None:
    """The layout's names are the caller's, so a missing one is echoed bounded."""
    _both_exits_bounded(
        lambda: annotate_windows(
            _windows(), _annotations(), columns=AnnotationColumns(start=_LOUD_NAME)
        )
    )


def test_a_wrong_kind_annotation_column_is_named_bounded_on_both_exits() -> None:
    """Both the caller's column name and its dtype are echoed bounded."""
    wide = _annotations((_TIME_BASE, _TIME_BASE + 1, "x")).with_columns(
        _wide_struct(1).alias(_LOUD_NAME)
    )
    _both_exits_bounded(
        lambda: annotate_windows(
            _windows(), wide, columns=AnnotationColumns(flag=_LOUD_NAME)
        )
    )


def test_a_wrong_kind_window_bound_is_named_bounded_on_both_exits() -> None:
    """A window bound of a wide dtype is refused with that dtype echoed bounded."""
    frame = _windows()
    wide = frame.with_columns(_wide_struct(frame.height).alias("close_time"))
    _both_exits_bounded(lambda: annotate_windows(wide, _annotations()))


def test_an_output_collision_is_named_bounded_on_both_exits() -> None:
    """A prefix whose columns already exist is refused with those names bounded."""
    frame = _windows().with_columns(pl.lit(0).alias(f"{_LOUD_NAME}_overlap_seconds"))
    _both_exits_bounded(
        lambda: annotate_windows(frame, _annotations(), prefix=_LOUD_NAME)
    )


def test_repeated_column_names_are_refused_bounded_on_both_exits() -> None:
    """Three roles need three names; a repeat is refused with the names bounded."""
    _both_exits_bounded(lambda: AnnotationColumns(start=_LOUD_NAME, end=_LOUD_NAME))


def test_reading_a_missing_file_is_refused_bounded_on_both_exits(
    tmp_path: Path,
) -> None:
    """The path is the caller's, and a very long one is echoed bounded."""
    deep = tmp_path.joinpath(*["p" * _DEEP_COMPONENT_CHARS] * _DEEP_PATH_COMPONENTS)
    logged: list[str] = []
    sink_id = annotations_module.logger.add(
        logged.append, level="ERROR", format="{message}"
    )
    try:
        with pytest.raises(FileNotFoundError) as raised:
            read_annotations(deep / "sidecar.csv")
    finally:
        annotations_module.logger.remove(sink_id)
    assert len(str(raised.value)) < _MAX_REFUSAL_CHARS
    assert logged, "the refusal logs before it raises; nothing was captured"
    assert len(logged[-1]) < _MAX_REFUSAL_CHARS


def test_reading_a_malformed_cell_is_refused_bounded_on_both_exits(
    tmp_path: Path,
) -> None:
    """The parser's own error text embeds the cell; it reaches the caller bounded."""
    path = tmp_path / "sidecar.csv"
    path.write_text(
        f"start_timestamp,end_timestamp,flag\n{'x' * _ENORMOUS_NAME_CHARS},2,a\n"
    )
    _both_exits_bounded(lambda: read_annotations(path), level="ERROR")


def test_reading_a_file_without_a_named_column_is_refused(tmp_path: Path) -> None:
    """A sidecar in another layout is refused unless its names are given."""
    path = tmp_path / "sidecar.csv"
    path.write_text("from,to,kind\n1,2,a\n")
    with pytest.raises(ConfigError, match="start_timestamp"):
        read_annotations(path)
    assert (
        read_annotations(
            path, columns=AnnotationColumns(start="from", end="to", flag="kind")
        ).height
        == 1
    )


@pytest.mark.network
def test_the_published_sidecar_reads_and_annotates_hour_windows(tmp_path: Path) -> None:
    """The real Bitstamp sidecar is read whole and lands where the fixture row did.

    Deselected by default like the rest of the network lane. The assertions
    about the file -- its columns, its row count, its flag vocabulary, that
    its first interval is the one the fixture copies -- are about the
    world, read off the published release rather than off this code.
    """
    release = SnapshotRelease(
        repository=BITSTAMP_BTCUSD_1M_REPOSITORY, tag=_PUBLISHED_TAG
    )
    transport = HttpAssetTransport()
    manifest_path = tmp_path / MANIFEST_ASSET_NAME
    transport.download(
        release.asset_url(MANIFEST_ASSET_NAME),
        manifest_path,
        max_bytes=MAX_MANIFEST_BYTES,
    )
    record = parse_manifest(manifest_path.read_bytes()).assets[_PUBLISHED_SIDECAR_ASSET]
    sidecar_path = tmp_path / _PUBLISHED_SIDECAR_ASSET
    transport.download(
        release.asset_url(_PUBLISHED_SIDECAR_ASSET),
        sidecar_path,
        max_bytes=record.size_bytes,
    )

    sidecar = read_annotations(sidecar_path)
    assert sidecar.columns == _PUBLISHED_SIDECAR_COLUMNS
    assert sidecar.height == _PUBLISHED_SIDECAR_ROWS
    assert (
        sorted(sidecar.get_column("flag").unique().to_list())
        == _PUBLISHED_SIDECAR_FLAGS
    )
    assert sidecar.get_column("start_timestamp").is_sorted()
    first = sidecar.row(0, named=True)
    assert (first["start_timestamp"], first["end_timestamp"], first["flag"]) == (
        _SIDECAR_START,
        _SIDECAR_END,
        _SIDECAR_FLAG,
    )

    annotated = annotate_windows(_hour_windows(), sidecar)
    assert annotated.get_column("annotation_overlap_seconds").to_list() == (
        _EXPECTED_HOURLY_OVERLAP
    )
