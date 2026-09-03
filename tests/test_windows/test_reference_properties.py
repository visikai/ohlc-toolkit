"""Property-based invariant suites for the brute-force window oracle.

Each suite re-derives what it checks from the specification, in the test,
with a second deliberately dumb implementation: the emit grid is found by
walking every integer second in the materialization range, and window
membership is decided by re-reading the two boundary inequalities against
the raw rows. Nothing here imports the oracle's own helpers, so a shared
mistake cannot cancel itself out.
"""

from dataclasses import dataclass
from datetime import timedelta
from itertools import pairwise

import polars as pl
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.windows import ExplicitRange, compute_reference_windows
from tests.test_windows.factories import SourceRow, frame_from_rows, profile_for

# A small fixed set of cadences rather than an unbounded search space: the
# oracle is cadence-agnostic, so these exercise the second-level, the
# minute-level, and the coarse-multiple arithmetic without inflating run
# time.
_CADENCE_CHOICES = (1, 5, 60, 300)

# The oracle is brute force by design, so a single example does real work.
# The deadline stays explicit (rather than disabled) to keep catching a
# pathological blow-up, but generous enough to survive a loaded machine.
_SETTINGS = settings(deadline=timedelta(seconds=2))


@dataclass(frozen=True)
class _Scenario:
    """One randomly drawn source frame plus a resolvable schedule over it."""

    profile: SourceProfile
    rows: tuple[SourceRow, ...]
    frame: pl.DataFrame
    window_seconds: int
    emit_seconds: int
    anchor_seconds: int
    range_start: int
    range_end: int


def _draw_scenario(
    data: st.DataObject, *, align_emit_to_window: bool = False
) -> _Scenario:
    """Draw a source frame and a schedule that satisfies every strict rule.

    The declared phase is drawn first and the grid is built from it, so the
    frame is always on the phase the profile declares. Window, emit
    cadence, and anchor are all drawn as whole multiples of the cadence
    (the anchor offset by the phase), which is exactly the region the
    strict resolution rules admit.

    Args:
        data: The hypothesis data object to draw from.
        align_emit_to_window: When True, force ``E == W``, the aligned
            tiling case.

    Returns:
        The drawn scenario.

    """
    cadence = data.draw(st.sampled_from(_CADENCE_CHOICES), label="cadence_seconds")
    phase = data.draw(
        st.integers(min_value=0, max_value=cadence - 1), label="phase_seconds"
    )
    # Start at least three cadence steps up the grid so the drawn
    # materialization range can reach below the data and stay non-negative.
    grid_index = data.draw(st.integers(min_value=3, max_value=40), label="grid_index")
    first_open = phase + cadence * grid_index

    slots = data.draw(
        st.lists(
            st.tuples(st.booleans(), st.integers(min_value=-8, max_value=8)),
            min_size=1,
            max_size=24,
        ).filter(lambda drawn: any(is_present for is_present, _ in drawn)),
        label="slots",
    )
    rows = tuple(
        (
            first_open + index * cadence,
            100.0 + jitter,
            110.0 + jitter,
            90.0 + jitter,
            105.0 + jitter,
            float(index + 1),
        )
        for index, (is_present, jitter) in enumerate(slots)
        if is_present
    )

    window_multiple = data.draw(
        st.integers(min_value=1, max_value=6), label="window_multiple"
    )
    emit_multiple = (
        window_multiple
        if align_emit_to_window
        else data.draw(st.integers(min_value=1, max_value=6), label="emit_multiple")
    )
    anchor_multiple = data.draw(
        st.integers(min_value=0, max_value=6), label="anchor_multiple"
    )
    lead = data.draw(st.integers(min_value=0, max_value=3), label="lead_steps")
    trail = data.draw(st.integers(min_value=0, max_value=3), label="trail_steps")

    last_close = rows[-1][0] + cadence
    return _Scenario(
        profile=profile_for(cadence, phase_seconds=phase),
        rows=rows,
        frame=frame_from_rows(rows),
        window_seconds=window_multiple * cadence,
        emit_seconds=emit_multiple * cadence,
        anchor_seconds=phase + anchor_multiple * cadence,
        range_start=first_open - lead * cadence,
        range_end=last_close + trail * cadence + 1,
    )


def _run(scenario: _Scenario, frame: pl.DataFrame | None = None) -> pl.DataFrame:
    """Run the oracle over a scenario, optionally against a replaced frame."""
    return compute_reference_windows(
        scenario.frame if frame is None else frame,
        scenario.profile,
        window=f"{scenario.window_seconds}s",
        emit_every=f"{scenario.emit_seconds}s",
        anchor=f"{scenario.anchor_seconds}s",
        materialization=ExplicitRange(
            start=scenario.range_start, end=scenario.range_end
        ),
    )


def _brute_force_ticks(scenario: _Scenario) -> list[int]:
    """Find the emit grid by testing every single second in the range.

    This is the definition of the grid read literally: the set of instants
    ``t`` in ``[start, end)`` with ``(t - anchor) mod E == 0``. It is far
    too slow for production use, which is exactly why it makes a good
    independent check on the oracle's arithmetic.
    """
    return [
        tick
        for tick in range(scenario.range_start, scenario.range_end)
        if (tick - scenario.anchor_seconds) % scenario.emit_seconds == 0
    ]


def _naive_included(scenario: _Scenario, tick: int) -> list[SourceRow]:
    """Re-read the inclusion rule against the raw rows, one row at a time."""
    cadence = scenario.profile.cadence.total_seconds
    window_open = tick - scenario.window_seconds
    included = []
    for row in scenario.rows:
        open_time = row[0]
        close_time = row[0] + cadence
        if open_time >= window_open and close_time <= tick:
            included.append(row)
    return included


def _naive_row(scenario: _Scenario, tick: int) -> tuple[object, ...]:
    """Build the whole expected output row for one tick, the dumb way."""
    cadence = scenario.profile.cadence.total_seconds
    included = _naive_included(scenario, tick)
    if not included:
        return (
            tick - scenario.window_seconds,
            tick,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
        )

    earliest = min(included, key=lambda row: row[0])
    latest = max(included, key=lambda row: row[0])
    return (
        tick - scenario.window_seconds,
        tick,
        earliest[1],
        max(row[2] for row in included),
        min(row[3] for row in included),
        latest[4],
        sum(row[5] for row in included),
        len(included),
        cadence * len(included),
    )


@_SETTINGS
@given(data=st.data())
def test_every_row_matches_an_independent_recomputation(data: st.DataObject) -> None:
    """The oracle agrees, cell for cell, with a second dumb implementation."""
    scenario = _draw_scenario(data)
    result = _run(scenario)

    expected = [_naive_row(scenario, tick) for tick in _brute_force_ticks(scenario)]
    assert result.rows() == expected


@_SETTINGS
@given(data=st.data())
def test_only_candles_inside_both_boundaries_are_counted(
    data: st.DataObject,
) -> None:
    """src_count counts exactly the candles passing BOTH boundary inequalities.

    The close-time test alone is not enough, and neither is the open-time
    test alone: both single-sided counts are computed here, and the oracle
    must match the two-sided one.
    """
    scenario = _draw_scenario(data)
    cadence = scenario.profile.cadence.total_seconds
    result = _run(scenario)

    for open_time, close_time, src_count in result.select(
        "open_time", "close_time", "src_count"
    ).iter_rows():
        both_sides = [
            row
            for row in scenario.rows
            if row[0] >= open_time and row[0] + cadence <= close_time
        ]
        assert src_count == len(both_sides)
        for row in both_sides:
            assert row[0] >= open_time
            assert row[0] + cadence <= close_time


@_SETTINGS
@given(data=st.data())
def test_a_tick_never_depends_on_a_candle_that_closes_after_it(
    data: st.DataObject,
) -> None:
    """Causality: deleting every candle that closes after t cannot change t's row."""
    scenario = _draw_scenario(data)
    cadence = scenario.profile.cadence.total_seconds
    full = _run(scenario)
    assume(full.height > 0)

    tick_index = data.draw(
        st.integers(min_value=0, max_value=full.height - 1), label="tick_index"
    )
    tick = full.get_column("close_time")[tick_index]

    past_only = frame_from_rows(
        [row for row in scenario.rows if row[0] + cadence <= tick]
    )
    truncated = _run(scenario, frame=past_only)

    assert truncated.height == full.height
    assert truncated.row(tick_index) == full.row(tick_index)


@_SETTINGS
@given(data=st.data())
def test_the_emitted_grid_is_total_and_uniformly_spaced(
    data: st.DataObject,
) -> None:
    """One row per grid tick in the range, ascending by E, with open = close - W."""
    scenario = _draw_scenario(data)
    result = _run(scenario)

    expected_ticks = _brute_force_ticks(scenario)
    close_times = result.get_column("close_time").to_list()
    open_times = result.get_column("open_time").to_list()

    assert result.height == len(expected_ticks)
    assert close_times == expected_ticks
    for previous, following in pairwise(close_times):
        assert following - previous == scenario.emit_seconds
    for open_time, close_time in zip(open_times, close_times, strict=True):
        assert open_time == close_time - scenario.window_seconds


@_SETTINGS
@given(data=st.data())
def test_coverage_seconds_is_the_exact_sum_of_included_durations(
    data: st.DataObject,
) -> None:
    """Coverage is arithmetic, never an estimate, and never exceeds the window."""
    scenario = _draw_scenario(data)
    cadence = scenario.profile.cadence.total_seconds
    result = _run(scenario)

    for tick, coverage, src_count in result.select(
        "close_time", "coverage_seconds", "src_count"
    ).iter_rows():
        included = _naive_included(scenario, tick)
        assert coverage == sum(cadence for _ in included)
        assert coverage == cadence * src_count
        assert 0 <= coverage <= scenario.window_seconds


@_SETTINGS
@given(data=st.data())
def test_price_and_volume_are_null_exactly_when_no_candle_was_included(
    data: st.DataObject,
) -> None:
    """Absence of data is null everywhere, including volume; never a zero price."""
    scenario = _draw_scenario(data)
    result = _run(scenario)

    for row in result.iter_rows(named=True):
        is_empty = row["src_count"] == 0
        for column in ("open", "high", "low", "close", "volume"):
            assert (row[column] is None) is is_empty
        if is_empty:
            assert row["coverage_seconds"] == 0


@_SETTINGS
@given(data=st.data())
def test_aligned_windows_assign_each_interior_candle_to_exactly_one_window(
    data: st.DataObject,
) -> None:
    """At E == W the emitted windows tile their span with no overlap and no hole."""
    scenario = _draw_scenario(data, align_emit_to_window=True)
    cadence = scenario.profile.cadence.total_seconds
    result = _run(scenario)
    assume(result.height > 0)

    windows = list(result.select("open_time", "close_time").iter_rows())
    span_open = windows[0][0]
    span_close = windows[-1][1]

    for row in scenario.rows:
        open_time = row[0]
        close_time = row[0] + cadence
        if open_time < span_open or close_time > span_close:
            continue
        matches = [
            (window_open, window_close)
            for window_open, window_close in windows
            if open_time >= window_open and close_time <= window_close
        ]
        assert len(matches) == 1


@_SETTINGS
@given(data=st.data())
def test_the_output_schema_never_varies(data: st.DataObject) -> None:
    """Column names, order, and dtypes are identical for every input."""
    scenario = _draw_scenario(data)
    result = _run(scenario)

    assert result.columns == [
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "src_count",
        "coverage_seconds",
    ]
    assert result.dtypes == [
        pl.Int64,
        pl.Int64,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.Float64,
        pl.UInt32,
        pl.Int64,
    ]


if __name__ == "__main__":
    pytest.main([__file__])
