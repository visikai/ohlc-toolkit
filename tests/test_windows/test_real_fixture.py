"""The window oracle over a committed slice of real published minute data.

The synthetic families pin behaviour at a scale a human can check by hand.
This module checks that the same behaviour survives twenty thousand real
candles and a window measured in thousands of them, against data nobody
tidied up for the test.
"""

import pytest

from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M
from ohlc_toolkit.windows import (
    ExplicitRange,
    MaterializationRule,
    compute_reference_windows,
)
from tests.test_windows.fixtures import (
    REAL_SLICE_CADENCE_SECONDS,
    REAL_SLICE_END,
    REAL_SLICE_ROW_COUNT,
    REAL_SLICE_START,
    load_real_slice,
)

_HOUR = 3_600

# A schedule-scale window: 2590 minutes is 43h10m, so each window spans
# 2590 source candles.
_WINDOW = "2590m"
_WINDOW_SECONDS = 2_590 * 60
_WINDOW_CANDLES = 2_590

# The first hour boundary at or after ``REAL_SLICE_START + _WINDOW_SECONDS``
# (1787080200), which is the first tick whose window the slice fully
# covers.
_FIRST_COVERED_TICK = 1_787_083_200

# The slice's final close time is itself an exact hour, so it is the last
# tick a 1h emit grid anchored at zero can place at or before the data.
_LAST_TICK = REAL_SLICE_END

_EXPECTED_TICK_COUNT = (_LAST_TICK - _FIRST_COVERED_TICK) // _HOUR + 1

# The tick count that arithmetic works out to, written down so a change in
# the fixture or the schedule cannot quietly redefine "total".
_DOCUMENTED_TICK_COUNT = 293

# One window, verified by hand against the committed CSV's own rows. See
# the docstring of test_a_hand_verified_window_of_the_real_slice for the
# four boundary rows these numbers were read off.
_SPOT_TICK = 1_787_227_200
_SPOT_OPEN_TIME = 1_787_071_800
_SPOT_OPEN = 64_768.44
_SPOT_HIGH = 72_408.08
_SPOT_LOW = 64_112.88
_SPOT_CLOSE = 71_872.32
_SPOT_VOLUME = 7_182.822_791_09

# The lead-in ticks of an explicit range starting at the data's first open:
# no candles, then one hour of them, then two.
_WARMUP_SRC_COUNTS = [0, 60, 120]
_WARMUP_COVERAGE_SECONDS = [0, 3_600, 7_200]


def test_the_committed_slice_is_a_complete_minute_grid() -> None:
    """The fixture is what the fixtures module says it is, or the read fails."""
    frame = load_real_slice()

    assert frame.height == REAL_SLICE_ROW_COUNT
    timestamps = frame.get_column("timestamp")
    assert timestamps[0] == REAL_SLICE_START
    assert timestamps[-1] == REAL_SLICE_END - REAL_SLICE_CADENCE_SECONDS
    # A complete grid holds exactly one candle per cadence step.
    assert frame.height == (REAL_SLICE_END - REAL_SLICE_START) // (
        REAL_SLICE_CADENCE_SECONDS
    )


def test_the_derived_tick_count_is_the_one_expected() -> None:
    """Pin the arithmetic the assertions below lean on."""
    assert _EXPECTED_TICK_COUNT == _DOCUMENTED_TICK_COUNT


def test_a_schedule_scale_window_emits_a_total_fully_covered_grid() -> None:
    """Every hourly tick from the first fully covered one is emitted, and full.

    Because the slice is a complete grid and ``skip_warmup`` starts at the
    first fully covered tick and stops at the last tick inside the data,
    there is no partly covered tick left in the result: all 293 windows
    hold exactly 2590 source candles.
    """
    frame = load_real_slice()

    result = compute_reference_windows(
        frame,
        BITSTAMP_BTCUSD_1M,
        window=_WINDOW,
        emit_every="1h",
        materialization=MaterializationRule.SKIP_WARMUP,
    )

    assert result.height == _EXPECTED_TICK_COUNT
    assert result.get_column("close_time").to_list() == list(
        range(_FIRST_COVERED_TICK, _LAST_TICK + 1, _HOUR)
    )
    assert result.get_column("open_time").to_list() == [
        tick - _WINDOW_SECONDS
        for tick in range(_FIRST_COVERED_TICK, _LAST_TICK + 1, _HOUR)
    ]
    assert result.get_column("coverage_seconds").unique().to_list() == [_WINDOW_SECONDS]
    assert result.get_column("src_count").unique().to_list() == [_WINDOW_CANDLES]
    assert result.get_column("volume").null_count() == 0


def test_a_hand_verified_window_of_the_real_slice() -> None:
    """One window, checked against the fixture's own rows by hand.

    The window is ``[1787071800, 1787227200)``. Its boundary rows, read
    straight out of the committed CSV:

    ==========  ========  ========  ========  ========
    timestamp   open      high      low       close
    ==========  ========  ========  ========  ========
    1787071740  64788.62  64788.62  64768.44  64768.44
    1787071800  64768.44  64768.61  64759.6   64759.61
    1787227140  71860.52  71872.32  71839.71  71872.32
    1787227200  71872.71  71913.71  71872.71  71882.13
    ==========  ========  ========  ========  ========

    The candle opening at 1787071740 closes exactly at the window open, so
    it lies entirely before the window and is excluded whole: were it
    wrongly included, ``open`` would read 64788.62 instead of 64768.44.
    The candle opening at 1787227140 closes exactly at the emit time, so
    it IS included, which is where ``close`` of 71872.32 comes from. The
    candle opening at 1787227200 is still open at the emit time and is
    excluded: were it wrongly included, ``close`` would read 71882.13.

    ``open`` and ``close`` are read straight off those two included
    boundary rows. ``high``, ``low``, ``src_count``, ``coverage_seconds``,
    and ``volume`` are independently recomputed over all 2590 candles in
    the window with a separate polars aggregation (max, min, count, count
    times the 60s cadence, and sum, respectively) -- not with the oracle
    under test -- so this test cannot pass merely because the oracle
    agrees with itself.
    """
    frame = load_real_slice()

    result = compute_reference_windows(
        frame,
        BITSTAMP_BTCUSD_1M,
        window=_WINDOW,
        emit_every="1h",
        materialization=MaterializationRule.SKIP_WARMUP,
    )
    row = result.filter(result.get_column("close_time") == _SPOT_TICK).rows(named=True)[
        0
    ]

    assert row["open_time"] == _SPOT_OPEN_TIME
    assert row["open"] == _SPOT_OPEN
    assert row["high"] == _SPOT_HIGH
    assert row["low"] == _SPOT_LOW
    assert row["close"] == _SPOT_CLOSE
    assert row["src_count"] == _WINDOW_CANDLES
    assert row["coverage_seconds"] == _WINDOW_SECONDS
    # Volume is the one aggregate compared approximately. Summing 2590
    # real float volumes gives a result that depends on the order they are
    # added in; an independent vectorized sum of the same rows lands
    # about 3.6e-12 away from the oracle's row-order accumulation. The
    # tolerance is about float addition, not about the window being fuzzy.
    assert row["volume"] == pytest.approx(_SPOT_VOLUME, rel=1e-12)


def test_an_explicit_range_keeps_the_warmup_that_skip_warmup_drops() -> None:
    """The first hours of the slice are emitted, honestly under-covered.

    Three ticks from the very start of the data: the first window lies
    entirely before it, the second holds one hour of candles, the third
    two. This is the region ``skip_warmup`` exists to remove.
    """
    frame = load_real_slice()

    result = compute_reference_windows(
        frame,
        BITSTAMP_BTCUSD_1M,
        window=_WINDOW,
        emit_every="1h",
        materialization=ExplicitRange(
            start=REAL_SLICE_START, end=REAL_SLICE_START + 3 * _HOUR
        ),
    )

    assert result.get_column("close_time").to_list() == [
        REAL_SLICE_START,
        REAL_SLICE_START + _HOUR,
        REAL_SLICE_START + 2 * _HOUR,
    ]
    # 0, then 60, then 120 one-minute candles.
    assert result.get_column("src_count").to_list() == _WARMUP_SRC_COUNTS
    assert result.get_column("coverage_seconds").to_list() == _WARMUP_COVERAGE_SECONDS
    assert result.get_column("open").to_list()[0] is None
    assert result.get_column("volume").to_list()[0] is None


if __name__ == "__main__":
    pytest.main([__file__])
