"""The Polars window engine, measured against the brute-force oracle.

The oracle in :mod:`ohlc_toolkit.windows.reference` is the normative
statement of the window contract. This module asserts that the fast engine
computes the same thing: the same rows, in the same order, with the same
dtypes, and -- everywhere the arithmetic allows it -- the same bits.

Exactness, and where it stops
-----------------------------

``open_time``, ``close_time``, ``src_count`` and ``coverage_seconds`` are
integer arithmetic, so they are compared exactly, always. ``open``,
``high``, ``low`` and ``close`` are a selection, a maximum and a minimum
of values that already exist in the input -- no arithmetic is performed
on them at all -- so they are compared exactly too.

``volume`` is the one sum, and the two implementations add the same
addends in different orders: the oracle folds the included volumes left
to right in the frame's own row order, one addend at a time, while the
engine hands the window's contiguous slice to polars, which folds it in
blocks. Both look only at the candles the window contains and neither
carries a term in from the rows before them, so they can differ only in
where the rounding lands.

Where every partial sum is exactly representable they cannot differ at
all. Every synthetic family draws volumes on a quarter-unit grid, which
is exactly that case, so those comparisons stay exact. Real volumes are
not on such a grid, so the real-data fixture is the one place a tolerance
is needed -- and it is derived from the oracle's own fold where it is
defined below, not picked to make a test pass.

Error parity counts as equivalence
----------------------------------

A schedule the oracle refuses to resolve must be refused by the engine
too, with the same message. Every matrix case below therefore compares
outcomes, not just frames: either both implementations raise the same
:class:`~ohlc_toolkit.temporal.ConfigError`, or neither does and the
frames match.
"""

from dataclasses import dataclass

import polars as pl
import pytest
from polars.testing import assert_frame_equal, assert_series_equal

from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.windows import (
    ExplicitRange,
    Materialization,
    MaterializationRule,
    compute_reference_windows,
    compute_windows,
)
from tests.test_windows.factories import (
    SourceRow,
    WindowRow,
    expected_frame,
    frame_from_rows,
    profile_for,
)
from tests.test_windows.fixtures import REAL_SLICE_CADENCE_SECONDS, load_real_slice
from tests.test_windows.synthetic import (
    FAMILY_NAMES,
    GOLDEN_CASES,
    GoldenCase,
    SyntheticFamily,
    build_family,
)

# (window, emit_every, anchor) triples, grouped by the cadence and phase
# they are legal for. Each group spans aligned tiling (E == W),
# overlapping emission (E < W), sparse emission (E > W), and a nonzero
# anchor, because those are the four shapes the emit grid can take.
_MINUTE_SCHEDULES = (
    ("1m", "1m", "0s"),
    ("5m", "1m", "0s"),
    ("5m", "5m", "0s"),
    ("3m", "2m", "1m"),
    ("10m", "5m", "0s"),
    ("4m", "2m", "0s"),
    ("20m", "20m", "0s"),
)
_SECOND_SCHEDULES = (
    ("1s", "1s", "0s"),
    ("10s", "1s", "0s"),
    ("5s", "5s", "3s"),
    ("7s", "3s", "2s"),
)
# A source at phase 7s only admits an emit grid that shares that phase.
_PHASED_SCHEDULES = (
    ("1m", "1m", "7s"),
    ("4m", "2m", "7s"),
    ("5m", "5m", "1m7s"),
)

_SCHEDULES_BY_FAMILY: dict[str, tuple[tuple[str, str, str], ...]] = {
    "complete_grid_1m": _MINUTE_SCHEDULES,
    "single_gap_1m": _MINUTE_SCHEDULES,
    "multi_gap_1m": _MINUTE_SCHEDULES,
    "straddling_1m": _MINUTE_SCHEDULES,
    "complete_grid_1s": _SECOND_SCHEDULES,
    "phased_grid_1m": _PHASED_SCHEDULES,
}

# How each case asks for its materialization range. ``full_span`` covers
# the data exactly, ``over_run`` reaches three emit steps past it at both
# ends so leading and trailing empty windows are exercised, and ``empty``
# is the legal caller-stated empty range.
_RANGE_KINDS = ("skip_warmup", "full_span", "over_run", "empty")

# How far an ``over_run`` range reaches past the data, in emit steps.
_OVER_RUN_STEPS = 3

# The gap frame in test_every_tick_over_a_gap_emits_exactly_one_row spans
# fourteen one-minute emit ticks, and only four of them (60, 120, 780 and
# 840) close a window that holds a candle. The other ten fall in the hole.
_GAP_TICK_COUNT = 14
_GAP_EMPTY_TICK_COUNT = 10


@dataclass(frozen=True)
class _MatrixCase:
    """One (family, schedule, materialization kind) combination."""

    family: str
    window: str
    emit_every: str
    anchor: str
    range_kind: str

    @property
    def label(self) -> str:
        """Name this case for a pytest parameter id."""
        return (
            f"{self.family}-{self.window}-every-{self.emit_every}"
            f"-anchor-{self.anchor}-{self.range_kind}"
        )


_MATRIX: tuple[_MatrixCase, ...] = tuple(
    _MatrixCase(
        family=family,
        window=window,
        emit_every=emit_every,
        anchor=anchor,
        range_kind=range_kind,
    )
    for family, schedules in _SCHEDULES_BY_FAMILY.items()
    for window, emit_every, anchor in schedules
    for range_kind in _RANGE_KINDS
)


def _span(family: SyntheticFamily) -> tuple[int, int]:
    """Return the family's first open time and its last close time."""
    timestamps = family.frame.get_column("timestamp")
    cadence_seconds = family.profile.cadence.total_seconds
    return int(timestamps.min()), int(timestamps.max()) + cadence_seconds  # type: ignore[arg-type]


def _materialization(case: _MatrixCase, family: SyntheticFamily) -> Materialization:
    """Build the materialization argument this case asks for."""
    if case.range_kind == "skip_warmup":
        return MaterializationRule.SKIP_WARMUP

    first_open, last_close = _span(family)
    if case.range_kind == "empty":
        return ExplicitRange(start=first_open, end=first_open)
    if case.range_kind == "full_span":
        return ExplicitRange(start=first_open, end=last_close + 1)

    emit_seconds = Duration.parse(case.emit_every).total_seconds
    return ExplicitRange(
        start=first_open - _OVER_RUN_STEPS * emit_seconds,
        end=last_close + _OVER_RUN_STEPS * emit_seconds + 1,
    )


def _outcome(
    case: _MatrixCase, family: SyntheticFamily, *, use_engine: bool
) -> tuple[str | None, pl.DataFrame | None]:
    """Run one implementation, returning either its error text or its frame."""
    compute = compute_windows if use_engine else compute_reference_windows
    try:
        frame = compute(
            family.frame,
            family.profile,
            window=case.window,
            emit_every=case.emit_every,
            anchor=case.anchor,
            materialization=_materialization(case, family),
        )
    except ConfigError as error:
        return str(error), None
    return None, frame


@pytest.mark.parametrize("case", _MATRIX, ids=lambda case: case.label)
def test_the_engine_agrees_with_the_oracle_across_the_family_matrix(
    case: _MatrixCase,
) -> None:
    """Same rows, same dtypes, same bits -- or the same refusal."""
    family = build_family(case.family)
    reference_error, reference_frame = _outcome(case, family, use_engine=False)
    engine_error, engine_frame = _outcome(case, family, use_engine=True)

    assert engine_error == reference_error
    if reference_frame is None:
        return
    assert engine_frame is not None
    assert_frame_equal(
        engine_frame,
        reference_frame,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


def test_the_matrix_exercises_every_synthetic_family() -> None:
    """A family with no matrix case is a family nothing is comparing."""
    assert {case.family for case in _MATRIX} == set(FAMILY_NAMES)


def test_the_matrix_reaches_both_materialization_forms() -> None:
    """Both the named rule and an explicit range are covered, or the matrix lies."""
    assert {case.range_kind for case in _MATRIX} == set(_RANGE_KINDS)


def _render_engine_csv(case: GoldenCase) -> str:
    """Render one committed golden case with the engine instead of the oracle."""
    family = build_family(case.family)
    return compute_windows(
        family.frame,
        family.profile,
        window=case.window,
        emit_every=case.emit_every,
        anchor=case.anchor,
        materialization=case.materialization,
    ).write_csv()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=lambda case: case.label)
def test_the_engine_reproduces_each_committed_golden_byte_for_byte(
    case: GoldenCase,
) -> None:
    """The engine writes the same CSV the oracle's committed golden holds.

    Comparing serialized bytes rather than frames is deliberate: it pins
    the dtypes, the column order, the null pattern and the exact decimal
    rendering of every float all at once, against a file that was
    reviewed by a human.
    """
    assert case.path.exists(), f"missing golden file {case.path}"
    assert _render_engine_csv(case).encode("utf-8") == case.path.read_bytes()


def test_every_tick_over_a_gap_emits_exactly_one_row() -> None:
    """The grid is total: gaps produce empty rows, never missing rows.

    The frame below holds two minutes of candles, then a ten-minute hole,
    then two more. Every one of the emit ticks in the requested range must
    come back, and the ticks whose window falls inside the hole must come
    back null-priced with a zero count and zero coverage.
    """
    rows = tuple(
        (open_time, 100.0, 110.0, 90.0, 105.0, 1.0) for open_time in (0, 60, 720, 780)
    )
    result = compute_windows(
        frame_from_rows(rows),
        profile_for(60),
        window="1m",
        emit_every="1m",
        materialization=ExplicitRange(start=60, end=900),
    )

    assert result.height == _GAP_TICK_COUNT
    assert result.get_column("close_time").to_list() == list(range(60, 900, 60))
    empty = result.filter(pl.col("src_count") == 0)
    assert empty.height == _GAP_EMPTY_TICK_COUNT
    assert empty.get_column("coverage_seconds").unique().to_list() == [0]
    for column in ("open", "high", "low", "close", "volume"):
        assert empty.get_column(column).null_count() == empty.height


def test_an_explicit_range_over_an_empty_frame_still_emits_its_grid() -> None:
    """No candles at all is data, not an error, when the caller stated the range.

    An empty frame is the degenerate end of the same rule the gap case
    above exercises: the emit grid is total, so every tick in the stated
    range comes back, null-priced. Only ``skip_warmup`` refuses an empty
    frame, and it refuses it because nobody asked for empty.
    """
    arguments = {
        "window": "5m",
        "emit_every": "1m",
        "materialization": ExplicitRange(start=0, end=300),
    }
    empty_frame = frame_from_rows(())

    expected = compute_reference_windows(empty_frame, profile_for(60), **arguments)  # type: ignore[arg-type]
    result = compute_windows(empty_frame, profile_for(60), **arguments)  # type: ignore[arg-type]

    assert result.get_column("close_time").to_list() == list(range(0, 300, 60))
    assert result.get_column("src_count").to_list() == [0] * 5
    assert_frame_equal(result, expected, check_exact=True, check_dtypes=True)


# Two candles sharing an open time, published in a fixed order and
# disagreeing about every price. A window ending at 120 holds both, so its
# ``close`` has to come from one of them; the contract says the first the
# provider published, and 201.0 rather than 301.0 is the whole assertion.
_TIED_OPEN_TIME_ROWS: tuple[SourceRow, ...] = (
    (0, 100.0, 110.0, 90.0, 101.0, 1.0),
    (60, 200.0, 210.0, 190.0, 201.0, 2.0),
    (60, 300.0, 310.0, 290.0, 301.0, 4.0),
    (120, 400.0, 410.0, 390.0, 401.0, 8.0),
)

# The five windows a 2m window emitted every minute over [0, 241) produces
# for those rows, worked out by hand. Volumes are whole numbers, so this
# comparison stays exact.
_TIED_OPEN_TIME_WINDOWS: tuple[WindowRow, ...] = (
    # Nothing has closed yet by t = 0.
    (-120, 0, None, None, None, None, None, 0, 0),
    # Only the first candle: [0, 60) closes at 60, inside [-60, 60).
    (-60, 60, 100.0, 110.0, 90.0, 101.0, 1.0, 1, 60),
    # [0, 120) holds all three of the first three rows. The latest open
    # time is 60, shared by two rows, so ``close`` is the earlier row's
    # 201.0 -- not 301.0, which is the last row of the slice.
    (0, 120, 100.0, 310.0, 90.0, 201.0, 7.0, 3, 180),
    # [60, 180) drops the first row and gains the fourth. The earliest
    # open time is now the tied one, so ``open`` is the earlier row's
    # 200.0 -- the same tie, read from the other end.
    (60, 180, 200.0, 410.0, 190.0, 401.0, 14.0, 3, 180),
    # [120, 240) holds the fourth row alone.
    (120, 240, 400.0, 410.0, 390.0, 401.0, 8.0, 1, 60),
)


def test_candles_sharing_an_open_time_break_ties_toward_the_earlier_row() -> None:
    """Duplicate open times resolve to the row the provider published first.

    Duplicate open times are invalid input -- strict source validation
    rejects them -- but both implementations still define an answer for
    them, and the property suites reach the case only by chance. This pins
    it deterministically, with the expected frame written out by hand
    rather than taken from either implementation.
    """
    frame = frame_from_rows(_TIED_OPEN_TIME_ROWS)
    arguments = {
        "window": "2m",
        "emit_every": "1m",
        "materialization": ExplicitRange(start=0, end=241),
    }

    expected = expected_frame(_TIED_OPEN_TIME_WINDOWS)
    oracle = compute_reference_windows(frame, profile_for(60), **arguments)  # type: ignore[arg-type]
    result = compute_windows(frame, profile_for(60), **arguments)  # type: ignore[arg-type]

    assert_frame_equal(result, expected, check_exact=True, check_dtypes=True)
    assert_frame_equal(result, oracle, check_exact=True, check_dtypes=True)


# Two representative schedules over the committed 14-day minute slice: one
# short window emitted often, one schedule-scale window emitted hourly.
# The oracle is quadratic, so the emit cadence is kept coarse enough that
# running it over 20160 real candles stays a few seconds, not minutes.
_REAL_SCHEDULES = (("15m", "15m"), ("2590m", "1h"))

# The unit roundoff of binary64: half an ulp at 1.0, and the size of the
# relative error one floating-point addition may introduce.
_UNIT_ROUNDOFF = 2.0**-53

# How far apart the two summations may land on real volumes. The gap
# belongs to the ORACLE, not to the window rule: folding m non-negative
# addends left to right in row order carries up to (m - 1) * _UNIT_ROUNDOFF
# relative rounding, which is 2.9e-13 for the 2590-candle schedule above,
# while the engine's block fold over the same slice stays within a few
# units in the last place of math.fsum. 1e-12 covers that derived bound
# with roughly threefold headroom.
#
# The measured disagreement over this fixture is far smaller: the
# 15-candle schedule matches bit for bit on all 1344 rows, and the
# 2590-candle schedule lands at most 3.9e-15 relative away. So the
# tolerance is headroom against float addition and nothing else: it is
# still some eleven orders of magnitude below the effect of dropping one
# average-sized candle from a 2590-candle window, so no membership
# mistake can hide inside it.
_VOLUME_RELATIVE_TOLERANCE = 1e-12


@pytest.mark.parametrize(("window", "emit_every"), _REAL_SCHEDULES)
def test_the_engine_agrees_with_the_oracle_on_the_real_slice(
    window: str, emit_every: str
) -> None:
    """Twenty thousand real candles, and the two implementations still agree.

    Every column but ``volume`` is compared exactly. ``volume`` is
    compared with the tolerance derived above, because real Bitstamp
    volumes are not exactly representable in binary floating point: the
    oracle's left-to-right fold over the frame's row order and the
    engine's block fold over the window's own slice are sums of the same
    addends and differ only in where the rounding lands.

    Measured over this committed fixture, the 15-candle schedule agrees
    bit for bit on all 1344 rows and the 2590-candle schedule disagrees by
    at most 3.9e-15 relative -- two orders of magnitude inside the
    tolerance, and nearly two inside the oracle's own derived rounding
    bound. The tolerance is about float addition, not about the window
    being fuzzy.
    """
    frame = load_real_slice()
    arguments = {
        "window": window,
        "emit_every": emit_every,
        "materialization": MaterializationRule.SKIP_WARMUP,
    }

    expected = compute_reference_windows(frame, BITSTAMP_BTCUSD_1M, **arguments)  # type: ignore[arg-type]
    result = compute_windows(frame, BITSTAMP_BTCUSD_1M, **arguments)  # type: ignore[arg-type]

    assert result.height > 0
    assert_frame_equal(
        result.drop("volume"),
        expected.drop("volume"),
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )
    assert_series_equal(
        result.get_column("volume"),
        expected.get_column("volume"),
        check_exact=False,
        rel_tol=_VOLUME_RELATIVE_TOLERANCE,
        abs_tol=0.0,
    )


def test_the_volume_tolerance_covers_the_oracles_own_rounding() -> None:
    """The tolerance is derived from the fold, not tuned until a test passed.

    Written as an assertion rather than only as a comment so that widening
    a schedule above -- a longer window means more addends means more
    rounding -- cannot silently leave the tolerance too tight to hold.
    """
    longest_window_candles = max(
        Duration.parse(window).total_seconds // REAL_SLICE_CADENCE_SECONDS
        for window, _ in _REAL_SCHEDULES
    )
    oracle_fold_bound = (longest_window_candles - 1) * _UNIT_ROUNDOFF

    assert _VOLUME_RELATIVE_TOLERANCE >= oracle_fold_bound


if __name__ == "__main__":
    pytest.main([__file__])
