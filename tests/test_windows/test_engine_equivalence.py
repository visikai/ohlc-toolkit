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

``volume`` is the one sum. The oracle folds the included volumes left to
right in the frame's own row order; the engine sums them with a
vectorized rolling sum. Floating-point addition is not associative, so
the two orders can differ in the last bits whenever a partial sum is not
exactly representable. Every synthetic family draws volumes on a
quarter-unit grid, where every partial sum IS exact, so those comparisons
are exact as well. Only the real-data fixture needs a tolerance, and it
is stated and justified at that test.

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
from tests.test_windows.factories import frame_from_rows, profile_for
from tests.test_windows.fixtures import load_real_slice
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


# How far apart the two summation orders are allowed to land on real
# volumes. A few parts in 1e13 is what float rounding actually produces
# between a vectorized rolling sum and a fresh left fold over these
# volumes; 1e-12 leaves an order of magnitude of headroom and still fails
# long before any real aggregation mistake could hide inside it.
_VOLUME_RELATIVE_TOLERANCE = 1e-12

# Two representative schedules over the committed 14-day minute slice: one
# short window emitted often, one schedule-scale window emitted hourly.
# The oracle is quadratic, so the emit cadence is kept coarse enough that
# running it over 20160 real candles stays a few seconds, not minutes.
_REAL_SCHEDULES = (("15m", "15m"), ("2590m", "1h"))


@pytest.mark.parametrize(("window", "emit_every"), _REAL_SCHEDULES)
def test_the_engine_agrees_with_the_oracle_on_the_real_slice(
    window: str, emit_every: str
) -> None:
    """Twenty thousand real candles, and the two implementations still agree.

    Every column but ``volume`` is compared exactly. ``volume`` is
    compared with a relative tolerance of 1e-12 because real Bitstamp
    volumes are not exactly representable in binary floating point: the
    oracle's left-to-right fold over the frame's row order and the
    engine's vectorized rolling sum are both correct sums of the same
    addends, and they differ only in the order the rounding happens. The
    observed disagreement is a handful of parts in 1e15; the tolerance is
    about float addition, not about the window being fuzzy.
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


if __name__ == "__main__":
    pytest.main([__file__])
