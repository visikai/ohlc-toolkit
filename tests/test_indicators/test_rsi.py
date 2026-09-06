"""Cutler's RSI against literals, against an oracle, and at its boundaries.

The three boundary conventions are pinned as exact literals rather than
as ranges: they are conventions, chosen rather than derived, and a test
that accepted `> 99` would accept an implementation that had quietly
started smoothing.
"""

import math
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ohlc_toolkit.indicators import CutlersRSI, IndicatorPrimitive, add_indicator
from ohlc_toolkit.temporal import ConfigError, DataValidationError
from tests.test_indicators.factories import phased_from_closes, rising

_PERIOD = 3
_LOOKBACK = _PERIOD + 1

# The primitive under test, held once so every test reads the same object
# a caller would.
_RSI = CutlersRSI()

# Both algebraic routes to the same number agree to far better than this;
# the tolerance exists because they are different routes, not because
# either is approximate.
_TOLERANCE = 1e-9

# The conventions, as literals in this file rather than as names imported
# from the module under test: an expectation that read its value from the
# code would agree with the code whatever the code said.
_ONLY_RISES = 100.0
_ONLY_FALLS = 0.0
_ALL_UNCHANGED = 50.0
_HAND_COMPUTED_MIXED = 62.5
_LOOKBACK_AT_FOURTEEN = 15
_LOOKBACK_AT_ONE = 2


def _oracle(closes: Sequence[float], period: int) -> float:
    """Compute the reading from the spelled-out formula, in Python floats.

    Deliberately the SPELLED-OUT form `100 - 100 / (1 + G / D)` with its
    conventions as explicit branches, while the implementation uses the
    closed form. Two different routes to one number is a check; the same
    route twice would only prove the code equals itself.
    """
    oldest_first = list(reversed(closes))
    changes = [later - earlier for earlier, later in pairwise(oldest_first)]
    gain = sum(change for change in changes if change > 0) / period
    loss = sum(-change for change in changes if change < 0) / period
    if gain == 0 and loss == 0:
        return 50.0
    if loss == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + gain / loss)


def _reading(closes: Sequence[float | None] | None, *, period: int = _PERIOD) -> object:
    """Run the primitive over one tick and return that tick's value."""
    phased = phased_from_closes([closes], lookback=period + 1)
    return _RSI.values(phased, period=period).item(0)


def test_the_primitive_satisfies_the_protocol() -> None:
    """Checked by the type checker, and by reading the declarations back."""
    primitive: IndicatorPrimitive = _RSI

    assert primitive.name == "rsi"
    assert primitive.normalization.value == "bounded_by_construction"


def test_the_lookback_is_the_period_plus_one() -> None:
    """`P` changes need `P + 1` closes, which is `L`."""
    assert _RSI.lookback(14) == _LOOKBACK_AT_FOURTEEN
    assert _RSI.lookback(1) == _LOOKBACK_AT_ONE


def test_a_period_that_is_not_a_positive_int_is_refused() -> None:
    """The lookback is the first thing a caller asks for, so it validates."""
    with pytest.raises(ConfigError, match="must be an int"):
        _RSI.lookback(True)


def test_a_strictly_rising_run_reads_exactly_one_hundred() -> None:
    """`D = 0` and `G > 0`: the first boundary convention."""
    assert _reading(rising(_LOOKBACK)) == _ONLY_RISES


def test_a_strictly_falling_run_reads_exactly_zero() -> None:
    """`G = 0` and `D > 0`: the second."""
    assert _reading(list(reversed(rising(_LOOKBACK)))) == _ONLY_FALLS


def test_identical_traded_closes_read_exactly_fifty() -> None:
    """Both means zero: the symmetric limit, and not a null.

    The inputs are all present -- this is a genuinely neutral window, not
    a window with too little trading in it, which the harness has already
    turned into a null before a primitive sees it.
    """
    assert _reading([100.0] * _LOOKBACK) == _ALL_UNCHANGED


def test_the_bound_holds_where_scaling_before_dividing_would_break_it() -> None:
    """The case the property test found, kept as a literal.

    `100 * G / (G + D)` with `D = 0` and `G = 256842.5 / 3` rounds the
    product before it divides and returns `100.00000000000001` -- a value
    outside the range this indicator's whole normalization class rests
    on. Dividing first is exact here.
    """
    assert _reading([256857.5, 15.0, 15.0, 15.0]) == _ONLY_RISES


def test_a_mixed_run_matches_the_hand_computed_number() -> None:
    """One case computed by hand, so the oracle is not the only witness.

    Closes newest first `[3, 1, 4, 1]` are `1, 4, 1, 3` in time, so the
    three changes are `+3, -3, +2`. `G = 5/3`, `D = 3/3 = 1`, and
    `100 - 100 / (1 + 5/3) = 62.5`.
    """
    assert _reading([3.0, 1.0, 4.0, 1.0]) == _HAND_COMPUTED_MIXED


@settings(max_examples=250, deadline=None)
@given(
    closes=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=_LOOKBACK,
        max_size=_LOOKBACK,
    )
)
def test_the_primitive_equals_the_brute_force_oracle(closes: list[float]) -> None:
    """The property: one route through polars, one through Python floats."""
    value = _reading(closes)

    assert isinstance(value, float)
    assert math.isclose(
        value, _oracle(closes, _PERIOD), rel_tol=_TOLERANCE, abs_tol=_TOLERANCE
    )
    assert _ONLY_FALLS <= value <= _ONLY_RISES
    assert math.isfinite(value)


def test_a_null_tick_reads_null_and_its_neighbours_do_not() -> None:
    """Null propagation is per tick: the row nulls, the column does not."""
    values = _RSI.values(
        phased_from_closes(
            [rising(_LOOKBACK), None, rising(_LOOKBACK)], lookback=_LOOKBACK
        ),
        period=_PERIOD,
    )

    assert values.null_count() == 1
    assert values.to_list() == [_ONLY_RISES, None, _ONLY_RISES]


def test_one_null_among_the_inputs_nulls_exactly_that_tick() -> None:
    """A hole in a list is not an average over what is left.

    The harness emits a complete list or a null one, so this shape only
    reaches a primitive from a caller that built the record by hand --
    and dropping the null would divide `P - 1` changes by `P`.
    """
    holed: list[float | None] = [*rising(_LOOKBACK)]
    holed[1] = None
    values = _RSI.values(
        phased_from_closes([rising(_LOOKBACK), holed], lookback=_LOOKBACK),
        period=_PERIOD,
    )

    assert values.to_list() == [_ONLY_RISES, None]


def test_the_column_is_float64_and_named_from_the_identity() -> None:
    """`{indicator}_{family}{period}_w{window}`, derived and not passed."""
    values = _RSI.values(
        phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK, window_seconds=180),
        period=_PERIOD,
    )

    assert values.name == "rsi_p3_w3m"
    assert values.dtype == pl.Float64


def test_a_non_finite_reading_is_refused_rather_than_written() -> None:
    """Unreachable from finite closes, which is why it refuses out loud."""
    enormous = [1e308, -1e308, 1e308, -1e308]

    with pytest.raises(DataValidationError, match="non-finite"):
        _RSI.values(phased_from_closes([enormous], lookback=_LOOKBACK), period=_PERIOD)


def test_the_derived_name_survives_a_round_trip_through_parquet(
    tmp_path: Path,
) -> None:
    """The first time a derived name meets a stored schema.

    Nothing had written one until now: `FeatureIdentity` derived
    names and tests read them back out of memory. Parquet has its own
    rules about column names, and the `d`-family character had never been
    near a file.
    """
    phased = phased_from_closes(
        [rising(_LOOKBACK), list(reversed(rising(_LOOKBACK)))], lookback=_LOOKBACK
    )
    written = add_indicator(phased, _RSI, period=_PERIOD)
    path = tmp_path / "indicators.parquet"
    written.write_parquet(path)

    read_back = pl.read_parquet(path)

    assert read_back.columns == written.columns
    assert read_back["rsi_p3_w3m"].to_list() == [_ONLY_RISES, _ONLY_FALLS]
    assert read_back["rsi_p3_w3m"].dtype == pl.Float64


if __name__ == "__main__":
    pytest.main([__file__])
