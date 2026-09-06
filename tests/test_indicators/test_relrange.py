"""Relative range: the gap term, the oracle, and the sign it can never take.

The load-bearing literal is the one where the previous close sits OUTSIDE
the current window's own high-low band. Every other fixture is satisfied
by a true range that has quietly degraded to `high - low`.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ohlc_toolkit.indicators import IndicatorPrimitive, RelativeRange, phased_lookback
from ohlc_toolkit.temporal import ConfigError, DataValidationError
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from tests.test_indicators.factories import BASE, phased_from_fields
from tests.test_windows.factories import frame_from_rows, profile_for

_MINUTE = 60
_PERIOD = 3
_LOOKBACK = _PERIOD + 1
_RELRANGE = RelativeRange()
_TOLERANCE = 1e-12

# The gap literal: the previous close is 20 while the current window
# traded between 6 and 10, so the true range is 20 - 6 = 14 and NOT the
# 10 - 6 a degraded implementation would report.
_GAP_HIGHS = [10.0, 9.0]
_GAP_LOWS = [6.0, 5.0]
_GAP_CLOSES = [7.0, 20.0]
_GAP_EXPECTED = 2.0
_DEGRADED_TO_THE_BAR = 4.0 / 7.0
_LOOKBACK_AT_FOURTEEN = 15


def _oracle(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    period: int,
) -> float:
    """Compute the reading from the three-candidate definition, in Python.

    Deliberately the spelled-out `max(high - low, |high - prev|,
    |low - prev|)` while the implementation uses
    `max(high, prev) - min(low, prev)`. The two are equal exactly rather
    than approximately, which is the claim the property test checks.
    """
    ranges = [
        max(
            highs[phase] - lows[phase],
            abs(highs[phase] - closes[phase + 1]),
            abs(lows[phase] - closes[phase + 1]),
        )
        for phase in range(period)
    ]
    return (sum(ranges) / period) / closes[0]


def _reading(
    highs: Sequence[float],
    lows: Sequence[float],
    closes: Sequence[float],
    *,
    period: int = _PERIOD,
) -> object:
    """Run the primitive over one tick and return that tick's value."""
    phased = phased_from_fields(
        {"high": [highs], "low": [lows], "close": [closes]}, lookback=period + 1
    )
    return _RELRANGE.values(phased, period=period).item(0)


def test_the_primitive_satisfies_the_protocol_and_declares_its_class() -> None:
    """Stationarized: a ratio of two prices, so the scale divides out."""
    primitive: IndicatorPrimitive = _RELRANGE

    assert primitive.name == "relrange"
    assert primitive.normalization.value == "stationarized"
    assert primitive.lookback(14) == _LOOKBACK_AT_FOURTEEN


def test_the_gap_term_dominates_when_the_previous_close_is_outside_the_bar() -> None:
    """The literal a degraded true range fails.

    One window, high 10 and low 6, whose PREVIOUS phased window closed at
    20. The true range is 20 - 6 = 14, not 10 - 6 = 4, and dividing by
    the current close of 7 gives exactly 2. An implementation that had
    forgotten the previous close entirely would read 4/7.
    """
    value = _reading(_GAP_HIGHS, _GAP_LOWS, _GAP_CLOSES, period=1)

    assert value == _GAP_EXPECTED
    assert value != _DEGRADED_TO_THE_BAR


def test_a_previous_close_inside_the_bar_reads_the_bar() -> None:
    """The other side of the same rule, so the gap term is not always on."""
    assert _reading([10.0, 9.0], [6.0, 5.0], [7.0, 8.0], period=1) == 4.0 / 7.0


def test_the_mean_is_over_the_windows_that_have_a_predecessor() -> None:
    """Hand-computed: true ranges 4, 15 and 4 over three windows.

    Highs `[10, 9, 8]`, lows `[6, 5, 4]` and closes `[8, 7, 20, 5]`
    newest first. The predecessors are 7, 20 and 5, so the ranges are
    `max(10,7)-min(6,7) = 4`, `max(9,20)-min(5,20) = 15` and
    `max(8,5)-min(4,5) = 4`. Their mean is 23/3, over a close of 8.
    """
    value = _reading([10.0, 9.0, 8.0, 7.0], [6.0, 5.0, 4.0, 3.0], [8.0, 7.0, 20.0, 5.0])

    assert value == (23.0 / 3.0) / 8.0


@settings(max_examples=2_000, deadline=None)
@given(
    lows=st.lists(
        st.floats(min_value=1.0, max_value=1e5, allow_nan=False, allow_infinity=False),
        min_size=_LOOKBACK,
        max_size=_LOOKBACK,
    ),
    heights=st.lists(
        st.floats(min_value=0.0, max_value=1e4, allow_nan=False, allow_infinity=False),
        min_size=_LOOKBACK,
        max_size=_LOOKBACK,
    ),
    fractions=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
        min_size=_LOOKBACK,
        max_size=_LOOKBACK,
    ),
)
def test_the_primitive_equals_the_brute_force_oracle(
    lows: list[float], heights: list[float], fractions: list[float]
) -> None:
    """Real candles: the close lies inside its own bar, and the bar is real."""
    highs = [low + height for low, height in zip(lows, heights, strict=True)]
    closes = [
        low + height * fraction
        for low, height, fraction in zip(lows, heights, fractions, strict=True)
    ]
    value = _reading(highs, lows, closes)

    assert isinstance(value, float)
    assert math.isclose(
        value, _oracle(highs, lows, closes, _PERIOD), rel_tol=_TOLERANCE, abs_tol=0.0
    )
    assert value >= 0.0
    assert math.isfinite(value)


def test_a_null_tick_reads_null_and_its_neighbours_do_not() -> None:
    """Null propagation is per tick."""
    values = _RELRANGE.values(
        phased_from_fields(
            {
                "high": [[10.0] * _LOOKBACK, None, [10.0] * _LOOKBACK],
                "low": [[6.0] * _LOOKBACK, None, [6.0] * _LOOKBACK],
                "close": [[8.0] * _LOOKBACK, None, [8.0] * _LOOKBACK],
            },
            lookback=_LOOKBACK,
        ),
        period=_PERIOD,
    )

    assert values.null_count() == 1
    assert values.to_list()[1] is None


@pytest.mark.parametrize("field", ["high", "low", "close"])
@pytest.mark.parametrize("position", range(_LOOKBACK))
def test_a_null_in_any_field_it_reads_nulls_the_tick(field: str, position: int) -> None:
    """All three fields are read, so a hole in any of them is a hole.

    Every position, and position `L - 1` in particular: that window is a
    PREDECESSOR ONLY -- its own true range is sliced off before the mean
    -- so a rule that looked at the averaged values rather than at the
    inputs would let a null there through. This is the one primitive with
    a predecessor slice, which is why its siblings' single-position test
    would not be enough here.
    """
    fields: dict[str, list[list[float | None] | None]] = {
        "high": [[10.0] * _LOOKBACK],
        "low": [[6.0] * _LOOKBACK],
        "close": [[8.0] * _LOOKBACK],
    }
    holed = list(fields[field][0])  # type: ignore[arg-type]
    holed[position] = None
    fields[field] = [holed]

    values = _RELRANGE.values(
        phased_from_fields(fields, lookback=_LOOKBACK), period=_PERIOD
    )

    assert values.to_list() == [None]


@pytest.mark.parametrize(
    "closes",
    [
        pytest.param([8.0, 0.0, 8.0, 8.0], id="a-zero-close"),
        pytest.param([8.0, -8.0, 8.0, 8.0], id="a-negative-close"),
    ],
)
def test_a_non_positive_close_is_refused(closes: list[float]) -> None:
    """The divisor. Both branches of the guard, not only the zero one.

    The guard is shared, so a fixture at `0.0` alone is killed by any
    sibling's negative case today -- and stops being killed the moment
    anyone specialises it for one primitive.
    """
    with pytest.raises(DataValidationError, match="traded price"):
        _reading([10.0] * _LOOKBACK, [6.0] * _LOOKBACK, closes)


def test_an_overflowing_true_range_is_refused_rather_than_averaged() -> None:
    """Its own test, because a guard wired into one caller protects one.

    `close` is the only field this primitive constrains to be positive;
    `high` and `low` are free, so a bar from -1.7e308 to 1.7e308 has an
    infinite true range while every input is finite. Averaging it would
    produce an infinite reading, which AC4 forbids.
    """
    enormous = 1.7e308

    with pytest.raises(DataValidationError, match="non-finite intermediate"):
        _reading([enormous, enormous], [-enormous, -enormous], [1.0, 1.0], period=1)


def test_the_true_range_is_exact_where_the_arithmetic_form_is_not() -> None:
    """The counter-example that rules out `clip(h - p, 0) + clip(p - l, 0)`.

    That form computes `(high - prev) + (prev - low)` when the previous
    close lies inside the bar, which can differ from `high - low` in the
    last place. Here it returns ...407 where the exact answer is ...409.
    The prohibition was a docstring until this test; substituting the
    form left the suite green.
    """
    high = 99533.67442462409
    previous = 24378.60928155538

    # A close of 1.0 makes the reading the true range itself, undivided.
    assert _reading([high, 1.0], [0.0, 1.0], [1.0, previous], period=1) == high


def test_the_primitive_runs_over_real_harness_output() -> None:
    """The packed-list path meets a frame the harness really produced.

    Every other test here builds the record by hand, so the packing, the
    newest-first order and the predecessor slice are only ever checked
    against a fixture that shares their assumptions.
    """
    rows = [
        (
            BASE + index * _MINUTE,
            100.0 + index,
            101.0 + index,
            99.0 + index,
            100.5 + index,
            5.0,
        )
        for index in range(60)
    ]
    frame = compute_windows(
        frame_from_rows(rows),
        profile_for(_MINUTE),
        window="3m",
        emit_every="1m",
        materialization=ExplicitRange(start=BASE + 180, end=BASE + 60 * _MINUTE + 1),
    )
    phased = phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)

    values = _RELRANGE.values(phased, period=_PERIOD)

    present = values.drop_nulls()
    assert present.len() > 0
    assert (present >= 0.0).all()
    assert present.is_finite().all()


def test_a_frame_resolved_for_another_lookback_is_refused() -> None:
    """`L = P + 1` here, and the grid says what it was assembled for."""
    phased = phased_from_fields(
        {"high": [[10.0] * 3], "low": [[6.0] * 3], "close": [[8.0] * 3]}, lookback=3
    )

    with pytest.raises(ConfigError, match="reads 4 phased window"):
        _RELRANGE.values(phased, period=_PERIOD)


def test_the_column_is_float64_and_named_from_the_identity() -> None:
    """The window comes from the grid, not from the primitive."""
    phased = phased_from_fields(
        {
            "high": [[10.0] * _LOOKBACK],
            "low": [[6.0] * _LOOKBACK],
            "close": [[8.0] * _LOOKBACK],
        },
        lookback=_LOOKBACK,
        window_seconds=8760,
    )

    values = _RELRANGE.values(phased, period=_PERIOD)

    assert values.name == "relrange_p3_w2h26m"
    assert values.dtype == pl.Float64


if __name__ == "__main__":
    pytest.main([__file__])
