"""Price to moving average: the degenerate case, the oracle, and the reason.

This primitive exists because a moving-average log SLOPE was shown to be
nearly proportional to a backward log return the package already
computes. The last test here reports both correlations over a seeded
series so that the replacement can be seen not to be a second copy of the
thing it replaced.
"""

import math
import random
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ohlc_toolkit.indicators import (
    IndicatorPrimitive,
    PriceToMovingAverage,
    phased_lookback,
)
from ohlc_toolkit.temporal import ConfigError, DataValidationError
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from tests.test_indicators.factories import BASE, phased_from_closes
from tests.test_windows.factories import frame_from_rows, profile_for

_MINUTE = 60
_PERIOD = 3
_MAPOS = PriceToMovingAverage()
_TOLERANCE = 1e-12
_LOOKBACK_AT_FOURTEEN = 14
_ON_THE_MEAN = 0.0


def _oracle(closes: Sequence[float], period: int) -> float:
    """Compute the reading in plain Python: log of price over its own mean."""
    window = closes[:period]
    return math.log(closes[0] / (sum(window) / period))


def _reading(closes: Sequence[float], *, period: int = _PERIOD) -> object:
    """Run the primitive over one tick and return that tick's value."""
    return _MAPOS.values(
        phased_from_closes([closes], lookback=period), period=period
    ).item(0)


def test_the_primitive_satisfies_the_protocol_and_declares_its_class() -> None:
    """`L = P`: the current window is inside its own mean, so no extra one."""
    primitive: IndicatorPrimitive = _MAPOS

    assert primitive.name == "mapos"
    assert primitive.normalization.value == "stationarized"
    assert primitive.lookback(14) == _LOOKBACK_AT_FOURTEEN


def test_identical_closes_read_exactly_zero() -> None:
    """Price on its own mean, and the log of one is exact."""
    assert _reading([7.0, 7.0, 7.0]) == _ON_THE_MEAN


def test_a_price_above_its_mean_reads_the_log_of_the_ratio() -> None:
    """Hand-computed: closes 4, 1, 1 have mean 2, and `ln(4/2)` is `ln 2`."""
    assert _reading([4.0, 1.0, 1.0]) == math.log(2.0)


def test_a_price_below_its_mean_reads_negative() -> None:
    """The sign convention, which a reversed ratio would invert."""
    value = _reading([1.0, 4.0, 4.0])

    assert isinstance(value, float)
    assert value < 0.0
    assert value == math.log(1.0 / 3.0)


def test_at_a_period_of_one_every_reading_is_identically_zero() -> None:
    """The degenerate case, and the reason a lookback slice starts above one.

    At `P = 1` the mean IS the current close, so the ratio is one and the
    log is zero whatever the price did. The primitive does not refuse it
    -- the contract says `L = P` and one window is a legal lookback --
    but a recipe that asked for it would be recording a column of zeros.
    """
    values = _MAPOS.values(
        phased_from_closes([[7.0], [9.0], [11.0]], lookback=1), period=1
    )

    assert values.to_list() == [_ON_THE_MEAN, _ON_THE_MEAN, _ON_THE_MEAN]


@settings(max_examples=2_000, deadline=None)
@given(
    closes=st.lists(
        st.floats(min_value=0.01, max_value=1e6, allow_nan=False, allow_infinity=False),
        min_size=_PERIOD,
        max_size=_PERIOD,
    )
)
def test_the_primitive_equals_the_brute_force_oracle(closes: list[float]) -> None:
    """One route through polars' list mean, one through a Python sum."""
    value = _reading(closes)

    assert isinstance(value, float)
    assert math.isclose(
        value, _oracle(closes, _PERIOD), rel_tol=_TOLERANCE, abs_tol=_TOLERANCE
    )
    assert math.isfinite(value)


@pytest.mark.parametrize("position", range(_PERIOD))
def test_a_null_among_the_closes_nulls_exactly_that_tick(position: int) -> None:
    """A hole is not a mean over what is left, at any position."""
    holed: list[float | None] = [4.0, 1.0, 1.0]
    holed[position] = None

    values = _MAPOS.values(
        phased_from_closes([[4.0, 1.0, 1.0], holed, [4.0, 1.0, 1.0]], lookback=_PERIOD),
        period=_PERIOD,
    )

    assert values.null_count() == 1
    assert values.to_list()[1] is None


def test_a_non_positive_close_is_refused() -> None:
    """The log's argument. A zero close is a violation upstream."""
    with pytest.raises(DataValidationError, match="traded price"):
        _reading([4.0, 0.0, 1.0])


def test_a_frame_resolved_for_another_lookback_is_refused() -> None:
    """`L = P` here, which is the whole difference from the other three."""
    with pytest.raises(ConfigError, match="reads 3 phased window"):
        _MAPOS.values(
            phased_from_closes([[4.0, 1.0, 1.0, 1.0]], lookback=4), period=_PERIOD
        )


def test_the_column_is_float64_and_named_from_the_identity() -> None:
    """The window comes from the grid, not from the primitive."""
    values = _MAPOS.values(
        phased_from_closes([[4.0, 1.0, 1.0]], lookback=_PERIOD, window_seconds=8760),
        period=_PERIOD,
    )

    assert values.name == "mapos_p3_w2h26m"
    assert values.dtype == pl.Float64


def test_the_trend_quadrant_is_not_a_second_copy_of_a_backward_return(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REPORT, not assert: the two correlations that justified the choice.

    A moving-average log slope is nearly proportional to the `P x W`
    backward log return, which is why it was dropped. This test computes
    all three quantities over one seeded random walk and prints their
    correlations with that return, so a reader can see that the
    replacement is not itself the duplicate.

    Nothing is asserted about the numbers themselves: they depend on the
    seed and on the walk's volatility, and an assertion on them would be
    a threshold nobody could justify. What IS asserted is that all three
    were computable over the same ticks -- an empty comparison would
    print two `null`s and look like agreement.
    """
    lookback = _PERIOD + 1
    random.seed(20260906)
    price = 30_000.0
    rows = []
    for index in range(600):
        price *= math.exp(random.gauss(0.0, 0.0015))
        rows.append(
            (
                BASE + index * _MINUTE,
                price,
                price * 1.001,
                price * 0.999,
                price,
                5.0,
            )
        )
    frame = compute_windows(
        frame_from_rows(rows),
        profile_for(_MINUTE),
        window="3m",
        emit_every="1m",
        materialization=ExplicitRange(start=BASE + 180, end=BASE + 600 * _MINUTE + 1),
    )
    phased = phased_lookback(frame, window="3m", emit_every="3m", lookback=lookback)

    # `L = P + 1` gives both the P closes the mean is over and the close
    # one `P x W` further back, so all three read the same ticks.
    closes = pl.col("close")
    measured = phased.frame.select(
        (closes.list.first() / closes.list.slice(0, _PERIOD).list.mean())
        .log()
        .alias("mapos"),
        (
            closes.list.slice(0, _PERIOD).list.mean()
            / closes.list.slice(1, _PERIOD).list.mean()
        )
        .log()
        .alias("ma_log_slope"),
        (closes.list.first() / closes.list.get(_PERIOD)).log().alias("backward_return"),
    ).drop_nulls()
    correlations = measured.select(
        pl.corr("mapos", "backward_return").alias("mapos_vs_return"),
        pl.corr("ma_log_slope", "backward_return").alias("slope_vs_return"),
    ).row(0, named=True)

    with capsys.disabled():
        print(
            f"\n  ticks compared: {measured.height}"
            f"\n  corr(mapos, backward return)        = "
            f"{correlations['mapos_vs_return']:+.6f}"
            f"\n  corr(MA log slope, backward return) = "
            f"{correlations['slope_vs_return']:+.6f}"
        )

    assert measured.height > 0
    assert correlations["mapos_vs_return"] is not None
    assert correlations["slope_vs_return"] is not None


if __name__ == "__main__":
    pytest.main([__file__])
