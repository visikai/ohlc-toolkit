"""Log volume ratio: the median, the exclusion, and the refusal.

Two things here can be silently wrong and produce a plausible number: a
MEAN baseline instead of a median, and a baseline that includes the
current window. Both have a literal chosen so that the wrong answer is a
different number rather than the same one.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ohlc_toolkit.indicators import (
    IndicatorPrimitive,
    LogVolumeRatio,
    phased_lookback,
)
from ohlc_toolkit.temporal import ConfigError, DataValidationError
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from tests.test_indicators.factories import BASE, phased_from_fields
from tests.test_windows.factories import frame_from_rows, profile_for

_MINUTE = 60
_PERIOD = 3
_LOOKBACK = _PERIOD + 1
_RATIO = LogVolumeRatio()
_TOLERANCE = 1e-12
_LOOKBACK_AT_FOURTEEN = 15

# Baseline volumes 1, 2 and 4: median 2, mean 7/3. A spike of 100 in the
# current window keeps the median at 2 and would drag a mean-based
# baseline nowhere near it -- and including the current window in the
# median would move it to 3.
_SPIKE = [100.0, 1.0, 2.0, 4.0]
_AGAINST_THE_MEDIAN = math.log(50.0)
_AGAINST_A_MEAN = math.log(100.0 / (7.0 / 3.0))
_INCLUDING_ITSELF = math.log(100.0 / 3.0)

# An even period, where the median is the average of the two middle
# values: 1, 2, 4, 8 has median 3, and a current volume of 3 reads zero.
_EVEN_BASELINE = [3.0, 1.0, 2.0, 4.0, 8.0]


def _oracle(volumes: Sequence[float], period: int) -> float:
    """Compute the reading in plain Python, median by sorting."""
    baseline = sorted(volumes[1:])
    middle = period // 2
    median = (
        baseline[middle]
        if period % 2
        else (baseline[middle - 1] + baseline[middle]) / 2.0
    )
    return math.log(volumes[0] / median)


def _reading(volumes: Sequence[float], *, period: int = _PERIOD) -> object:
    """Run the primitive over one tick and return that tick's value."""
    phased = phased_from_fields({"volume": [volumes]}, lookback=period + 1)
    return _RATIO.values(phased, period=period).item(0)


def test_the_primitive_satisfies_the_protocol_and_declares_its_class() -> None:
    """Stationarized: a ratio of two volumes, logged."""
    primitive: IndicatorPrimitive = _RATIO

    assert primitive.name == "logvolratio"
    assert primitive.normalization.value == "stationarized"
    assert primitive.lookback(14) == _LOOKBACK_AT_FOURTEEN


def test_the_baseline_is_the_median_and_not_the_mean() -> None:
    """Odd period, and a fixture where the two disagree.

    Baseline volumes 1, 2, 4: the median is 2 and the mean is 7/3. A
    current volume of 100 reads `ln(50)` against the median and
    `ln(300/7)` against the mean -- different numbers, so the assertion
    can see which was used.
    """
    value = _reading(_SPIKE)

    assert value == _AGAINST_THE_MEDIAN
    assert value != _AGAINST_A_MEAN


def test_the_current_window_is_excluded_from_its_own_baseline() -> None:
    """A spike must not shrink the ratio it is supposed to produce.

    The same fixture: including the current 100 in the median moves it
    from 2 to 3, and the reading from `ln(50)` to `ln(100/3)`. The
    exclusion is one `list.slice(1)`, and this is what holds it there.

    Asserted as an equality against the right answer rather than as an
    inequality against the wrong one: any third number would satisfy
    `!= ln(100/3)`.
    """
    value = _reading(_SPIKE)

    assert value == _AGAINST_THE_MEDIAN
    assert value != _INCLUDING_ITSELF


def test_an_even_period_averages_the_two_middle_volumes() -> None:
    """Baseline 1, 2, 4, 8 has median 3, so a current 3 reads zero."""
    assert _reading(_EVEN_BASELINE, period=4) == 0.0


def test_a_volume_equal_to_its_baseline_reads_zero() -> None:
    """The odd case too, so zero is not an artifact of the even branch."""
    assert _reading([2.0, 1.0, 2.0, 4.0]) == 0.0


@pytest.mark.parametrize("period", [3, 4])
@settings(max_examples=2_000, deadline=None)
@given(data=st.data())
def test_the_primitive_equals_the_brute_force_oracle(
    data: st.DataObject, period: int
) -> None:
    """One route through polars' median, one through a Python sort.

    Both parities of the period: the even branch averages the two middle
    volumes, and polars' quantile interpolation is where the two routes
    could disagree. A single literal was the only thing exercising it.
    """
    volumes = data.draw(
        st.lists(
            st.floats(
                min_value=1e-6, max_value=1e9, allow_nan=False, allow_infinity=False
            ),
            min_size=period + 1,
            max_size=period + 1,
        )
    )
    value = _reading(volumes, period=period)

    assert isinstance(value, float)
    assert math.isclose(
        value, _oracle(volumes, period), rel_tol=_TOLERANCE, abs_tol=_TOLERANCE
    )
    assert math.isfinite(value)


def test_a_null_tick_reads_null_and_its_neighbours_do_not() -> None:
    """Null propagation is per tick."""
    values = _RATIO.values(
        phased_from_fields(
            {"volume": [[8.0, 1.0, 2.0, 4.0], None, [8.0, 1.0, 2.0, 4.0]]},
            lookback=_LOOKBACK,
        ),
        period=_PERIOD,
    )

    assert values.null_count() == 1
    assert values.to_list()[1] is None


@pytest.mark.parametrize("position", range(_LOOKBACK))
def test_a_null_among_the_volumes_nulls_exactly_that_tick(position: int) -> None:
    """A hole is not a median over what is left, at any position."""
    holed: list[float | None] = [8.0, 1.0, 2.0, 4.0]
    holed[position] = None

    values = _RATIO.values(
        phased_from_fields(
            {"volume": [[8.0, 1.0, 2.0, 4.0], holed, [8.0, 1.0, 2.0, 4.0]]},
            lookback=_LOOKBACK,
        ),
        period=_PERIOD,
    )

    assert values.null_count() == 1
    assert values.to_list()[1] is None


@pytest.mark.parametrize(
    "volumes",
    [
        pytest.param([0.0, 1.0, 2.0, 4.0], id="the-current-window"),
        pytest.param([8.0, 1.0, 0.0, 4.0], id="one-of-the-baseline-windows"),
        pytest.param([8.0, 1.0, -2.0, 4.0], id="a-negative-volume"),
    ],
)
def test_a_non_positive_volume_is_refused_rather_than_read_through(
    volumes: list[float],
) -> None:
    """The contract, quoted in the refusal.

    A present window has traded seconds at or above the threshold and
    therefore volume above zero, so a zero here is a violation upstream.
    Reading through it would give a null, a negative infinity or a NaN,
    each of which a reader would then have to interpret.
    """
    with pytest.raises(DataValidationError, match="traded seconds"):
        _reading(volumes)


def test_an_infinite_volume_is_refused_by_the_finiteness_guard() -> None:
    """Its own test, because a guard wired into one caller protects one.

    This primitive selects rather than computes -- a first element and a
    median -- so no arithmetic here can overflow, and it would be easy to
    conclude that the guard is unreachable. It is reachable, by one path:
    an INFINITE volume passes the positivity guard, since `inf` is not
    `<= 0`, and arrives at the finiteness check as the current volume.
    The source layer refuses non-finite values, so this needs a
    hand-built record -- which is exactly the threat model the guards
    state.
    """
    with pytest.raises(DataValidationError, match="non-finite intermediate"):
        _reading([math.inf, 1.0, 2.0, 4.0])


def test_the_primitive_runs_over_real_harness_output() -> None:
    """The median path meets a frame the harness really produced."""
    rows = [
        (
            BASE + index * _MINUTE,
            100.0 + index,
            101.0 + index,
            99.0 + index,
            100.5 + index,
            5.0 + index,
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

    values = _RATIO.values(phased, period=_PERIOD)

    present = values.drop_nulls()
    assert present.len() > 0
    assert present.is_finite().all()
    # Volume rises throughout, so every reading is above its own baseline.
    assert (present > 0.0).all()


def test_a_frame_resolved_for_another_lookback_is_refused() -> None:
    """`L = P + 1`, and the baseline is the `P` behind the current one."""
    phased = phased_from_fields({"volume": [[8.0, 1.0, 2.0]]}, lookback=3)

    with pytest.raises(ConfigError, match="reads 4 phased window"):
        _RATIO.values(phased, period=_PERIOD)


def test_the_column_is_float64_and_named_from_the_identity() -> None:
    """The window comes from the grid, not from the primitive."""
    phased = phased_from_fields(
        {"volume": [[8.0, 1.0, 2.0, 4.0]]}, lookback=_LOOKBACK, window_seconds=8760
    )

    values = _RATIO.values(phased, period=_PERIOD)

    assert values.name == "logvolratio_p3_w2h26m"
    assert values.dtype == pl.Float64


if __name__ == "__main__":
    pytest.main([__file__])
