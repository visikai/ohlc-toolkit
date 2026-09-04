"""What comes out when the arithmetic has no real answer.

Real closes produce ordinary returns. A window frame's closes are not
guaranteed to be ordinary: the aggregator emits a null close for a window
holding no source candle, and nothing between a provider and this step
promises a close is positive, non-zero, or of a sane magnitude. Left
alone, polars is happy to hand back ``inf``, ``-inf``, or ``NaN`` from a
division or a logarithm, and each of those travels through a downstream
fit as a number rather than as an absence.

The claim under test is a single one: every emitted return is a finite
float or null, whatever the closes were. The cases below are the ways to
reach the exception, each checked on its own so that a regression names
the case it broke, plus one sweep asserting the claim outright over a
fixture that contains all of them at once.

The one case deliberately NOT nulled is a simple return over a negative
denominator: it is a finite number arrived at by the stated formula, and
this step has no opinion about the sign of a price. That belongs upstream
in source validation, where a negative price is a finding about the data
rather than a shrug about one row.
"""

import math
import sys

import polars as pl
import pytest

from ohlc_toolkit.returns import (
    ReturnMethod,
    add_backward_returns,
    add_forward_returns,
    backward_return_column,
    forward_available_at_column,
    forward_return_column,
)
from tests.test_returns.factories import CADENCE, TIME_BASE, return_frame

# One close per 1m tick, walking through every way an ordinary formula
# stops having a real answer. Index by index:
#
#   0  128.0   an ordinary close, for the pair below it
#   1  160.0   ordinary:            160 / 128        is a real ratio
#   2    0.0   a zero NUMERATOR:      0 / 160        is real; ln(0) is not
#   3  100.0   a zero DENOMINATOR:  100 / 0          is +inf
#   4  -50.0   a negative numerator: -50 / 100       is real; its ln is not
#   5  100.0   a negative DENOMINATOR: 100 / -50     is real; its ln is not
#   6   None   a window that reported no close at all
#   7  100.0   the row after that one, whose denominator is the null
_DEGENERATE_CLOSES = (128.0, 160.0, 0.0, 100.0, -50.0, 100.0, None, 100.0)
_DEGENERATE_OFFSETS = (0, 60, 120, 180, 240, 300, 360, 420)

# Backward simple returns over that fixture at a one-cadence horizon.
# Derived one row at a time from the pairs named above:
#
#   t=0    no counterpart                          -> null
#   t=60   160 / 128 - 1 = 1.25 - 1                -> 0.25
#   t=120    0 / 160 - 1 = 0    - 1                -> -1.0
#   t=180  100 / 0   is +inf, which is not a return -> null
#   t=240  -50 / 100 - 1 = -0.5 - 1                -> -1.5
#   t=300  100 / -50 - 1 = -2   - 1                -> -3.0
#   t=360  the numerator is null                   -> null
#   t=420  the denominator is null                 -> null
_DEGENERATE_BACKWARD_SIMPLE = [None, 0.25, -1.0, None, -1.5, -3.0, None, None]

# The same fixture under the log formula. Only the first pair has a
# positive finite ratio; ln(0) is -inf and the logarithm of a negative
# ratio is not a real number at all:
#
#   t=60   ln(160 / 128) = ln(1.25)               -> the one real value
#   t=120  ln(0)   is -inf                        -> null
#   t=180  ln(+inf) is +inf                       -> null
#   t=240  ln(-0.5) is not real                   -> null
#   t=300  ln(-2)   is not real                   -> null
_DEGENERATE_LOG_DEFINED_INDEX = 1
_DEGENERATE_LOG_DEFINED_RATIO = 1.25

# Two independent `ln` implementations may disagree in the last place.
_LOG_TOLERANCE = 1e-15

# The extremes of the double range, as far apart as two closes can be.
# Their quotient overflows to infinity in one order and underflows to
# zero in the other, and both are reachable from a corrupted feed.
_LARGEST_DOUBLE = sys.float_info.max
_SMALLEST_DOUBLE = 5e-324


def _degenerate_frame() -> pl.DataFrame:
    """Build the fixture holding every degenerate pair at once."""
    return return_frame(_DEGENERATE_OFFSETS, _DEGENERATE_CLOSES)


def _values(frame: pl.DataFrame, column: str) -> list[float | None]:
    """Read one column out as a plain Python list, nulls included."""
    return frame.get_column(column).to_list()


class TestSimpleReturnsOverDegenerateCloses:
    """A simple return is emitted when it is a finite number, and only then."""

    def test_every_row_of_the_degenerate_fixture(self) -> None:
        """One assertion covering all six pairs, against hand-derived literals."""
        result = add_backward_returns(
            _degenerate_frame(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(result, column) == _DEGENERATE_BACKWARD_SIMPLE

    def test_a_zero_denominator_is_null_and_not_infinity(self) -> None:
        """A division by zero is ``+/-inf`` in polars; a return is neither."""
        frame = return_frame((0, 60, 120), (0.0, 100.0, -100.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        # 100 / 0 is +inf and -100 / 100 - 1 is exactly -2.0.
        assert _values(result, backward_return_column(ReturnMethod.SIMPLE, "1m")) == [
            None,
            None,
            -2.0,
        ]

    def test_zero_over_zero_is_null_and_not_nan(self) -> None:
        """``0 / 0`` is ``NaN`` in polars, which compares equal to nothing."""
        frame = return_frame((0, 60), (0.0, 0.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert _values(result, backward_return_column(ReturnMethod.SIMPLE, "1m")) == [
            None,
            None,
        ]

    def test_a_zero_numerator_is_a_return_of_minus_one(self) -> None:
        """A price that fell to zero fell by 100%: that is a real answer."""
        frame = return_frame((0, 60), (100.0, 0.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert _values(result, backward_return_column(ReturnMethod.SIMPLE, "1m")) == [
            None,
            -1.0,
        ]

    def test_a_negative_denominator_is_reported_rather_than_nulled(self) -> None:
        """Finite is finite. Judging the sign of a price is not this step's job.

        Source validation is where a negative price is a finding about
        the data; here it is one row of arithmetic that came out to an
        ordinary number, and hiding it would hide the input too.
        """
        frame = return_frame((0, 60), (-50.0, 100.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        # 100 / -50 - 1 = -2 - 1, exactly.
        assert _values(result, backward_return_column(ReturnMethod.SIMPLE, "1m")) == [
            None,
            -3.0,
        ]

    def test_a_ratio_that_overflows_is_null_and_not_infinity(self) -> None:
        """Two closes far enough apart divide to ``inf``, which is not a return."""
        frame = return_frame((0, 60), (_SMALLEST_DOUBLE, _LARGEST_DOUBLE))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert _values(result, backward_return_column(ReturnMethod.SIMPLE, "1m")) == [
            None,
            None,
        ]

    def test_a_ratio_that_underflows_is_a_return_of_minus_one(self) -> None:
        """The other order underflows to zero, which is finite and reported."""
        frame = return_frame((0, 60), (_LARGEST_DOUBLE, _SMALLEST_DOUBLE))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert _values(result, backward_return_column(ReturnMethod.SIMPLE, "1m")) == [
            None,
            -1.0,
        ]

    def test_the_chosen_extremes_really_do_overflow_and_underflow(self) -> None:
        """Guard the fixture: the two constants must behave as claimed."""
        assert math.isinf(_LARGEST_DOUBLE / _SMALLEST_DOUBLE)
        assert _SMALLEST_DOUBLE / _LARGEST_DOUBLE == 0.0


class TestLogReturnsOverDegenerateCloses:
    """A log return exists only where the ratio is positive and finite."""

    def test_only_the_one_positive_finite_ratio_survives(self) -> None:
        """Every other pair in the fixture has no real logarithm."""
        result = add_backward_returns(
            _degenerate_frame(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.LOG,
        )
        values = _values(result, backward_return_column(ReturnMethod.LOG, "1m"))

        for index, value in enumerate(values):
            if index == _DEGENERATE_LOG_DEFINED_INDEX:
                assert value == pytest.approx(
                    math.log(_DEGENERATE_LOG_DEFINED_RATIO), rel=_LOG_TOLERANCE
                )
            else:
                assert value is None

    def test_the_logarithm_of_zero_is_null_and_not_negative_infinity(self) -> None:
        """A price that fell to zero has a simple return but no log return."""
        frame = return_frame((0, 60), (100.0, 0.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        assert _values(result, backward_return_column(ReturnMethod.LOG, "1m")) == [
            None,
            None,
        ]

    def test_the_logarithm_of_a_negative_ratio_is_null_and_not_nan(self) -> None:
        """A sign change has no real logarithm, and ``NaN`` is not a value."""
        frame = return_frame((0, 60), (-50.0, 100.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        assert _values(result, backward_return_column(ReturnMethod.LOG, "1m")) == [
            None,
            None,
        ]


class TestForwardReturnsOverDegenerateCloses:
    """The forward direction nulls the same values, and still states availability."""

    def test_the_forward_values_are_the_backward_values_one_row_earlier(self) -> None:
        """The consistency identity survives every degenerate pair.

        Both directions null the same rows for the same reason, so the
        forward column over this fixture is the backward column with its
        leading null moved to the end.
        """
        result = add_forward_returns(
            _degenerate_frame(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        expected = [*_DEGENERATE_BACKWARD_SIMPLE[1:], None]
        assert _values(result, forward_return_column(ReturnMethod.SIMPLE, "1m")) == (
            expected
        )

    def test_availability_is_still_stated_where_the_value_is_degenerate(self) -> None:
        """A value nulled by its own arithmetic still belongs to an instant.

        The row at ``t=120`` divides by zero, so it has no value; ``t +
        H`` is still exactly when it would have been known, and nulling
        that too would conflate an unfound counterpart with an
        unrepresentable number.
        """
        result = add_forward_returns(
            _degenerate_frame(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        available_at = forward_available_at_column(ReturnMethod.SIMPLE, "1m")
        assert result.get_column(available_at).null_count() == 0
        assert _values(result, available_at) == [
            TIME_BASE + offset + 60 for offset in _DEGENERATE_OFFSETS
        ]


class TestNothingNonFiniteIsEverEmitted:
    """The whole claim, stated once, over everything that could break it."""

    @pytest.mark.parametrize("method", list(ReturnMethod), ids=lambda m: m.value)
    @pytest.mark.parametrize(
        "closes",
        [
            _DEGENERATE_CLOSES,
            (_SMALLEST_DOUBLE, _LARGEST_DOUBLE),
            (_LARGEST_DOUBLE, _SMALLEST_DOUBLE),
            (0.0, 0.0),
            (-1.0, -1.0),
        ],
        ids=["degenerate", "overflow", "underflow", "zeroes", "negatives"],
    )
    def test_no_emitted_value_is_infinite_or_nan(
        self, method: ReturnMethod, closes: tuple[float | None, ...]
    ) -> None:
        """Every return column holds finite floats and nulls, and nothing else."""
        frame = return_frame(tuple(60 * index for index in range(len(closes))), closes)
        composed = add_forward_returns(
            add_backward_returns(frame, horizon="1m", cadence=CADENCE, method=method),
            horizon="1m",
            cadence=CADENCE,
            method=method,
        )

        for column in (
            backward_return_column(method, "1m"),
            forward_return_column(method, "1m"),
        ):
            series = composed.get_column(column)
            assert series.is_infinite().fill_null(value=False).sum() == 0
            assert series.is_nan().fill_null(value=False).sum() == 0


if __name__ == "__main__":
    pytest.main([__file__])
