"""Forward returns, and the availability that has to travel with them.

A forward return at close time ``t`` over ``H`` describes the interval
``[t, t + H]``. Its value is not available at ``t``, and the tests here
are as much about that fact being impossible to miss as about the
arithmetic being right: the column is named so it does not read like a
feature, and every row states the instant its value may first be used.

As in :mod:`tests.test_returns.test_backward`, the expected values are
exact literals derived by hand from the fixture closes, with the
arithmetic in a comment beside them. The fixture closes are dyadic, so
the simple returns are compared with no tolerance at all.
"""

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ohlc_toolkit.returns import (
    ReturnMethod,
    add_backward_returns,
    add_forward_returns,
    backward_return_column,
    forward_available_at_column,
    forward_return_column,
)
from ohlc_toolkit.temporal import Duration
from tests.test_returns.factories import (
    CADENCE,
    GAP_FREE_OFFSETS,
    GAPPED_OFFSETS,
    TIME_BASE,
    gap_free_frame,
    gapped_frame,
)

# Forward simple returns over the gapped fixture at a one-cadence
# horizon. Row by row, from closes (128, 160, 320, 80, 100) at offsets
# (0, 60, 120, 240, 300):
#
#   t=0    160 / 128 - 1 = 1.25 - 1                -> 0.25
#   t=60   320 / 160 - 1 = 2    - 1                -> 1.0
#   t=120  counterpart t=180 is the missing tick   -> null
#   t=240  100 / 80  - 1 = 1.25 - 1                -> 0.25
#   t=300  counterpart t=360 is past the frame     -> null
_GAPPED_FORWARD_1M = [0.25, 1.0, None, 0.25, None]

# The same fixture at a two-cadence horizon:
#
#   t=0    320 / 128 - 1 = 2.5  - 1                -> 1.5
#   t=60   counterpart t=180 is the missing tick   -> null
#   t=120  80  / 320 - 1 = 0.25 - 1                -> -0.75
#   t=240  counterpart t=360 is past the frame     -> null
#   t=300  counterpart t=420 is past the frame     -> null
_GAPPED_FORWARD_2M = [1.5, None, -0.75, None, None]

# What a shift by two ROWS forward would have produced instead. The
# frame's rows are two cadences apart across the gap, so row i+2 is not
# the row at t+120 for every i:
#
#   i=1 (t=60)  -> row 3 (t=240): 80  / 160 - 1    -> -0.5    (not null)
#   i=2 (t=120) -> row 4 (t=300): 100 / 320 - 1    -> -0.6875 (not -0.75)
_GAPPED_ROW_SHIFT_2M = [1.5, -0.5, -0.6875, None, None]

# Forward simple returns over the gap-free fixture at a one-cadence
# horizon, from closes (128, 160, 320, 80, 100, 25):
#
#   t=0    160 / 128 - 1 = 1.25 - 1                -> 0.25
#   t=60   320 / 160 - 1 = 2    - 1                -> 1.0
#   t=120  80  / 320 - 1 = 0.25 - 1                -> -0.75
#   t=180  100 / 80  - 1 = 1.25 - 1                -> 0.25
#   t=240  25  / 100 - 1 = 0.25 - 1                -> -0.75
#   t=300  counterpart t=360 is past the frame     -> null
_GAP_FREE_FORWARD_1M = [0.25, 1.0, -0.75, 0.25, -0.75, None]


def _values(frame: pl.DataFrame, column: str) -> list[float | None]:
    """Read one column out as a plain Python list, nulls included."""
    return frame.get_column(column).to_list()


class TestForwardSimpleReturns:
    """A forward return at ``t`` relates close(t + H) to close(t)."""

    def test_one_cadence_horizon_over_a_gapped_frame(self) -> None:
        """The counterpart is the row at exactly ``t + H``, or nothing."""
        result = add_forward_returns(
            gapped_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = forward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(result, column) == _GAPPED_FORWARD_1M

    def test_two_cadence_horizon_reaches_across_the_gap(self) -> None:
        """Looking two cadences ahead skips the missing tick, or lands on it."""
        result = add_forward_returns(
            gapped_frame(), horizon="2m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = forward_return_column(ReturnMethod.SIMPLE, "2m")
        assert _values(result, column) == _GAPPED_FORWARD_2M

    def test_a_shift_by_rows_would_have_disagreed(self) -> None:
        """Guard the discriminator: the two answers must actually differ."""
        assert _GAPPED_ROW_SHIFT_2M != _GAPPED_FORWARD_2M
        assert _GAPPED_ROW_SHIFT_2M[1] is not None
        assert _GAPPED_FORWARD_2M[1] is None
        assert _GAPPED_ROW_SHIFT_2M[2] == -0.6875  # noqa: PLR2004 - shift, not time
        assert _GAPPED_FORWARD_2M[2] == -0.75  # noqa: PLR2004 - time, not shift

    def test_one_cadence_horizon_over_a_gap_free_frame(self) -> None:
        """With every tick present, only the last row has nothing ahead of it."""
        result = add_forward_returns(
            gap_free_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = forward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(result, column) == _GAP_FREE_FORWARD_1M

    def test_both_columns_are_appended_and_the_input_columns_are_untouched(
        self,
    ) -> None:
        """A forward call adds the value and its availability, in that order."""
        frame = gap_free_frame()
        result = add_forward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert result.columns == [
            "close_time",
            "close",
            forward_return_column(ReturnMethod.SIMPLE, "1m"),
            forward_available_at_column(ReturnMethod.SIMPLE, "1m"),
        ]
        assert_frame_equal(
            result.select("close_time", "close"), frame, check_exact=True
        )

    def test_the_input_frame_is_never_mutated(self) -> None:
        """The frame handed in is byte-for-byte the same afterwards."""
        frame = gapped_frame()
        before = frame.clone()
        result = add_forward_returns(
            frame, horizon="2m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert_frame_equal(frame, before, check_exact=True)
        assert result is not frame

    def test_an_empty_frame_gains_both_columns_with_the_right_dtypes(self) -> None:
        """No rows is not an error; the schema still says what was asked for."""
        result = add_forward_returns(
            gap_free_frame().clear(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        assert result.height == 0
        assert result.schema[forward_return_column(ReturnMethod.SIMPLE, "1m")] == (
            pl.Float64
        )
        assert result.schema[
            forward_available_at_column(ReturnMethod.SIMPLE, "1m")
        ] == (pl.Int64)


class TestAvailability:
    """Every forward value carries the instant it may first be used."""

    @pytest.mark.parametrize(
        ("horizon", "horizon_seconds"), [("1m", 60), ("2m", 120), ("15m", 900)]
    )
    def test_available_at_is_close_time_plus_the_horizon_on_every_row(
        self, horizon: str, horizon_seconds: int
    ) -> None:
        """Derived here from the fixture's own offsets, not read back."""
        result = add_forward_returns(
            gapped_frame(),
            horizon=horizon,
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        expected = [TIME_BASE + offset + horizon_seconds for offset in GAPPED_OFFSETS]
        column = forward_available_at_column(ReturnMethod.SIMPLE, horizon)
        assert _values(result, column) == expected

    def test_available_at_shares_the_close_time_dtype(self) -> None:
        """The two columns are the same kind of instant, so the same kind."""
        result = add_forward_returns(
            gapped_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = forward_available_at_column(ReturnMethod.SIMPLE, "1m")
        assert result.schema[column] == result.schema["close_time"] == pl.Int64

    def test_available_at_is_stated_even_where_the_value_is_null(self) -> None:
        """A null return does not make the availability column null too.

        The horizon and the row's own close time are both known whatever
        the lookup found, so ``t + H`` is always statable. Nulling it as
        well would make one column's null mean two different things and
        would throw away the only fact still standing.
        """
        result = add_forward_returns(
            gapped_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        value_column = forward_return_column(ReturnMethod.SIMPLE, "1m")
        available_column = forward_available_at_column(ReturnMethod.SIMPLE, "1m")

        null_rows = result.filter(pl.col(value_column).is_null())
        assert null_rows.height > 0, "fixture must include an unfound counterpart"
        assert null_rows.get_column(available_column).null_count() == 0
        assert _values(null_rows, available_column) == [
            TIME_BASE + 120 + 60,
            TIME_BASE + 300 + 60,
        ]

    def test_the_availability_column_is_never_the_close_time(self) -> None:
        """A strictly positive horizon always moves the instant forward."""
        result = add_forward_returns(
            gap_free_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = forward_available_at_column(ReturnMethod.SIMPLE, "1m")
        assert (result.get_column(column) > result.get_column("close_time")).all()


class TestForwardColumnNames:
    """A forward value must not read like a feature in a column list."""

    def test_the_value_column_is_prefixed_forward(self) -> None:
        """The prefix is the warning that survives being read out of context."""
        assert forward_return_column(ReturnMethod.SIMPLE, "1m") == (
            "forward_return_simple_1m"
        )
        assert forward_return_column(ReturnMethod.LOG, "2m") == "forward_return_log_2m"

    def test_the_availability_column_is_derived_from_the_value_column(self) -> None:
        """Pairing them by name is what keeps them together in a projection."""
        value = forward_return_column(ReturnMethod.SIMPLE, "1m")
        assert forward_available_at_column(ReturnMethod.SIMPLE, "1m") == (
            f"{value}_available_at"
        )

    def test_the_two_directions_never_name_the_same_column(self) -> None:
        """Backward and forward over one horizon compose onto one frame."""
        assert forward_return_column(
            ReturnMethod.SIMPLE, "1m"
        ) != backward_return_column(ReturnMethod.SIMPLE, "1m")

        frame = add_backward_returns(
            gap_free_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        both = add_forward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert both.columns == [
            "close_time",
            "close",
            backward_return_column(ReturnMethod.SIMPLE, "1m"),
            forward_return_column(ReturnMethod.SIMPLE, "1m"),
            forward_available_at_column(ReturnMethod.SIMPLE, "1m"),
        ]

    def test_the_horizon_in_the_name_is_the_canonical_spelling(self) -> None:
        """Two spellings of one horizon name one pair of columns, not two."""
        assert forward_return_column(
            ReturnMethod.SIMPLE, "90s"
        ) == forward_return_column(ReturnMethod.SIMPLE, Duration(90))
        assert forward_available_at_column(
            ReturnMethod.SIMPLE, "90s"
        ) == forward_available_at_column(ReturnMethod.SIMPLE, Duration(90))

    def test_the_two_methods_carry_their_own_availability_columns(self) -> None:
        """Each value column owns its sidecar, so two methods compose."""
        simple = add_forward_returns(
            gap_free_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        both = add_forward_returns(
            simple, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        assert both.columns[-2:] == [
            forward_return_column(ReturnMethod.LOG, "1m"),
            forward_available_at_column(ReturnMethod.LOG, "1m"),
        ]
        assert _values(
            both, forward_available_at_column(ReturnMethod.SIMPLE, "1m")
        ) == _values(both, forward_available_at_column(ReturnMethod.LOG, "1m"))


class TestForwardIsNotAvailableYet:
    """The lookahead the naming warns about is demonstrated, not asserted."""

    def test_truncating_the_future_away_empties_every_forward_value(self) -> None:
        """A row that knows only its own past has no forward return at all.

        This is the mirror image of the causality test for backward
        returns: cutting the frame off at ``t`` leaves every backward
        value intact and every forward value null, which is exactly the
        difference the two column namings exist to make visible.
        """
        frame = gap_free_frame()
        column = forward_return_column(ReturnMethod.SIMPLE, "1m")

        for close_time in frame.get_column("close_time").to_list():
            truncated = add_forward_returns(
                frame.filter(pl.col("close_time") <= close_time),
                horizon="1m",
                cadence=CADENCE,
                method=ReturnMethod.SIMPLE,
            )
            assert _values(truncated.tail(1), column) == [None]


class TestForwardBackwardIdentity:
    """The two directions are one relation read from opposite ends."""

    @pytest.mark.parametrize(("horizon", "horizon_seconds"), [("1m", 60), ("2m", 120)])
    @pytest.mark.parametrize("method", list(ReturnMethod), ids=lambda m: m.value)
    @pytest.mark.parametrize("gap_free", [True, False], ids=["gap_free", "gapped"])
    def test_forward_at_t_equals_backward_at_t_plus_horizon(
        self,
        horizon: str,
        horizon_seconds: int,
        method: ReturnMethod,
        gap_free: bool,
    ) -> None:
        """Bit for bit: both are close(t + H) over close(t) by one formula.

        Checked in both directions of the implication. Where ``t + H`` is
        a row of the frame, the two values must be identical -- not close,
        identical, since they are the same expression over the same two
        doubles. Where it is not, the forward value must be null.
        """
        frame = gap_free_frame() if gap_free else gapped_frame()
        offsets = GAP_FREE_OFFSETS if gap_free else GAPPED_OFFSETS

        forward = add_forward_returns(
            frame, horizon=horizon, cadence=CADENCE, method=method
        )
        backward = add_backward_returns(
            frame, horizon=horizon, cadence=CADENCE, method=method
        )
        forward_by_time = dict(
            zip(
                forward.get_column("close_time").to_list(),
                _values(forward, forward_return_column(method, horizon)),
                strict=True,
            )
        )
        backward_by_time = dict(
            zip(
                backward.get_column("close_time").to_list(),
                _values(backward, backward_return_column(method, horizon)),
                strict=True,
            )
        )

        matched = 0
        for offset in offsets:
            close_time = TIME_BASE + offset
            available_at = close_time + horizon_seconds
            if available_at in backward_by_time:
                matched += 1
                assert forward_by_time[close_time] == backward_by_time[available_at]
            else:
                assert forward_by_time[close_time] is None
        assert matched > 0, "fixture must pair at least one row with its future"

    def test_the_identity_holds_at_the_row_the_availability_column_names(self) -> None:
        """The row a value becomes available at is the row that reproduces it.

        Stated through ``available_at`` rather than through arithmetic
        repeated in the test, so the availability column is what is being
        checked and not just the horizon.
        """
        frame = gap_free_frame()
        forward = add_forward_returns(
            frame, horizon="2m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        backward = add_backward_returns(
            frame, horizon="2m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )

        joined = forward.join(
            backward.select(
                pl.col("close_time").alias(
                    forward_available_at_column(ReturnMethod.SIMPLE, "2m")
                ),
                pl.col(backward_return_column(ReturnMethod.SIMPLE, "2m")),
            ),
            on=forward_available_at_column(ReturnMethod.SIMPLE, "2m"),
            how="inner",
        )
        assert joined.height > 0, "fixture must pair at least one row with its future"
        assert _values(
            joined, forward_return_column(ReturnMethod.SIMPLE, "2m")
        ) == _values(joined, backward_return_column(ReturnMethod.SIMPLE, "2m"))


if __name__ == "__main__":
    pytest.main([__file__])
