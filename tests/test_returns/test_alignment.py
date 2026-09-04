"""What a return primitive refuses at its own boundary, and why.

Both directions share one set of refusals, so every test that is about
the frame rather than about the arithmetic is parametrized over both
entry points: a rule one direction enforced and the other did not would
be worse than no rule.

The scenarios that are not refusals are here too, because a boundary is
defined as much by what it lets through: a frame carrying only the two
columns this step reads, a frame carrying all nine the aggregator emits,
a frame whose rows are not in time order, and a frame with null closes in
it -- an absent observation is data, not a malformed input.
"""

from collections.abc import Callable

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
from ohlc_toolkit.temporal import ConfigError
from ohlc_toolkit.windows import compute_windows
from tests.test_returns.factories import (
    CADENCE,
    CADENCE_SECONDS,
    GAP_FREE_CLOSES,
    GAP_FREE_OFFSETS,
    TIME_BASE,
    gap_free_frame,
    return_frame,
)
from tests.test_windows.factories import frame_from_rows, profile_for

# The two entry points every frame-shaped rule must fire in.
AddReturns = Callable[..., pl.DataFrame]
_ENTRY_POINTS = [add_backward_returns, add_forward_returns]
_ENTRY_POINT_IDS = ["backward", "forward"]

# Each entry point paired with the helper that names its value column.
_ENTRY_POINTS_AND_COLUMNS = [
    (add_backward_returns, backward_return_column),
    (add_forward_returns, forward_return_column),
]

# The bounds of the Int64 column close times are held in.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# A permutation of the six-row gap-free fixture that leaves no row in its
# original position (a derangement, and a test below checks that it is),
# so a call that quietly sorted or assumed order would be caught wherever
# it looked.
_SHUFFLE = [3, 0, 5, 2, 1, 4]


def _add(
    entry_point: AddReturns,
    frame: pl.DataFrame,
    *,
    horizon: str = "1m",
    method: ReturnMethod = ReturnMethod.SIMPLE,
) -> pl.DataFrame:
    """Call one of the two entry points with this suite's usual arguments."""
    return entry_point(frame, horizon=horizon, cadence=CADENCE, method=method)


class TestRequiredColumns:
    """The two columns this step reads are required; nothing else is."""

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    @pytest.mark.parametrize("dropped", ["close_time", "close"])
    def test_a_frame_missing_a_read_column_is_refused(
        self, entry_point: AddReturns, dropped: str
    ) -> None:
        """A column that cannot be read cannot be silently worked around."""
        with pytest.raises(ConfigError, match=dropped):
            _add(entry_point, gap_free_frame().drop(dropped))

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_frame_missing_both_names_them_both(
        self, entry_point: AddReturns
    ) -> None:
        """One refusal reports every missing column, not just the first."""
        frame = pl.DataFrame({"volume": [1.0, 2.0]})
        with pytest.raises(ConfigError) as caught:
            _add(entry_point, frame)
        assert "close_time" in str(caught.value)
        assert "close" in str(caught.value)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_frame_carrying_only_the_columns_read_is_accepted(
        self, entry_point: AddReturns
    ) -> None:
        """Nothing beyond close_time and close is required.

        A caller who has projected a window frame down to what this step
        consults is not doing anything wrong, and must not be refused for
        dropping columns no rule reads.
        """
        result = _add(entry_point, gap_free_frame())
        assert result.height == len(GAP_FREE_OFFSETS)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_columns_the_step_does_not_read_are_carried_through_untouched(
        self, entry_point: AddReturns
    ) -> None:
        """Every other column survives with its values and its position."""
        frame = gap_free_frame().with_columns(
            pl.Series("open", [1.0] * len(GAP_FREE_OFFSETS), dtype=pl.Float64),
            pl.Series("src_count", list(range(len(GAP_FREE_OFFSETS))), dtype=pl.UInt32),
        )
        result = _add(entry_point, frame)
        assert result.columns[: frame.width] == frame.columns
        assert_frame_equal(result.select(frame.columns), frame, check_exact=True)


class TestRequiredDtypes:
    """Exactly the dtypes the aggregator emits, and no other width."""

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    @pytest.mark.parametrize(
        "dtype",
        [pl.Int8, pl.Int16, pl.Int32, pl.UInt32, pl.UInt64, pl.Float64],
        ids=["Int8", "Int16", "Int32", "UInt32", "UInt64", "Float64"],
    )
    def test_a_non_int64_close_time_is_refused(
        self, entry_point: AddReturns, dtype: pl.DataType
    ) -> None:
        """Only the width the aggregator emits will do.

        A narrower column wraps when a horizon is added to it, a
        ``UInt64`` cannot be widened safely near the top of its range,
        and a floating close time has no exact equality to join on.
        Refusing every other width keeps all three failures at this
        module's boundary and in its own words.

        The close times here are small enough to be held by every dtype
        under test, ``Int8`` included, so the refusal is about the width
        the column is declared in and not about values that would not
        fit in it.
        """
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [0, 60, 120], dtype=dtype),
                pl.Series("close", [100.0, 110.0, 120.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64"):
            _add(entry_point, frame)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    @pytest.mark.parametrize(
        "dtype",
        [pl.Float32, pl.Int64, pl.String],
        ids=["Float32", "Int64", "String"],
    )
    def test_a_non_float64_close_is_refused(
        self, entry_point: AddReturns, dtype: pl.DataType
    ) -> None:
        """A return is Float64 arithmetic over a Float64 column."""
        frame = gap_free_frame().with_columns(pl.col("close").cast(dtype))
        with pytest.raises(ConfigError, match="Float64"):
            _add(entry_point, frame)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_an_unwieldy_dtype_is_echoed_within_a_bound(
        self, entry_point: AddReturns
    ) -> None:
        """A refusal never turns one bad column into an unbounded message."""
        wide_struct = {f"field_number_{index}": index for index in range(40)}
        frame = gap_free_frame().with_columns(
            pl.Series("close_time", [wide_struct] * len(GAP_FREE_OFFSETS))
        )
        with pytest.raises(ConfigError) as caught:
            _add(entry_point, frame)
        assert len(str(caught.value)) < 400  # noqa: PLR2004 - a bound, not a value
        assert "chars total" in str(caught.value)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_the_aggregators_own_output_is_accepted(
        self, entry_point: AddReturns
    ) -> None:
        """The dtypes required are the dtypes a real window frame carries.

        Built by running the aggregator rather than written out by hand,
        so this cannot drift away from what it actually emits.
        """
        source = frame_from_rows(
            tuple(
                (TIME_BASE + offset, 1.0, 2.0, 0.5, close, 1.0)
                for offset, close in zip(GAP_FREE_OFFSETS, GAP_FREE_CLOSES, strict=True)
            )
        )
        window_frame = compute_windows(
            source,
            profile_for(CADENCE_SECONDS),
            window=CADENCE,
            emit_every=CADENCE,
            materialization="skip_warmup",
        )
        assert window_frame.schema["close_time"] == pl.Int64
        assert window_frame.schema["close"] == pl.Float64

        result = _add(entry_point, window_frame)
        assert result.width > window_frame.width
        assert_frame_equal(
            result.select(window_frame.columns), window_frame, check_exact=True
        )


class TestCloseTimeIsAKey:
    """A counterpart join needs a key: one close time per row, and present."""

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_duplicate_close_time_is_refused(self, entry_point: AddReturns) -> None:
        """Two rows claiming one instant would multiply rows, not pick one.

        A self-join over a duplicated key fans out: the output would be
        longer than the input, and the surplus rows would look exactly
        like data. There is no defensible tie-break between two closes
        claiming the same instant, so none is invented.
        """
        frame = return_frame((0, 60, 60, 120), (100.0, 110.0, 111.0, 120.0))
        with pytest.raises(ConfigError, match="unique"):
            _add(entry_point, frame)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_the_duplicate_refusal_names_the_repeated_close_time(
        self, entry_point: AddReturns
    ) -> None:
        """The offending instant is reported, not just its existence."""
        frame = return_frame((0, 60, 60, 120), (100.0, 110.0, 111.0, 120.0))
        with pytest.raises(ConfigError) as caught:
            _add(entry_point, frame)
        assert str(TIME_BASE + 60) in str(caught.value)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_null_close_time_is_refused(self, entry_point: AddReturns) -> None:
        """A row with no close time has neither a counterpart nor availability."""
        frame = pl.DataFrame(
            [
                pl.Series(
                    "close_time", [TIME_BASE, None, TIME_BASE + 120], dtype=pl.Int64
                ),
                pl.Series("close", [100.0, 110.0, 120.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="null"):
            _add(entry_point, frame)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_null_close_is_not_refused(self, entry_point: AddReturns) -> None:
        """An absent observation is data. The aggregator emits these itself.

        A window holding no source candle reports a null close, and a
        return over one is simply unknown -- which is a null value, not a
        malformed frame.
        """
        frame = return_frame((0, 60, 120), (100.0, None, 120.0))
        result = _add(entry_point, frame)
        assert result.height == frame.height


class TestRowOrderIsNotAssumed:
    """The join is on equality, so it does not care what order rows arrive in."""

    @pytest.mark.parametrize(
        ("entry_point", "column_for"), _ENTRY_POINTS_AND_COLUMNS, ids=_ENTRY_POINT_IDS
    )
    def test_an_unsorted_frame_gets_the_same_values_per_close_time(
        self, entry_point: AddReturns, column_for: Callable[..., str]
    ) -> None:
        """Shuffling the input permutes the output and changes no value."""
        column = column_for(ReturnMethod.SIMPLE, "1m")
        ordered = _add(entry_point, gap_free_frame())
        by_close_time = dict(
            zip(
                ordered.get_column("close_time").to_list(),
                ordered.get_column(column).to_list(),
                strict=True,
            )
        )

        result = _add(entry_point, gap_free_frame()[_SHUFFLE])
        for close_time, value in zip(
            result.get_column("close_time").to_list(),
            result.get_column(column).to_list(),
            strict=True,
        ):
            assert value == by_close_time[close_time]

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_the_input_row_order_is_handed_back_unchanged(
        self, entry_point: AddReturns
    ) -> None:
        """No sort happens, so the caller's row order survives the call."""
        shuffled = gap_free_frame()[_SHUFFLE]
        result = _add(entry_point, shuffled)
        assert (
            result.get_column("close_time").to_list()
            == shuffled.get_column("close_time").to_list()
        )

    def test_the_shuffle_really_does_displace_every_row(self) -> None:
        """Guard the fixture: a fixed point would exempt one row from the claim."""
        assert sorted(_SHUFFLE) == list(range(len(_SHUFFLE)))
        assert all(index != target for index, target in enumerate(_SHUFFLE))

    def test_every_row_of_a_large_shuffled_frame_gets_its_own_counterpart(
        self,
    ) -> None:
        """The value on row i belongs to row i, pinned against a dict oracle.

        The counterpart column is attached to the caller's frame
        POSITIONALLY, so this module is correct only if the join hands
        rows back in exactly the left frame's order -- a guarantee the
        join is asked for explicitly, because the default is documented
        as unspecified. The six-row fixtures cannot catch a reorder that
        only shows up at scale, so this one is a few thousand shuffled
        rows with gaps, and every expected value is derived from a plain
        Python dict keyed by close time -- never from the implementation
        run in a friendlier order.
        """
        spacing = 60
        gap_stride, gap_phase = 7, 3  # every 7th tick, offset 3, is missing
        times = [
            TIME_BASE + spacing * tick
            for tick in range(4001)
            if tick % gap_stride != gap_phase
        ]
        closes = [float(1000 + 3 * (tick % 997)) for tick in range(len(times))]
        # A fixed-stride permutation: gcd(1597, len) == 1 makes it a
        # bijection, and consecutive rows land far apart.
        order = [(index * 1597) % len(times) for index in range(len(times))]
        assert sorted(order) == list(range(len(times)))
        shuffled_times = [times[position] for position in order]
        shuffled_closes = [closes[position] for position in order]

        frame = pl.DataFrame(
            [
                pl.Series("close_time", shuffled_times, dtype=pl.Int64),
                pl.Series("close", shuffled_closes, dtype=pl.Float64),
            ]
        )
        close_by_time = dict(zip(shuffled_times, shuffled_closes, strict=True))
        expected = [
            (close - close_by_time[time - spacing]) / close_by_time[time - spacing]
            if (time - spacing) in close_by_time
            else None
            for time, close in zip(shuffled_times, shuffled_closes, strict=True)
        ]

        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert result.get_column(column).to_list() == expected


class TestHorizonsThatLeaveTheInt64Range:
    """A wrapped close time does not fail to match -- it matches the wrong row."""

    def test_a_forward_horizon_past_the_top_of_the_range_is_refused(self) -> None:
        """Polars wraps Int64 addition silently; this refuses before it can."""
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MAX - 30], dtype=pl.Int64),
                pl.Series("close", [100.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            add_forward_returns(
                frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
            )

    def test_a_backward_horizon_past_the_bottom_of_the_range_is_refused(self) -> None:
        """The same wrap happens downwards, and is refused the same way."""
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MIN + 30], dtype=pl.Int64),
                pl.Series("close", [100.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            add_backward_returns(
                frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
            )

    def test_a_shift_landing_one_past_the_top_is_refused(self) -> None:
        """The refusal boundary is exact: one second past the largest Int64.

        The tests above land 30 seconds past the bound, so a constant
        moved by one would still catch them; this one lands the shifted
        key on exactly ``2**63``, where an off-by-one in the bound is
        the difference between a refusal and a silently wrapped key that
        matches another row.
        """
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MAX - 59], dtype=pl.Int64),
                pl.Series("close", [100.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            add_forward_returns(
                frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
            )

    def test_a_shift_landing_exactly_on_the_top_is_accepted(self) -> None:
        """The largest Int64 itself is a representable key, so it is allowed."""
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MAX - 60], dtype=pl.Int64),
                pl.Series("close", [100.0], dtype=pl.Float64),
            ]
        )
        result = add_forward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        availability = next(c for c in result.columns if c.endswith("_available_at"))
        assert result.get_column(availability).to_list() == [_INT64_MAX]

    def test_a_shift_landing_one_past_the_bottom_is_refused(self) -> None:
        """The bottom boundary is exact too: one second below the smallest Int64."""
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MIN + 59], dtype=pl.Int64),
                pl.Series("close", [100.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            add_backward_returns(
                frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
            )

    def test_a_shift_landing_exactly_on_the_bottom_is_accepted(self) -> None:
        """The smallest Int64 itself is a representable key, so it is allowed."""
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MIN + 60], dtype=pl.Int64),
                pl.Series("close", [100.0], dtype=pl.Float64),
            ]
        )
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        value = next(c for c in result.columns if c.startswith("backward_return"))
        assert result.get_column(value).to_list() == [None]

    def test_only_the_largest_close_time_overflowing_forward_is_refused(self) -> None:
        """Both extremes guard the shift, not just one of them.

        Here the SMALLEST close time has all the room in the world going
        forward and only the LARGEST wraps -- so a guard that consulted
        the minimum alone would let the shift proceed, the wrapped key
        would land deep in the negatives, and it could match an
        unrelated row: over this frame it matches row 0 and emits
        ``7.0 / 9.0 - 1``, a plausible finite return built from the
        wrong close. Refusal is the only honest answer.
        """
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MIN + 99, _INT64_MAX], dtype=pl.Int64),
                pl.Series("close", [7.0, 9.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            add_forward_returns(
                frame, horizon="100s", cadence="100s", method=ReturnMethod.SIMPLE
            )

    def test_only_the_smallest_close_time_overflowing_backward_is_refused(
        self,
    ) -> None:
        """The mirror: looking back, it is the smallest close time that wraps."""
        frame = pl.DataFrame(
            [
                pl.Series("close_time", [_INT64_MIN, _INT64_MAX - 99], dtype=pl.Int64),
                pl.Series("close", [7.0, 9.0], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            add_backward_returns(
                frame, horizon="100s", cadence="100s", method=ReturnMethod.SIMPLE
            )

    def test_the_opposite_direction_over_the_same_frame_is_fine(self) -> None:
        """The bound is the shift this call performs, not the horizon alone.

        A close time near the top of the range has room to look back and
        none to look forward, and the refusal has to tell those apart or
        it is refusing arithmetic that would have been exact.
        """
        frame = pl.DataFrame(
            [
                pl.Series(
                    "close_time", [_INT64_MAX - 90, _INT64_MAX - 30], dtype=pl.Int64
                ),
                pl.Series("close", [100.0, 125.0], dtype=pl.Float64),
            ]
        )
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        # 125 / 100 - 1 = 1.25 - 1, exactly, at the row 60s after the first.
        assert result.get_column(
            backward_return_column(ReturnMethod.SIMPLE, "1m")
        ).to_list() == [None, 0.25]

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_horizon_too_large_to_be_a_close_time_at_all_is_refused(
        self, entry_point: AddReturns
    ) -> None:
        """A horizon polars cannot hold is refused in this module's words.

        Unrefused, a shift by more than Int64 raises a bare
        ``OverflowError`` from inside polars about a C type -- a foreign
        exception far from the input that caused it. The refusal keeps
        the failure at this boundary.
        """
        with pytest.raises(ConfigError, match="Int64 range"):
            _add(entry_point, gap_free_frame(), horizon="99999999999999999w")

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_an_unrepresentable_horizon_is_refused_even_on_an_empty_frame(
        self, entry_point: AddReturns
    ) -> None:
        """The horizon's own bound does not depend on the frame's contents.

        An empty frame has no extremes to shift, so only the horizon
        literal's own check stands between this configuration and the
        arithmetic; it must hold on its own.
        """
        empty = pl.DataFrame(
            [
                pl.Series("close_time", [], dtype=pl.Int64),
                pl.Series("close", [], dtype=pl.Float64),
            ]
        )
        with pytest.raises(ConfigError, match="Int64 range"):
            _add(entry_point, empty, horizon="99999999999999999w")

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_large_but_representable_horizon_is_accepted(
        self, entry_point: AddReturns
    ) -> None:
        """The bound is the Int64 range, not an opinion about plausibility."""
        result = _add(entry_point, gap_free_frame(), horizon="52w")
        assert result.height == len(GAP_FREE_OFFSETS)

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_an_empty_frame_has_no_close_time_to_overflow(
        self, entry_point: AddReturns
    ) -> None:
        """With no rows there is no arithmetic to do and nothing to refuse."""
        result = _add(entry_point, gap_free_frame().clear(), horizon="52w")
        assert result.height == 0


class TestOutputColumnsAreNeverOverwritten:
    """A step that silently replaced a column would destroy a caller's work."""

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_repeating_the_same_call_is_refused(self, entry_point: AddReturns) -> None:
        """The second call would overwrite the first call's column."""
        once = _add(entry_point, gap_free_frame())
        with pytest.raises(ConfigError, match="already carries"):
            _add(entry_point, once)

    def test_an_existing_availability_column_is_refused_too(self) -> None:
        """Both columns a forward call writes are protected, not just the value."""
        column = forward_available_at_column(ReturnMethod.SIMPLE, "1m")
        frame = gap_free_frame().with_columns(pl.lit(0, dtype=pl.Int64).alias(column))
        with pytest.raises(ConfigError, match=column):
            add_forward_returns(
                frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
            )

    @pytest.mark.parametrize("entry_point", _ENTRY_POINTS, ids=_ENTRY_POINT_IDS)
    def test_a_different_horizon_on_the_same_frame_is_accepted(
        self, entry_point: AddReturns
    ) -> None:
        """Only a real collision is refused; composing horizons is the point."""
        once = _add(entry_point, gap_free_frame(), horizon="1m")
        twice = _add(entry_point, once, horizon="2m")
        assert twice.width > once.width


if __name__ == "__main__":
    pytest.main([__file__])
