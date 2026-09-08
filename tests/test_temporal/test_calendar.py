"""Tests for the pure Int64-seconds decomposition and its cyclic columns."""

import math

import polars as pl
import pytest

from ohlc_toolkit.temporal.calendar import (
    CAL_DOW_COS,
    CAL_DOW_SIN,
    CAL_MOD_COS,
    CAL_MOD_SIN,
    SECONDS_PER_DAY,
    SECONDS_PER_WEEK,
    add_calendar_columns,
    day_of_week,
    second_of_day,
    second_of_week,
)
from ohlc_toolkit.temporal.errors import ConfigError

# Unix time 0 is a Thursday.
_EPOCH = 0
_EPOCH_DOW = 3
# 1970-01-03 00:00:00 UTC, a Saturday.
_A_SATURDAY = 2 * SECONDS_PER_DAY
_SATURDAY_DOW = 5
# 1970-01-05 00:00:00 UTC, a Monday.
_A_MONDAY = 4 * SECONDS_PER_DAY

_SOME_TIMESTAMPS = (
    0,
    1,
    86_399,
    86_400,
    604_799,
    604_800,
    1_700_000_000,
    1_700_000_000 - 10 * SECONDS_PER_WEEK,
    -1,
    -86_401,
)


class TestScalarDecomposition:
    """The three pinned facts the whole module is built from."""

    def test_epoch_is_a_thursday_at_midnight(self):
        """Epoch time zero is a Thursday: sod = 0, dow = 3."""
        assert second_of_day(_EPOCH) == 0
        assert day_of_week(_EPOCH) == _EPOCH_DOW

    def test_a_monday_midnight_has_second_of_week_zero(self):
        """A Monday 00:00 UTC has sow = 0."""
        assert second_of_week(_A_MONDAY) == 0
        assert day_of_week(_A_MONDAY) == 0

    def test_a_saturday_has_day_of_week_five(self):
        """A Saturday has dow = 5."""
        assert day_of_week(_A_SATURDAY) == _SATURDAY_DOW
        assert second_of_day(_A_SATURDAY) == 0


class TestScalarAndExpressionAgree:
    """The same helper works as a Python function and as a polars expression."""

    @pytest.mark.parametrize("t", _SOME_TIMESTAMPS)
    def test_second_of_day_matches_between_scalar_and_expression(self, t: int):
        """second_of_day agrees between the scalar and the expression form."""
        frame = pl.DataFrame({"t": [t]}, schema={"t": pl.Int64})
        got = frame.select(second_of_day(pl.col("t")).alias("v")).item()
        assert got == second_of_day(t)

    @pytest.mark.parametrize("t", _SOME_TIMESTAMPS)
    def test_second_of_week_matches_between_scalar_and_expression(self, t: int):
        """second_of_week agrees between the scalar and the expression form."""
        frame = pl.DataFrame({"t": [t]}, schema={"t": pl.Int64})
        got = frame.select(second_of_week(pl.col("t")).alias("v")).item()
        assert got == second_of_week(t)

    @pytest.mark.parametrize("t", _SOME_TIMESTAMPS)
    def test_day_of_week_matches_between_scalar_and_expression(self, t: int):
        """day_of_week agrees between the scalar and the expression form."""
        frame = pl.DataFrame({"t": [t]}, schema={"t": pl.Int64})
        got = frame.select(day_of_week(pl.col("t")).alias("v")).item()
        assert got == day_of_week(t)


class TestScalarRanges:
    """Every decomposed value stays inside its stated bounds."""

    @pytest.mark.parametrize("t", _SOME_TIMESTAMPS)
    def test_second_of_day_is_in_range(self, t: int):
        """second_of_day is always in [0, 86400)."""
        assert 0 <= second_of_day(t) < SECONDS_PER_DAY

    @pytest.mark.parametrize("t", _SOME_TIMESTAMPS)
    def test_second_of_week_is_in_range(self, t: int):
        """second_of_week is always in [0, 604800)."""
        assert 0 <= second_of_week(t) < SECONDS_PER_WEEK

    @pytest.mark.parametrize("t", _SOME_TIMESTAMPS)
    def test_day_of_week_is_in_range(self, t: int):
        """day_of_week is always in [0, 6]."""
        assert 0 <= day_of_week(t) <= 6  # noqa: PLR2004 - Sunday, named in the docstring


def _frame(timestamps: tuple[int, ...]) -> pl.DataFrame:
    """Build a minimal Int64 close_time frame from ``timestamps``."""
    return pl.DataFrame(
        {"close_time": list(timestamps)}, schema={"close_time": pl.Int64}
    )


class TestAddCalendarColumns:
    """The frame function: four cyclic columns, correct and reproducible."""

    def test_uses_close_time_by_default(self):
        """With no column name given, close_time is read."""
        out = add_calendar_columns(_frame(_SOME_TIMESTAMPS))
        for column in (CAL_MOD_SIN, CAL_MOD_COS, CAL_DOW_SIN, CAL_DOW_COS):
            assert column in out.columns

    def test_reads_a_named_column_instead(self):
        """A caller-named column is read when one is given."""
        frame = pl.DataFrame(
            {"my_seconds": list(_SOME_TIMESTAMPS)}, schema={"my_seconds": pl.Int64}
        )
        out = add_calendar_columns(frame, "my_seconds")
        assert CAL_MOD_SIN in out.columns

    def test_preserves_existing_columns_and_their_order(self):
        """The frame's own columns are unchanged and come first."""
        frame = _frame(_SOME_TIMESTAMPS).with_columns(pl.lit(1.0).alias("close"))
        out = add_calendar_columns(frame)
        assert out.columns[:2] == ["close_time", "close"]
        assert out.columns[2:] == [CAL_MOD_SIN, CAL_MOD_COS, CAL_DOW_SIN, CAL_DOW_COS]

    def test_does_not_mutate_the_input_frame(self):
        """The input frame is untouched; a new frame is returned."""
        frame = _frame(_SOME_TIMESTAMPS)
        add_calendar_columns(frame)
        assert frame.columns == ["close_time"]

    def test_each_pair_is_on_the_unit_circle_every_row(self):
        """sin**2 + cos**2 == 1 to float tolerance, on every row, both pairs."""
        out = add_calendar_columns(_frame(_SOME_TIMESTAMPS))
        for sin_col, cos_col in (
            (CAL_MOD_SIN, CAL_MOD_COS),
            (CAL_DOW_SIN, CAL_DOW_COS),
        ):
            for sin_value, cos_value in zip(
                out.get_column(sin_col), out.get_column(cos_col), strict=True
            ):
                assert math.isclose(
                    sin_value**2 + cos_value**2, 1.0, rel_tol=0, abs_tol=1e-12
                )

    def test_invariant_under_a_day_cycle_shift(self):
        """Adding a multiple of 86400 leaves the day pair unchanged."""
        base = add_calendar_columns(_frame(_SOME_TIMESTAMPS))
        shifted = add_calendar_columns(
            _frame(tuple(t + 5 * SECONDS_PER_DAY for t in _SOME_TIMESTAMPS))
        )
        for column in (CAL_MOD_SIN, CAL_MOD_COS):
            for a, b in zip(
                base.get_column(column), shifted.get_column(column), strict=True
            ):
                assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9)

    def test_invariant_under_a_week_cycle_shift(self):
        """Adding a multiple of 604800 leaves the week pair unchanged."""
        base = add_calendar_columns(_frame(_SOME_TIMESTAMPS))
        shifted = add_calendar_columns(
            _frame(tuple(t + 7 * SECONDS_PER_WEEK for t in _SOME_TIMESTAMPS))
        )
        for column in (CAL_DOW_SIN, CAL_DOW_COS):
            for a, b in zip(
                base.get_column(column), shifted.get_column(column), strict=True
            ):
                assert math.isclose(a, b, rel_tol=0, abs_tol=1e-9)

    def test_a_monday_midnight_reads_zero_phase_on_the_week_pair(self):
        """A Monday 00:00 UTC is the zero angle of the week cycle."""
        out = add_calendar_columns(_frame((_A_MONDAY,)))
        assert math.isclose(out.get_column(CAL_DOW_SIN).item(), 0.0, abs_tol=1e-12)
        assert math.isclose(out.get_column(CAL_DOW_COS).item(), 1.0, abs_tol=1e-12)

    def test_cal_mod_is_the_angle_of_second_of_day(self):
        """cal_mod_sin/cos is sin/cos of 2*pi * second_of_day(t) / 86400.

        Picked away from any boundary shared with second_of_week, so a
        formula that read the wrong one of the two would not coincide by
        accident.
        """
        t = _A_SATURDAY + 3_600
        out = add_calendar_columns(_frame((t,)))
        angle = 2 * math.pi * second_of_day(t) / SECONDS_PER_DAY
        assert math.isclose(out.get_column(CAL_MOD_SIN).item(), math.sin(angle))
        assert math.isclose(out.get_column(CAL_MOD_COS).item(), math.cos(angle))

    def test_cal_dow_is_the_angle_of_second_of_week(self):
        """cal_dow_sin/cos is sin/cos of 2*pi * second_of_week(t) / 604800.

        Picked away from any boundary shared with second_of_day, so a
        formula that read the wrong one of the two would not coincide by
        accident.
        """
        t = _A_SATURDAY + 3_600
        out = add_calendar_columns(_frame((t,)))
        angle = 2 * math.pi * second_of_week(t) / SECONDS_PER_WEEK
        assert math.isclose(out.get_column(CAL_DOW_SIN).item(), math.sin(angle))
        assert math.isclose(out.get_column(CAL_DOW_COS).item(), math.cos(angle))


class TestRefusals:
    """Each refusal by name, with the message it must carry."""

    def test_refuses_a_missing_input_column(self):
        """The default column name is refused by name when absent."""
        frame = pl.DataFrame({"other": [1]}, schema={"other": pl.Int64})
        with pytest.raises(ConfigError, match="close_time"):
            add_calendar_columns(frame)

    def test_refuses_a_named_missing_column(self):
        """A caller-named column is refused by name when absent."""
        frame = pl.DataFrame({"close_time": [1]}, schema={"close_time": pl.Int64})
        with pytest.raises(ConfigError, match="my_seconds"):
            add_calendar_columns(frame, "my_seconds")

    def test_refuses_a_float_column(self):
        """A non-integer dtype is refused and named in the message."""
        frame = pl.DataFrame({"close_time": [1.0]}, schema={"close_time": pl.Float64})
        with pytest.raises(ConfigError, match="Float64"):
            add_calendar_columns(frame)

    def test_refuses_a_datetime_column_without_converting_it(self):
        """A Datetime column is refused, never silently cast."""
        frame = pl.DataFrame(
            {"close_time": [1_700_000_000]}, schema={"close_time": pl.Int64}
        ).with_columns(pl.from_epoch("close_time", time_unit="s").alias("close_time"))
        assert frame.schema["close_time"] == pl.Datetime("us")

        with pytest.raises(ConfigError, match="Datetime") as caught:
            add_calendar_columns(frame)
        assert "never converted" in str(caught.value)

    def test_refuses_a_narrower_integer_column_without_widening_it(self):
        """Only Int64 is accepted: an Int32 clock is refused, never widened."""
        frame = pl.DataFrame(
            {"close_time": [1_700_000_000]}, schema={"close_time": pl.Int32}
        )

        with pytest.raises(ConfigError, match="Int64") as caught:
            add_calendar_columns(frame)
        assert "Int32" in str(caught.value)

    def test_refuses_an_already_present_output_column(self):
        """A single colliding output column is named in the refusal."""
        frame = _frame(_SOME_TIMESTAMPS).with_columns(pl.lit(0.0).alias(CAL_MOD_SIN))
        with pytest.raises(ConfigError, match=CAL_MOD_SIN):
            add_calendar_columns(frame)

    @pytest.mark.parametrize(
        "column", (CAL_MOD_SIN, CAL_MOD_COS, CAL_DOW_SIN, CAL_DOW_COS)
    )
    def test_refuses_each_output_column_by_name(self, column: str):
        """Each of the four output columns is refused when already present."""
        frame = _frame(_SOME_TIMESTAMPS).with_columns(pl.lit(0.0).alias(column))
        with pytest.raises(ConfigError, match=column):
            add_calendar_columns(frame)
