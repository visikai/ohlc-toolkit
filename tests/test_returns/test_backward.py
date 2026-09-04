"""Backward (causal) returns over a typed duration horizon.

Every expected value below is derived by hand from the fixture closes in
:mod:`tests.test_returns.factories` and written as an exact literal with
its arithmetic in a comment. Nothing here is read back from the
implementation, and nothing calls the implementation twice to compare it
against itself.

The fixture closes are dyadic rationals, so every ratio the tests take is
exact in an IEEE-754 double and so is subtracting one from it. The simple
returns are therefore compared with ``==`` and no tolerance at all. Log
returns are not exactly representable, so those are compared against
:func:`math.log` of the same independently derived ratio, within a
tolerance that covers the last-place freedom two ``ln`` implementations
have.
"""

import decimal
import math

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ohlc_toolkit.returns import (
    ReturnMethod,
    add_backward_returns,
    backward_return_column,
)
from ohlc_toolkit.temporal import ConfigError, Duration
from tests.test_returns.factories import (
    CADENCE,
    CADENCE_SECONDS,
    GAPPED_CLOSES,
    GAPPED_OFFSETS,
    TIME_BASE,
    gap_free_frame,
    gapped_frame,
    return_frame,
)

# Backward simple returns over the gapped fixture at a one-cadence
# horizon. Row by row, from closes (128, 160, 320, 80, 100) at offsets
# (0, 60, 120, 240, 300):
#
#   t=0    counterpart t=-60  is absent            -> null
#   t=60   160 / 128 - 1 = 1.25 - 1                -> 0.25
#   t=120  320 / 160 - 1 = 2    - 1                -> 1.0
#   t=240  counterpart t=180 is the missing tick   -> null
#   t=300  100 / 80  - 1 = 1.25 - 1                -> 0.25
_GAPPED_BACKWARD_1M = [None, 0.25, 1.0, None, 0.25]

# The same fixture at a two-cadence horizon. This is the row the gap
# exists for:
#
#   t=0    counterpart t=-120 is absent            -> null
#   t=60   counterpart t=-60  is absent            -> null
#   t=120  320 / 128 - 1 = 2.5  - 1                -> 1.5
#   t=240  80  / 320 - 1 = 0.25 - 1                -> -0.75
#   t=300  counterpart t=180 is the missing tick   -> null
_GAPPED_BACKWARD_2M = [None, None, 1.5, -0.75, None]

# What a shift by two ROWS would have produced over the same fixture --
# the defect this module exists to replace. The frame's rows are two
# cadences apart across the gap, so row i-2 is not the row at t-120 for
# every i:
#
#   i=3 (t=240) -> row 1 (t=60):  80  / 160 - 1    -> -0.5    (not -0.75)
#   i=4 (t=300) -> row 2 (t=120): 100 / 320 - 1    -> -0.6875 (not null)
_GAPPED_ROW_SHIFT_2M = [None, None, 1.5, -0.5, -0.6875]

# Backward simple returns over the gap-free fixture at a one-cadence
# horizon, from closes (128, 160, 320, 80, 100, 25):
#
#   t=0    no counterpart                          -> null
#   t=60   160 / 128 - 1 = 1.25 - 1                -> 0.25
#   t=120  320 / 160 - 1 = 2    - 1                -> 1.0
#   t=180  80  / 320 - 1 = 0.25 - 1                -> -0.75
#   t=240  100 / 80  - 1 = 1.25 - 1                -> 0.25
#   t=300  25  / 100 - 1 = 0.25 - 1                -> -0.75
_GAP_FREE_BACKWARD_1M = [None, 0.25, 1.0, -0.75, 0.25, -0.75]

# The exact ratios behind the log returns over the gap-free fixture at a
# one-cadence horizon, as literal dyadic rationals rather than as a
# quotient of two closes the implementation also divides.
_GAP_FREE_RATIOS_1M = [None, 1.25, 2.0, 0.25, 1.25, 0.25]

# Two independent `ln` implementations may disagree in the last place or
# two. Every log return checked here has magnitude between 0.22 and 1.39,
# so a relative bound is the right shape and this one is roughly four
# units in the last place.
_LOG_TOLERANCE = 1e-15


def _values(frame: pl.DataFrame, column: str) -> list[float | None]:
    """Read one column out as a plain Python list, nulls included."""
    return frame.get_column(column).to_list()


class TestBackwardSimpleReturns:
    """A backward simple return relates close(t) to close(t - H)."""

    def test_one_cadence_horizon_over_a_gapped_frame(self) -> None:
        """Each row's counterpart is found by exact close_time, or is null."""
        result = add_backward_returns(
            gapped_frame(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(result, column) == _GAPPED_BACKWARD_1M

    def test_two_cadence_horizon_reaches_across_the_gap(self) -> None:
        """A counterpart two cadences back is used when it is present.

        The row at ``t=240`` has no neighbour one cadence back, but it
        does have one two cadences back, and that is the row this test
        pins. The row after it has neither.
        """
        result = add_backward_returns(
            gapped_frame(),
            horizon="2m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "2m")
        assert _values(result, column) == _GAPPED_BACKWARD_2M

    def test_a_shift_by_rows_would_have_disagreed(self) -> None:
        """Guard the discriminator: the gap is what separates the two lookups.

        Both answers are DERIVED here from the fixture itself -- the
        row-shift one by shifting ``horizon // cadence`` rows, the
        defect this module replaces, and the time-based one from a dict
        of the fixture's own closes. If the fixture ever stopped
        separating them (a filled gap, an added tick), the two
        derivations would agree and this test would fail loudly, instead
        of two stale literals continuing to agree with themselves.
        """
        horizon_seconds = 120
        rows = horizon_seconds // CADENCE_SECONDS
        row_shifted = [
            None
            if index < rows
            else (close - GAPPED_CLOSES[index - rows]) / GAPPED_CLOSES[index - rows]
            for index, close in enumerate(GAPPED_CLOSES)
        ]
        close_by_offset = dict(zip(GAPPED_OFFSETS, GAPPED_CLOSES, strict=True))
        time_based = [
            (close - close_by_offset[offset - horizon_seconds])
            / close_by_offset[offset - horizon_seconds]
            if (offset - horizon_seconds) in close_by_offset
            else None
            for offset, close in zip(GAPPED_OFFSETS, GAPPED_CLOSES, strict=True)
        ]

        assert time_based == _GAPPED_BACKWARD_2M
        assert row_shifted == _GAPPED_ROW_SHIFT_2M
        assert row_shifted != time_based

    def test_one_cadence_horizon_over_a_gap_free_frame(self) -> None:
        """With every tick present, only the first row lacks a counterpart."""
        result = add_backward_returns(
            gap_free_frame(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(result, column) == _GAP_FREE_BACKWARD_1M

    def test_the_column_is_appended_and_the_input_columns_are_untouched(self) -> None:
        """Composition adds one column and rewrites none of the others."""
        frame = gap_free_frame()
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert result.columns == [
            "close_time",
            "close",
            backward_return_column(ReturnMethod.SIMPLE, "1m"),
        ]
        assert_frame_equal(
            result.select("close_time", "close"), frame, check_exact=True
        )

    def test_the_input_frame_is_never_mutated(self) -> None:
        """The frame handed in is byte-for-byte the same afterwards."""
        frame = gap_free_frame()
        before = frame.clone()
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert_frame_equal(frame, before, check_exact=True)
        assert result is not frame

    def test_the_column_name_states_direction_method_and_horizon(self) -> None:
        """A reader can tell what a column holds from its name alone."""
        assert backward_return_column(ReturnMethod.SIMPLE, "1m") == (
            "backward_return_simple_1m"
        )
        assert backward_return_column(ReturnMethod.LOG, "2m") == (
            "backward_return_log_2m"
        )

    def test_the_horizon_in_the_name_is_the_canonical_spelling(self) -> None:
        """Two spellings of one horizon name one column, not two."""
        assert backward_return_column(
            ReturnMethod.SIMPLE, "90s"
        ) == backward_return_column(ReturnMethod.SIMPLE, Duration(90))

    def test_two_horizons_compose_onto_one_frame(self) -> None:
        """Distinct horizons write distinct columns, so calls compose."""
        first = add_backward_returns(
            gapped_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        both = add_backward_returns(
            first, horizon="2m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        assert (
            _values(both, backward_return_column(ReturnMethod.SIMPLE, "1m"))
            == _GAPPED_BACKWARD_1M
        )
        assert (
            _values(both, backward_return_column(ReturnMethod.SIMPLE, "2m"))
            == _GAPPED_BACKWARD_2M
        )

    def test_an_empty_frame_gains_the_column_with_the_right_dtype(self) -> None:
        """No rows is not an error; the schema still says what was asked for."""
        result = add_backward_returns(
            gap_free_frame().clear(),
            horizon="1m",
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert result.height == 0
        assert result.schema[column] == pl.Float64


class TestBackwardLogReturns:
    """The log return is the same relation under a different formula."""

    def test_log_returns_over_a_gap_free_frame(self) -> None:
        """Each value is ``ln`` of the independently derived exact ratio."""
        result = add_backward_returns(
            gap_free_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        column = backward_return_column(ReturnMethod.LOG, "1m")

        for got, ratio in zip(
            _values(result, column), _GAP_FREE_RATIOS_1M, strict=True
        ):
            if ratio is None:
                assert got is None
            else:
                assert got == pytest.approx(math.log(ratio), rel=_LOG_TOLERANCE)

    def test_an_unchanged_price_has_a_log_return_of_exactly_zero(self) -> None:
        """``ln(1)`` is exact, so this one needs no tolerance at all."""
        frame = return_frame((0, 60), (100.0, 100.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        column = backward_return_column(ReturnMethod.LOG, "1m")
        assert _values(result, column) == [None, 0.0]

    def test_an_unchanged_price_has_a_simple_return_of_exactly_zero(self) -> None:
        """The two formulas agree exactly at the one point they must."""
        frame = return_frame((0, 60), (100.0, 100.0))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(result, column) == [None, 0.0]

    def test_a_positive_ratio_at_the_quotient_rounding_boundary_is_nulled(
        self,
    ) -> None:
        """LOG nulls a band of tiny POSITIVE ratios; this pins that as chosen.

        At a price ratio of ``2**-54`` the difference quotient
        ``(a - b) / b`` rounds to exactly ``-1.0`` (``2**-54`` is half an
        ulp of 1, and ties round to even), so ``log1p`` sees ``-1`` and
        the value is nulled -- even though the true log return, about
        ``-37.43``, is a real number. SIMPLE reports that same rounded
        quotient, ``-1.0``, as a value. Nulling here is a deliberate
        fail-safe for a collapse the formula can no longer resolve, not
        an accident; the module docstring states the band, and this test
        keeps the statement honest. One ulp higher, at ``2**-53``, the
        quotient is representable and LOG is finite again.
        """
        at_boundary = return_frame((0, 60), (1.0, 2.0**-54))
        log_result = add_backward_returns(
            at_boundary, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        simple_result = add_backward_returns(
            at_boundary, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        log_column = backward_return_column(ReturnMethod.LOG, "1m")
        simple_column = backward_return_column(ReturnMethod.SIMPLE, "1m")
        assert _values(log_result, log_column) == [None, None]
        assert _values(simple_result, simple_column) == [None, -1.0]

        above_boundary = return_frame((0, 60), (1.0, 2.0**-53))
        finite_result = add_backward_returns(
            above_boundary, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        finite_value = _values(finite_result, log_column)[1]
        assert finite_value is not None
        assert finite_value == pytest.approx(
            math.log1p(2.0**-53 - 1.0), rel=_LOG_TOLERANCE
        )

    def test_the_two_formulas_are_each_others_log_and_exp(self) -> None:
        """``log_return == ln(1 + simple_return)`` on every defined row.

        This is the cross-check between the two methods: they are
        computed by different expressions from the same two closes, and
        the relation between them is stated here rather than assumed.
        """
        frame = gap_free_frame()
        simple = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        logarithmic = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )

        pairs = zip(
            _values(simple, backward_return_column(ReturnMethod.SIMPLE, "1m")),
            _values(logarithmic, backward_return_column(ReturnMethod.LOG, "1m")),
            strict=True,
        )
        defined = 0
        for simple_value, log_value in pairs:
            if simple_value is None:
                assert log_value is None
                continue
            defined += 1
            assert log_value is not None
            assert log_value == pytest.approx(
                math.log1p(simple_value), rel=_LOG_TOLERANCE
            )
            assert math.exp(log_value) == pytest.approx(
                1.0 + simple_value, rel=_LOG_TOLERANCE
            )
        assert defined == len(_GAP_FREE_BACKWARD_1M) - 1


# Near-one close pairs where a return is smallest and a careless log
# formula sheds the most digits. The reference for each is
# ``ln(numerator / denominator)`` computed with 60-significant-digit
# decimal arithmetic over the EXACT values of the two doubles -- the same
# numbers the implementation receives -- so the assertion measures the
# formula, not the fixture.
_NEAR_ONE_PAIRS = [
    (100.0, 100.0001),
    (1.0, 1.0000001),
    (43210.5, 43210.4999),
]
_NEAR_ONE_IDS = ["100_vs_100.0001", "1_vs_1.0000001", "43210.5_vs_43210.4999"]


class TestLogReturnPrecision:
    """The log formula must not lose digits where returns are smallest.

    A log return near zero is the common case -- most closes barely move
    over one cadence -- and it is exactly where a careless formula loses
    six or more significant digits: the ratio of two nearby closes lands
    within half an ulp of 1, and ``ln(1 + x)`` amplifies that half-ulp
    by ``1 / x``. Holding the emitted value to within ``1e-15`` of a
    60-digit reference pins the formula itself: ``ln(a) - ln(b)`` and
    ``ln(a / b)`` both fail this bound by four or more orders of
    magnitude on the first pair below.
    """

    @pytest.mark.parametrize(
        ("numerator", "denominator"), _NEAR_ONE_PAIRS, ids=_NEAR_ONE_IDS
    )
    def test_a_tiny_log_return_is_accurate_to_a_60_digit_reference(
        self, numerator: float, denominator: float
    ) -> None:
        """The emitted log return sits within 1e-15 of exact."""
        frame = return_frame((0, 60), (denominator, numerator))
        result = add_backward_returns(
            frame, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        got = _values(result, backward_return_column(ReturnMethod.LOG, "1m"))[1]
        assert got is not None

        with decimal.localcontext() as context:
            context.prec = 60
            reference = (decimal.Decimal(numerator) / decimal.Decimal(denominator)).ln()
            relative_error = abs((decimal.Decimal(got) - reference) / reference)
        assert relative_error < decimal.Decimal("1e-15")


class TestBackwardReturnsAreCausal:
    """A backward return uses only information available at its own row."""

    @pytest.mark.parametrize("horizon", ["1m", "2m"])
    def test_truncating_the_future_away_changes_nothing(self, horizon: str) -> None:
        """Every row keeps its value when every later row is deleted.

        This is the causality claim stated as an experiment rather than
        as a comment: if any backward value depended on a close that had
        not arrived yet, dropping the rows carrying it would change that
        value.
        """
        frame = gapped_frame()
        column = backward_return_column(ReturnMethod.SIMPLE, horizon)
        full = add_backward_returns(
            frame, horizon=horizon, cadence=CADENCE, method=ReturnMethod.SIMPLE
        )

        for close_time in frame.get_column("close_time").to_list():
            truncated = add_backward_returns(
                frame.filter(pl.col("close_time") <= close_time),
                horizon=horizon,
                cadence=CADENCE,
                method=ReturnMethod.SIMPLE,
            )
            expected = full.filter(pl.col("close_time") == close_time)
            assert _values(truncated.tail(1), column) == _values(expected, column)


class TestHorizonAndCadenceResolution:
    """The horizon is checked against the cadence the caller states."""

    def test_a_horizon_that_is_not_a_whole_multiple_of_the_cadence_is_refused(
        self,
    ) -> None:
        """A 90s horizon over a 1m cadence lands between two ticks."""
        with pytest.raises(ConfigError, match="whole multiple"):
            add_backward_returns(
                gap_free_frame(),
                horizon="90s",
                cadence=CADENCE,
                method=ReturnMethod.SIMPLE,
            )

    def test_a_horizon_shorter_than_the_cadence_is_refused(self) -> None:
        """A 30s horizon over a 1m cadence never lands on a tick either."""
        with pytest.raises(ConfigError, match="whole multiple"):
            add_backward_returns(
                gap_free_frame(),
                horizon="30s",
                cadence=CADENCE,
                method=ReturnMethod.SIMPLE,
            )

    def test_a_zero_horizon_is_refused(self) -> None:
        """A zero horizon relates every close to itself: a constant zero."""
        with pytest.raises(ConfigError, match="strictly positive"):
            add_backward_returns(
                gap_free_frame(),
                horizon="0s",
                cadence=CADENCE,
                method=ReturnMethod.SIMPLE,
            )

    def test_a_zero_cadence_is_refused(self) -> None:
        """No horizon is a whole multiple of a cadence that never advances."""
        with pytest.raises(ConfigError, match="strictly positive"):
            add_backward_returns(
                gap_free_frame(),
                horizon="1m",
                cadence="0s",
                method=ReturnMethod.SIMPLE,
            )

    @pytest.mark.parametrize(
        "horizon",
        ["not-a-duration", "1M", "5", "", "1s1m"],
        ids=["words", "wrong_case_unit", "no_unit", "empty", "misordered"],
    )
    def test_a_malformed_horizon_string_is_refused(self, horizon: str) -> None:
        """The horizon goes through the same grammar every duration does."""
        with pytest.raises(ConfigError):
            add_backward_returns(
                gap_free_frame(),
                horizon=horizon,
                cadence=CADENCE,
                method=ReturnMethod.SIMPLE,
            )

    def test_a_horizon_equal_to_the_cadence_is_accepted(self) -> None:
        """One cadence is a whole multiple of itself: the shortest horizon."""
        result = add_backward_returns(
            gap_free_frame(),
            horizon=CADENCE,
            cadence=CADENCE,
            method=ReturnMethod.SIMPLE,
        )
        assert (
            _values(result, backward_return_column(ReturnMethod.SIMPLE, CADENCE))
            == _GAP_FREE_BACKWARD_1M
        )

    def test_duration_objects_are_accepted_alongside_strings(self) -> None:
        """The boundary type is ``Duration | str``, as everywhere else."""
        from_strings = add_backward_returns(
            gapped_frame(), horizon="2m", cadence="1m", method=ReturnMethod.SIMPLE
        )
        from_durations = add_backward_returns(
            gapped_frame(),
            horizon=Duration(120),
            cadence=Duration(60),
            method=ReturnMethod.SIMPLE,
        )
        assert_frame_equal(from_durations, from_strings, check_exact=True)


class TestMethodIsAlwaysStated:
    """The formula is a recorded choice, never an inferred default."""

    def test_a_non_member_method_is_refused(self) -> None:
        """A plain string is not accepted in place of a ReturnMethod."""
        with pytest.raises(ConfigError, match="method"):
            add_backward_returns(
                gap_free_frame(),
                horizon="1m",
                cadence=CADENCE,
                method="simple",  # type: ignore[arg-type]
            )

    def test_the_two_methods_write_different_columns(self) -> None:
        """Both formulas can sit on one frame without either hiding the other."""
        simple = add_backward_returns(
            gap_free_frame(), horizon="1m", cadence=CADENCE, method=ReturnMethod.SIMPLE
        )
        both = add_backward_returns(
            simple, horizon="1m", cadence=CADENCE, method=ReturnMethod.LOG
        )
        assert both.columns[-2:] == [
            backward_return_column(ReturnMethod.SIMPLE, "1m"),
            backward_return_column(ReturnMethod.LOG, "1m"),
        ]

    def test_the_method_values_are_the_two_named_formulas(self) -> None:
        """The enum has exactly the two members the module documents."""
        assert {member.value for member in ReturnMethod} == {"simple", "log"}


def test_the_fixture_time_base_is_on_the_cadence_grid() -> None:
    """Guard the fixture: its close times read as real 1m emit ticks."""
    assert TIME_BASE % 60 == 0


if __name__ == "__main__":
    pytest.main([__file__])
