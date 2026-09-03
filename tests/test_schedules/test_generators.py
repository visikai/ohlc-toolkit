"""Window-scale schedule generators: the resolved durations they produce.

Every expected value here is computed independently of the generator
under test -- with exact rationals in this module, or pinned as a
literal list -- so an implementation that agreed with itself but not
with the arithmetic would still be caught.
"""

import math
import sys
from fractions import Fraction

import pytest

from ohlc_toolkit.schedules import (
    MAX_RESOLVED_WINDOWS,
    ExplicitSpec,
    GeneratorKind,
    LogSpacedSpec,
    MetallicRecurrenceSpec,
    RoundingRule,
    explicit,
    log_spaced,
    metallic_recurrence,
)
from ohlc_toolkit.temporal import ConfigError, Duration

_MINUTE_SECONDS = 60
_TWO_WEEKS_SECONDS = 14 * 86400

# sqrt(e + sqrt(5)). An irrational coefficient, chosen because no term
# of the sequence it generates lands anywhere near a rounding tie, so
# the pinned schedule below does not depend on the tie rule at all.
_COEFFICIENT = math.sqrt(math.e + math.sqrt(5))

# The resolved schedule, in whole minutes, for that coefficient with a
# 1m seed on a 1m grain bounded by 2w. Pinned as a literal: an expected
# value read back out of the generator would confirm the generator
# rather than check it. ``_real_terms`` below re-derives the same list
# from exact rationals, and the first test asserts the two agree.
_PINNED_MINUTES = (1, 3, 8, 21, 56, 146, 380, 993, 2590, 6758, 17632)

# The seven-decimal drift fixture: c = 1.0000001 from a 1m seed on a 1s
# grain reaches 2w in exactly this many windows, and the second-to-last
# window is the first place a coefficient rounded to six decimals (that
# is, to 1.0) lands on a different whole second (656760 instead of
# 656761). Both numbers re-derived below from exact rationals before use.
_SEVEN_DECIMAL_WINDOW_COUNT = 21
_SEVEN_DECIMAL_PENULTIMATE_SECONDS = 656_761


def _real_terms(
    coefficient: float, seed_seconds: int, maximum_seconds: int
) -> list[Fraction]:
    """Compute the unquantized recurrence terms as exact rationals.

    Seeded with two copies of the seed, and stopped as soon as the next
    REAL term would exceed the maximum. ``Fraction(float)`` is the
    float's exact value, so this carries no rounding of its own and can
    be read as the arithmetic the generator is measured against.

    Args:
        coefficient: The recurrence coefficient.
        seed_seconds: The seed duration, in whole seconds.
        maximum_seconds: The bound the sequence must not pass.

    Returns:
        The terms, in order, in seconds.

    """
    exact_coefficient = Fraction(coefficient)
    terms = [Fraction(seed_seconds), Fraction(seed_seconds)]
    while True:
        following = exact_coefficient * terms[-1] + terms[-2]
        if following > maximum_seconds:
            return terms
        terms.append(following)


def _round_nearest_ties_away(value: Fraction, grain_seconds: int) -> int:
    """Round an exact number of seconds to the nearest whole grain.

    Ties go away from zero. Written out here rather than imported, so
    the expected quantization is this module's own statement of the rule.
    """
    scaled = value / grain_seconds
    whole = scaled.numerator // scaled.denominator
    if scaled - whole >= Fraction(1, 2):
        whole += 1
    return whole * grain_seconds


def _dedup(values: list[int]) -> list[int]:
    """Drop later repeats, preserving the order of first appearance."""
    kept: list[int] = []
    for value in values:
        if value not in kept:
            kept.append(value)
    return kept


def _minutes(windows: tuple[Duration, ...]) -> list[int]:
    """Render resolved windows as whole minutes, for readable assertions."""
    return [window.total_seconds // _MINUTE_SECONDS for window in windows]


def _pinned_schedule_windows() -> tuple[Duration, ...]:
    """Resolve the pinned metallic schedule, the fixture most tests start from."""
    return metallic_recurrence(
        coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="2w"
    ).windows


class TestMetallicOracle:
    """The pinned sequence, and the exact arithmetic it comes from."""

    def test_the_pinned_schedule_matches_an_exact_rational_recurrence(self) -> None:
        """Guard the fixture itself: the literal list is the arithmetic.

        If this ever fails, the pinned list is wrong and every
        expectation built on it is worthless -- so it is checked against
        a from-scratch rational computation before it is used.
        """
        terms = _real_terms(_COEFFICIENT, _MINUTE_SECONDS, _TWO_WEEKS_SECONDS)
        quantized = [_round_nearest_ties_away(t, _MINUTE_SECONDS) for t in terms]
        expected = [minutes * _MINUTE_SECONDS for minutes in _PINNED_MINUTES]
        assert _dedup(quantized) == expected

    def test_no_term_of_the_pinned_sequence_sits_near_a_tie(self) -> None:
        """The pinned schedule must not depend on the tie rule.

        Every term is at least a hundredth of a grain away from a half
        grain, so both tie directions round every one of them the same
        way and the pinned list stays valid under either rule.
        """
        terms = _real_terms(_COEFFICIENT, _MINUTE_SECONDS, _TWO_WEEKS_SECONDS)
        for term in terms:
            scaled = term / _MINUTE_SECONDS
            fractional_part = scaled - scaled.numerator // scaled.denominator
            assert abs(fractional_part - Fraction(1, 2)) > Fraction(1, 100)


class TestMetallicRecurrence:
    """The two-term recurrence generator, quantized once at the end."""

    def test_resolves_the_pinned_windows(self) -> None:
        """The generator reproduces the independently computed schedule."""
        assert _minutes(_pinned_schedule_windows()) == list(_PINNED_MINUTES)

    def test_the_kind_is_recorded(self) -> None:
        """A schedule always states which generator produced it."""
        schedule = metallic_recurrence(
            coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="2w"
        )
        assert schedule.spec.kind is GeneratorKind.METALLIC_RECURRENCE

    def test_the_two_seeds_collapse_to_one_window(self) -> None:
        """The sequence starts with the seed twice; the schedule names it once."""
        windows = _pinned_schedule_windows()
        assert windows[0] == Duration.parse("1m")
        assert windows[1] != Duration.parse("1m")

    def test_windows_are_never_repeated(self) -> None:
        """Deduplication leaves each window named exactly once."""
        windows = _pinned_schedule_windows()
        assert len(set(windows)) == len(windows)

    def test_the_bound_stops_the_recurrence_on_the_real_term(self) -> None:
        """The sequence runs while the next REAL term is within the bound.

        The last term kept is 17632.177... minutes and the first term
        refused is 46004.28... minutes, so the recurrence produces one
        more term than the schedule names -- the extra one being the
        duplicated seed that deduplication drops.
        """
        terms = _real_terms(_COEFFICIENT, _MINUTE_SECONDS, _TWO_WEEKS_SECONDS)
        windows = _pinned_schedule_windows()
        assert len(terms) == len(windows) + 1
        assert windows[-1] == Duration.parse("17632m")

    def test_a_term_landing_exactly_on_the_maximum_is_kept(self) -> None:
        """The bound is inclusive: a term equal to the maximum is within it.

        With a coefficient of one, a 1m seed and a 2m maximum, the third
        real term is exactly 120s -- exactly the maximum. The recurrence
        must keep it (the bound reads "at most", not "below") and stop
        at the term after it.
        """
        schedule = metallic_recurrence(
            coefficient=1, seed="1m", grain="1m", maximum="2m"
        )
        assert [str(window) for window in schedule.windows] == ["1m", "2m"]

    def test_the_term_past_the_cap_is_the_one_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The cap admits exactly its own number of terms, then refuses.

        At the real cap of 512 an integer-second maximum cannot land
        between two consecutive terms of any recurrence slow enough to
        need that many, so the boundary is pinned with the cap lowered
        to six. A coefficient-one recurrence from a 1m seed needs a
        seventh term to reach 13m: refused. The same recurrence bounded
        at 8m needs exactly six terms: accepted.
        """
        monkeypatch.setattr("ohlc_toolkit.schedules.generators.MAX_RESOLVED_WINDOWS", 6)
        with pytest.raises(ConfigError, match="terms"):
            metallic_recurrence(coefficient=1, seed="1m", grain="1m", maximum="13m")

        schedule = metallic_recurrence(
            coefficient=1, seed="1m", grain="1m", maximum="8m"
        )
        assert [str(window) for window in schedule.windows] == [
            "1m",
            "2m",
            "3m",
            "5m",
            "8m",
        ]

    def test_a_seven_decimal_coefficient_resolves_by_its_full_value(self) -> None:
        """No hidden rounding of the coefficient survives into the schedule.

        With c = 1.0000001 from a 1m seed on a 1s grain bounded by 2w,
        the drift a slow recurrence accumulates from the seventh decimal
        reaches a whole second by the twentieth term: the full value
        yields 656761s where a coefficient rounded to six decimals
        yields 656760s. The expected list is re-derived from exact
        rationals, so the pin is on the arithmetic, not the generator.
        """
        coefficient = 1.0000001
        terms = _real_terms(coefficient, _MINUTE_SECONDS, _TWO_WEEKS_SECONDS)
        expected = _dedup([_round_nearest_ties_away(term, 1) for term in terms])
        assert len(expected) == _SEVEN_DECIMAL_WINDOW_COUNT
        assert expected[-2] == _SEVEN_DECIMAL_PENULTIMATE_SECONDS

        schedule = metallic_recurrence(
            coefficient=coefficient, seed="1m", grain="1s", maximum="2w"
        )
        assert [window.total_seconds for window in schedule.windows] == expected

    def test_a_value_that_rounds_past_the_maximum_is_dropped(self) -> None:
        """Rounding must never carry a window over the bound the caller set.

        With a 45s seed on a 1m grain, the third real term is exactly
        90s -- exactly the stated maximum, so the recurrence keeps it --
        but it is exactly one and a half grains, and rounding takes it
        to 2m, which is past that maximum. A schedule bounded by 1m30s
        must not name a 2m window.
        """
        schedule = metallic_recurrence(
            coefficient=1.0, seed="45s", grain="1m", maximum="1m30s"
        )
        assert schedule.windows == (Duration.parse("1m"),)

    def test_a_minimum_drops_the_short_windows(self) -> None:
        """A lower bound excludes the early, small scales."""
        schedule = metallic_recurrence(
            coefficient=_COEFFICIENT,
            seed="1m",
            grain="1m",
            minimum="1h",
            maximum="2w",
        )
        one_hour_minutes = 60
        expected = [m for m in _PINNED_MINUTES if m >= one_hour_minutes]
        assert _minutes(schedule.windows) == expected

    def test_bounds_that_exclude_every_value_are_refused(self) -> None:
        """A schedule naming no window at all is a configuration error."""
        with pytest.raises(ConfigError, match="no windows"):
            metallic_recurrence(
                coefficient=_COEFFICIENT,
                seed="1m",
                grain="1m",
                minimum="17633m",
                maximum="2w",
            )

    def test_a_minimum_above_the_maximum_is_refused(self) -> None:
        """Inverted bounds are refused in their own words, not as an empty result."""
        with pytest.raises(ConfigError, match="minimum"):
            metallic_recurrence(
                coefficient=_COEFFICIENT,
                seed="1m",
                grain="1m",
                minimum="2w",
                maximum="1w",
            )


class TestMetallicRounding:
    """One quantization at the end, with a recorded tie rule."""

    def test_the_recurrence_is_not_truncated_step_by_step(self) -> None:
        """Every term comes from the real sequence, not from rounded predecessors.

        Quantizing each term before feeding it back is what an earlier
        implementation did, and it drifts: with this coefficient the two
        readings disagree well before the bound is reached.
        """
        rounded_each_step = [_MINUTE_SECONDS, _MINUTE_SECONDS]
        exact_coefficient = Fraction(_COEFFICIENT)
        while True:
            following = _round_nearest_ties_away(
                exact_coefficient * rounded_each_step[-1] + rounded_each_step[-2],
                _MINUTE_SECONDS,
            )
            if following > _TWO_WEEKS_SECONDS:
                break
            rounded_each_step.append(following)

        drifted = _dedup([s // _MINUTE_SECONDS for s in rounded_each_step])
        assert drifted != list(_PINNED_MINUTES)
        assert _minutes(_pinned_schedule_windows()) != drifted

    @pytest.mark.parametrize(
        ("rounding", "expected_minutes"),
        [
            (RoundingRule.NEAREST_TIES_AWAY, [3, 5, 8]),
            (RoundingRule.NEAREST_TIES_EVEN, [2, 5, 8]),
        ],
        ids=["ties_away", "ties_even"],
    )
    def test_an_exact_tie_follows_the_recorded_rule(
        self, rounding: RoundingRule, expected_minutes: list[int]
    ) -> None:
        """A term at exactly half a grain rounds by the rule the caller named.

        The seed is 2m30s on a 1m grain: exactly two and a half grains,
        which is the case the two rules disagree about. Away from zero
        gives 3m, to even gives 2m. The later terms are 5m and exactly
        seven and a half grains, and both rules take the latter to 8m,
        so this pins the disagreement and nothing else.
        """
        schedule = metallic_recurrence(
            coefficient=1.0,
            seed="2m30s",
            grain="1m",
            maximum="10m",
            rounding=rounding,
        )
        assert _minutes(schedule.windows) == expected_minutes

    def test_the_rounding_rule_is_recorded(self) -> None:
        """The tie rule is part of the identity, not an implementation detail."""
        schedule = metallic_recurrence(
            coefficient=_COEFFICIENT,
            seed="1m",
            grain="1m",
            maximum="2w",
            rounding=RoundingRule.NEAREST_TIES_EVEN,
        )
        assert isinstance(schedule.spec, MetallicRecurrenceSpec)
        assert schedule.spec.rounding is RoundingRule.NEAREST_TIES_EVEN

    def test_a_term_that_quantizes_to_nothing_is_refused(self) -> None:
        """A grain coarser than the seed would silently drop the small scales."""
        with pytest.raises(ConfigError, match="0s"):
            metallic_recurrence(
                coefficient=_COEFFICIENT, seed="20s", grain="1m", maximum="2w"
            )


class TestMetallicLimitingRatio:
    """The ratio the recurrence actually converges to, recorded with it."""

    def test_the_recorded_ratio_is_the_one_the_terms_converge_to(self) -> None:
        """The stored ratio is checked against the sequence, not its formula.

        Successive real terms converge to the recurrence's limiting
        ratio, so the ratio of the two latest terms is an independent
        measurement of the number the identity records. Twelve terms in,
        that measurement has converged to about eight decimal places,
        which is the tolerance below -- still far tighter than the gap
        between the ratio and any other number in the identity.
        """
        schedule = metallic_recurrence(
            coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="2w"
        )
        terms = _real_terms(_COEFFICIENT, _MINUTE_SECONDS, _TWO_WEEKS_SECONDS)
        observed = float(terms[-1] / terms[-2])
        assert schedule.spec.limiting_ratio == pytest.approx(observed, abs=1e-7)

    def test_the_golden_ratio_is_the_unit_coefficient(self) -> None:
        """A coefficient of 1 is the Fibonacci recurrence, whose ratio is known."""
        schedule = metallic_recurrence(
            coefficient=1.0, seed="1m", grain="1m", maximum="1d"
        )
        golden_ratio = (1 + math.sqrt(5)) / 2
        assert schedule.spec.limiting_ratio == pytest.approx(golden_ratio, abs=1e-12)


class TestMetallicRefusals:
    """Inputs no schedule can be resolved from at all."""

    @pytest.mark.parametrize(
        "coefficient",
        [0.0, -1.0, -0.5],
        ids=["zero", "negative_one", "negative_fraction"],
    )
    def test_a_non_positive_coefficient_is_refused(self, coefficient: float) -> None:
        """A coefficient of zero never grows: the recurrence would not terminate."""
        with pytest.raises(ConfigError, match="coefficient"):
            metallic_recurrence(
                coefficient=coefficient, seed="1m", grain="1m", maximum="2w"
            )

    @pytest.mark.parametrize(
        "coefficient",
        [math.nan, math.inf, -math.inf],
        ids=["nan", "infinity", "negative_infinity"],
    )
    def test_a_non_finite_coefficient_is_refused(self, coefficient: float) -> None:
        """Neither a NaN nor an infinity names a recurrence."""
        with pytest.raises(ConfigError, match="coefficient"):
            metallic_recurrence(
                coefficient=coefficient, seed="1m", grain="1m", maximum="2w"
            )

    def test_a_boolean_coefficient_is_refused(self) -> None:
        """``bool`` is an ``int`` subtype in Python, and is refused anyway."""
        with pytest.raises(ConfigError, match="coefficient"):
            metallic_recurrence(coefficient=True, seed="1m", grain="1m", maximum="2w")

    def test_a_coefficient_whose_square_overflows_is_refused(self) -> None:
        """Above the square root of the float maximum, the ratio overflows.

        The recurrence itself would resolve (a single seed window,
        since the first real term already passes any bound), but the
        recorded identity computes ``(c + sqrt(c**2 + 4)) / 2`` -- so
        without this refusal the object constructs successfully and then
        cannot serialize or hash itself, failing later, elsewhere, and
        in a foreign exception type.
        """
        with pytest.raises(ConfigError, match="coefficient"):
            metallic_recurrence(
                coefficient=1.35e154, seed="1m", grain="1m", maximum="2w"
            )

    def test_the_largest_squarable_coefficient_still_serializes(self) -> None:
        """The bound is exact: the square root of the float maximum works.

        At exactly ``sqrt(float max)`` the squared value still fits, so
        this coefficient must construct, resolve, serialize, and hash --
        pinning that the refusal above does not overreach.
        """
        schedule = metallic_recurrence(
            coefficient=math.sqrt(sys.float_info.max),
            seed="1m",
            grain="1m",
            maximum="2w",
        )
        assert [str(window) for window in schedule.windows] == ["1m"]
        assert isinstance(schedule.spec, MetallicRecurrenceSpec)
        assert math.isfinite(schedule.spec.limiting_ratio)
        assert (
            schedule.schedule_id
            == type(schedule).from_dict(schedule.to_dict()).schedule_id
        )

    def test_a_coefficient_too_small_to_terminate_is_refused(self) -> None:
        """A vanishing coefficient grows so slowly it is unbounded in practice.

        The sequence does still terminate mathematically, but only after
        billions of terms; the cap turns that into an immediate,
        explained refusal instead of a hang.
        """
        with pytest.raises(ConfigError, match="512"):
            metallic_recurrence(coefficient=1e-9, seed="1m", grain="1m", maximum="2w")

    @pytest.mark.parametrize(
        ("seed", "grain", "maximum"),
        [
            ("0s", "1m", "2w"),
            ("1m", "0s", "2w"),
            ("1m", "1m", "0s"),
        ],
        ids=["zero_seed", "zero_grain", "zero_maximum"],
    )
    def test_a_zero_duration_is_refused(
        self, seed: str, grain: str, maximum: str
    ) -> None:
        """Seed, grain, and bounds are all strictly positive durations."""
        with pytest.raises(ConfigError, match="strictly positive"):
            metallic_recurrence(
                coefficient=_COEFFICIENT, seed=seed, grain=grain, maximum=maximum
            )

    def test_a_zero_minimum_is_refused(self) -> None:
        """A stated lower bound is a duration like any other."""
        with pytest.raises(ConfigError, match="strictly positive"):
            metallic_recurrence(
                coefficient=_COEFFICIENT,
                seed="1m",
                grain="1m",
                minimum="0s",
                maximum="2w",
            )

    def test_a_malformed_duration_string_is_refused(self) -> None:
        """Durations are coerced at the boundary, in the temporal grammar."""
        with pytest.raises(ConfigError):
            metallic_recurrence(
                coefficient=_COEFFICIENT,
                seed="not-a-duration",
                grain="1m",
                maximum="2w",
            )

    def test_durations_may_be_given_as_duration_instances(self) -> None:
        """The boundary accepts ``Duration | str``, like the rest of the package."""
        schedule = metallic_recurrence(
            coefficient=_COEFFICIENT,
            seed=Duration.parse("1m"),
            grain=Duration(60),
            maximum=Duration.parse("2w"),
        )
        assert _minutes(schedule.windows) == list(_PINNED_MINUTES)


def _log_points(minimum_seconds: int, maximum_seconds: int, count: int) -> list[float]:
    """Compute log-spaced points between two bounds, endpoints included.

    Written in ordinary floating point, which is a different route to
    the same numbers than the generator takes, and enough to check them:
    the cases below are all far from a rounding boundary, so the two
    routes cannot disagree after quantization.

    Args:
        minimum_seconds: The first point, in seconds.
        maximum_seconds: The last point, in seconds.
        count: How many points to place, endpoints included.

    Returns:
        The points, in ascending order, in seconds.

    """
    span = math.log(maximum_seconds / minimum_seconds)
    interior = [
        minimum_seconds * math.exp(span * step / (count - 1))
        for step in range(1, count - 1)
    ]
    return [float(minimum_seconds), *interior, float(maximum_seconds)]


def _expected_log_minutes(
    minimum_minutes: int, maximum_minutes: int, count: int
) -> list[int]:
    """Quantize independently computed log-spaced points to whole minutes."""
    points = _log_points(
        minimum_minutes * _MINUTE_SECONDS, maximum_minutes * _MINUTE_SECONDS, count
    )
    quantized = [
        _round_nearest_ties_away(Fraction(point), _MINUTE_SECONDS) for point in points
    ]
    return [seconds // _MINUTE_SECONDS for seconds in _dedup(quantized)]


class TestLogSpaced:
    """A fixed count of log-spaced durations between two bounds."""

    def test_a_power_of_two_ladder_is_exact(self) -> None:
        """Eleven points from 1m to 1024m are the powers of two, by hand.

        The ratio between neighbours is 1024 ** (1/10) = 2 exactly, so
        this case has a closed-form answer that owes nothing to any
        implementation of logarithms.
        """
        schedule = log_spaced(count=11, minimum="1m", maximum="1024m", grain="1m")
        assert _minutes(schedule.windows) == [2**step for step in range(11)]

    def test_both_endpoints_are_the_bounds_themselves(self) -> None:
        """The first and last windows are exactly the bounds the caller gave."""
        schedule = log_spaced(count=7, minimum="5m", maximum="1d", grain="1m")
        assert schedule.windows[0] == Duration.parse("5m")
        assert schedule.windows[-1] == Duration.parse("1d")

    def test_matches_independently_computed_points(self) -> None:
        """A ladder whose points do not land on whole grains still agrees."""
        schedule = log_spaced(count=5, minimum="1m", maximum="1h", grain="1m")
        assert _minutes(schedule.windows) == _expected_log_minutes(1, 60, 5)

    def test_the_kind_is_recorded(self) -> None:
        """A schedule always states which generator produced it."""
        schedule = log_spaced(count=5, minimum="1m", maximum="1h", grain="1m")
        assert schedule.spec.kind is GeneratorKind.LOG_SPACED

    def test_the_count_is_recorded(self) -> None:
        """The count is a parameter of the identity, not just of the call.

        The spec is reached through an isinstance check rather than an
        attribute lookup on the union: a generator's own parameters are
        its own, and only a spec of the right kind has a count at all.
        """
        count = 5
        schedule = log_spaced(count=count, minimum="1m", maximum="1h", grain="1m")
        assert isinstance(schedule.spec, LogSpacedSpec)
        assert schedule.spec.count == count

    def test_a_two_point_ladder_is_just_the_endpoints(self) -> None:
        """The smallest legal count places the bounds and nothing between."""
        schedule = log_spaced(count=2, minimum="1m", maximum="1h", grain="1m")
        assert _minutes(schedule.windows) == [1, 60]

    def test_points_that_collapse_onto_one_grain_are_deduplicated(self) -> None:
        """A count too high for the range is not an error, just a shorter list.

        Ten points between 1m and 3m cannot be told apart on a 1m grain:
        the quantized ladder repeats, and the schedule names each
        surviving window once.
        """
        schedule = log_spaced(count=10, minimum="1m", maximum="3m", grain="1m")
        assert _minutes(schedule.windows) == _expected_log_minutes(1, 3, 10)
        assert _minutes(schedule.windows) == [1, 2, 3]

    def test_equal_bounds_resolve_to_a_single_window(self) -> None:
        """A zero-width range is legal and names one window."""
        schedule = log_spaced(count=5, minimum="1h", maximum="1h", grain="1m")
        assert schedule.windows == (Duration.parse("1h"),)
        assert schedule.spec.limiting_ratio == pytest.approx(1.0, abs=1e-12)


class TestLogSpacedRatio:
    """The spacing a log-spaced ladder implies, recorded with it."""

    def test_the_recorded_ratio_raised_to_the_steps_spans_the_bounds(self) -> None:
        """The stored ratio is checked by its defining property, not its formula.

        A ladder of ``count`` points has ``count - 1`` steps, so its
        ratio raised to that power must take the minimum to the maximum.
        """
        schedule = log_spaced(count=7, minimum="5m", maximum="1d", grain="1m")
        assert isinstance(schedule.spec, LogSpacedSpec)
        spanned = schedule.spec.limiting_ratio**6
        assert spanned == pytest.approx(86400 / 300, rel=1e-12)

    def test_the_power_of_two_ladder_records_a_ratio_of_two(self) -> None:
        """The closed-form case has a closed-form ratio."""
        schedule = log_spaced(count=11, minimum="1m", maximum="1024m", grain="1m")
        assert schedule.spec.limiting_ratio == pytest.approx(2.0, abs=1e-12)


class TestLogSpacedRefusals:
    """Inputs no log-spaced ladder can be resolved from."""

    @pytest.mark.parametrize("count", [1, 0, -1], ids=["one", "zero", "negative"])
    def test_a_count_below_two_is_refused(self, count: int) -> None:
        """With fewer than two points there are no two endpoints to span."""
        with pytest.raises(ConfigError, match="at least 2"):
            log_spaced(count=count, minimum="1m", maximum="1h", grain="1m")

    def test_a_boolean_count_is_refused(self) -> None:
        """``bool`` is an ``int`` subtype in Python, and is refused anyway."""
        with pytest.raises(ConfigError, match="count"):
            log_spaced(count=True, minimum="1m", maximum="1h", grain="1m")

    def test_a_non_integer_count_is_refused(self) -> None:
        """A count of points is a whole number of points."""
        with pytest.raises(ConfigError, match="count"):
            log_spaced(
                count=5.0,  # type: ignore[arg-type]
                minimum="1m",
                maximum="1h",
                grain="1m",
            )

    def test_a_count_past_the_cap_is_refused(self) -> None:
        """No schedule may name more windows than the package's one cap."""
        with pytest.raises(ConfigError, match="512"):
            log_spaced(count=513, minimum="1m", maximum="2w", grain="1s")

    def test_a_count_of_exactly_the_cap_is_accepted(self) -> None:
        """The cap is inclusive: exactly 512 points is a valid ladder.

        Together with the refusal above, this pins which side of the
        boundary the cap sits on; without it, an off-by-one that refused
        512 as well would go unnoticed.
        """
        schedule = log_spaced(count=512, minimum="1s", maximum="2w", grain="1s")
        assert 0 < len(schedule.windows) <= MAX_RESOLVED_WINDOWS

    def test_a_minimum_above_the_maximum_is_refused(self) -> None:
        """Inverted bounds are refused in the same words as anywhere else."""
        with pytest.raises(ConfigError, match="minimum"):
            log_spaced(count=5, minimum="1h", maximum="1m", grain="1m")

    @pytest.mark.parametrize(
        ("minimum", "maximum", "grain"),
        [
            ("0s", "1h", "1m"),
            ("1m", "0s", "1m"),
            ("1m", "1h", "0s"),
        ],
        ids=["zero_minimum", "zero_maximum", "zero_grain"],
    )
    def test_a_zero_duration_is_refused(
        self, minimum: str, maximum: str, grain: str
    ) -> None:
        """Both bounds and the grain are strictly positive durations."""
        with pytest.raises(ConfigError, match="strictly positive"):
            log_spaced(count=5, minimum=minimum, maximum=maximum, grain=grain)

    def test_a_point_that_quantizes_to_nothing_is_refused(self) -> None:
        """A grain coarser than the smallest point would drop it silently."""
        with pytest.raises(ConfigError, match="0s"):
            log_spaced(count=5, minimum="20s", maximum="1h", grain="1m")


class TestExplicit:
    """A caller-supplied resolved list, generated by nothing."""

    def test_the_given_durations_are_the_windows(self) -> None:
        """An explicit schedule is exactly what it was handed."""
        schedule = explicit(["1m", "5m", "1h"])
        assert schedule.windows == (
            Duration.parse("1m"),
            Duration.parse("5m"),
            Duration.parse("1h"),
        )

    def test_the_kind_is_recorded(self) -> None:
        """A schedule always states which generator produced it."""
        assert explicit(["1m"]).spec.kind is GeneratorKind.EXPLICIT

    def test_the_caller_order_is_preserved(self) -> None:
        """Nothing is sorted: a control list is used in the order it was written."""
        schedule = explicit(["1h", "1m", "30m"])
        assert _minutes(schedule.windows) == [60, 1, 30]

    def test_duration_instances_are_accepted(self) -> None:
        """The boundary accepts ``Duration | str``, like the rest of the package."""
        schedule = explicit([Duration.parse("1m"), Duration(300)])
        assert _minutes(schedule.windows) == [1, 5]

    def test_a_name_is_recorded_when_given(self) -> None:
        """A registered list carries the name it is asked for by."""
        schedule = explicit(["1m"], name="control-single-minute")
        assert isinstance(schedule.spec, ExplicitSpec)
        assert schedule.spec.name == "control-single-minute"

    def test_an_unnamed_list_records_no_name(self) -> None:
        """An ad-hoc list is not forced to invent a name."""
        schedule = explicit(["1m"])
        assert isinstance(schedule.spec, ExplicitSpec)
        assert schedule.spec.name is None

    def test_no_limiting_ratio_is_implied(self) -> None:
        """A hand-written list implies no constant ratio, and says so."""
        assert explicit(["1m", "5m", "1h"]).spec.limiting_ratio is None

    def test_an_empty_list_is_refused(self) -> None:
        """The empty schedule is refused here for the same reason as anywhere."""
        with pytest.raises(ConfigError, match="no windows"):
            explicit([])

    def test_a_list_of_exactly_the_cap_is_accepted(self) -> None:
        """The cap is inclusive: exactly 512 windows is a valid schedule.

        The refusal tests pin that 513 is too many; this pins that the
        boundary itself is on the accepted side.
        """
        schedule = explicit([f"{i}s" for i in range(1, MAX_RESOLVED_WINDOWS + 1)])
        assert len(schedule.windows) == MAX_RESOLVED_WINDOWS

    def test_a_repeated_window_is_refused(self) -> None:
        """A caller-supplied list is not silently deduplicated.

        The generated kinds deduplicate because their arithmetic
        produces repeats; a hand-written repeat is a mistake in the
        hand-written list, and correcting it silently would hide it.
        """
        with pytest.raises(ConfigError, match="once"):
            explicit(["1m", "5m", "1m"])

    def test_a_zero_window_is_refused(self) -> None:
        """A zero-length window carries no data, in any kind of schedule."""
        with pytest.raises(ConfigError, match="strictly positive"):
            explicit(["1m", "0s"])

    def test_a_malformed_duration_string_is_refused(self) -> None:
        """Durations are coerced at the boundary, in the temporal grammar."""
        with pytest.raises(ConfigError):
            explicit(["1m", "not-a-duration"])

    def test_a_list_past_the_cap_is_refused(self) -> None:
        """No schedule may name more windows than the package's one cap."""
        with pytest.raises(ConfigError, match="512"):
            explicit([f"{seconds}s" for seconds in range(1, 515)])

    def test_a_bare_string_is_not_a_list_of_windows(self) -> None:
        """A string is iterable, and iterating it would be nonsense."""
        with pytest.raises(ConfigError, match="list"):
            explicit("1m")


class TestRecordedRules:
    """The rounding and dedup rules are recorded, so they must be real rules."""

    def test_a_rounding_rule_that_is_not_a_rule_is_refused(self) -> None:
        """A plain string is not accepted in place of a RoundingRule member.

        The value is recorded in the identity and read back by name, so
        a string that merely looks like a member would serialize into a
        payload nothing can read.
        """
        with pytest.raises(ConfigError, match="rounding"):
            metallic_recurrence(
                coefficient=_COEFFICIENT,
                seed="1m",
                grain="1m",
                maximum="2w",
                rounding="nearest_ties_away",  # type: ignore[arg-type]
            )

    def test_a_log_spaced_rounding_rule_is_checked_the_same_way(self) -> None:
        """Every kind that quantizes checks its tie rule in the same words."""
        with pytest.raises(ConfigError, match="rounding"):
            log_spaced(
                count=5,
                minimum="1m",
                maximum="1h",
                grain="1m",
                rounding="nearest_ties_even",  # type: ignore[arg-type]
            )

    def test_a_dedup_rule_that_is_not_a_rule_is_refused(self) -> None:
        """The dedup rule is recorded too, and checked the same way."""
        with pytest.raises(ConfigError, match="dedup"):
            MetallicRecurrenceSpec(
                coefficient=2.0,
                seed=Duration.parse("1m"),
                grain=Duration.parse("1m"),
                maximum=Duration.parse("1d"),
                dedup="drop_later_repeats",  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    pytest.main([__file__])
