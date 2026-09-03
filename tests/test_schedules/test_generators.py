"""Window-scale schedule generators: the resolved durations they produce.

Every expected value here is computed independently of the generator
under test -- with exact rationals in this module, or pinned as a
literal list -- so an implementation that agreed with itself but not
with the arithmetic would still be caught.
"""

import math
from fractions import Fraction

import pytest
from ohlc_toolkit.schedules import (
    GeneratorKind,
    RoundingRule,
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


if __name__ == "__main__":
    pytest.main([__file__])
