"""The count-valued lookback schedule.

Three things are worth more than the rest here. The two published lists
are pinned as literal numbers rather than recomputed from the formula,
because a test that recomputes the thing under test agrees with it
whatever it does. The identity is proved DIFFERENT from a duration
schedule's over the same numbers, which is what stops a lookback of
``{1, 3, 8}`` and a window schedule of ``{1m, 3m, 8m}`` being recorded as
the same schedule. And the two readers are proved to refuse each other's
payloads rather than coerce them.
"""

import math

import pytest

from ohlc_toolkit.schedules import (
    MAX_RESOLVED_WINDOWS,
    WindowSchedule,
    metallic_recurrence,
)
from ohlc_toolkit.schedules.lookback import (
    ExplicitLookbackSpec,
    LogSpacedLookbackSpec,
    LookbackSchedule,
    MetallicLookbackSpec,
    explicit_lookback,
    log_spaced_lookback,
    metallic_lookback,
)
from ohlc_toolkit.temporal import ConfigError

# The coefficient the indicator ontology names for the lookback ladder:
# sqrt(e + sqrt(5)), the same one the window schedule uses.
_COEFFICIENT = math.sqrt(math.e + math.sqrt(5))


def test_the_metallic_lookback_ladder_is_exactly_the_published_list() -> None:
    """1, 3, 8, 21, 56 -- written out, not recomputed.

    A test that re-ran the recurrence to build its expectation would
    agree with the generator whatever the generator did. These are the
    numbers the decision names, typed in.
    """
    schedule = metallic_lookback(coefficient=_COEFFICIENT, seed=1, grain=1, maximum=100)

    assert schedule.periods == (1, 3, 8, 21, 56)


def test_the_recipe_selection_is_exactly_three_eight_twenty_one() -> None:
    """Bounded [3, 21], the ladder yields {3, 8, 21}.

    21 is the case that matters. Its unrounded term is 21.434, which is
    ABOVE the maximum; it belongs in the list because it quantizes to
    exactly 21, which is inside it. Generation therefore produces the
    first term past the bound and lets resolution decide, rather than
    comparing a bound against a number the schedule never uses.
    """
    schedule = metallic_lookback(
        coefficient=_COEFFICIENT, seed=1, grain=1, minimum=3, maximum=21
    )

    assert schedule.periods == (3, 8, 21)


def test_the_log_spaced_control_is_exactly_seven_fourteen_twenty_eight() -> None:
    """7, 14, 28 -- the control the ladder is measured against."""
    schedule = log_spaced_lookback(count=3, minimum=7, maximum=28)

    assert schedule.periods == (7, 14, 28)


def test_only_one_term_past_the_bound_is_ever_generated() -> None:
    """The extra term is one, not a run of them.

    Quantization is monotone, so a term far past the maximum rounds past
    it too and resolution drops it. This pins that the generosity is
    exactly one term: a maximum of 9 admits 8 and nothing above it, even
    though 21.434 exists further along the recurrence.
    """
    schedule = metallic_lookback(coefficient=_COEFFICIENT, seed=1, grain=1, maximum=9)

    assert schedule.periods == (1, 3, 8)


def test_a_lookback_and_a_window_schedule_over_the_same_numbers_differ() -> None:
    """The identity carries the type, so {1, 3, 8} is not {1m, 3m, 8m}.

    Both are content hashes over a kind, its parameters and its resolved
    list. What separates them is the key the members are recorded under
    and the form of the members themselves -- ``"periods": [1, 3, 8]``
    against ``"windows": ["1m", "3m", "8m"]``. Neither is a coincidence
    of the numbers chosen: the payloads cannot be made equal.
    """
    counts = explicit_lookback([1, 3, 8])
    durations = metallic_recurrence(
        coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="8m"
    )

    assert durations.windows == tuple(
        type(durations.windows[0])(period * 60) for period in (1, 3, 8)
    )
    assert counts.schedule_id != durations.schedule_id
    assert "periods" in counts.to_dict()
    assert "windows" in durations.to_dict()


def test_a_lookback_round_trips_and_requires_its_recorded_id() -> None:
    """The recorded id is checked, so an edited payload is refused."""
    schedule = metallic_lookback(coefficient=_COEFFICIENT, seed=1, grain=1, maximum=100)
    payload = schedule.to_dict()

    assert LookbackSchedule.from_dict(payload) == schedule

    tampered = {**payload, "periods": [1, 3, 8, 21, 57]}
    with pytest.raises(ConfigError, match="schedule_id"):
        LookbackSchedule.from_dict(tampered)


def test_a_window_schedule_payload_is_refused_by_the_lookback_reader() -> None:
    """Refused for a missing key, not coerced into counts."""
    durations = metallic_recurrence(
        coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="8m"
    )

    with pytest.raises(ConfigError, match="periods"):
        LookbackSchedule.from_dict(durations.to_dict())


def test_a_lookback_payload_is_refused_by_the_window_reader() -> None:
    """And the reverse, which is the half a one-directional guard misses."""
    counts = explicit_lookback([1, 3, 8])

    with pytest.raises(ConfigError, match="windows"):
        WindowSchedule.from_dict(counts.to_dict())


@pytest.mark.parametrize(
    ("value", "match"),
    [
        (0, r"strictly positive"),
        (-1, r"strictly positive"),
        (True, r"must be an int"),
        (1.0, r"must be an int"),
        ("3", r"must be an int"),
    ],
)
def test_a_period_that_is_not_a_positive_int_is_refused(
    value: object, match: str
) -> None:
    """A lookback of ``True`` is nobody's intention, and would mean one."""
    with pytest.raises(ConfigError, match=match):
        ExplicitLookbackSpec(periods=(value,))  # type: ignore[arg-type]


def test_a_recorded_list_of_durations_is_refused_as_periods() -> None:
    """The member form is checked, not only the key name."""
    counts = explicit_lookback([1, 3, 8])
    payload = {**counts.to_dict(), "periods": ["1m", "3m", "8m"]}

    with pytest.raises(ConfigError, match="must be an int"):
        LookbackSchedule.from_dict(payload)


def test_crossed_bounds_are_refused() -> None:
    """A minimum above the maximum names nothing, and says so."""
    with pytest.raises(ConfigError, match="must not exceed"):
        MetallicLookbackSpec(
            coefficient=_COEFFICIENT, seed=1, grain=1, minimum=20, maximum=10
        )
    with pytest.raises(ConfigError, match="must not exceed"):
        LogSpacedLookbackSpec(count=3, minimum=20, maximum=10)


def test_bounds_that_leave_nothing_are_refused_in_the_lookback_s_own_words() -> None:
    """The refusal counts periods, not seconds.

    The resolution rule is shared with the window schedule; only the
    prose differs, and this is what proves the prose followed the unit
    rather than staying in seconds.
    """
    with pytest.raises(ConfigError, match="lookback") as caught:
        metallic_lookback(
            coefficient=_COEFFICIENT, seed=1, grain=1, minimum=90, maximum=100
        )

    assert "s]" not in str(caught.value)


def test_a_repeated_count_is_refused() -> None:
    """Each count once, as each window appears once."""
    with pytest.raises(ConfigError, match="twice"):
        explicit_lookback([3, 8, 3])


def test_an_empty_lookback_schedule_is_refused() -> None:
    """A schedule naming nothing is not a schedule."""
    with pytest.raises(ConfigError, match="at least one"):
        explicit_lookback([])


def test_a_log_spaced_lookback_round_trips_through_its_own_reader() -> None:
    """The control's parameters survive a round trip, id and all."""
    schedule = log_spaced_lookback(count=3, minimum=7, maximum=28)

    assert LookbackSchedule.from_dict(schedule.to_dict()) == schedule
    assert schedule.to_dict()["parameters"] == {
        "count": 3,
        "minimum": 7,
        "maximum": 28,
        "grain": 1,
        "rounding": "nearest_ties_away",
        "dedup": "drop_later_repeats",
    }


def test_a_recorded_lower_bound_reads_back_present() -> None:
    """The optional bound survives a round trip when it was given.

    Paired with the absent case below: an optional field needs both
    directions, and only one of them is the branch a default would hide.
    """
    lower = 3
    schedule = metallic_lookback(
        coefficient=_COEFFICIENT, seed=1, grain=1, minimum=lower, maximum=21
    )

    assert schedule.to_dict()["parameters"]["minimum"] == lower  # type: ignore[index]
    assert LookbackSchedule.from_dict(schedule.to_dict()) == schedule


def test_an_absent_optional_bound_reads_back_absent() -> None:
    """`None` means no lower bound, and survives being recorded as one."""
    schedule = metallic_lookback(coefficient=_COEFFICIENT, seed=1, grain=1, maximum=100)

    assert schedule.to_dict()["parameters"]["minimum"] is None  # type: ignore[index]
    assert LookbackSchedule.from_dict(schedule.to_dict()) == schedule


@pytest.mark.parametrize("value", [{"3": 1}, "138", 3])
def test_a_period_list_that_is_not_a_list_is_refused(value: object) -> None:
    """A string is a sequence, and a string of digits is not a schedule."""
    counts = explicit_lookback([1, 3, 8])

    with pytest.raises(ConfigError, match=r"list of period counts|must be an int"):
        LookbackSchedule.from_dict({**counts.to_dict(), "periods": value})


def test_a_lookback_longer_than_the_cap_is_refused() -> None:
    """The same cap the window schedule carries, counted in the same way."""
    with pytest.raises(ConfigError, match="at most"):
        explicit_lookback(range(1, MAX_RESOLVED_WINDOWS + 2))


if __name__ == "__main__":
    pytest.main([__file__])
