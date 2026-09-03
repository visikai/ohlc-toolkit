"""Named registrations, and the absence of anything chosen for the caller.

Everything in the registry is data a caller asks for by name. These
tests pin both halves of that: the registered lists are exactly what
they claim to be, and no entry point in the package supplies a
schedule, a coefficient, or a divisor on anybody's behalf.
"""

import hashlib
import inspect
import json
import math
from collections.abc import Callable
from fractions import Fraction

import pytest

from ohlc_toolkit import schedules as schedules_namespace
from ohlc_toolkit.schedules import (
    GeneratorKind,
    WindowSchedule,
    explicit_pairs,
    log_spaced,
    metallic_recurrence,
    named_schedule,
    named_schedule_names,
    w_over_k,
)
from ohlc_toolkit.temporal import ConfigError, Duration

_MINUTE_SECONDS = 60
_TWO_WEEKS_MINUTES = 14 * 24 * 60
_COEFFICIENT = math.sqrt(math.e + math.sqrt(5))

_LEGACY_NAME = "metallic-legacy-2025"

# A refusal echoing a rejected name must stay readable in a log line.
_MAX_REASONABLE_MESSAGE_CHARS = 1_000

# The legacy schedule, in whole minutes. Pinned as a literal, and
# re-derived below from the per-step-truncating recurrence it came from.
_LEGACY_MINUTES = (1, 3, 7, 18, 47, 122, 318, 829, 2163, 5643, 14723)


def _truncated_each_step() -> list[int]:
    """Recompute the recurrence the way an implementation that truncates would.

    The coefficient is held exactly, so the only rounding here is the
    deliberate one: the floor taken at every step, before the term is
    fed back in. That is what makes this sequence differ from the
    quantize-once one -- and what the registered list is a record of.
    """
    coefficient = Fraction(_COEFFICIENT)
    terms = [1, 1]
    while True:
        following = coefficient * terms[-1] + terms[-2]
        if following > _TWO_WEEKS_MINUTES:
            return terms
        terms.append(math.floor(following))


def _dedup(values: list[int]) -> list[int]:
    """Drop later repeats, preserving the order of first appearance."""
    kept: list[int] = []
    for value in values:
        if value not in kept:
            kept.append(value)
    return kept


class TestLegacyRegistration:
    """The one registered window list, and what it is a record of."""

    def test_the_pinned_list_is_the_truncating_recurrence(self) -> None:
        """Guard the fixture: the literal list is that arithmetic.

        Checked before it is used to check anything else, so a wrong
        literal cannot quietly become the expectation.
        """
        assert _dedup(_truncated_each_step()) == list(_LEGACY_MINUTES)

    def test_the_registered_schedule_holds_those_windows(self) -> None:
        """The registration is exactly the list it claims to preserve."""
        schedule = named_schedule(_LEGACY_NAME)
        assert [
            window.total_seconds // _MINUTE_SECONDS for window in schedule.windows
        ] == list(_LEGACY_MINUTES)

    def test_it_is_an_explicit_schedule_under_its_own_name(self) -> None:
        """It is data, not a generator: nothing recomputes it."""
        schedule = named_schedule(_LEGACY_NAME)
        assert schedule.spec.kind is GeneratorKind.EXPLICIT
        assert schedule.to_dict()["parameters"] == {
            "name": _LEGACY_NAME,
            "limiting_ratio": None,
        }

    def test_it_differs_from_the_quantize_once_schedule(self) -> None:
        """The drift it preserves is the whole reason it is kept.

        The same coefficient, seed, grain, and bound, quantized once at
        the end instead of at every step, is a different schedule from
        the fourth window on.
        """
        generated = metallic_recurrence(
            coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="2w"
        )
        assert named_schedule(_LEGACY_NAME).windows != generated.windows

    def test_its_identity_is_the_hash_of_its_payload(self) -> None:
        """A registered schedule is named by content like any other.

        The stored windows are canonical compact durations, so 122m is
        written as 2h2m and 14723m as 10d5h23m: a payload states a
        duration one way, whatever spelling built it.
        """
        schedule = named_schedule(_LEGACY_NAME)
        payload = {
            "kind": "explicit",
            "parameters": {"name": _LEGACY_NAME, "limiting_ratio": None},
            "windows": [
                str(Duration(minutes * _MINUTE_SECONDS)) for minutes in _LEGACY_MINUTES
            ],
        }
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        assert schedule.schedule_id == hashlib.sha256(text.encode("utf-8")).hexdigest()

    def test_it_round_trips_like_any_other_schedule(self) -> None:
        """Nothing about being registered makes it a special case."""
        schedule = named_schedule(_LEGACY_NAME)
        assert WindowSchedule.from_dict(json.loads(json.dumps(schedule.to_dict()))) == (
            schedule
        )

    def test_it_is_listed_by_name(self) -> None:
        """A caller can discover what there is to ask for."""
        assert _LEGACY_NAME in named_schedule_names()

    def test_asking_twice_gives_an_equal_schedule(self) -> None:
        """A registration is a value, so handing it out twice is safe."""
        assert named_schedule(_LEGACY_NAME) == named_schedule(_LEGACY_NAME)


class TestLookup:
    """Asking for something that is not there."""

    def test_an_unknown_name_is_refused(self) -> None:
        """A typo does not silently fall back to anything."""
        with pytest.raises(ConfigError, match="metallic-legacy-2024"):
            named_schedule("metallic-legacy-2024")

    def test_the_refusal_lists_what_there_is(self) -> None:
        """The message says what could have been asked for instead."""
        with pytest.raises(ConfigError, match=_LEGACY_NAME):
            named_schedule("nope")

    def test_an_oversized_name_is_echoed_within_bounds(self) -> None:
        """A pathological name cannot produce a pathological message."""
        with pytest.raises(ConfigError) as caught:
            named_schedule("x" * 10_000)
        assert len(str(caught.value)) < _MAX_REASONABLE_MESSAGE_CHARS

    def test_a_non_string_name_is_refused(self) -> None:
        """A name is text; anything else never named a registration."""
        with pytest.raises(ConfigError, match="name"):
            named_schedule(7)  # type: ignore[arg-type]


class TestNoDefaults:
    """The package ships mechanisms, and chooses nothing for the caller."""

    @pytest.mark.parametrize(
        ("entry_point", "required"),
        [
            (metallic_recurrence, ("coefficient", "seed", "grain", "maximum")),
            (log_spaced, ("count", "minimum", "maximum", "grain")),
            (w_over_k, ("windows", "divisor", "allowed", "source_cadence")),
            (explicit_pairs, ("pairs",)),
        ],
        ids=["metallic_recurrence", "log_spaced", "w_over_k", "explicit_pairs"],
    )
    def test_every_shaping_parameter_must_be_stated(
        self, entry_point: Callable[..., object], required: tuple[str, ...]
    ) -> None:
        """No coefficient, no bounds, no divisor is supplied on anyone's behalf.

        A default here would be a policy the package has no business
        holding: it would silently decide what windows a caller gets,
        and every recipe that took the default would record a choice
        nobody made.
        """
        parameters = inspect.signature(entry_point).parameters
        for name in required:
            assert parameters[name].default is inspect.Parameter.empty, (
                f"{name} must have no default"
            )

    def test_the_namespace_exports_no_default(self) -> None:
        """Not even by name: there is no DEFAULT_ anything to reach for."""
        assert not [
            name for name in schedules_namespace.__all__ if "DEFAULT" in name.upper()
        ]

    def test_the_registry_is_not_consulted_unless_asked(self) -> None:
        """Registered names are data, and reaching them takes asking by name."""
        assert "metallic" not in named_schedule_names()
        assert named_schedule_names() == (_LEGACY_NAME,)


if __name__ == "__main__":
    pytest.main([__file__])
