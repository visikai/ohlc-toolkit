"""Horizon schedules: the same durations as windows, never the same identity."""

import inspect
import math
import re
from collections.abc import Callable
from dataclasses import fields
from functools import partial

import pytest

import ohlc_toolkit.schedules as schedules_namespace
from ohlc_toolkit.schedules import (
    ExplicitSpec,
    HorizonSchedule,
    LogSpacedSpec,
    LookbackSchedule,
    MetallicRecurrenceSpec,
    RoundingRule,
    WindowSchedule,
    explicit,
    explicit_horizons,
    explicit_lookback,
    log_spaced,
    log_spaced_horizons,
    metallic_horizons,
    metallic_recurrence,
)
from ohlc_toolkit.schedules.identity import content_hash
from ohlc_toolkit.temporal import ConfigError, Duration

# The parameters the temporal core contract names for the metallic ladder:
# coefficient sqrt(e + sqrt(5)), seed 1m, grain 1m, round to nearest, drop
# later repeats. The three horizons the target ontology selects are
# members of THIS recurrence, and the test below shows it by running it.
_COEFFICIENT = math.sqrt(math.e + math.sqrt(5))
_SELECTED = ("2h26m", "6h20m", "16h33m")
# The two generators over the contract's ladder, differing only in what
# they are a schedule of.
_horizon_ladder = partial(
    metallic_horizons, coefficient=_COEFFICIENT, seed="1m", grain="1m"
)
_window_ladder = partial(
    metallic_recurrence, coefficient=_COEFFICIENT, seed="1m", grain="1m"
)


def test_the_selected_horizons_are_members_of_the_metallic_recurrence() -> None:
    """2h26m, 6h20m and 16h33m fall out of the recurrence; nothing types them in.

    Proved by construction: the generator is run with the contract's
    parameters up to a day and the three are asserted to be among what it
    produced, in the recurrence's own order, spelled by ``Duration``.
    """
    ladder = _horizon_ladder(maximum="1d")

    spelled = tuple(str(horizon) for horizon in ladder.horizons)
    assert spelled == ("1m", "3m", "8m", "21m", "56m", "2h26m", "6h20m", "16h33m")
    assert all(selected in spelled for selected in _SELECTED)
    assert spelled.index("2h26m") < spelled.index("6h20m") < spelled.index("16h33m")


def test_a_horizon_spells_exactly_like_the_window_of_the_same_duration() -> None:
    """Same generator, same parameters: the same durations, letter for letter."""
    horizons = _horizon_ladder(maximum="1d")
    windows = _window_ladder(maximum="1d")

    assert horizons.horizons == windows.windows
    assert [str(h) for h in horizons.horizons] == [str(w) for w in windows.windows]
    assert all(isinstance(horizon, Duration) for horizon in horizons.horizons)


def test_identical_parameters_give_a_horizon_and_a_window_schedule_different_ids() -> (
    None
):
    """The identity carries what the schedule is OF, not only its numbers."""
    horizons = _horizon_ladder(maximum="1d")
    windows = _window_ladder(maximum="1d")

    assert horizons.spec == windows.spec, "the fixture is not identical parameters"
    assert horizons.schedule_id != windows.schedule_id
    assert "horizons" in horizons.to_dict()
    assert "windows" not in horizons.to_dict()
    assert "windows" in windows.to_dict()
    assert "horizons" not in windows.to_dict()


def test_a_horizon_payload_is_refused_by_the_window_reader() -> None:
    """Refused for a missing key, before any value is read as a window."""
    horizons = _horizon_ladder(maximum="1d")

    with pytest.raises(ConfigError, match="windows"):
        WindowSchedule.from_dict(horizons.to_dict())


def test_a_window_payload_is_refused_by_the_horizon_reader() -> None:
    """And the reverse, which is the half a one-directional guard misses."""
    windows = _window_ladder(maximum="1d")

    with pytest.raises(ConfigError, match="horizons"):
        HorizonSchedule.from_dict(windows.to_dict())


def test_horizon_and_lookback_payloads_refuse_each_other() -> None:
    """Three kinds, three keys: no pair of readers accepts the other's payload."""
    horizons = explicit_horizons(["1m", "3m", "8m"])
    lookback = explicit_lookback([1, 3, 8])

    with pytest.raises(ConfigError, match="periods"):
        LookbackSchedule.from_dict(horizons.to_dict())
    with pytest.raises(ConfigError, match="horizons"):
        HorizonSchedule.from_dict(lookback.to_dict())
    assert horizons.schedule_id != lookback.schedule_id


@pytest.mark.parametrize(
    "schedule",
    [
        _horizon_ladder(minimum="2h26m", maximum="16h33m"),
        log_spaced_horizons(count=3, minimum="1h", maximum="4h", grain="1h"),
        explicit_horizons(["2h26m", "6h20m", "16h33m"], name="selected"),
    ],
    ids=["metallic", "log_spaced", "explicit"],
)
def test_every_kind_round_trips_through_its_payload(schedule: HorizonSchedule) -> None:
    """to_dict then from_dict is the identity, for each generator kind."""
    payload = schedule.to_dict()

    assert tuple(payload) == ("kind", "parameters", "horizons", "schedule_id")
    assert HorizonSchedule.from_dict(payload) == schedule
    assert HorizonSchedule.from_dict(payload).schedule_id == schedule.schedule_id


def test_an_edited_payload_is_refused_by_its_recorded_id() -> None:
    """The recorded id is checked against the derived one, not trusted."""
    schedule = _horizon_ladder(maximum="1d")
    payload = schedule.to_dict()

    recorded = payload["horizons"]
    assert isinstance(recorded, list)
    tampered = {**payload, "horizons": [*recorded[:-1], "17h"]}
    with pytest.raises(ConfigError, match="schedule_id"):
        HorizonSchedule.from_dict(tampered)


def test_a_recorded_list_that_breaks_an_invariant_is_refused() -> None:
    """A payload whose horizons repeat is refused even with a matching id.

    The constructors cannot produce such a list, so the reader is the only
    door, and the invariant check on construction is what closes it.
    """
    schedule = explicit_horizons(["1m", "3m"])
    payload = schedule._identity_payload()
    payload["horizons"] = ["1m", "1m"]
    forged = {**payload, "schedule_id": content_hash(payload)}

    with pytest.raises(ConfigError, match=r"once|twice|repeat"):
        HorizonSchedule.from_dict(forged)


def test_a_non_default_rounding_is_recorded_on_a_metallic_horizon_schedule() -> None:
    """The tie rule is part of the identity, not an implementation detail.

    A dropped pass-through would record the default rule and the default
    rule's id for a schedule the caller built with another, so the id is
    asserted against the default as well as the field.
    """
    default = _horizon_ladder(minimum="2h26m", maximum="16h33m")
    ties_even = _horizon_ladder(
        minimum="2h26m", maximum="16h33m", rounding=RoundingRule.NEAREST_TIES_EVEN
    )
    assert isinstance(ties_even.spec, MetallicRecurrenceSpec)
    assert ties_even.spec.rounding is RoundingRule.NEAREST_TIES_EVEN
    assert ties_even.schedule_id != default.schedule_id, (
        "two tie rules share an id: the rounding was not recorded"
    )


def test_a_non_default_rounding_is_recorded_on_a_log_spaced_horizon_schedule() -> None:
    """The tie rule is part of the identity for the log-spaced generator too."""
    default = log_spaced_horizons(count=3, minimum="1h", maximum="4h", grain="1h")
    ties_even = log_spaced_horizons(
        count=3,
        minimum="1h",
        maximum="4h",
        grain="1h",
        rounding=RoundingRule.NEAREST_TIES_EVEN,
    )
    assert isinstance(ties_even.spec, LogSpacedSpec)
    assert ties_even.spec.rounding is RoundingRule.NEAREST_TIES_EVEN
    assert ties_even.schedule_id != default.schedule_id, (
        "two tie rules share an id: the rounding was not recorded"
    )


def test_a_name_is_recorded_on_an_explicit_horizon_schedule() -> None:
    """A registered list carries the name it is asked for by; an ad-hoc one none."""
    named = explicit_horizons(["2h26m"], name="control-single-horizon")
    unnamed = explicit_horizons(["2h26m"])
    assert isinstance(named.spec, ExplicitSpec)
    assert named.spec.name == "control-single-horizon"
    assert isinstance(unnamed.spec, ExplicitSpec)
    assert unnamed.spec.name is None
    assert named.schedule_id != unnamed.schedule_id, (
        "a named and an unnamed list share an id: the name was not recorded"
    )


def test_a_bare_string_is_not_a_list_of_horizons() -> None:
    """A string is iterable, and iterating it would be nonsense here too."""
    with pytest.raises(ConfigError, match="list"):
        explicit_horizons("2h26m")


@pytest.mark.parametrize(
    ("generate", "kwargs"),
    [
        (
            metallic_horizons,
            {"coefficient": 1.618, "seed": "0s", "grain": "1h", "maximum": "4h"},
        ),
        (
            metallic_horizons,
            {"coefficient": 1.618, "seed": "1h", "grain": "1h", "maximum": "0s"},
        ),
        (
            metallic_horizons,
            {
                "coefficient": 1.618,
                "seed": "1h",
                "grain": "1h",
                "maximum": "4h",
                "minimum": "0s",
            },
        ),
        (
            log_spaced_horizons,
            {"count": 3, "minimum": "0s", "maximum": "4h", "grain": "1h"},
        ),
        (
            log_spaced_horizons,
            {"count": 3, "minimum": "1h", "maximum": "0s", "grain": "1h"},
        ),
        (explicit_horizons, {"horizons": ["1h", "0s"]}),
    ],
    ids=[
        "metallic-seed",
        "metallic-maximum",
        "metallic-minimum",
        "log-spaced-minimum",
        "log-spaced-maximum",
        "explicit-member",
    ],
)
def test_a_zero_length_horizon_is_refused_as_a_horizon(
    generate: Callable[..., HorizonSchedule], kwargs: dict[str, object]
) -> None:
    """A zero-length duration is refused in the name of what it was passed as.

    A horizon schedule is built from the window generators' own parameter
    classes, whose validators name a "Window duration". Every duration is
    validated as a horizon before those classes see it, so the refusal a
    caller reads names the thing they actually passed.
    """
    with pytest.raises(ConfigError) as refused:
        generate(**kwargs)
    assert str(refused.value).startswith("Horizon duration must be strictly positive")
    assert "window" not in str(refused.value).lower()


def test_a_horizon_schedule_carries_no_cadence() -> None:
    """Two fields, neither a cadence; no constructor takes one.

    The frame a target is computed on supplies the cadence. A horizon
    schedule that carried one would let a target be defined against a
    grid the frame does not have.
    """
    assert [field.name for field in fields(HorizonSchedule)] == ["spec", "horizons"]
    for constructor in (metallic_horizons, log_spaced_horizons, explicit_horizons):
        parameters = inspect.signature(constructor).parameters
        offending = [
            name
            for name in parameters
            if "cadence" in name or "emit" in name or "pair" in name
        ]
        assert not offending, f"{constructor.__name__} takes {offending}"


# A whole-word "window(s)" or "horizon(s)", replaced by a common
# placeholder so two messages that differ only in which of those nouns
# they name compare equal underneath it.
_NOUN = re.compile(r"\b(?:windows?|horizons?)\b")


def _without_the_noun(message: str) -> str:
    """Strip the schedule noun, leaving only the shared arithmetic."""
    return _NOUN.sub("_", message)


@pytest.mark.parametrize(
    ("horizon_constructor", "window_constructor", "arguments"),
    [
        (
            _horizon_ladder,
            _window_ladder,
            {"minimum": "2d", "maximum": "1d"},
        ),
        (
            log_spaced_horizons,
            log_spaced,
            {"count": 3, "minimum": "1h", "maximum": "4h", "grain": "3h"},
        ),
        (explicit_horizons, explicit, {"horizons": []}),
        (explicit_horizons, explicit, {"horizons": ["1m", "1m"]}),
    ],
    ids=["crossed-bounds", "endpoint-off-the-grain", "empty-list", "repeated-member"],
)
def test_a_horizon_schedule_has_the_window_generator_s_refusals(
    horizon_constructor: Callable[..., object],
    window_constructor: Callable[..., object],
    arguments: dict[str, object],
) -> None:
    """The same parameters are refused under the same condition, by construction.

    Not with the same words: the window path names "window" and the
    horizon path names "horizon", by design. What must stay identical is
    everything else in the message -- the bounds, the counts, the
    quantized value -- which is the shared arithmetic neither path is
    allowed to drift from. Stripping the noun from both messages before
    comparing pins that arithmetic without pinning which noun raised it.
    """
    window_arguments = {
        ("windows" if key == "horizons" else key): value
        for key, value in arguments.items()
    }
    with pytest.raises(ConfigError) as from_windows:
        window_constructor(**window_arguments)
    with pytest.raises(ConfigError) as from_horizons:
        horizon_constructor(**arguments)

    assert _without_the_noun(str(from_horizons.value)) == _without_the_noun(
        str(from_windows.value)
    )
    assert "window" not in str(from_horizons.value)
    assert "horizon" not in str(from_windows.value)


def test_the_public_names_are_exported() -> None:
    """The horizon schedule is reachable from the package, not only the module."""
    for name in (
        "HorizonSchedule",
        "explicit_horizons",
        "log_spaced_horizons",
        "metallic_horizons",
    ):
        assert name in schedules_namespace.__all__, f"{name} is not exported"
        assert hasattr(schedules_namespace, name)


if __name__ == "__main__":
    pytest.main([__file__])
