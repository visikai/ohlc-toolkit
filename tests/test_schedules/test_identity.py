"""The persisted identity every schedule carries: payload, round trip, id.

The canonical payload and the hash over it are re-stated here rather
than imported, so these tests check the serialization rule instead of
agreeing with whatever the implementation happens to do.
"""

import hashlib
import json
import math

import pytest

from ohlc_toolkit.schedules import (
    ExplicitSpec,
    GeneratorKind,
    LogSpacedSpec,
    RoundingRule,
    WindowSchedule,
    explicit,
    log_spaced,
    metallic_recurrence,
)
from ohlc_toolkit.temporal import ConfigError, Duration

_COEFFICIENT = math.sqrt(math.e + math.sqrt(5))
_SHA256_HEX_LENGTH = 64


def _metallic() -> WindowSchedule:
    """Resolve the metallic schedule the payload tests are written against."""
    return metallic_recurrence(
        coefficient=_COEFFICIENT, seed="1m", grain="1m", maximum="2w"
    )


def _log() -> WindowSchedule:
    """Resolve a log-spaced schedule."""
    return log_spaced(count=5, minimum="1m", maximum="1h", grain="1m")


def _explicit() -> WindowSchedule:
    """Resolve a named explicit schedule."""
    return explicit(["1m", "5m", "1h"], name="control-three")


def _every_kind() -> list[WindowSchedule]:
    """One resolved schedule of each generator kind."""
    return [_metallic(), _log(), _explicit()]


def _hash_of(payload: dict[str, object]) -> str:
    """Hash an identity payload the way a schedule id is defined to be hashed.

    Written out here -- sorted keys, no whitespace, sha256 over the
    UTF-8 bytes -- so the id is checked against the rule rather than
    against the code that implements it.
    """
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestPayload:
    """What a schedule serializes to, key by key."""

    def test_the_metallic_payload_records_every_parameter(self) -> None:
        """Kind, parameters, resolved windows, and the id over them.

        The limiting ratio is recomputed here from its closed form; that
        the closed form is the right number is pinned separately,
        against the sequence itself.
        """
        schedule = _metallic()
        expected_parameters = {
            "coefficient": _COEFFICIENT,
            "limiting_ratio": (_COEFFICIENT + math.sqrt(_COEFFICIENT**2 + 4)) / 2,
            "seed": "1m",
            "grain": "1m",
            "minimum": None,
            "maximum": "2w",
            "rounding": "nearest_ties_away",
            "dedup": "drop_later_repeats",
        }
        expected_windows = [str(window) for window in schedule.windows]

        payload = schedule.to_dict()
        assert payload["kind"] == "metallic_recurrence"
        assert payload["parameters"] == expected_parameters
        assert payload["windows"] == expected_windows
        assert payload["schedule_id"] == _hash_of(
            {
                "kind": "metallic_recurrence",
                "parameters": expected_parameters,
                "windows": expected_windows,
            }
        )

    def test_the_log_spaced_payload_records_the_count(self) -> None:
        """A log-spaced ladder records the count in place of a coefficient."""
        schedule = _log()
        parameters = schedule.to_dict()["parameters"]
        assert parameters == {
            "count": 5,
            "limiting_ratio": schedule.spec.limiting_ratio,
            "minimum": "1m",
            "maximum": "1h",
            "grain": "1m",
            "rounding": "nearest_ties_away",
            "dedup": "drop_later_repeats",
        }

    def test_the_explicit_payload_records_the_name_and_no_ratio(self) -> None:
        """A caller-supplied list has a name and implies no ratio."""
        parameters = _explicit().to_dict()["parameters"]
        assert parameters == {"name": "control-three", "limiting_ratio": None}

    def test_the_resolved_windows_are_embedded_in_every_kind(self) -> None:
        """The list a schedule resolved to travels with the parameters.

        A payload that only named the generator would have to be re-run
        to be read, and would then depend on the arithmetic of whatever
        machine read it.
        """
        for schedule in _every_kind():
            assert schedule.to_dict()["windows"] == [
                str(window) for window in schedule.windows
            ]

    def test_the_payload_survives_real_json(self) -> None:
        """Every value is a plain JSON type, in every kind."""
        for schedule in _every_kind():
            text = json.dumps(schedule.to_dict())
            assert json.loads(text) == schedule.to_dict()


class TestScheduleId:
    """The content hash over that payload."""

    def test_the_id_is_a_sha256_hex_digest(self) -> None:
        """A schedule id looks like what it is."""
        for schedule in _every_kind():
            schedule_id = schedule.schedule_id
            assert len(schedule_id) == _SHA256_HEX_LENGTH
            assert set(schedule_id) <= set("0123456789abcdef")

    def test_equal_specs_hash_equal(self) -> None:
        """Two independently resolved, identical schedules share an id."""
        assert _metallic().schedule_id == _metallic().schedule_id
        assert _metallic() == _metallic()

    def test_the_id_is_stable_across_calls(self) -> None:
        """The id is a function of the identity, not of when it was asked for."""
        schedule = _metallic()
        assert schedule.schedule_id == schedule.schedule_id

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (
                metallic_recurrence(
                    coefficient=2.0, seed="1m", grain="1m", maximum="1d"
                ),
                metallic_recurrence(
                    coefficient=2.5, seed="1m", grain="1m", maximum="1d"
                ),
            ),
            (
                metallic_recurrence(
                    coefficient=2.0, seed="1m", grain="1m", maximum="1d"
                ),
                metallic_recurrence(
                    coefficient=2.0, seed="2m", grain="1m", maximum="1d"
                ),
            ),
            (
                metallic_recurrence(
                    coefficient=2.0, seed="2m", grain="1m", maximum="1d"
                ),
                metallic_recurrence(
                    coefficient=2.0, seed="2m", grain="2m", maximum="1d"
                ),
            ),
            (
                metallic_recurrence(
                    coefficient=2.0, seed="1m", grain="1m", maximum="1d"
                ),
                metallic_recurrence(
                    coefficient=2.0, seed="1m", grain="1m", maximum="2d"
                ),
            ),
            (
                metallic_recurrence(
                    coefficient=2.0, seed="1m", grain="1m", maximum="1d"
                ),
                metallic_recurrence(
                    coefficient=2.0,
                    seed="1m",
                    grain="1m",
                    minimum="5m",
                    maximum="1d",
                ),
            ),
            (
                log_spaced(count=5, minimum="1m", maximum="1h", grain="1m"),
                log_spaced(count=6, minimum="1m", maximum="1h", grain="1m"),
            ),
            (
                explicit(["1m", "5m"]),
                explicit(["1m", "10m"]),
            ),
            (
                explicit(["1m", "5m"]),
                explicit(["5m", "1m"]),
            ),
            (
                explicit(["1m", "5m"]),
                explicit(["1m", "5m"], name="control"),
            ),
            (
                explicit(["1m", "5m"], name="control"),
                explicit(["1m", "5m"], name="baseline"),
            ),
        ],
        ids=[
            "coefficient",
            "seed",
            "grain",
            "maximum",
            "minimum",
            "count",
            "windows",
            "window_order",
            "name_added",
            "name_changed",
        ],
    )
    def test_any_parameter_difference_changes_the_id(
        self, first: WindowSchedule, second: WindowSchedule
    ) -> None:
        """One field apart is a different schedule, and a different id."""
        assert first.schedule_id != second.schedule_id
        assert first != second

    def test_a_different_tie_rule_changes_the_id_even_with_equal_windows(
        self,
    ) -> None:
        """The id covers the parameters, not only the list they produced.

        This coefficient never lands on a tie, so both rules resolve the
        same windows -- and the two schedules are still not the same
        schedule, because they would not agree on a sequence that did
        hit one.
        """
        away = metallic_recurrence(
            coefficient=_COEFFICIENT,
            seed="1m",
            grain="1m",
            maximum="2w",
            rounding=RoundingRule.NEAREST_TIES_AWAY,
        )
        even = metallic_recurrence(
            coefficient=_COEFFICIENT,
            seed="1m",
            grain="1m",
            maximum="2w",
            rounding=RoundingRule.NEAREST_TIES_EVEN,
        )
        assert away.windows == even.windows
        assert away.schedule_id != even.schedule_id

    def test_a_metallic_and_an_explicit_schedule_of_the_same_windows_differ(
        self,
    ) -> None:
        """How a list was arrived at is part of what it is."""
        generated = _metallic()
        copied = explicit(list(generated.windows))
        assert copied.windows == generated.windows
        assert copied.schedule_id != generated.schedule_id


class TestRoundTrip:
    """from_dict(to_dict(x)) == x, through real JSON."""

    def test_every_kind_round_trips_through_json(self) -> None:
        """A schedule read back out of a JSON string is the same schedule."""
        for schedule in _every_kind():
            restored = WindowSchedule.from_dict(
                json.loads(json.dumps(schedule.to_dict()))
            )
            assert restored == schedule
            assert restored.schedule_id == schedule.schedule_id

    def test_the_restored_spec_is_the_same_kind(self) -> None:
        """The discriminator survives the round trip."""
        for schedule, kind in zip(
            _every_kind(),
            [
                GeneratorKind.METALLIC_RECURRENCE,
                GeneratorKind.LOG_SPACED,
                GeneratorKind.EXPLICIT,
            ],
            strict=True,
        ):
            restored = WindowSchedule.from_dict(schedule.to_dict())
            assert restored.spec.kind is kind

    def test_a_schedule_with_a_minimum_round_trips(self) -> None:
        """The optional bound survives as a bound, not as a missing key."""
        schedule = metallic_recurrence(
            coefficient=2.0, seed="1m", grain="1m", minimum="5m", maximum="1d"
        )
        assert WindowSchedule.from_dict(schedule.to_dict()) == schedule

    def test_the_restored_windows_are_durations(self) -> None:
        """Windows come back as Durations, not as the strings they were stored as."""
        restored = WindowSchedule.from_dict(_explicit().to_dict())
        assert all(isinstance(window, Duration) for window in restored.windows)


class TestFromDictRefusals:
    """A payload that cannot be trusted is refused, never patched up."""

    def test_a_missing_key_is_refused(self) -> None:
        """A truncated payload is not silently defaulted."""
        payload = _metallic().to_dict()
        del payload["windows"]
        with pytest.raises(ConfigError, match="missing"):
            WindowSchedule.from_dict(payload)

    def test_a_missing_parameter_is_refused(self) -> None:
        """The same rule applies inside the parameters."""
        payload = _metallic().to_dict()
        parameters = dict(payload["parameters"])  # type: ignore[call-overload]
        del parameters["seed"]
        payload["parameters"] = parameters
        with pytest.raises(ConfigError, match="missing"):
            WindowSchedule.from_dict(payload)

    def test_an_unknown_kind_is_refused(self) -> None:
        """A payload naming a generator this package does not have is refused."""
        payload = _metallic().to_dict()
        payload["kind"] = "harmonic_series"
        with pytest.raises(ConfigError, match="kind"):
            WindowSchedule.from_dict(payload)

    def test_an_unknown_rounding_rule_is_refused(self) -> None:
        """A rule nothing implements is not accepted just because it is a string."""
        payload = _metallic().to_dict()
        parameters = dict(payload["parameters"])  # type: ignore[call-overload]
        parameters["rounding"] = "nearest_ties_sideways"
        payload["parameters"] = parameters
        with pytest.raises(ConfigError, match="rounding"):
            WindowSchedule.from_dict(payload)

    def test_a_non_string_window_is_refused(self) -> None:
        """Windows are stored as compact duration strings, and read back as such."""
        payload = _metallic().to_dict()
        payload["windows"] = [60, 180]
        with pytest.raises(ConfigError, match="duration"):
            WindowSchedule.from_dict(payload)

    def test_a_malformed_window_string_is_refused(self) -> None:
        """A stored window still has to parse in the duration grammar."""
        payload = _metallic().to_dict()
        payload["windows"] = ["1m", "not-a-duration"]
        with pytest.raises(ConfigError):
            WindowSchedule.from_dict(payload)

    def test_an_empty_window_list_is_refused(self) -> None:
        """The empty schedule is refused on the way in as well as on the way out."""
        payload = _explicit().to_dict()
        payload["windows"] = []
        with pytest.raises(ConfigError, match="no windows"):
            WindowSchedule.from_dict(payload)

    def test_a_windows_value_that_is_not_a_list_is_refused(self) -> None:
        """A single string is not a one-window list."""
        payload = _explicit().to_dict()
        payload["windows"] = "1m"
        with pytest.raises(ConfigError, match="list"):
            WindowSchedule.from_dict(payload)

    def test_parameters_that_are_not_a_mapping_are_refused(self) -> None:
        """The parameters are a nested object, not a scalar."""
        payload = _explicit().to_dict()
        payload["parameters"] = "control-three"
        with pytest.raises(ConfigError, match="parameters"):
            WindowSchedule.from_dict(payload)

    def test_an_edited_window_list_is_caught_by_the_recorded_id(self) -> None:
        """The id is checked on the way in, so a hand-edited payload is refused.

        This is the realistic corruption: someone changes a window in a
        stored recipe and does not think to change the id beside it.
        """
        payload = _metallic().to_dict()
        payload["windows"] = ["1m", "3m"]
        with pytest.raises(ConfigError, match="schedule_id"):
            WindowSchedule.from_dict(payload)

    def test_an_edited_parameter_is_caught_by_the_recorded_id(self) -> None:
        """A parameter edited without re-deriving the id is refused too."""
        payload = _metallic().to_dict()
        parameters = dict(payload["parameters"])  # type: ignore[call-overload]
        parameters["seed"] = "2m"
        payload["parameters"] = parameters
        with pytest.raises(ConfigError, match="schedule_id"):
            WindowSchedule.from_dict(payload)


class TestValueSemantics:
    """A schedule is a value: frozen, comparable, hashable."""

    def test_a_schedule_is_frozen(self) -> None:
        """A resolved schedule is a record of what happened, not a mutable box."""
        schedule = _metallic()
        with pytest.raises(AttributeError):
            schedule.windows = ()  # type: ignore[misc]

    def test_a_spec_is_frozen(self) -> None:
        """So are the parameters it records."""
        schedule = _log()
        assert isinstance(schedule.spec, LogSpacedSpec)
        with pytest.raises(AttributeError):
            schedule.spec.grain = Duration.parse("1h")  # type: ignore[misc]

    def test_a_schedule_is_hashable(self) -> None:
        """A schedule can be a dict key, so a recipe can index by it."""
        assert len({_metallic(), _metallic(), _log()}) == 2  # noqa: PLR2004

    def test_equal_schedules_hash_equal(self) -> None:
        """Python's hash agrees with equality, as it must."""
        assert hash(_metallic()) == hash(_metallic())


class TestResolvedListInvariants:
    """A window list is checked wherever it comes from, payload included."""

    def test_a_zero_window_in_a_payload_is_refused(self) -> None:
        """A stored 0s window is refused on the way in.

        The generators cannot produce one -- they refuse a term that
        quantizes to nothing -- but a payload is not a generator, and a
        zero-length window read back out of one would carry no data.

        The message is about the window rather than about the id it no
        longer matches: a payload is checked for what it says before it
        is checked against what it is called.
        """
        payload = _explicit().to_dict()
        payload["windows"] = ["1m", "0s"]
        with pytest.raises(ConfigError, match="strictly positive"):
            WindowSchedule.from_dict(payload)

    def test_a_non_duration_window_is_refused_at_construction(self) -> None:
        """A schedule built directly still has to hold Durations."""
        with pytest.raises(ConfigError, match="Duration"):
            WindowSchedule(spec=ExplicitSpec(name=None), windows=("1m",))  # type: ignore[arg-type]

    def test_a_non_string_name_in_a_payload_is_refused(self) -> None:
        """A name is text or nothing; a number is neither."""
        payload = _explicit().to_dict()
        payload["parameters"] = {"name": 7, "limiting_ratio": None}
        with pytest.raises(ConfigError, match="name"):
            WindowSchedule.from_dict(payload)


if __name__ == "__main__":
    pytest.main([__file__])
