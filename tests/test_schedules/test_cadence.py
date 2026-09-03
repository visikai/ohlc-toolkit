"""Emit-cadence rules: how a window scale is turned into an emit cadence.

Every expected cadence here is worked out by hand from the rule -- the
largest allowed cadence at or below W/K, then clamped up to the source
cadence -- rather than read back out of the code that computes it.
"""

import hashlib
import json

import pytest

from ohlc_toolkit.schedules import (
    CadenceKind,
    CadenceRule,
    ExplicitPairsSpec,
    WindowEmitPair,
    WOverKSpec,
    explicit_pairs,
    metallic_recurrence,
    w_over_k,
)
from ohlc_toolkit.temporal import ConfigError, Duration

# A plausible allowed set for a minute-cadence source: the cadences an
# operator is willing to emit on, written out of order and with a
# repeat, so the tests below also pin how the set is normalized.
_ALLOWED = ["15m", "1m", "5m", "1h", "1m"]
_SOURCE_CADENCE = "1m"


def _schedule_windows() -> tuple[Duration, ...]:
    """Resolve a window schedule whose smallest scale a rule can still divide.

    Seeded at 5m rather than at the source cadence, so that W/K stays at
    or above the finest allowed cadence for every window in it.
    """
    return metallic_recurrence(
        coefficient=2.0, seed="5m", grain="5m", maximum="1d"
    ).windows


def _texts(rule: CadenceRule) -> list[tuple[str, str]]:
    """Render resolved pairs as compact duration strings, for readable asserts."""
    return [(str(pair.window), str(pair.emit_every)) for pair in rule.pairs]


def _hash_of(payload: dict[str, object]) -> str:
    """Hash an identity payload the way a rule id is defined to be hashed."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class TestWOverK:
    """E is the largest allowed cadence at or below W/K, never below d."""

    def test_an_exact_division_lands_on_the_allowed_member(self) -> None:
        """W/K is 15m and 1h exactly, and both are in the allowed set."""
        rule = w_over_k(
            ["1h", "4h"],
            divisor=4,
            allowed=_ALLOWED,
            source_cadence=_SOURCE_CADENCE,
        )
        assert _texts(rule) == [("1h", "15m"), ("4h", "1h")]

    def test_a_ratio_between_two_allowed_members_rounds_down(self) -> None:
        """W/K of 7m30s takes 5m, the largest member at or below it.

        Rounding up would emit more often than the caller asked for,
        which is the direction that costs work and overlaps windows; the
        rule is stated as "at or below" for that reason, and this is the
        case that tells the two directions apart.
        """
        rule = w_over_k(
            ["30m"], divisor=4, allowed=_ALLOWED, source_cadence=_SOURCE_CADENCE
        )
        assert _texts(rule) == [("30m", "5m")]

    def test_a_member_exactly_equal_to_the_ratio_is_taken(self) -> None:
        """A member equal to the ratio qualifies: an exact hit is not passed over."""
        rule = w_over_k(
            ["1h"], divisor=4, allowed=["15m"], source_cadence=_SOURCE_CADENCE
        )
        assert _texts(rule) == [("1h", "15m")]

    def test_a_divisor_of_one_emits_at_the_window_scale(self) -> None:
        """K of 1 means W/K is W itself: non-overlapping windows."""
        rule = w_over_k(
            ["1h"], divisor=1, allowed=_ALLOWED, source_cadence=_SOURCE_CADENCE
        )
        assert _texts(rule) == [("1h", "1h")]

    def test_a_ratio_below_the_source_cadence_clamps_up(self) -> None:
        """Nothing can be emitted faster than the source produces candles.

        W/K here is 30s over a 1m source. The allowed set does contain a
        30s member, and it is still not usable: the clamp to the source
        cadence wins over the allowed set, because emitting below the
        source cadence is impossible while emitting at a cadence the
        caller did not list is merely not preferred. The resolved
        cadence is therefore coarser than W/K, which is the one case
        where that happens.
        """
        rule = w_over_k(["2m"], divisor=4, allowed=["30s", "1m"], source_cadence="1m")
        assert _texts(rule) == [("2m", "1m")]

    def test_the_windows_of_a_schedule_can_be_fed_straight_in(self) -> None:
        """A resolved schedule's windows are the intended input to a rule."""
        rule = w_over_k(
            _schedule_windows(),
            divisor=4,
            allowed=_ALLOWED,
            source_cadence=_SOURCE_CADENCE,
        )
        assert [pair.window for pair in rule.pairs] == list(_schedule_windows())

    def test_every_resolved_cadence_is_at_least_the_source_cadence(self) -> None:
        """The clamp holds across a whole schedule, not just at its small end.

        The allowed set here reaches below the source cadence, so the
        smallest window's W/K really does select a sub-cadence member --
        and every resolved cadence still comes back at or above the
        source cadence.
        """
        rule = w_over_k(
            _schedule_windows(),
            divisor=8,
            allowed=["10s", "30s", *_ALLOWED],
            source_cadence=_SOURCE_CADENCE,
        )
        source_cadence = Duration.parse(_SOURCE_CADENCE)
        assert all(pair.emit_every >= source_cadence for pair in rule.pairs)
        assert rule.pairs[0].emit_every == source_cadence

    def test_a_self_similar_allowed_set_resolves_every_window(self) -> None:
        """Feeding a schedule in as its own allowed set resolves in full.

        The natural way to use the rule is to hand a resolved schedule
        in twice: once as the windows, once as the allowed set. With a
        large divisor the smallest windows put ``W / K`` below the
        smallest member -- 1m/48 is 1.25s -- and the source-cadence
        floor must answer for them, exactly as it answers for an allowed
        member that is too fine. Every expected cadence below is worked
        out by hand from ``a * 48 <= W``: the largest member at or below
        the ratio where one exists (146m/48 takes 3m; 56m/48 takes 1m
        with no clamping involved), and the source cadence where none
        does (the first four windows).
        """
        minutes = [1, 3, 8, 21, 56, 146, 380, 993, 2590, 6758, 17632]
        windows = [f"{m}m" for m in minutes]
        expected_emit_minutes = [1, 1, 1, 1, 1, 3, 3, 8, 21, 56, 146]

        rule = w_over_k(windows, divisor=48, allowed=windows, source_cadence="1m")

        assert [pair.emit_every for pair in rule.pairs] == [
            Duration.parse(f"{m}m") for m in expected_emit_minutes
        ]

    def test_a_window_too_small_for_the_divisor_is_refused(self) -> None:
        """A schedule reaching below K times the finest allowed cadence refuses.

        This is the shape the refusal takes in practice: a metallic
        schedule seeded at the source cadence starts at 1m, and 1m/4 is
        below every member of a minute-and-up allowed set, so the whole
        rule is refused rather than quietly dropping that window.
        """
        schedule = metallic_recurrence(
            coefficient=2.0, seed="1m", grain="1m", maximum="1d"
        )
        with pytest.raises(ConfigError, match="allowed"):
            w_over_k(
                schedule.windows,
                divisor=4,
                allowed=_ALLOWED,
                source_cadence=_SOURCE_CADENCE,
            )

    def test_the_parameters_are_recorded(self) -> None:
        """The divisor, the allowed set, and the source cadence are the identity."""
        divisor = 4
        rule = w_over_k(
            ["1h"], divisor=divisor, allowed=_ALLOWED, source_cadence=_SOURCE_CADENCE
        )
        assert isinstance(rule.spec, WOverKSpec)
        assert rule.spec.kind is CadenceKind.W_OVER_K
        assert rule.spec.divisor == divisor
        assert rule.spec.source_cadence == Duration.parse("1m")

    def test_the_allowed_set_is_normalized(self) -> None:
        """A set is a set: stored ascending, with repeats dropped.

        Two callers who list the same cadences in different orders have
        asked for the same rule, and must get the same identity.
        """
        rule = w_over_k(
            ["1h"], divisor=4, allowed=_ALLOWED, source_cadence=_SOURCE_CADENCE
        )
        assert isinstance(rule.spec, WOverKSpec)
        assert [str(allowed) for allowed in rule.spec.allowed] == [
            "1m",
            "5m",
            "15m",
            "1h",
        ]

    def test_the_allowed_set_order_does_not_change_the_id(self) -> None:
        """Normalization is what makes that true, and it is worth pinning."""
        first = w_over_k(
            ["1h"], divisor=4, allowed=["1m", "5m"], source_cadence=_SOURCE_CADENCE
        )
        second = w_over_k(
            ["1h"],
            divisor=4,
            allowed=["5m", "1m", "5m"],
            source_cadence=_SOURCE_CADENCE,
        )
        assert first.schedule_id == second.schedule_id


class TestWOverKRefusals:
    """A cadence that cannot be resolved is refused, never approximated."""

    @pytest.mark.parametrize(
        "divisor", [0, -1, -4], ids=["zero", "minus_one", "minus_four"]
    )
    def test_a_non_positive_divisor_is_refused(self, divisor: int) -> None:
        """W/K is meaningless at or below zero."""
        with pytest.raises(ConfigError, match="divisor"):
            w_over_k(
                ["1h"],
                divisor=divisor,
                allowed=_ALLOWED,
                source_cadence=_SOURCE_CADENCE,
            )

    def test_a_boolean_divisor_is_refused(self) -> None:
        """``bool`` is an ``int`` subtype in Python, and is refused anyway."""
        with pytest.raises(ConfigError, match="divisor"):
            w_over_k(
                ["1h"],
                divisor=True,
                allowed=_ALLOWED,
                source_cadence=_SOURCE_CADENCE,
            )

    def test_a_non_integer_divisor_is_refused(self) -> None:
        """A divisor is a whole number of windows per emit."""
        with pytest.raises(ConfigError, match="divisor"):
            w_over_k(
                ["1h"],
                divisor=2.5,  # type: ignore[arg-type]
                allowed=_ALLOWED,
                source_cadence=_SOURCE_CADENCE,
            )

    def test_an_empty_allowed_set_is_refused(self) -> None:
        """With nothing allowed, no window can be resolved at all."""
        with pytest.raises(ConfigError, match="allowed"):
            w_over_k(["1h"], divisor=4, allowed=[], source_cadence=_SOURCE_CADENCE)

    def test_an_allowed_set_entirely_above_the_ratio_is_refused(self) -> None:
        """Nothing at or below W/K means there is no answer to give.

        Emitting at the smallest allowed cadence anyway would be a
        cadence coarser than the caller asked for, silently; refusing
        says so instead.
        """
        with pytest.raises(ConfigError, match="allowed"):
            w_over_k(
                ["1h"], divisor=4, allowed=["1h", "4h"], source_cadence=_SOURCE_CADENCE
            )

    def test_the_refusal_names_the_window_it_could_not_resolve(self) -> None:
        """A rule over many windows says which one failed."""
        with pytest.raises(ConfigError, match="1h"):
            w_over_k(
                ["4h", "1h"],
                divisor=4,
                allowed=["1h"],
                source_cadence=_SOURCE_CADENCE,
            )

    def test_an_empty_window_list_is_refused(self) -> None:
        """A rule that maps nothing is not a rule."""
        with pytest.raises(ConfigError, match="no windows"):
            w_over_k([], divisor=4, allowed=_ALLOWED, source_cadence=_SOURCE_CADENCE)

    def test_a_repeated_window_is_refused(self) -> None:
        """A window mapped twice would be two answers to the same question."""
        with pytest.raises(ConfigError, match="once"):
            w_over_k(
                ["1h", "1h"],
                divisor=4,
                allowed=_ALLOWED,
                source_cadence=_SOURCE_CADENCE,
            )

    @pytest.mark.parametrize(
        ("windows", "allowed", "source_cadence"),
        [
            (["0s"], _ALLOWED, _SOURCE_CADENCE),
            (["1h"], ["0s"], _SOURCE_CADENCE),
            (["1h"], _ALLOWED, "0s"),
        ],
        ids=["zero_window", "zero_allowed", "zero_source_cadence"],
    )
    def test_a_zero_duration_is_refused(
        self, windows: list[str], allowed: list[str], source_cadence: str
    ) -> None:
        """Windows, allowed cadences, and the source cadence are all positive."""
        with pytest.raises(ConfigError, match="strictly positive"):
            w_over_k(
                windows,
                divisor=4,
                allowed=allowed,
                source_cadence=source_cadence,
            )

    def test_a_bare_string_is_not_an_allowed_set(self) -> None:
        """The allowed set is a list of durations, not one duration."""
        with pytest.raises(ConfigError, match="allowed"):
            w_over_k(["1h"], divisor=4, allowed="15m", source_cadence=_SOURCE_CADENCE)

    def test_a_bare_string_is_not_a_list_of_windows(self) -> None:
        """A string is iterable, and iterating it would be nonsense."""
        with pytest.raises(ConfigError, match="list"):
            w_over_k("1h", divisor=4, allowed=_ALLOWED, source_cadence=_SOURCE_CADENCE)


class TestExplicitPairs:
    """A caller-supplied mapping from window to emit cadence."""

    def test_the_given_pairs_are_the_rule(self) -> None:
        """Nothing is derived: the pairs are what was written down."""
        rule = explicit_pairs([("1h", "15m"), ("4h", "30m")])
        assert _texts(rule) == [("1h", "15m"), ("4h", "30m")]

    def test_the_kind_and_name_are_recorded(self) -> None:
        """A registered mapping carries the name it is asked for by."""
        rule = explicit_pairs([("1h", "15m")], name="control-pairs")
        assert isinstance(rule.spec, ExplicitPairsSpec)
        assert rule.spec.kind is CadenceKind.EXPLICIT_PAIRS
        assert rule.spec.name == "control-pairs"

    def test_an_unnamed_mapping_records_no_name(self) -> None:
        """An ad-hoc mapping is not forced to invent a name."""
        rule = explicit_pairs([("1h", "15m")])
        assert isinstance(rule.spec, ExplicitPairsSpec)
        assert rule.spec.name is None

    def test_the_caller_order_is_preserved(self) -> None:
        """Nothing is sorted: the pairs stay in the order they were written."""
        rule = explicit_pairs([("4h", "1h"), ("1h", "15m")])
        assert _texts(rule) == [("4h", "1h"), ("1h", "15m")]

    def test_duration_instances_are_accepted(self) -> None:
        """The boundary accepts ``Duration | str``, like the rest of the package."""
        rule = explicit_pairs([(Duration.parse("1h"), Duration(900))])
        assert _texts(rule) == [("1h", "15m")]

    def test_an_emit_cadence_longer_than_its_window_is_allowed(self) -> None:
        """No relation between W and E is stated here, and none is invented.

        Whether an emit cadence makes sense against a source is decided
        where the schedule meets the source, not here.
        """
        rule = explicit_pairs([("1h", "1d")])
        assert _texts(rule) == [("1h", "1d")]

    def test_an_empty_mapping_is_refused(self) -> None:
        """A rule that maps nothing is not a rule."""
        with pytest.raises(ConfigError, match="no windows"):
            explicit_pairs([])

    def test_a_repeated_window_is_refused(self) -> None:
        """One window, one answer."""
        with pytest.raises(ConfigError, match="once"):
            explicit_pairs([("1h", "15m"), ("1h", "30m")])

    def test_a_zero_duration_is_refused(self) -> None:
        """Neither half of a pair may be zero."""
        with pytest.raises(ConfigError, match="strictly positive"):
            explicit_pairs([("1h", "0s")])

    def test_a_pair_that_is_not_a_pair_is_refused(self) -> None:
        """A mapping is written as two-element pairs, and checked as such."""
        with pytest.raises(ConfigError, match="pair"):
            explicit_pairs([("1h", "15m", "extra")])  # type: ignore[list-item]


class TestCadenceIdentity:
    """The same identity machinery the window schedules use."""

    def test_the_payload_records_the_parameters_and_the_resolved_pairs(self) -> None:
        """Kind, parameters, resolved pairs, and the id over them."""
        rule = w_over_k(
            ["1h", "4h"], divisor=4, allowed=["15m", "1h"], source_cadence="1m"
        )
        expected_parameters = {
            "divisor": 4,
            "allowed": ["15m", "1h"],
            "source_cadence": "1m",
        }
        expected_pairs = [
            {"window": "1h", "emit_every": "15m"},
            {"window": "4h", "emit_every": "1h"},
        ]

        payload = rule.to_dict()
        assert payload["kind"] == "w_over_k"
        assert payload["parameters"] == expected_parameters
        assert payload["pairs"] == expected_pairs
        assert payload["schedule_id"] == _hash_of(
            {
                "kind": "w_over_k",
                "parameters": expected_parameters,
                "pairs": expected_pairs,
            }
        )

    def test_the_explicit_payload_records_the_name(self) -> None:
        """A caller-supplied mapping has no parameters but its name."""
        rule = explicit_pairs([("1h", "15m")], name="control-pairs")
        assert rule.to_dict()["parameters"] == {"name": "control-pairs"}

    def test_both_kinds_round_trip_through_json(self) -> None:
        """from_dict(to_dict(x)) == x, through real JSON, for both kinds."""
        rules = [
            w_over_k(["1h", "4h"], divisor=4, allowed=_ALLOWED, source_cadence="1m"),
            explicit_pairs([("1h", "15m")], name="control-pairs"),
        ]
        for rule in rules:
            restored = CadenceRule.from_dict(json.loads(json.dumps(rule.to_dict())))
            assert restored == rule
            assert restored.schedule_id == rule.schedule_id

    @pytest.mark.parametrize(
        ("first", "second"),
        [
            (
                w_over_k(["1h"], divisor=4, allowed=["15m"], source_cadence="1m"),
                w_over_k(
                    ["1h"], divisor=2, allowed=["15m", "30m"], source_cadence="1m"
                ),
            ),
            (
                w_over_k(["1h"], divisor=4, allowed=["15m"], source_cadence="1m"),
                w_over_k(["1h"], divisor=4, allowed=["15m", "1h"], source_cadence="1m"),
            ),
            (
                w_over_k(["1h"], divisor=4, allowed=["15m"], source_cadence="1m"),
                w_over_k(["1h"], divisor=4, allowed=["15m"], source_cadence="5m"),
            ),
            (
                explicit_pairs([("1h", "15m")]),
                explicit_pairs([("1h", "30m")]),
            ),
            (
                explicit_pairs([("1h", "15m")]),
                explicit_pairs([("1h", "15m")], name="control"),
            ),
        ],
        ids=["divisor", "allowed", "source_cadence", "pairs", "name"],
    )
    def test_any_parameter_difference_changes_the_id(
        self, first: CadenceRule, second: CadenceRule
    ) -> None:
        """One field apart is a different rule, and a different id."""
        assert first.schedule_id != second.schedule_id
        assert first != second

    def test_the_two_kinds_differ_even_with_the_same_resolved_pairs(self) -> None:
        """How a mapping was arrived at is part of what it is."""
        derived = w_over_k(["1h"], divisor=4, allowed=["15m"], source_cadence="1m")
        copied = explicit_pairs([("1h", "15m")])
        assert _texts(copied) == _texts(derived)
        assert copied.schedule_id != derived.schedule_id

    def test_a_missing_key_is_refused(self) -> None:
        """A truncated payload is not silently defaulted."""
        payload = explicit_pairs([("1h", "15m")]).to_dict()
        del payload["pairs"]
        with pytest.raises(ConfigError, match="missing"):
            CadenceRule.from_dict(payload)

    def test_an_unknown_kind_is_refused(self) -> None:
        """A payload naming a rule this package does not have is refused."""
        payload = explicit_pairs([("1h", "15m")]).to_dict()
        payload["kind"] = "w_over_log_k"
        with pytest.raises(ConfigError, match="kind"):
            CadenceRule.from_dict(payload)

    def test_an_edited_pair_is_caught_by_the_recorded_id(self) -> None:
        """The id is checked on the way in, so a hand-edited payload is refused."""
        payload = explicit_pairs([("1h", "15m")]).to_dict()
        payload["pairs"] = [{"window": "1h", "emit_every": "30m"}]
        with pytest.raises(ConfigError, match="schedule_id"):
            CadenceRule.from_dict(payload)

    def test_a_stored_allowed_set_that_is_not_a_list_is_refused(self) -> None:
        """A payload's allowed set is read as the list it was written as."""
        rule = w_over_k(["1h"], divisor=4, allowed=["15m"], source_cadence="1m")
        payload = rule.to_dict()
        payload["parameters"] = {
            "divisor": 4,
            "allowed": "15m",
            "source_cadence": "1m",
        }
        with pytest.raises(ConfigError, match="allowed"):
            CadenceRule.from_dict(payload)

    def test_stored_pairs_that_are_not_a_list_are_refused(self) -> None:
        """So is the mapping itself."""
        payload = explicit_pairs([("1h", "15m")]).to_dict()
        payload["pairs"] = {"window": "1h", "emit_every": "15m"}
        with pytest.raises(ConfigError, match="pairs"):
            CadenceRule.from_dict(payload)

    def test_a_pair_that_is_not_an_object_is_refused(self) -> None:
        """Each stored pair is an object with both halves named."""
        payload = explicit_pairs([("1h", "15m")]).to_dict()
        payload["pairs"] = [["1h", "15m"]]
        with pytest.raises(ConfigError, match="pair"):
            CadenceRule.from_dict(payload)

    def test_a_pair_missing_a_half_is_refused(self) -> None:
        """Both halves are required: a window with no cadence maps nothing."""
        payload = explicit_pairs([("1h", "15m")]).to_dict()
        payload["pairs"] = [{"window": "1h"}]
        with pytest.raises(ConfigError, match="missing"):
            CadenceRule.from_dict(payload)

    def test_a_rule_is_frozen(self) -> None:
        """A resolved rule is a record of what happened, not a mutable box."""
        rule = explicit_pairs([("1h", "15m")])
        with pytest.raises(AttributeError):
            rule.pairs = ()  # type: ignore[misc]

    def test_a_pair_is_frozen(self) -> None:
        """So is each pair inside it."""
        rule = explicit_pairs([("1h", "15m")])
        with pytest.raises(AttributeError):
            rule.pairs[0].emit_every = Duration.parse("1h")  # type: ignore[misc]

    def test_a_rule_is_hashable(self) -> None:
        """A rule can be a dict key, so a recipe can index by it."""
        first = explicit_pairs([("1h", "15m")])
        second = explicit_pairs([("1h", "15m")])
        assert len({first, second}) == 1
        assert hash(first) == hash(second)

    def test_a_pair_names_both_halves(self) -> None:
        """The pair is a named record, not an order-ambiguous tuple."""
        rule = explicit_pairs([("1h", "15m")])
        assert rule.pairs[0] == WindowEmitPair(
            window=Duration.parse("1h"), emit_every=Duration.parse("15m")
        )


if __name__ == "__main__":
    pytest.main([__file__])
