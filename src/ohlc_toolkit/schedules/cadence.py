"""Emit-cadence rules: turning a window scale into how often to emit.

A window schedule says what scales to aggregate over. It does not say
how often to produce a row at each scale, and the two questions have
different answers: a one-day window emitted every day and the same
window emitted every hour are different datasets. A
:class:`CadenceRule` records that second decision the same way a
schedule records the first -- as parameters, a fully resolved mapping,
and a content hash over both.

The W/K rule
------------

``E = quantize_down(W / K, allowed)``, snapped up WITHIN the allowed
set when that lands below the source cadence ``d``.

Reading it in order:

- ``W / K`` is the cadence a caller asks for indirectly: K emits per
  window. It is compared exactly, in integers -- an allowed cadence
  ``a`` is at or below ``W / K`` exactly when ``a * K <= W`` -- so no
  division and no float rounding enters the decision.
- ``quantize_down`` takes the LARGEST allowed cadence at or below that
  ratio. Down, not nearest: rounding up would emit more often than the
  caller asked for, which is the direction that costs work and overlaps
  windows.
- The source cadence ``d`` is a floor, checked last: nothing can be
  emitted more often than the source produces candles. When the
  quantized cadence lands below that floor -- or nothing is at or below
  the ratio at all -- the answer SNAPS UP to the smallest allowed member
  at or above ``d``, never to ``d`` itself unless ``d`` is a member:
  every resolved cadence is a member of the set the rule's identity
  records, so a persisted rule can never claim an allowed set it did not
  emit on. A set with no member at or above ``d`` is refused rather than
  approximated. For the natural configuration of feeding a resolved
  schedule in as its own allowed set with the source cadence as its
  smallest member, the snap lands on that smallest member and the rule
  stays total. These snapped cases are the only ones where the resolved
  cadence is COARSER than ``W / K``, and they are deliberate.

What this layer does not decide
-------------------------------

Nothing here relates E to W, or checks either against a source frame's
grid. A rule may resolve a pair that
:func:`~ohlc_toolkit.windows.resolution.resolve_schedule` will later
refuse -- an emit cadence that is not a whole multiple of the source
cadence, say. That check belongs where the schedule meets the source,
and duplicating it here would give two places to keep in step.

The allowed set is a set
------------------------

It is stored ascending with repeats dropped, so two callers who listed
the same cadences in different orders have asked for the same rule and
get the same id.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, unique
from typing import ClassVar, Self

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.schedules.generators import require_resolved_windows
from ohlc_toolkit.schedules.identity import (
    content_hash,
    duration_from_payload,
    enum_from_payload,
    mapping_from_payload,
    optional_text_from_payload,
    require_keys,
    require_recorded_id,
)
from ohlc_toolkit.temporal import (
    ConfigError,
    Duration,
    validate_cadence,
    validate_window_duration,
)

logger = get_logger(__name__)

_PAIR_LENGTH = 2


@unique
class CadenceKind(Enum):
    """Which rule produced a cadence mapping.

    Attributes:
        W_OVER_K: The largest allowed cadence at or below ``W / K``,
            clamped to at least the source cadence.
        EXPLICIT_PAIRS: A caller-supplied mapping, derived from nothing.

    """

    W_OVER_K = "w_over_k"
    EXPLICIT_PAIRS = "explicit_pairs"


@dataclass(frozen=True)
class WindowEmitPair:
    """One window scale and the cadence rows are emitted at for it.

    A named record rather than a bare two-tuple, so no reader has to
    remember which half came first, and frozen so a resolved mapping
    cannot be edited out from under its own id.

    Attributes:
        window: The window duration ``W``.
        emit_every: The emit cadence ``E`` for that window.

    """

    window: Duration
    emit_every: Duration

    def __post_init__(self) -> None:
        """Coerce and check both halves.

        Raises:
            ConfigError: If either half is not a strictly positive
                duration. A zero window carries no data and a zero
                cadence never advances, so neither is a pair worth
                recording.

        """
        object.__setattr__(self, "window", validate_window_duration(self.window))
        object.__setattr__(self, "emit_every", validate_cadence(self.emit_every))


def _validated_divisor(value: object) -> int:
    """Return an emit divisor as an int, refusing anything unusable.

    Args:
        value: The candidate divisor, of any type.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ConfigError: If ``value`` is not an ``int`` (``bool`` is refused
            too, even though it is an ``int`` subtype) or is not
            strictly positive. ``W / K`` names no cadence at or below
            zero.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("Rejecting a non-integer emit divisor: {!r}", value)
        raise ConfigError(f"divisor must be an int, got {type(value).__name__}")
    if value <= 0:
        logger.warning("Rejecting a non-positive emit divisor: {}", value)
        raise ConfigError(f"divisor must be strictly positive, got {value}.")
    return value


def _normalized_allowed(allowed: Sequence[Duration | str]) -> tuple[Duration, ...]:
    """Coerce, check, sort, and deduplicate the allowed cadence set.

    Args:
        allowed: The cadences a caller is willing to emit on.

    Returns:
        The allowed cadences, ascending, with repeats dropped.

    Raises:
        ConfigError: If ``allowed`` is not a list or tuple, is empty, or
            holds anything but a strictly positive duration. An empty
            set can resolve no window at all, so it is refused up front
            rather than once per window.

    """
    if isinstance(allowed, str) or not isinstance(allowed, Sequence):
        logger.warning("Rejecting an allowed set that is not a list: {!r}", allowed)
        raise ConfigError(
            f"The allowed cadence set must be a list of durations, got "
            f"{type(allowed).__name__}"
        )
    if not allowed:
        logger.warning("Rejecting an empty allowed cadence set.")
        raise ConfigError(
            "The allowed cadence set must name at least one cadence; an empty "
            "set can resolve no window at all."
        )
    coerced = {validate_cadence(cadence) for cadence in allowed}
    return tuple(sorted(coerced, key=lambda cadence: cadence.total_seconds))


@dataclass(frozen=True)
class WOverKSpec:
    """The parameters of one W/K emit-cadence rule.

    Attributes:
        divisor: The ``K`` in ``W / K``: how many emits a caller wants
            per window. Strictly positive.
        allowed: The cadences a caller is willing to emit on, stored
            ascending with repeats dropped.
        source_cadence: The source's own candle cadence ``d``, the floor
            no resolved cadence may fall below.

    """

    divisor: int
    allowed: tuple[Duration, ...]
    source_cadence: Duration

    kind: ClassVar[CadenceKind] = CadenceKind.W_OVER_K

    def __post_init__(self) -> None:
        """Normalize and check every parameter.

        Raises:
            ConfigError: If the divisor is not a strictly positive int,
                the allowed set is empty or holds a non-positive
                duration, or the source cadence is not strictly
                positive.

        """
        object.__setattr__(self, "divisor", _validated_divisor(self.divisor))
        object.__setattr__(self, "allowed", _normalized_allowed(self.allowed))
        object.__setattr__(
            self, "source_cadence", validate_cadence(self.source_cadence)
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict.

        Returns:
            A dict holding the divisor, the normalized allowed set, and
            the source cadence, in that fixed key order.

        """
        return {
            "divisor": self.divisor,
            "allowed": [str(cadence) for cadence in self.allowed],
            "source_cadence": str(self.source_cadence),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their :meth:`to_dict` form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed parameters.

        Raises:
            ConfigError: If a key is missing, or any value fails the
                same validation it would have failed at construction.

        """
        require_keys(data, _W_OVER_K_KEYS, label="w_over_k parameters")
        allowed = data["allowed"]
        if not isinstance(allowed, list | tuple):
            logger.warning("Rejecting a stored allowed set that is not a list.")
            raise ConfigError(
                f"The allowed cadence set must be a list of compact duration "
                f"strings, got {type(allowed).__name__}"
            )
        return cls(
            divisor=_validated_divisor(data["divisor"]),
            allowed=tuple(
                duration_from_payload(cadence, label="allowed cadence")
                for cadence in allowed
            ),
            source_cadence=duration_from_payload(
                data["source_cadence"], label="source cadence"
            ),
        )


@dataclass(frozen=True)
class ExplicitPairsSpec:
    """The parameters of a caller-supplied cadence mapping: at most a name.

    Attributes:
        name: The name a registered mapping is asked for by, or None for
            an ad-hoc one.

    """

    name: str | None = None

    kind: ClassVar[CadenceKind] = CadenceKind.EXPLICIT_PAIRS

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict.

        Returns:
            A dict holding just the name.

        """
        return {"name": self.name}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their :meth:`to_dict` form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed parameters.

        Raises:
            ConfigError: If the key is missing, or the name is neither a
                string nor null.

        """
        require_keys(data, _EXPLICIT_PAIRS_KEYS, label="explicit_pairs parameters")
        return cls(name=optional_text_from_payload(data["name"], label="rule name"))


# What a cadence rule's parameters may be, discriminated by kind, for
# the same reason the generator specs are.
CadenceSpec = WOverKSpec | ExplicitPairsSpec

_SPEC_TYPES: dict[CadenceKind, type[WOverKSpec] | type[ExplicitPairsSpec]] = {
    CadenceKind.W_OVER_K: WOverKSpec,
    CadenceKind.EXPLICIT_PAIRS: ExplicitPairsSpec,
}

_W_OVER_K_KEYS = ("divisor", "allowed", "source_cadence")
_EXPLICIT_PAIRS_KEYS = ("name",)
_RULE_KEYS = ("kind", "parameters", "pairs", "schedule_id")
_PAIR_KEYS = ("window", "emit_every")


@dataclass(frozen=True)
class CadenceRule:
    """A resolved cadence mapping: its rule's parameters and its pairs.

    Frozen and hashable, like a :class:`~ohlc_toolkit.schedules.WindowSchedule`,
    and carrying the same shape of identity: what was asked for, what
    that resolved to, and a content hash over both.

    Attributes:
        spec: The rule parameters that produced ``pairs``.
        pairs: The fully resolved window-to-cadence mapping, in the
            order the windows were given. Never empty, never naming a
            window twice.

    """

    spec: CadenceSpec
    pairs: tuple[WindowEmitPair, ...]

    def __post_init__(self) -> None:
        """Check the resolved mapping against the invariants it shares with a schedule.

        Raises:
            ConfigError: If ``pairs`` is empty, names a window twice,
                names more windows than a schedule may, or -- for a W/K
                rule -- pairs any window with a cadence that is not a
                member of the rule's own allowed set. ``w_over_k``
                cannot produce such a pair, and ``from_dict`` is guarded
                by the recorded id, but the dataclass itself is public:
                a hand-assembled or ``replace``-built rule must not
                claim an allowed set it does not emit on.

        """
        require_resolved_windows(tuple(pair.window for pair in self.pairs))
        if isinstance(self.spec, WOverKSpec):
            members = set(self.spec.allowed)
            for pair in self.pairs:
                if pair.emit_every not in members:
                    logger.warning(
                        "Rejecting a W/K rule pairing {} with {}: not a member "
                        "of its allowed set.",
                        pair.window,
                        pair.emit_every,
                    )
                    raise ConfigError(
                        f"The pair ({pair.window} -> {pair.emit_every}) names a "
                        "cadence that is not a member of the rule's allowed "
                        f"set; the largest member is {self.spec.allowed[-1]}."
                    )

    @property
    def schedule_id(self) -> str:
        """The content hash naming this rule.

        A sha256 over the canonical JSON of the rule kind, its
        parameters, and the resolved pairs -- named ``schedule_id``, and
        derived exactly as a window schedule's is, because a recipe
        records the two the same way.
        """
        return content_hash(self._identity_payload())

    def _identity_payload(self) -> dict[str, object]:
        """Build the payload the rule id is the hash of."""
        return {
            "kind": self.spec.kind.value,
            "parameters": self.spec.to_dict(),
            "pairs": [
                {"window": str(pair.window), "emit_every": str(pair.emit_every)}
                for pair in self.pairs
            ],
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize this rule to a deterministic, JSON-compatible dict.

        Returns:
            A dict with exactly the keys ``"kind"``, ``"parameters"``,
            ``"pairs"``, and ``"schedule_id"``, in that fixed key order.

        """
        payload = self._identity_payload()
        return {**payload, "schedule_id": content_hash(payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct a cadence rule from its :meth:`to_dict` form.

        The resolved pairs are read from the payload rather than
        recomputed from the parameters, for the reason
        :meth:`~ohlc_toolkit.schedules.WindowSchedule.from_dict` gives:
        the embedded mapping is the record of what was used, and the
        recorded id is what proves it has not been edited since.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed rule.

        Raises:
            ConfigError: If a key is missing, the kind names no rule,
                any parameter or pair is malformed, the mapping breaks
                an invariant, or the recorded id does not match.

        """
        require_keys(data, _RULE_KEYS, label="cadence rule")
        kind = enum_from_payload(CadenceKind, data["kind"], label="cadence rule kind")
        parameters = mapping_from_payload(data["parameters"], label="rule parameters")
        rule = cls(
            spec=_SPEC_TYPES[kind].from_dict(parameters),
            pairs=_pairs_from_payload(data["pairs"]),
        )
        require_recorded_id(data["schedule_id"], rule.schedule_id, label="schedule_id")
        return rule


def _pairs_from_payload(value: object) -> tuple[WindowEmitPair, ...]:
    """Read the resolved window-to-cadence mapping out of a payload.

    Args:
        value: The candidate value, of any type.

    Returns:
        The parsed pairs, in stored order.

    Raises:
        ConfigError: If the value is not a list, if an entry is not an
            object, if an entry is missing either half, or if either
            half is not a valid duration string.

    """
    if not isinstance(value, list | tuple):
        logger.warning("Rejecting stored cadence pairs that are not a list.")
        raise ConfigError(
            f"The cadence pairs must be a list of objects, got {type(value).__name__}"
        )

    pairs = []
    for entry in value:
        pair = mapping_from_payload(entry, label="cadence pair")
        require_keys(pair, _PAIR_KEYS, label="cadence pair")
        pairs.append(
            WindowEmitPair(
                window=duration_from_payload(pair["window"], label="window"),
                emit_every=duration_from_payload(
                    pair["emit_every"], label="emit cadence"
                ),
            )
        )
    return tuple(pairs)


def _resolve_emit(window: Duration, spec: WOverKSpec) -> Duration:
    """Resolve one window's emit cadence under the W/K rule.

    Args:
        window: The window duration ``W``.
        spec: The validated rule parameters.

    Returns:
        The largest allowed cadence at or below ``W / K`` when that
        cadence is at or above the source cadence. Otherwise -- the
        quantized cadence undercuts the source cadence, or nothing is at
        or below ``W / K`` at all -- the SMALLEST allowed member at or
        above the source cadence, so the answer is always a member of
        the recorded set. The source cadence itself is never the answer
        unless it is a member.

    Raises:
        ConfigError: If no allowed member is at or above the source
            cadence, so there is nothing the snap could land on. A
            cadence outside the recorded set is never invented.

    """
    # `a <= W / K` without dividing: exact, in whole seconds, whatever
    # the remainder would have been.
    window_seconds = window.total_seconds
    source_seconds = spec.source_cadence.total_seconds
    candidates = [
        cadence
        for cadence in spec.allowed
        if cadence.total_seconds * spec.divisor <= window_seconds
    ]
    if candidates:
        # `allowed` is sorted ascending, so the last survivor is the largest.
        chosen = candidates[-1]
        if chosen.total_seconds >= source_seconds:
            return chosen

    # Nothing usable at or below the ratio: snap UP to the smallest
    # member the source can keep up with, so the resolved cadence stays
    # inside the set the rule's identity records.
    for member in spec.allowed:
        if member.total_seconds >= source_seconds:
            logger.debug(
                "Snapping the emit cadence for {} up to the {} member; the "
                "ratio {}/{} yields nothing at or above the {} source cadence.",
                window,
                member,
                window,
                spec.divisor,
                spec.source_cadence,
            )
            return member

    logger.warning(
        "Rejecting the W/{} rule for {}: no allowed cadence is at or above "
        "the {} source cadence.",
        spec.divisor,
        window,
        spec.source_cadence,
    )
    raise ConfigError(
        f"No allowed cadence is at or above the {spec.source_cadence} source "
        f"cadence, so the window {window} has no emit cadence to snap up to; "
        f"the largest allowed cadence is {spec.allowed[-1]}."
    )


def _require_sequence(value: object, *, label: str) -> None:
    """Refuse an argument that is not a list or tuple.

    Args:
        value: The candidate argument, of any type.
        label: What the argument should have been, for the message.

    Raises:
        ConfigError: If ``value`` is a bare string or is not a sequence.
            A string is iterable, and iterating it would read one
            character per entry.

    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        logger.warning("Rejecting an argument that is not a list: {!r}", value)
        raise ConfigError(f"A cadence rule takes a {label}, got {type(value).__name__}")


def w_over_k(
    windows: Sequence[Duration | str],
    *,
    divisor: int,
    allowed: Sequence[Duration | str],
    source_cadence: Duration | str,
) -> CadenceRule:
    """Resolve an emit cadence for each window as ``W / K``, quantized down.

    See the module docstring for the rule in full, including why the
    quantization goes down and why a result below the source cadence
    snaps UP to the smallest allowed member rather than to the source
    cadence itself.

    Args:
        windows: The window scales to resolve, typically a resolved
            schedule's ``windows``.
        divisor: The ``K`` in ``W / K``, strictly positive.
        allowed: The cadences a caller is willing to emit on.
        source_cadence: The source's own candle cadence, the floor no
            resolved cadence may fall below.

    Returns:
        The resolved rule, carrying both its pairs and the parameters
        that produced them.

    Raises:
        ConfigError: If any parameter is invalid, if the window list is
            empty, repeats a window, or is too long, or if no allowed
            member is at or above the source cadence when a window needs
            the snap.

    """
    _require_sequence(windows, label="list of window durations")
    spec = WOverKSpec(
        divisor=divisor,
        allowed=_normalized_allowed(allowed),
        source_cadence=validate_cadence(source_cadence),
    )

    pairs = []
    for window in windows:
        duration = validate_window_duration(window)
        pairs.append(
            WindowEmitPair(window=duration, emit_every=_resolve_emit(duration, spec))
        )

    logger.debug("Resolved {} window(s) at W/{}.", len(pairs), spec.divisor)
    return CadenceRule(spec=spec, pairs=tuple(pairs))


def explicit_pairs(
    pairs: Sequence[Sequence[Duration | str]], *, name: str | None = None
) -> CadenceRule:
    """Build a cadence rule from a caller-supplied window-to-cadence mapping.

    Nothing is derived: the pairs are taken as given, in the order
    given, and only checked against the invariants every rule shares.

    Args:
        pairs: The ``(window, emit_every)`` pairs, as Durations or
            compact duration strings.
        name: The name a registered mapping is asked for by. Defaults to
            None, for an ad-hoc mapping.

    Returns:
        The rule, carrying the given pairs and the name.

    Raises:
        ConfigError: If ``pairs`` is not a list, if an entry is not a
            two-element pair, if either half is not a strictly positive
            duration, or if the mapping is empty, names a window twice,
            or is too long.

    """
    _require_sequence(pairs, label="list of (window, emit_every) pairs")
    resolved = tuple(_coerced_pair(entry) for entry in pairs)
    logger.debug("Recorded an explicit cadence mapping of {} pair(s).", len(resolved))
    return CadenceRule(spec=ExplicitPairsSpec(name=name), pairs=resolved)


def _coerced_pair(entry: Sequence[Duration | str]) -> WindowEmitPair:
    """Coerce one caller-supplied ``(window, emit_every)`` entry.

    Raises:
        ConfigError: If the entry is not a two-element pair, or either
            half is not a strictly positive duration.

    """
    if (
        isinstance(entry, str)
        or not isinstance(entry, Sequence)
        or len(entry) != _PAIR_LENGTH
    ):
        logger.warning("Rejecting a cadence entry that is not a pair: {!r}", entry)
        raise ConfigError(
            "Each entry must be a (window, emit_every) pair of durations."
        )
    window, emit_every = entry
    return WindowEmitPair(
        window=validate_window_duration(window),
        emit_every=validate_cadence(emit_every),
    )
