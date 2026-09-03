"""Window-scale schedule generators and the identity each one carries.

A schedule is a list of window durations -- the scales a recipe wants a
frame aggregated over -- together with the parameters that produced it.
Nothing here touches candle values, reads a frame, or picks a schedule
for anybody: these are generators a caller asks for by name, with no
default coefficient, no default bounds, and no default schedule.

Real arithmetic, quantized once
-------------------------------

The metallic recurrence is computed in REAL numbers all the way down and
quantized ONCE, at the end. Rounding each term before feeding it back
into the recurrence is a different sequence, not a rounder rendering of
the same one: the rounding error is multiplied by the coefficient at
every step and compounds. With ``coefficient = sqrt(e + sqrt(5))``, a
1m seed and a 1m grain, the two readings part company at the fifth term
(55.888... minutes rounds to 56m, while the per-step reading has
already drifted to 55m) and end 2909 minutes apart at the two-week
bound.

"Real" here means exact rationals, not floats.
:class:`~fractions.Fraction` holds a float coefficient's exact value, so
the recurrence carries no rounding of its own and the tie rule below
decides an exact tie rather than whatever a double happened to land on.
This is the same reasoning
:mod:`ohlc_toolkit.windows.quality` applies to its coverage threshold.

A log-spaced ladder cannot be exact -- its points are irrational powers
-- so it is computed in :class:`~decimal.Decimal` at a fixed working
precision instead, and converted to exact rationals from there. Not for
the extra digits, but for reproducibility: the resolved list is part of
the identity that gets hashed, and a schedule must hash the same on
every machine that resolves it. ``Decimal``'s ``ln`` and ``exp`` are
correctly rounded to the working precision by specification, whereas a
libm ``pow`` is only nearly so, and may differ in the last place from
one platform's C library to another.

Bounds
------

The recurrence stops as soon as the next REAL term would exceed the
maximum, so the bound is read against the sequence itself rather than
against its rounded shadow. Quantization is then applied, and a value
that ROUNDS past a bound is dropped: a caller who asked for windows no
longer than ``2w`` must never be handed a longer one because the grain
rounded up. Both rules are needed, and they are not the same rule.

A lower bound, when given, is applied the same way: quantized values
below it are dropped.

Rounding and deduplication
--------------------------

Quantization is round-to-nearest-grain. Exact ties are unreachable for
an irrational coefficient, but they are reachable from the seed and from
a whole-number coefficient, so the tie rule is a stated,
:class:`RoundingRule` member recorded in the identity rather than an
accident of the implementation.

Deduplication drops later repeats and keeps the order of first
appearance. It is always applied: the recurrence is seeded with two
copies of the seed, so the first two terms are always equal, and a
coarse grain can collapse further neighbours. A schedule that named the
same window twice would aggregate it twice for no reason.

The empty schedule
------------------

Bounds that exclude every generated value are refused, in every kind,
rather than resolving to an empty schedule. A schedule with no windows
is not a statement about data -- nothing downstream can consume it, and
every later stage would silently do nothing -- so it fails closed here,
where the parameters that produced it are still in hand. This differs
deliberately from
:class:`~ohlc_toolkit.windows.resolution.ExplicitRange`, where an empty
range IS a meaningful statement about a stretch of time.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, localcontext
from enum import Enum, unique
from fractions import Fraction
from typing import ClassVar, Self

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.schedules.identity import (
    content_hash,
    duration_from_payload,
    durations_from_payload,
    enum_from_payload,
    mapping_from_payload,
    optional_duration_from_payload,
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

# No schedule may name more than this many windows, and no generator may
# run for more terms than this. The cap is a safety rule, not a taste
# one: a two-term recurrence with a vanishing coefficient does still
# terminate, but only after billions of terms, and a generator that
# hangs is worse than one that refuses. One number for every kind, so
# there is a single answer to "how long can a schedule be".
MAX_RESOLVED_WINDOWS = 512

# The working precision, in significant digits, for the log-spaced
# ladder. Fifty is far more than any duration needs -- a two-week window
# in seconds is seven digits -- and the surplus is the point: a point
# computed this far out lands unambiguously on one side of a
# quantization boundary, so the schedule does not depend on the last
# digits of the arithmetic that produced it.
_DECIMAL_DIGITS = 50

_HALF = Fraction(1, 2)


@unique
class GeneratorKind(Enum):
    """Which generator produced a schedule's resolved window list.

    Attributes:
        METALLIC_RECURRENCE: A two-term linear recurrence
            ``x[n+1] = coefficient * x[n] + x[n-1]``, seeded with two
            copies of a seed duration.
        LOG_SPACED: A fixed count of log-spaced durations between two
            bounds, both endpoints included.
        EXPLICIT: A caller-supplied resolved list, generated by nothing.

    """

    METALLIC_RECURRENCE = "metallic_recurrence"
    LOG_SPACED = "log_spaced"
    EXPLICIT = "explicit"


@unique
class RoundingRule(Enum):
    """How quantization breaks an exact tie between two grain multiples.

    Only ties are at stake: every other value rounds to the nearer
    multiple under both rules. Which rule applied is recorded in the
    identity, because two schedules that differ only in their tie
    handling are genuinely different schedules.

    Attributes:
        NEAREST_TIES_AWAY: At an exact half grain, round away from zero
            -- upward, durations being non-negative. The rule the
            schedules shipped with this package use.
        NEAREST_TIES_EVEN: At an exact half grain, round to the even
            multiple of the grain. This is IEEE-754's default and what
            Python's built-in ``round`` does, offered so a schedule
            reproducing a system that rounds that way can say so.

    """

    NEAREST_TIES_AWAY = "nearest_ties_away"
    NEAREST_TIES_EVEN = "nearest_ties_even"


@unique
class DedupRule(Enum):
    """How a generator treats a duration it has already produced.

    One rule exists today. Modelling it as an enum member rather than
    leaving it implicit means a payload states which rule produced its
    list, and that a future rule is a new member instead of a second
    meaning bolted onto this one.

    Attributes:
        DROP_LATER_REPEATS: Keep the first appearance of each value and
            drop every later one, leaving the surviving values in the
            order the generator produced them.

    """

    DROP_LATER_REPEATS = "drop_later_repeats"


def _validated_coefficient(value: object) -> float:
    """Return a recurrence coefficient as a float, refusing anything unusable.

    Args:
        value: The candidate coefficient, of any type.

    Returns:
        ``value`` as a ``float``.

    Raises:
        ConfigError: If ``value`` is not an ``int``/``float`` (``bool``
            is refused too, even though it is an ``int`` subtype), is
            NaN or an infinity, or is not strictly positive. A
            non-positive coefficient does not name a growing sequence:
            at zero the recurrence repeats its seeds forever and would
            never reach any bound.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        logger.warning("Rejecting non-numeric recurrence coefficient: {!r}", value)
        raise ConfigError(
            f"coefficient must be an int or float, got {type(value).__name__}"
        )
    if not math.isfinite(value):
        logger.warning("Rejecting non-finite recurrence coefficient: {}", value)
        raise ConfigError(f"coefficient must be finite, got {value}.")
    if value <= 0:
        logger.warning("Rejecting non-positive recurrence coefficient: {}", value)
        raise ConfigError(f"coefficient must be strictly positive, got {value}.")
    return float(value)


def _require_ordered_bounds(minimum: Duration, maximum: Duration) -> None:
    """Check that a schedule's bounds are the right way round.

    Args:
        minimum: The lower bound, already validated as a duration.
        maximum: The upper bound, already validated as a duration.

    Raises:
        ConfigError: If the minimum is above the maximum. Equal bounds
            are legal: they name a single scale.

    """
    if minimum.total_seconds > maximum.total_seconds:
        logger.warning(
            "Rejecting inverted schedule bounds: minimum {} above maximum {}.",
            minimum,
            maximum,
        )
        raise ConfigError(
            f"A schedule's minimum must not exceed its maximum, got "
            f"{minimum} > {maximum}."
        )


def _validated_count(value: object) -> int:
    """Return a point count as an int, refusing anything unusable.

    Args:
        value: The candidate count, of any type.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ConfigError: If ``value`` is not an ``int`` (``bool`` is refused
            too, even though it is an ``int`` subtype), is below two, or
            is above :data:`MAX_RESOLVED_WINDOWS`. Two is the smallest
            count that means anything: with a single point, naming both
            endpoints does not say which of them it is.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("Rejecting non-integer point count: {!r}", value)
        raise ConfigError(f"count must be an int, got {type(value).__name__}")
    if value < 2:  # noqa: PLR2004 - the two endpoints, named in the docstring
        logger.warning("Rejecting a point count below two: {}", value)
        raise ConfigError(
            f"count must be at least 2, one point per endpoint, got {value}."
        )
    if value > MAX_RESOLVED_WINDOWS:
        logger.warning("Rejecting a point count past the cap: {}", value)
        raise ConfigError(f"count must be at most {MAX_RESOLVED_WINDOWS}, got {value}.")
    return value


def _require_rules(rounding: RoundingRule, dedup: DedupRule) -> None:
    """Check that the recorded rounding and dedup rules are enum members.

    Both are written into the identity by name and read back the same
    way, so a plain string that merely looks like a member -- the
    realistic mistake -- would serialize into a payload nothing can
    read.

    Args:
        rounding: The candidate rounding rule.
        dedup: The candidate dedup rule.

    Raises:
        ConfigError: If either is not a member of its enum.

    """
    if not isinstance(rounding, RoundingRule):
        logger.warning("Rejecting non-RoundingRule rounding: {!r}", rounding)
        raise ConfigError(
            f"rounding must be a RoundingRule, got {type(rounding).__name__}"
        )
    if not isinstance(dedup, DedupRule):
        logger.warning("Rejecting non-DedupRule dedup: {!r}", dedup)
        raise ConfigError(f"dedup must be a DedupRule, got {type(dedup).__name__}")


@dataclass(frozen=True)
class MetallicRecurrenceSpec:
    """The parameters of one metallic-recurrence schedule.

    Frozen and hashable, so a resolved schedule can be compared, stored
    in a set, and serialized as the record of what produced it.

    Construction: the module-level :func:`metallic_recurrence` is the
    documented boundary, accepting ``Duration | str`` and resolving the
    schedule in one call. Direct construction expects ``Duration``
    instances, and revalidates them anyway -- the checks are what make
    the frozen value trustworthy, so they run however it was built.

    Attributes:
        coefficient: The ``c`` in ``x[n+1] = c * x[n] + x[n-1]``.
            Strictly positive and finite. A whole ``n`` gives the n-th
            metallic mean's recurrence -- 1 is the golden ratio's, 2 the
            silver -- which is where the name comes from, but any
            positive real is accepted.
        seed: The duration the recurrence starts from. Both of its first
            two terms are this value.
        grain: The quantization grain every resolved duration is a whole
            multiple of.
        maximum: The upper bound. The recurrence stops as soon as the
            next real term would pass it, and any value that rounds past
            it is dropped.
        minimum: An optional lower bound. Quantized values below it are
            dropped. ``None`` means no lower bound beyond the grain
            itself.
        rounding: The tie rule quantization applies. Defaults to
            :attr:`RoundingRule.NEAREST_TIES_AWAY`.
        dedup: The rule applied to repeated quantized values. Defaults
            to :attr:`DedupRule.DROP_LATER_REPEATS`, which is the only
            rule today.

    """

    coefficient: float
    seed: Duration
    grain: Duration
    maximum: Duration
    minimum: Duration | None = None
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY
    dedup: DedupRule = DedupRule.DROP_LATER_REPEATS

    kind: ClassVar[GeneratorKind] = GeneratorKind.METALLIC_RECURRENCE

    def __post_init__(self) -> None:
        """Normalize the durations and check every parameter.

        Raises:
            ConfigError: If the coefficient is not a strictly positive
                finite number, if the seed, grain, or either bound is
                not a strictly positive duration, if the minimum exceeds
                the maximum, or if ``rounding``/``dedup`` are not enum
                members.

        """
        object.__setattr__(
            self, "coefficient", _validated_coefficient(self.coefficient)
        )
        object.__setattr__(self, "seed", validate_window_duration(self.seed))
        object.__setattr__(self, "grain", validate_cadence(self.grain))
        object.__setattr__(self, "maximum", validate_window_duration(self.maximum))
        if self.minimum is not None:
            object.__setattr__(self, "minimum", validate_window_duration(self.minimum))
            _require_ordered_bounds(self.minimum, self.maximum)
        _require_rules(self.rounding, self.dedup)

    @property
    def limiting_ratio(self) -> float:
        """The ratio successive terms of this recurrence converge to.

        For ``x[n+1] = c * x[n] + x[n-1]`` that limit is the positive
        root of ``r^2 - c*r - 1 = 0``, namely ``(c + sqrt(c^2 + 4)) / 2``
        -- the metallic mean of ``c``. It is what the schedule's spacing
        actually tends to, which is rarely the coefficient itself, so it
        is recorded alongside the coefficient rather than left to be
        re-derived.
        """
        return (self.coefficient + math.sqrt(self.coefficient**2 + 4)) / 2

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict.

        The coefficient is stored as a plain float. ``repr`` of a float
        is the shortest string that reads back as the same double, and
        that is what ``json`` writes, so a stored coefficient names the
        same recurrence after a round trip as it did before one --
        which is what makes the id stable across one.

        Returns:
            A dict holding the coefficient, the implied ratio, the seed,
            the grain, both bounds, and both rules, in that fixed key
            order.

        """
        return {
            "coefficient": self.coefficient,
            "limiting_ratio": self.limiting_ratio,
            "seed": str(self.seed),
            "grain": str(self.grain),
            "minimum": None if self.minimum is None else str(self.minimum),
            "maximum": str(self.maximum),
            "rounding": self.rounding.value,
            "dedup": self.dedup.value,
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
        require_keys(data, _METALLIC_KEYS, label="metallic_recurrence parameters")
        return cls(
            coefficient=_validated_coefficient(data["coefficient"]),
            seed=duration_from_payload(data["seed"], label="seed"),
            grain=duration_from_payload(data["grain"], label="grain"),
            maximum=duration_from_payload(data["maximum"], label="maximum"),
            minimum=optional_duration_from_payload(data["minimum"], label="minimum"),
            rounding=enum_from_payload(
                RoundingRule, data["rounding"], label="rounding rule"
            ),
            dedup=enum_from_payload(DedupRule, data["dedup"], label="dedup rule"),
        )


@dataclass(frozen=True)
class LogSpacedSpec:
    """The parameters of one log-spaced schedule.

    Frozen and hashable, like every other spec here. The module-level
    :func:`log_spaced` is the documented boundary; direct construction
    expects ``Duration`` instances and revalidates them anyway.

    Attributes:
        count: How many points to place, both endpoints included. At
            least two, and never more than
            :data:`MAX_RESOLVED_WINDOWS`.
        minimum: The first point, and the lower bound.
        maximum: The last point, and the upper bound. Equal to
            ``minimum`` is legal and names a single scale.
        grain: The quantization grain every resolved duration is a whole
            multiple of.
        rounding: The tie rule quantization applies. Defaults to
            :attr:`RoundingRule.NEAREST_TIES_AWAY`.
        dedup: The rule applied to repeated quantized values. Defaults
            to :attr:`DedupRule.DROP_LATER_REPEATS`. A count too high
            for the range relies on it: several neighbouring points can
            land on the same grain multiple.

    """

    count: int
    minimum: Duration
    maximum: Duration
    grain: Duration
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY
    dedup: DedupRule = DedupRule.DROP_LATER_REPEATS

    kind: ClassVar[GeneratorKind] = GeneratorKind.LOG_SPACED

    def __post_init__(self) -> None:
        """Normalize the durations and check every parameter.

        Raises:
            ConfigError: If the count is not an int in
                ``[2, MAX_RESOLVED_WINDOWS]``, if a bound or the grain is
                not a strictly positive duration, if the minimum exceeds
                the maximum, or if ``rounding``/``dedup`` are not enum
                members.

        """
        object.__setattr__(self, "count", _validated_count(self.count))
        object.__setattr__(self, "minimum", validate_window_duration(self.minimum))
        object.__setattr__(self, "maximum", validate_window_duration(self.maximum))
        object.__setattr__(self, "grain", validate_cadence(self.grain))
        _require_ordered_bounds(self.minimum, self.maximum)
        _require_rules(self.rounding, self.dedup)

    @property
    def limiting_ratio(self) -> float:
        """The constant ratio between neighbouring points of this ladder.

        A ladder of ``count`` points has ``count - 1`` steps, so the
        ratio is ``(maximum / minimum) ** (1 / (count - 1))`` -- the
        number that, raised to the step count, spans the bounds. Equal
        bounds give exactly 1.0.
        """
        with localcontext() as context:
            context.prec = _DECIMAL_DIGITS
            span = _log_span(self.minimum, self.maximum)
            return float((span / (self.count - 1)).exp())

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict.

        Returns:
            A dict holding the count, the implied ratio, both bounds,
            the grain, and both rules, in that fixed key order.

        """
        return {
            "count": self.count,
            "limiting_ratio": self.limiting_ratio,
            "minimum": str(self.minimum),
            "maximum": str(self.maximum),
            "grain": str(self.grain),
            "rounding": self.rounding.value,
            "dedup": self.dedup.value,
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
        require_keys(data, _LOG_SPACED_KEYS, label="log_spaced parameters")
        return cls(
            count=_validated_count(data["count"]),
            minimum=duration_from_payload(data["minimum"], label="minimum"),
            maximum=duration_from_payload(data["maximum"], label="maximum"),
            grain=duration_from_payload(data["grain"], label="grain"),
            rounding=enum_from_payload(
                RoundingRule, data["rounding"], label="rounding rule"
            ),
            dedup=enum_from_payload(DedupRule, data["dedup"], label="dedup rule"),
        )


@dataclass(frozen=True)
class ExplicitSpec:
    """The parameters of a caller-supplied window list: at most a name.

    An explicit schedule is generated by nothing -- the caller states
    the resolved list, which :class:`WindowSchedule` holds -- so the
    only parameter it has is what to call it. It exists so that a
    control list, or a list an older implementation produced, can be
    recorded and compared the same way a generated one is.

    Attributes:
        name: The name a registered list is asked for by, or None for
            an ad-hoc one.

    """

    name: str | None = None

    kind: ClassVar[GeneratorKind] = GeneratorKind.EXPLICIT

    @property
    def limiting_ratio(self) -> None:
        """Always None: a hand-written list implies no constant ratio.

        Recorded as None rather than omitted, so every kind's payload
        answers the same question, and answering "none" is different
        from not being asked.
        """
        return None

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict.

        Returns:
            A dict holding the name and the (always null) implied ratio,
            in that fixed key order.

        """
        return {"name": self.name, "limiting_ratio": self.limiting_ratio}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their :meth:`to_dict` form.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed parameters.

        Raises:
            ConfigError: If a key is missing, or the name is neither a
                string nor null.

        """
        require_keys(data, _EXPLICIT_KEYS, label="explicit schedule parameters")
        return cls(name=optional_text_from_payload(data["name"], label="schedule name"))


# What a schedule's generator parameters may be. A discriminated union
# rather than one class with a nullable field per generator: each kind
# then validates and serializes only the parameters it actually has, and
# a spec that names a coefficient it does not use is unrepresentable.
GeneratorSpec = MetallicRecurrenceSpec | LogSpacedSpec | ExplicitSpec

# Which parameter class reads which kind's payload. The one place the
# discriminator is turned back into a type.
_SPEC_TYPES: dict[
    GeneratorKind,
    type[MetallicRecurrenceSpec] | type[LogSpacedSpec] | type[ExplicitSpec],
] = {
    GeneratorKind.METALLIC_RECURRENCE: MetallicRecurrenceSpec,
    GeneratorKind.LOG_SPACED: LogSpacedSpec,
    GeneratorKind.EXPLICIT: ExplicitSpec,
}

# Every key each kind's payload carries, and therefore every key its
# reader requires. `limiting_ratio` is required and then ignored: it is
# derived from the other parameters, so it is written for a reader that
# wants the number without re-deriving it, and recomputed rather than
# trusted on the way back in.
_METALLIC_KEYS = (
    "coefficient",
    "limiting_ratio",
    "seed",
    "grain",
    "minimum",
    "maximum",
    "rounding",
    "dedup",
)
_LOG_SPACED_KEYS = (
    "count",
    "limiting_ratio",
    "minimum",
    "maximum",
    "grain",
    "rounding",
    "dedup",
)
_EXPLICIT_KEYS = ("name", "limiting_ratio")
_SCHEDULE_KEYS = ("kind", "parameters", "windows", "schedule_id")


def _log_span(minimum: Duration, maximum: Duration) -> Decimal:
    """Return ``ln(maximum / minimum)``, at the caller's decimal precision."""
    return (Decimal(maximum.total_seconds) / Decimal(minimum.total_seconds)).ln()


@dataclass(frozen=True)
class WindowSchedule:
    """A resolved window schedule: its generator's parameters and its windows.

    Frozen and hashable, and every field is either a value type or a
    tuple of one, so a schedule can be compared, used as a dict key, and
    recorded as-is.

    Attributes:
        spec: The generator parameters that produced ``windows``.
        windows: The fully resolved window durations, in generated
            order. Never empty, never repeating, and never longer than
            :data:`MAX_RESOLVED_WINDOWS`.

    """

    spec: GeneratorSpec
    windows: tuple[Duration, ...]

    def __post_init__(self) -> None:
        """Check the resolved list against the invariants every kind shares.

        Raises:
            ConfigError: If ``windows`` is empty, holds anything but a
                strictly positive Duration, holds a repeat, or names
                more windows than :data:`MAX_RESOLVED_WINDOWS`.

        """
        _require_resolved_windows(self.windows)

    @property
    def schedule_id(self) -> str:
        """The content hash naming this schedule.

        A sha256 over the canonical JSON of the generator kind, its
        parameters, and the resolved window list -- everything that
        makes this schedule the schedule it is, and nothing else. Two
        independently resolved but identical schedules therefore share
        an id, and any difference in any recorded parameter changes it,
        including one that happens not to change the resolved list.
        """
        return content_hash(self._identity_payload())

    def _identity_payload(self) -> dict[str, object]:
        """Build the payload the schedule id is the hash of."""
        return {
            "kind": self.spec.kind.value,
            "parameters": self.spec.to_dict(),
            "windows": [str(window) for window in self.windows],
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize this schedule to a deterministic, JSON-compatible dict.

        Returns:
            A dict with exactly the keys ``"kind"``, ``"parameters"``,
            ``"windows"``, and ``"schedule_id"``, in that fixed key
            order. The id is the hash of the other three, so it is
            written last and never hashed into itself.

        """
        payload = self._identity_payload()
        return {**payload, "schedule_id": content_hash(payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct a schedule from its :meth:`to_dict` form.

        The resolved windows are read from the payload rather than
        regenerated from the parameters. The embedded list is the record
        of what the schedule actually used; re-running the generator
        here would make deserialization depend on the reading machine's
        arithmetic, and a disagreement between the two would have no
        obvious winner. The recorded id is checked instead, which
        catches the payload having been edited.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed schedule.

        Raises:
            ConfigError: If a key is missing, the kind names no
                generator, any parameter or window is malformed, the
                window list breaks a schedule invariant, or the recorded
                id does not match the payload it names.

        """
        require_keys(data, _SCHEDULE_KEYS, label="schedule")
        kind = enum_from_payload(GeneratorKind, data["kind"], label="generator kind")
        parameters = mapping_from_payload(
            data["parameters"], label="schedule parameters"
        )
        schedule = cls(
            spec=_SPEC_TYPES[kind].from_dict(parameters),
            windows=durations_from_payload(data["windows"], label="schedule window"),
        )
        require_recorded_id(
            data["schedule_id"], schedule.schedule_id, label="schedule_id"
        )
        return schedule


def _require_resolved_windows(windows: tuple[Duration, ...]) -> None:
    """Check a resolved window list, whichever generator produced it.

    Raises:
        ConfigError: If the list is empty, holds anything but a strictly
            positive Duration, repeats a value, or is longer than the
            cap.

    """
    if not windows:
        logger.warning("Rejecting a schedule that resolved no windows.")
        raise ConfigError(
            "A schedule must name at least one window; this one resolved no "
            "windows at all."
        )
    if len(windows) > MAX_RESOLVED_WINDOWS:
        logger.warning("Rejecting a schedule of {} windows.", len(windows))
        raise ConfigError(
            f"A schedule must name at most {MAX_RESOLVED_WINDOWS} windows, got "
            f"{len(windows)}."
        )
    seen: set[Duration] = set()
    for window in windows:
        if not isinstance(window, Duration):
            logger.warning("Rejecting non-Duration schedule window: {!r}", window)
            raise ConfigError(
                f"Schedule windows must be Durations, got {type(window).__name__}"
            )
        if window.total_seconds == 0:
            logger.warning("Rejecting a zero-length schedule window.")
            raise ConfigError("Schedule windows must be strictly positive, got 0s.")
        if window in seen:
            logger.warning("Rejecting a schedule repeating the window {}.", window)
            raise ConfigError(
                f"A schedule must name each window once, got {window} twice."
            )
        seen.add(window)


def _quantize(value: Fraction, grain_seconds: int, rounding: RoundingRule) -> int:
    """Round an exact number of seconds to the nearest whole grain.

    Args:
        value: The exact, non-negative duration in seconds.
        grain_seconds: The grain, in seconds.
        rounding: The rule to apply at an exact tie.

    Returns:
        The nearest whole multiple of the grain, in seconds.

    """
    scaled = value / grain_seconds
    whole = math.floor(scaled)
    remainder = scaled - whole
    if remainder > _HALF:
        return (whole + 1) * grain_seconds
    if remainder == _HALF and (
        rounding is RoundingRule.NEAREST_TIES_AWAY or whole % 2 == 1
    ):
        return (whole + 1) * grain_seconds
    return whole * grain_seconds


def _resolve_windows(
    values: list[Fraction],
    *,
    grain: Duration,
    rounding: RoundingRule,
    minimum: Duration | None,
    maximum: Duration,
) -> tuple[Duration, ...]:
    """Quantize, bound, and deduplicate a generator's real-valued output.

    The single place every kind's resolved list is produced, so the
    quantize/bound/dedup rules cannot drift apart between generators.

    Args:
        values: The generated durations in seconds, exact and unrounded.
        grain: The quantization grain.
        rounding: The tie rule.
        minimum: The optional lower bound, applied after quantization.
        maximum: The upper bound, applied after quantization.

    Returns:
        The resolved durations, in generated order.

    Raises:
        ConfigError: If a value quantizes to nothing, or if the bounds
            leave no value at all.

    """
    grain_seconds = grain.total_seconds
    minimum_seconds = 0 if minimum is None else minimum.total_seconds
    maximum_seconds = maximum.total_seconds

    kept: list[int] = []
    for value in values:
        seconds = _quantize(value, grain_seconds, rounding)
        if seconds == 0:
            logger.warning(
                "Rejecting a term of {}s that quantizes to nothing at a {} grain.",
                float(value),
                grain,
            )
            raise ConfigError(
                f"A generated duration of {float(value)}s quantizes to 0s at a "
                f"{grain} grain; every scheduled window must be strictly "
                "positive, so choose a finer grain or longer durations."
            )
        if minimum_seconds <= seconds <= maximum_seconds and seconds not in kept:
            kept.append(seconds)

    if not kept:
        logger.warning(
            "Rejecting a schedule whose bounds excluded all {} generated value(s).",
            len(values),
        )
        raise ConfigError(
            f"The bounds left no windows: all {len(values)} generated duration(s) "
            f"fall outside [{minimum or Duration(0)}, {maximum}]."
        )
    return tuple(Duration(seconds) for seconds in kept)


def _recurrence_terms(spec: MetallicRecurrenceSpec) -> list[Fraction]:
    """Run the two-term recurrence in exact rationals, bounded by the maximum.

    Args:
        spec: The validated recurrence parameters.

    Returns:
        The real terms in seconds, starting with two copies of the seed
        and stopping before the first term that would exceed the
        maximum.

    Raises:
        ConfigError: If the recurrence produces more than
            :data:`MAX_RESOLVED_WINDOWS` terms before reaching the
            maximum, which a coefficient near zero will do.

    """
    coefficient = Fraction(spec.coefficient)
    maximum_seconds = spec.maximum.total_seconds
    seed = Fraction(spec.seed.total_seconds)

    terms = [seed, seed]
    while True:
        following = coefficient * terms[-1] + terms[-2]
        if following > maximum_seconds:
            return terms
        if len(terms) >= MAX_RESOLVED_WINDOWS:
            logger.warning(
                "Rejecting a recurrence still below {} after {} terms.",
                spec.maximum,
                len(terms),
            )
            raise ConfigError(
                f"The recurrence produced more than {MAX_RESOLVED_WINDOWS} terms "
                f"without reaching {spec.maximum}; a coefficient of "
                f"{spec.coefficient} grows too slowly to bound the schedule."
            )
        terms.append(following)


def metallic_recurrence(  # noqa: PLR0913 - one keyword per recorded parameter
    *,
    coefficient: float,
    seed: Duration | str,
    grain: Duration | str,
    maximum: Duration | str,
    minimum: Duration | str | None = None,
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY,
) -> WindowSchedule:
    """Resolve a schedule from a two-term linear recurrence.

    The sequence ``x[n+1] = coefficient * x[n] + x[n-1]`` is seeded with
    two copies of ``seed`` and run in exact rationals until the next
    term would exceed ``maximum``. Only then is it quantized to
    ``grain``, bounded, and deduplicated. See the module docstring for
    why the rounding happens once, at the end.

    Args:
        coefficient: The recurrence coefficient, strictly positive.
        seed: The duration the recurrence starts from.
        grain: The quantization grain.
        maximum: The upper bound on the resolved durations.
        minimum: An optional lower bound. Defaults to None, meaning no
            lower bound.
        rounding: The tie rule for quantization. Defaults to
            :attr:`RoundingRule.NEAREST_TIES_AWAY`.

    Returns:
        The resolved schedule, carrying both its windows and the
        parameters that produced them.

    Raises:
        ConfigError: If any parameter is invalid, if the recurrence runs
            past the term cap, if a term quantizes to nothing, or if the
            bounds leave no windows.

    """
    spec = MetallicRecurrenceSpec(
        coefficient=coefficient,
        seed=validate_window_duration(seed),
        grain=validate_cadence(grain),
        maximum=validate_window_duration(maximum),
        minimum=None if minimum is None else validate_window_duration(minimum),
        rounding=rounding,
    )
    windows = _resolve_windows(
        _recurrence_terms(spec),
        grain=spec.grain,
        rounding=spec.rounding,
        minimum=spec.minimum,
        maximum=spec.maximum,
    )
    logger.debug(
        "Resolved a metallic schedule of {} window(s) from a {} seed.",
        len(windows),
        spec.seed,
    )
    return WindowSchedule(spec=spec, windows=windows)


def _log_spaced_terms(spec: LogSpacedSpec) -> list[Fraction]:
    """Place the log-spaced points between the bounds, in exact rationals.

    The two endpoints are the caller's bounds themselves, not the
    logarithm round trip of them: ``exp(ln(x))`` is a hair away from
    ``x``, and a schedule whose last window is a hair under the maximum
    the caller named would be a puzzle with no benefit.

    Args:
        spec: The validated ladder parameters.

    Returns:
        The points in seconds, ascending, endpoints included.

    """
    minimum_seconds = spec.minimum.total_seconds
    with localcontext() as context:
        context.prec = _DECIMAL_DIGITS
        span = _log_span(spec.minimum, spec.maximum)
        low = Decimal(minimum_seconds)
        steps = spec.count - 1
        interior = [
            Fraction(low * (span * step / steps).exp()) for step in range(1, steps)
        ]
    return [
        Fraction(minimum_seconds),
        *interior,
        Fraction(spec.maximum.total_seconds),
    ]


def log_spaced(
    *,
    count: int,
    minimum: Duration | str,
    maximum: Duration | str,
    grain: Duration | str,
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY,
) -> WindowSchedule:
    """Resolve a schedule of log-spaced durations between two bounds.

    ``count`` points are placed so that the ratio between neighbours is
    constant, with both endpoints included, and the result is quantized,
    bounded, and deduplicated by the same machinery every other
    generator here uses.

    Args:
        count: How many points to place, endpoints included. At least
            two.
        minimum: The first point, and the lower bound.
        maximum: The last point, and the upper bound.
        grain: The quantization grain.
        rounding: The tie rule for quantization. Defaults to
            :attr:`RoundingRule.NEAREST_TIES_AWAY`.

    Returns:
        The resolved schedule, carrying both its windows and the
        parameters that produced them. It may name fewer windows than
        ``count`` asked for: neighbouring points that land on the same
        grain multiple are deduplicated.

    Raises:
        ConfigError: If any parameter is invalid, if a point quantizes
            to nothing, or if the bounds leave no windows.

    """
    spec = LogSpacedSpec(
        count=count,
        minimum=validate_window_duration(minimum),
        maximum=validate_window_duration(maximum),
        grain=validate_cadence(grain),
        rounding=rounding,
    )
    windows = _resolve_windows(
        _log_spaced_terms(spec),
        grain=spec.grain,
        rounding=spec.rounding,
        minimum=spec.minimum,
        maximum=spec.maximum,
    )
    logger.debug(
        "Resolved a log-spaced schedule of {} window(s) from {} point(s).",
        len(windows),
        spec.count,
    )
    return WindowSchedule(spec=spec, windows=windows)


def explicit(
    windows: Sequence[Duration | str], *, name: str | None = None
) -> WindowSchedule:
    """Build a schedule from a caller-supplied resolved list.

    Nothing is generated, quantized, sorted, or deduplicated here: the
    list is taken as given, in the order given, and only checked against
    the invariants every schedule shares. A repeat is refused rather
    than collapsed -- the generated kinds deduplicate because their
    arithmetic produces repeats, whereas a repeat in a hand-written list
    is a mistake in that list, and correcting it silently would hide it.

    Args:
        windows: The window durations, as Durations or compact duration
            strings.
        name: The name a registered list is asked for by. Defaults to
            None, for an ad-hoc list.

    Returns:
        The schedule, carrying the given windows and the name.

    Raises:
        ConfigError: If ``windows`` is not a list or tuple, if any entry
            is not a strictly positive duration, if the list is empty,
            repeats a window, or is longer than
            :data:`MAX_RESOLVED_WINDOWS`.

    """
    if isinstance(windows, str) or not isinstance(windows, Sequence):
        logger.warning(
            "Rejecting an explicit schedule that is not a list: {!r}", windows
        )
        raise ConfigError(
            f"An explicit schedule takes a list of durations, got "
            f"{type(windows).__name__}"
        )
    resolved = tuple(validate_window_duration(window) for window in windows)
    logger.debug("Recorded an explicit schedule of {} window(s).", len(resolved))
    return WindowSchedule(spec=ExplicitSpec(name=name), windows=resolved)
