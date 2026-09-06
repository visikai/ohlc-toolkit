"""A count-valued lookback schedule, generated the same way windows are.

A lookback is a number of PERIODS, not a duration. "21" means twenty-one
of whatever period the frame it is applied to carries; the same 21 is
twenty-one minutes on a 1m frame and twenty-one weeks on a 1w one. That
is the whole reason this is a separate schedule rather than a setting on
the window schedule: a lookback that borrowed the window schedule's
durations would be silently coupled to it, and changing one would change
the other.

What it shares with the window schedule is the ARITHMETIC, not the type.
The recurrence, the log-spaced placement, and the quantize/bound/dedup
rule all live in :mod:`ohlc_toolkit.schedules.generators` and are called
from here with counts where that module calls them with seconds. Two
copies of a rounding rule are two rounding rules.

Identity works as it does for a window schedule -- a sha256 over the
canonical JSON of the kind, the parameters, and the resolved list -- with
one difference that is load-bearing: the member list is recorded under
``"periods"`` rather than ``"windows"``. A count schedule and a duration
schedule over the same numbers therefore hash differently, and a payload
written by one is refused by the other's reader for a missing key rather
than read as if the numbers meant the same thing.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar, Protocol, Self

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.schedules.generators import (
    MAX_RESOLVED_WINDOWS,
    DedupRule,
    GeneratorKind,
    RoundingRule,
    ScheduleUnits,
    _require_rules,
    _validated_coefficient,
    _validated_count,
    log_spaced_values,
    recurrence_values,
    require_endpoints_on_the_grain,
    resolve_values,
)
from ohlc_toolkit.schedules.identity import (
    content_hash,
    mapping_from_payload,
    require_keys,
    require_recorded_id,
)
from ohlc_toolkit.temporal import ConfigError
from ohlc_toolkit.temporal.echo import enum_from_payload

logger = get_logger(__name__)

#: How a lookback describes itself when it refuses. A period count has no
#: unit to render, so both renderers are the plain integer.
PERIOD_UNITS = ScheduleUnits(
    noun="lookback",
    render=str,
    render_grain=str,
)

_LOOKBACK_KEYS = ("kind", "parameters", "periods", "schedule_id")
_METALLIC_KEYS = (
    "coefficient",
    "seed",
    "grain",
    "minimum",
    "maximum",
    "rounding",
    "dedup",
)
_LOG_SPACED_KEYS = ("count", "minimum", "maximum", "grain", "rounding", "dedup")
_EXPLICIT_KEYS = ("periods",)


def validated_period(value: object, *, label: str) -> int:
    """Return a period count, rejecting anything that is not a positive int.

    ``bool`` is rejected despite being an ``int`` subtype: a lookback of
    ``True`` is nobody's intention, and it would silently mean one.

    Args:
        value: The candidate count, of any type.
        label: What to call it when refusing.

    Returns:
        ``value`` as an ``int``.

    Raises:
        ConfigError: If it is not an ``int``, or is not strictly
            positive.

    """
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("Rejecting non-int {}: {}", label, type(value).__name__)
        raise ConfigError(f"{label} must be an int, got {type(value).__name__}")
    if value <= 0:
        logger.warning("Rejecting non-positive {}: {}", label, value)
        raise ConfigError(f"{label} must be strictly positive, got {value}")
    return value


def optional_period(value: object, *, label: str) -> int | None:
    """Return a period count, or ``None`` for an absent optional bound."""
    if value is None:
        return None
    return validated_period(value, label=label)


def periods_from_payload(value: object, *, label: str) -> tuple[int, ...]:
    """Read a recorded list of period counts.

    Refuses a payload that is not a list of positive integers, which is
    what a duration schedule's ``"windows"`` list looks like from here:
    strings, not counts.

    Raises:
        ConfigError: If the value is not a sequence, or any member fails
            :func:`validated_period`.

    """
    if isinstance(value, str) or not isinstance(value, Sequence):
        logger.warning("Rejecting non-sequence {}: {}", label, type(value).__name__)
        raise ConfigError(
            f"{label} must be a list of period counts, got {type(value).__name__}"
        )
    return tuple(validated_period(member, label=label) for member in value)


class LookbackSpec(Protocol):
    """What every lookback generator's recorded parameters provide.

    Both directions are declared, because the reader needs both: without
    ``from_dict`` here, reconstructing a spec from a payload needs a
    ``type: ignore`` at the one place the protocol exists to type.
    """

    kind: ClassVar[GeneratorKind]

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict."""
        ...

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their serialized form."""
        ...


@dataclass(frozen=True)
class MetallicLookbackSpec:
    """The parameters of one metallic-recurrence lookback schedule.

    The same recurrence the window schedule uses, in counts. A
    coefficient of ``sqrt(e + sqrt(5))`` seeded at 1 with a grain of 1
    gives the ladder 1, 3, 8, 21, 56.

    Attributes:
        coefficient: The ``c`` in ``x[n+1] = c * x[n] + x[n-1]``.
        seed: The count the recurrence starts from; its first two terms.
        grain: The quantization grain every resolved count is a whole
            multiple of. ``1`` means whole periods, which is the only
            grain with an obvious meaning for a count.
        maximum: The upper bound.
        minimum: An optional lower bound.
        rounding: The tie rule quantization applies.
        dedup: The rule applied to repeated quantized values.

    """

    coefficient: float
    seed: int
    grain: int
    maximum: int
    minimum: int | None = None
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY
    dedup: DedupRule = DedupRule.DROP_LATER_REPEATS

    kind: ClassVar[GeneratorKind] = GeneratorKind.METALLIC_RECURRENCE

    def __post_init__(self) -> None:
        """Check every parameter, however this was constructed.

        Raises:
            ConfigError: If the coefficient is not strictly positive and
                finite, if any count is not a positive int, if the
                minimum exceeds the maximum, or if the rules are not
                enum members.

        """
        object.__setattr__(
            self, "coefficient", _validated_coefficient(self.coefficient)
        )
        object.__setattr__(self, "seed", validated_period(self.seed, label="seed"))
        object.__setattr__(self, "grain", validated_period(self.grain, label="grain"))
        object.__setattr__(
            self, "maximum", validated_period(self.maximum, label="maximum")
        )
        if self.minimum is not None:
            object.__setattr__(
                self, "minimum", validated_period(self.minimum, label="minimum")
            )
            _require_ordered_periods(self.minimum, self.maximum)
        _require_rules(self.rounding, self.dedup)

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict."""
        return {
            "coefficient": self.coefficient,
            "seed": self.seed,
            "grain": self.grain,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "rounding": self.rounding.value,
            "dedup": self.dedup.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their :meth:`to_dict` form.

        Raises:
            ConfigError: If a key is missing, or any value fails the
                validation it would have failed at construction.

        """
        require_keys(data, _METALLIC_KEYS, label="metallic lookback parameters")
        return cls(
            coefficient=_validated_coefficient(data["coefficient"]),
            seed=validated_period(data["seed"], label="seed"),
            grain=validated_period(data["grain"], label="grain"),
            maximum=validated_period(data["maximum"], label="maximum"),
            minimum=optional_period(data["minimum"], label="minimum"),
            rounding=enum_from_payload(
                RoundingRule, data["rounding"], label="rounding rule"
            ),
            dedup=enum_from_payload(DedupRule, data["dedup"], label="dedup rule"),
        )


@dataclass(frozen=True)
class LogSpacedLookbackSpec:
    """The parameters of one log-spaced lookback schedule.

    A ladder wants something to be measured against. Three points between
    7 and 28 -- a ratio of 2 -- give the control 7, 14, 28.

    Attributes:
        count: How many points, endpoints included.
        minimum: The lower bound, and the first point.
        maximum: The upper bound, and the last point.
        grain: The quantization grain.
        rounding: The tie rule quantization applies.
        dedup: The rule applied to repeated quantized values.

    """

    count: int
    minimum: int
    maximum: int
    grain: int = 1
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY
    dedup: DedupRule = DedupRule.DROP_LATER_REPEATS

    kind: ClassVar[GeneratorKind] = GeneratorKind.LOG_SPACED

    def __post_init__(self) -> None:
        """Check every parameter, however this was constructed.

        Raises:
            ConfigError: If the count is not at least two, if any count
                is not a positive int, if the minimum is not below the
                maximum, or if the rules are not enum members.

        """
        object.__setattr__(self, "count", _validated_count(self.count))
        object.__setattr__(
            self, "minimum", validated_period(self.minimum, label="minimum")
        )
        object.__setattr__(
            self, "maximum", validated_period(self.maximum, label="maximum")
        )
        object.__setattr__(self, "grain", validated_period(self.grain, label="grain"))
        _require_ordered_periods(self.minimum, self.maximum)
        _require_rules(self.rounding, self.dedup)

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict."""
        return {
            "count": self.count,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "grain": self.grain,
            "rounding": self.rounding.value,
            "dedup": self.dedup.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their :meth:`to_dict` form.

        Raises:
            ConfigError: If a key is missing, or any value fails the
                validation it would have failed at construction.

        """
        require_keys(data, _LOG_SPACED_KEYS, label="log_spaced lookback parameters")
        return cls(
            count=_validated_count(data["count"]),
            minimum=validated_period(data["minimum"], label="minimum"),
            maximum=validated_period(data["maximum"], label="maximum"),
            grain=validated_period(data["grain"], label="grain"),
            rounding=enum_from_payload(
                RoundingRule, data["rounding"], label="rounding rule"
            ),
            dedup=enum_from_payload(DedupRule, data["dedup"], label="dedup rule"),
        )


@dataclass(frozen=True)
class ExplicitLookbackSpec:
    """A lookback schedule stated rather than generated.

    Attributes:
        periods: The counts, exactly as the caller named them.

    """

    periods: tuple[int, ...]

    kind: ClassVar[GeneratorKind] = GeneratorKind.EXPLICIT

    def __post_init__(self) -> None:
        """Normalize the counts to a tuple and check each one.

        Raises:
            ConfigError: If any member is not a strictly positive int.

        """
        object.__setattr__(
            self,
            "periods",
            tuple(
                validated_period(period, label="explicit period")
                for period in self.periods
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize these parameters to a JSON-compatible dict."""
        return {"periods": list(self.periods)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct these parameters from their :meth:`to_dict` form.

        Raises:
            ConfigError: If the key is missing or any member is
                malformed.

        """
        require_keys(data, _EXPLICIT_KEYS, label="explicit lookback parameters")
        return cls(periods=periods_from_payload(data["periods"], label="period"))


_SPEC_TYPES: dict[GeneratorKind, type[LookbackSpec]] = {
    GeneratorKind.METALLIC_RECURRENCE: MetallicLookbackSpec,
    GeneratorKind.LOG_SPACED: LogSpacedLookbackSpec,
    GeneratorKind.EXPLICIT: ExplicitLookbackSpec,
}


def _require_ordered_periods(minimum: int, maximum: int) -> None:
    """Refuse bounds that cross.

    Raises:
        ConfigError: If ``minimum`` exceeds ``maximum``.

    """
    if minimum > maximum:
        logger.warning("Rejecting lookback bounds {} > {}.", minimum, maximum)
        raise ConfigError(
            f"A lookback schedule's minimum must not exceed its maximum, got "
            f"{minimum} > {maximum}."
        )


def require_resolved_periods(periods: tuple[int, ...]) -> None:
    """Check a resolved lookback list, wherever it came from.

    The count-valued twin of
    :func:`~ohlc_toolkit.schedules.generators.require_resolved_windows`,
    and deliberately its own function rather than a generalization of it:
    that one refuses anything but a ``Duration``, which is the guard that
    keeps a count out of a window schedule.

    Raises:
        ConfigError: If the list is empty, holds anything but a strictly
            positive int, repeats a value, or is longer than the cap.

    """
    if not periods:
        logger.warning("Rejecting a lookback schedule that resolved nothing.")
        raise ConfigError(
            "A lookback schedule must name at least one period count; this one "
            "resolved none at all."
        )
    if len(periods) > MAX_RESOLVED_WINDOWS:
        logger.warning("Rejecting a lookback schedule of {} counts.", len(periods))
        raise ConfigError(
            f"A lookback schedule must name at most {MAX_RESOLVED_WINDOWS} period "
            f"counts, got {len(periods)}."
        )
    seen: set[int] = set()
    for period in periods:
        validated_period(period, label="lookback period")
        if period in seen:
            logger.warning("Rejecting a lookback repeating the count {}.", period)
            raise ConfigError(
                f"A lookback schedule must name each count once, got {period} twice."
            )
        seen.add(period)


@dataclass(frozen=True)
class LookbackSchedule:
    """A resolved lookback schedule: its parameters and its period counts.

    Attributes:
        spec: The generator parameters that produced ``periods``.
        periods: The fully resolved counts, in generated order.

    """

    spec: LookbackSpec
    periods: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check the resolved list against the invariants every kind shares.

        Raises:
            ConfigError: If ``periods`` is empty, holds anything but a
                strictly positive int, holds a repeat, or is longer than
                :data:`~ohlc_toolkit.schedules.generators.MAX_RESOLVED_WINDOWS`.

        """
        require_resolved_periods(self.periods)

    @property
    def schedule_id(self) -> str:
        """The content hash naming this lookback schedule.

        The payload records its members under ``"periods"``, where a
        window schedule records ``"windows"``. That is what makes a
        collision between the two structurally impossible rather than
        merely unlikely: the key differs, and so does the form of every
        value in it -- ``3`` against ``"3m"``.
        """
        return content_hash(self._identity_payload())

    def _identity_payload(self) -> dict[str, object]:
        """Build the payload the schedule id is the hash of."""
        return {
            "kind": self.spec.kind.value,
            "parameters": self.spec.to_dict(),
            "periods": list(self.periods),
        }

    def to_dict(self) -> dict[str, object]:
        """Serialize this schedule to a deterministic, JSON-compatible dict.

        Returns:
            A dict with exactly the keys ``"kind"``, ``"parameters"``,
            ``"periods"`` and ``"schedule_id"``, in that fixed key order.

        """
        payload = self._identity_payload()
        return {**payload, "schedule_id": content_hash(payload)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct a lookback schedule from its :meth:`to_dict` form.

        The resolved counts are read from the payload rather than
        regenerated, for the reason
        :meth:`~ohlc_toolkit.schedules.generators.WindowSchedule.from_dict`
        gives: the recorded list is what the schedule actually used.

        Args:
            data: A mapping as produced by :meth:`to_dict`.

        Returns:
            The reconstructed schedule.

        Raises:
            ConfigError: If a key is missing -- which is what a window
                schedule's payload looks like here, since it records
                ``"windows"`` and not ``"periods"`` -- the kind names no
                generator, any parameter or count is malformed, the list
                breaks an invariant, or the recorded id does not match
                the payload it names.

        """
        require_keys(data, _LOOKBACK_KEYS, label="lookback schedule")
        kind = enum_from_payload(GeneratorKind, data["kind"], label="generator kind")
        parameters = mapping_from_payload(
            data["parameters"], label="lookback parameters"
        )
        schedule = cls(
            spec=_SPEC_TYPES[kind].from_dict(parameters),
            periods=periods_from_payload(data["periods"], label="period"),
        )
        require_recorded_id(
            data["schedule_id"], schedule.schedule_id, label="schedule_id"
        )
        return schedule


def metallic_lookback(  # noqa: PLR0913 - one keyword per recorded parameter
    *,
    coefficient: float,
    seed: int,
    grain: int,
    maximum: int,
    minimum: int | None = None,
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY,
    dedup: DedupRule = DedupRule.DROP_LATER_REPEATS,
) -> LookbackSchedule:
    """Resolve a metallic-recurrence lookback schedule.

    Args:
        coefficient: The ``c`` in ``x[n+1] = c * x[n] + x[n-1]``.
        seed: The count the recurrence starts from.
        grain: The quantization grain, in whole periods.
        maximum: The upper bound.
        minimum: An optional lower bound.
        rounding: The tie rule quantization applies.
        dedup: The rule applied to repeated quantized values.

    Returns:
        The resolved schedule.

    Raises:
        ConfigError: For any parameter the spec refuses, a recurrence
            that will not reach the maximum, or bounds that leave
            nothing.

    """
    spec = MetallicLookbackSpec(
        coefficient=coefficient,
        seed=seed,
        grain=grain,
        maximum=maximum,
        minimum=minimum,
        rounding=rounding,
        dedup=dedup,
    )
    values = recurrence_values(
        coefficient=spec.coefficient,
        seed=spec.seed,
        maximum=spec.maximum,
        units=PERIOD_UNITS,
    )
    return LookbackSchedule(
        spec=spec,
        periods=resolve_values(
            values,
            grain=spec.grain,
            rounding=spec.rounding,
            minimum=spec.minimum,
            maximum=spec.maximum,
            units=PERIOD_UNITS,
        ),
    )


def log_spaced_lookback(  # noqa: PLR0913 - one keyword per recorded parameter
    *,
    count: int,
    minimum: int,
    maximum: int,
    grain: int = 1,
    rounding: RoundingRule = RoundingRule.NEAREST_TIES_AWAY,
    dedup: DedupRule = DedupRule.DROP_LATER_REPEATS,
) -> LookbackSchedule:
    """Resolve a log-spaced lookback schedule.

    Args:
        count: How many points, endpoints included.
        minimum: The lower bound, and the first point.
        maximum: The upper bound, and the last point.
        grain: The quantization grain, in whole periods.
        rounding: The tie rule quantization applies.
        dedup: The rule applied to repeated quantized values.

    Returns:
        The resolved schedule.

    Raises:
        ConfigError: For any parameter the spec refuses, for an endpoint
            that quantizes outside the range it defines, or for bounds
            that leave nothing after quantization.

    """
    spec = LogSpacedLookbackSpec(
        count=count,
        minimum=minimum,
        maximum=maximum,
        grain=grain,
        rounding=rounding,
        dedup=dedup,
    )
    require_endpoints_on_the_grain(
        minimum=spec.minimum,
        maximum=spec.maximum,
        grain=spec.grain,
        rounding=spec.rounding,
        units=PERIOD_UNITS,
    )
    values = log_spaced_values(
        count=spec.count, minimum=spec.minimum, maximum=spec.maximum
    )
    return LookbackSchedule(
        spec=spec,
        periods=resolve_values(
            values,
            grain=spec.grain,
            rounding=spec.rounding,
            minimum=spec.minimum,
            maximum=spec.maximum,
            units=PERIOD_UNITS,
        ),
    )


def explicit_lookback(periods: Sequence[int]) -> LookbackSchedule:
    """State a lookback schedule rather than generating one.

    Args:
        periods: The counts, in the order they should be recorded.

    Returns:
        The resolved schedule.

    Raises:
        ConfigError: If any count is not strictly positive, if a count
            repeats, or if the list is empty or over the cap.

    """
    spec = ExplicitLookbackSpec(periods=tuple(periods))
    return LookbackSchedule(spec=spec, periods=spec.periods)
