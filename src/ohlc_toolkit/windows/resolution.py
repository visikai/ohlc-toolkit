"""Schedule, materialization, and emit-grid resolution shared by both engines.

Nothing in this module touches candle values. It decides only what a
caller asked for: whether a (cadence, window, emit cadence, anchor)
combination is legal at all, what the emit grid is, and which of the two
materialization forms was requested.

That separation matters because this package holds two implementations of
the same window contract -- a brute-force oracle and a vectorized engine
-- and a resolution rule that fired in one but not the other would make
the oracle useless as a correctness reference. Both import this module,
so there is exactly one statement of every rule and exactly one wording
of every refusal.

The emit grid
-------------

Emit ticks are the instants ``{t : (t - anchor) mod E == 0}``, an
epoch-anchored grid rather than a grid anchored to the data: the same
schedule over two different frames lands on the same instants. The anchor
is normalized to ``anchor mod E`` during resolution, so two spellings of
the same grid resolve to the same value.

Grid arithmetic lives here as scalar helpers -- the first tick at or
after a bound, the last tick at or before one, the number of ticks in a
range -- because the two implementations materialize those ticks very
differently (a Python tuple, a polars series) but must agree on exactly
which instants they are.
"""

from dataclasses import dataclass
from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.temporal import (
    ConfigError,
    Duration,
    bounded_echo,
    coerce_duration,
    validate_cadence,
    validate_window_duration,
)

logger = get_logger(__name__)

# The five OHLCV roles, read from the source frame by these exact names.
# A profile declares which raw columns exist and what kind they are; the
# role each one plays is fixed by this convention.
OHLCV_COLUMNS = ("open", "high", "low", "close", "volume")


@unique
class MaterializationRule(Enum):
    """A named rule for deriving the materialization range from the data.

    Only one rule exists today. Modelling it as an enum member rather than
    a bare string flag means a future rule is a new member, with its own
    documented derivation, instead of a second meaning bolted onto an
    existing one.

    Attributes:
        SKIP_WARMUP: A defined policy, not an inferred convenience: start
            at the first emit tick whose window is fully covered by
            source data, and stop one past the last emit tick at or
            before the source's final close time. If no tick is ever
            fully covered, resolving this rule raises
            :class:`~ohlc_toolkit.temporal.ConfigError` -- a deliberate
            fail-closed choice, unlike an explicit, caller-stated empty
            range, which is legal. See the
            :mod:`ohlc_toolkit.windows.reference` module docstring's
            "skip_warmup materialization rule" section for the full
            statement.

    """

    SKIP_WARMUP = "skip_warmup"


@dataclass(frozen=True)
class ExplicitRange:
    """A half-open materialization range given as exact Unix seconds.

    Every emit tick ``t`` with ``start <= t < end`` produces a row,
    including ticks whose windows lie entirely before or entirely after
    the source data: the emit grid is total, and a tick with no data is
    reported as an empty window rather than dropped.

    Attributes:
        start: The first Unix second that may hold an emit tick,
            inclusive.
        end: The first Unix second past the range, exclusive. Equal to
            ``start`` means an empty range, which is legal and yields zero
            rows.

    """

    start: int
    end: int

    def __post_init__(self) -> None:
        """Reject bounds that are not exact seconds, or that run backwards.

        Raises:
            ConfigError: If either bound is not an ``int`` (``bool`` is
                rejected too, even though it is an ``int`` subtype), or if
                ``end`` precedes ``start``.

        """
        for label, value in (("start", self.start), ("end", self.end)):
            if isinstance(value, bool) or not isinstance(value, int):
                logger.warning(
                    "Rejecting non-integer materialization range {}: {}",
                    label,
                    type(value).__name__,
                )
                raise ConfigError(
                    f"Materialization range {label} must be an int of Unix "
                    f"seconds, got {type(value).__name__}."
                )
        if self.end < self.start:
            logger.warning(
                "Rejecting inverted materialization range [{}, {}).",
                self.start,
                self.end,
            )
            raise ConfigError(
                f"Materialization range end must not precede its start, got "
                f"[{self.start}, {self.end})."
            )


# What a caller may pass as the materialization argument: an explicit
# range, a named rule, or that rule's name as a plain string.
Materialization = ExplicitRange | MaterializationRule | str


@dataclass(frozen=True)
class ResolvedSchedule:
    """A schedule whose every strict resolution rule has already passed.

    The source cadence and phase are not carried here: both are only
    needed during resolution itself (to check the schedule against the
    profile), and every later stage reads only ``window``, ``emit_every``,
    and ``anchor``.

    Attributes:
        window: The window duration ``W``.
        emit_every: The emit cadence ``E``.
        anchor: The emit-grid anchor offset, already normalized to
            ``anchor mod E`` so that two spellings of the same grid are
            the same value.

    """

    window: Duration
    emit_every: Duration
    anchor: Duration


def resolve_schedule(
    profile: SourceProfile,
    *,
    window: Duration | str,
    emit_every: Duration | str,
    anchor: Duration | str,
) -> ResolvedSchedule:
    """Coerce and check a schedule against the source's cadence and phase.

    The checks run shortest-first and only then divisibility, so each rule
    can fire on its own and report the most specific reason. Checking
    divisibility first would make the two shortest-than-cadence rules
    unreachable (a positive whole multiple of ``d`` is never smaller than
    ``d``) and leave a caller who asked for a 30s window over a 60s
    source reading about remainders instead of about size.

    Args:
        profile: The profile declaring the source cadence ``d`` and grid
            phase ``p``.
        window: The window duration ``W``.
        emit_every: The emit cadence ``E``.
        anchor: The emit-grid anchor offset, normalized here to
            ``anchor mod E``.

    Returns:
        The resolved schedule.

    Raises:
        ConfigError: If any strict resolution rule fails.

    """
    window_duration = validate_window_duration(window)
    emit_duration = validate_cadence(emit_every)
    anchor_duration = coerce_duration(anchor)

    window_seconds = window_duration.total_seconds
    emit_seconds = emit_duration.total_seconds
    cadence_seconds = profile.cadence.total_seconds
    phase_seconds = profile.phase.total_seconds

    if window_seconds < cadence_seconds:
        logger.warning(
            "Rejecting window of {}s: shorter than the {}s source cadence.",
            window_seconds,
            cadence_seconds,
        )
        raise ConfigError(
            f"Window duration must not be shorter than the source cadence, "
            f"got {window_seconds}s < {cadence_seconds}s."
        )
    if emit_seconds < cadence_seconds:
        logger.warning(
            "Rejecting emit cadence of {}s: shorter than the {}s source cadence.",
            emit_seconds,
            cadence_seconds,
        )
        raise ConfigError(
            f"Emit cadence must not be shorter than the source cadence, got "
            f"{emit_seconds}s < {cadence_seconds}s."
        )
    if window_seconds % cadence_seconds != 0:
        logger.warning(
            "Rejecting window of {}s: not a whole multiple of the {}s source cadence.",
            window_seconds,
            cadence_seconds,
        )
        raise ConfigError(
            f"Window duration must be a whole multiple of the source cadence, "
            f"got {window_seconds}s over a {cadence_seconds}s cadence."
        )
    if emit_seconds % cadence_seconds != 0:
        logger.warning(
            "Rejecting emit cadence of {}s: not a whole multiple of the {}s "
            "source cadence.",
            emit_seconds,
            cadence_seconds,
        )
        raise ConfigError(
            f"Emit cadence must be a whole multiple of the source cadence, got "
            f"{emit_seconds}s over a {cadence_seconds}s cadence."
        )

    # Two spellings of the same grid must resolve to the same anchor, so
    # the offset is reduced into [0, E) once, here.
    anchor_seconds = anchor_duration.total_seconds % emit_seconds

    # Every emit tick must be a possible source close time. Source opens
    # sit at `p` modulo `d`, so closes do too. Because `d` divides `E`,
    # every tick is congruent to the anchor modulo `d`, so testing the
    # anchor tests the whole grid -- including grids whose materialized
    # range turns out to be empty.
    if (anchor_seconds - phase_seconds) % cadence_seconds != 0:
        logger.warning(
            "Rejecting emit grid anchored at {}s: off the {}s close-time grid "
            "of a source at phase {}s.",
            anchor_seconds,
            cadence_seconds,
            phase_seconds,
        )
        raise ConfigError(
            f"The emit grid does not land on the source close-time grid: an "
            f"anchor of {anchor_seconds}s over a {cadence_seconds}s cadence at "
            f"phase {phase_seconds}s leaves every tick between two source "
            "close times."
        )

    return ResolvedSchedule(
        window=window_duration,
        emit_every=emit_duration,
        anchor=Duration(anchor_seconds),
    )


def require_source_columns(frame: pl.DataFrame, profile: SourceProfile) -> None:
    """Check that every column the aggregation reads exists to be read.

    Args:
        frame: The raw source frame.
        profile: The profile declaring the timestamp column and raw
            schema.

    Raises:
        ConfigError: If the profile does not declare, or the frame does not
            contain, the timestamp column or one of the five OHLCV
            columns.

    """
    required = (profile.timestamp_column, *OHLCV_COLUMNS)

    undeclared = [name for name in required if name not in profile.raw_schema]
    if undeclared:
        logger.warning(
            "Source profile {!r} does not declare the column(s) {}.",
            profile.name,
            undeclared,
        )
        raise ConfigError(
            f"Source profile {profile.name!r} must declare the column(s) "
            f"{undeclared} to be aggregated into windows."
        )

    absent = [name for name in required if name not in frame.columns]
    if absent:
        logger.warning(
            "Source frame for profile {!r} is missing the column(s) {}.",
            profile.name,
            absent,
        )
        raise ConfigError(
            f"The source frame does not contain the declared column(s) {absent}."
        )


def coerce_materialization(
    value: Materialization,
) -> ExplicitRange | MaterializationRule:
    """Coerce the boundary materialization argument to one of its two forms.

    Args:
        value: An :class:`ExplicitRange`, a :class:`MaterializationRule`,
            or a rule name as a plain string.

    Returns:
        The coerced explicit range or rule.

    Raises:
        ConfigError: If ``value`` is a string that names no rule, or is
            neither an :class:`ExplicitRange`, a
            :class:`MaterializationRule`, nor such a string.

    """
    if isinstance(value, ExplicitRange | MaterializationRule):
        return value
    if isinstance(value, str):
        try:
            return MaterializationRule(value)
        except ValueError as error:
            quoted = bounded_echo(value)
            logger.warning("Rejecting unknown materialization rule name: {}", quoted)
            raise ConfigError(
                f"Unknown materialization rule {quoted}. Supported: "
                f"{[rule.value for rule in MaterializationRule]}."
            ) from error

    logger.warning(
        "Rejecting materialization of unsupported type: {}", type(value).__name__
    )
    raise ConfigError(
        f"Expected an ExplicitRange, a MaterializationRule, or a rule name, "
        f"got {type(value).__name__}."
    )


def first_tick_at_or_after(bound: int, schedule: ResolvedSchedule) -> int:
    """Return the earliest emit tick that is not before ``bound``.

    Args:
        bound: A Unix second, on the grid or not.
        schedule: The resolved schedule supplying ``E`` and the anchor.

    Returns:
        The smallest ``t >= bound`` with ``(t - anchor) mod E == 0``.

    """
    emit_seconds = schedule.emit_every.total_seconds
    anchor_seconds = schedule.anchor.total_seconds
    return bound + (anchor_seconds - bound) % emit_seconds


def last_tick_at_or_before(bound: int, schedule: ResolvedSchedule) -> int:
    """Return the latest emit tick that is not after ``bound``.

    Args:
        bound: A Unix second, on the grid or not.
        schedule: The resolved schedule supplying ``E`` and the anchor.

    Returns:
        The greatest ``t <= bound`` with ``(t - anchor) mod E == 0``.

    """
    emit_seconds = schedule.emit_every.total_seconds
    anchor_seconds = schedule.anchor.total_seconds
    return bound - (bound - anchor_seconds) % emit_seconds


def count_ticks(first_tick: int, end: int, emit_seconds: int) -> int:
    """Count the emit ticks in ``[first_tick, end)``, stepping by ``E``.

    Args:
        first_tick: The first tick of the grid at or after the range
            start, as returned by :func:`first_tick_at_or_after`.
        end: The first Unix second past the range, exclusive.
        emit_seconds: The emit cadence ``E``.

    Returns:
        The number of ticks, never negative.

    """
    return max(0, -((first_tick - end) // emit_seconds))
