"""What a phased lookback needs to be true of its frame, checked once.

Both the reference harness and the fast one resolve their inputs through
here, so the refusals cannot drift apart between them and neither can
quietly accept a frame the other would reject.

Every check is a REFUSAL at resolution time, never a warning: a phase
error does not show up as a wrong number in one column, it shows up as
every indicator built on the frame being wrong together, and by then the
artifact has been written.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import ConfigError, Duration, coerce_duration
from ohlc_toolkit.temporal.echo import bounded_echo

logger = get_logger(__name__)

#: The phased fields, and the dtype each list column carries. `open_time`
#: is not among them: it is `close_time - W` for every row, so carrying it
#: would be recording the same fact twice and inviting the two to
#: disagree. `close_time` is the frame's key rather than a phased field.
_PHASED_COLUMN_TYPES: dict[str, pl.DataType] = {
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "src_count": pl.UInt32(),
    "coverage_seconds": pl.Int64(),
    "traded_seconds": pl.Int64(),
}

#: Exposed as a read-only mapping rather than the dict itself: it is a
#: public name on a 2.0 surface, and a caller that mutated it would change
#: what every later call reads.
PHASED_COLUMNS: Mapping[str, pl.DataType] = MappingProxyType(_PHASED_COLUMN_TYPES)

#: What a windowed-candle frame must carry for any of this to mean
#: anything. A frame without `traded_seconds` is a schema v1 frame, and
#: no reader invents one for it.
REQUIRED_COLUMNS = ("open_time", "close_time", *PHASED_COLUMNS)


@dataclass(frozen=True, slots=True)
class PhasedGrid:
    """One resolved phased-lookback request.

    Attributes:
        window_seconds: The window duration ``W``, in whole seconds.
        emit_seconds: The emit cadence ``E``, in whole seconds.
        cadence_seconds: The source cadence the frame is materialized at,
            measured from the frame rather than declared by the caller.
        lookback: How many phased windows ``L`` each tick consumes.
        min_traded_seconds: The threshold below which a window is a null
            input.
        ticks: Every emit tick inside the frame's range, ascending.

    """

    window_seconds: int
    emit_seconds: int
    cadence_seconds: int
    lookback: int
    min_traded_seconds: int
    ticks: tuple[int, ...]

    @property
    def effective_history(self) -> Duration:
        """How far back one tick's inputs reach: ``L * W``.

        Reported rather than left to a caller, because it is the number a
        recipe records and compares between indicators, and two callers
        multiplying it themselves are two chances to multiply by the
        emit cadence instead. The phased windows do not overlap, so the
        span really is the product and not something shorter.
        """
        return Duration(self.lookback * self.window_seconds)


@dataclass(frozen=True, slots=True)
class PhasedLookback:
    """One tick grid's phased inputs, and what the run reaches back to.

    Attributes:
        frame: One row per emit tick, carrying ``close_time`` and one
            list column per phased field.
        grid: The resolved request, including the effective history the
            run covers.

    """

    frame: pl.DataFrame
    grid: PhasedGrid

    @property
    def effective_history(self) -> Duration:
        """How far back one tick's inputs reach: ``L * W``."""
        return self.grid.effective_history


def resolve_phased_grid(  # noqa: PLR0913 - one keyword per resolution input
    frame: pl.DataFrame,
    *,
    window: Duration | str,
    emit_every: Duration | str,
    anchor: Duration | str,
    lookback: int,
    min_traded_seconds: int,
) -> PhasedGrid:
    """Check a frame and a schedule, and list the ticks to emit on.

    Args:
        frame: The windowed-candle frame at source cadence.
        window: The window duration ``W``.
        emit_every: The emit cadence ``E``.
        anchor: The emit-grid anchor offset.
        lookback: How many phased windows each tick consumes.
        min_traded_seconds: The recipe's traded threshold.

    Returns:
        The resolved grid.

    Raises:
        ConfigError: If ``lookback`` or ``min_traded_seconds`` is
            unusable, if the frame is missing a required column or is a
            schema v1 frame, if its ``close_time`` spacing is not a
            single constant cadence, if any row's span is not ``W``, if
            that cadence does not divide ``W``, or if ``E`` is not a
            whole multiple of that cadence.

    """
    resolved_lookback = _validated_lookback(lookback)
    threshold = _validated_threshold(min_traded_seconds)
    window_seconds = _positive_seconds(window, label="window")
    emit_seconds = _positive_seconds(emit_every, label="emit_every")
    anchor_seconds = coerce_duration(anchor).total_seconds

    _require_columns(frame)
    cadence_seconds = _require_regular_grid(frame)
    _require_window_span(frame, window_seconds)
    _require_cadence_divides_window(cadence_seconds, window_seconds)

    if emit_seconds % cadence_seconds:
        logger.warning(
            "Rejecting an emit cadence of {}s against a {}s source grid.",
            emit_seconds,
            cadence_seconds,
        )
        raise ConfigError(
            f"The emit cadence must be a whole multiple of the frame's {cadence_seconds}s "
            f"cadence, got {emit_seconds}s; a tick off the grid can never be looked up."
        )

    ticks = _emit_ticks(frame, emit_seconds, anchor_seconds)
    _require_ticks_on_the_grid(frame, ticks, cadence_seconds)

    return PhasedGrid(
        window_seconds=window_seconds,
        emit_seconds=emit_seconds,
        cadence_seconds=cadence_seconds,
        lookback=resolved_lookback,
        min_traded_seconds=threshold,
        ticks=ticks,
    )


def _validated_lookback(value: object) -> int:
    """Return a lookback count, rejecting anything that is not a positive int."""
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("Rejecting non-int lookback: {}", type(value).__name__)
        raise ConfigError(f"lookback must be an int, got {type(value).__name__}")
    if value <= 0:
        logger.warning("Rejecting non-positive lookback: {}", value)
        raise ConfigError(f"lookback must be strictly positive, got {value}")
    if value > MAX_LOOKBACK:
        logger.warning("Rejecting a lookback of {}.", value)
        raise ConfigError(
            f"lookback must be at most {MAX_LOOKBACK}, got {value}; each phase is "
            "a separate join, so an unbounded count is an unbounded query."
        )
    return value


def _validated_threshold(value: object) -> int:
    """Return a traded threshold, rejecting anything that is not a non-negative int.

    The same check :mod:`ohlc_toolkit.windows.quality` applies to its own
    ``min_traded_seconds``, and deliberately a second copy rather than an
    import: this module reads a threshold a RECIPE states, that one reads
    a threshold a POLICY states, and the two are separate numbers that
    happen to share a grammar. Importing would couple a recipe's
    validation to a policy's, so that relaxing one would silently relax
    the other. A test compares the two messages instead.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("Rejecting non-int min_traded_seconds: {}", type(value).__name__)
        raise ConfigError(
            f"min_traded_seconds must be an int, got {type(value).__name__}"
        )
    if value < 0:
        logger.warning("Rejecting negative min_traded_seconds: {}", value)
        raise ConfigError(f"min_traded_seconds must not be negative, got {value}")
    return value


def _positive_seconds(value: Duration | str, *, label: str) -> int:
    """Coerce a duration and refuse a zero one."""
    seconds = coerce_duration(value).total_seconds
    if seconds == 0:
        logger.warning("Rejecting a zero {}.", label)
        raise ConfigError(f"{label} must be strictly positive, got 0s.")
    return seconds


def _require_columns(frame: pl.DataFrame) -> None:
    """Refuse a frame missing anything the phasing reads.

    Raises:
        ConfigError: If a required column is absent. A frame carrying
            everything but ``traded_seconds`` is named as the schema v1
            frame it is, because that is the shape every artifact written
            before schema v2 has.

    """
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if not missing:
        return
    logger.warning("Rejecting a frame missing phased column(s): {}", missing)
    superseded = (
        " A frame carrying the others but not traded_seconds is a schema v1"
        " windowed-candle frame, which this supersedes rather than"
        " reinterprets: nothing invents a traded_seconds for it."
        if missing == ["traded_seconds"]
        else ""
    )
    raise ConfigError(
        f"A phased lookback requires column(s) {missing}; apply it to a "
        f"windowed-candle frame materialized at source cadence.{superseded}"
    )


def _require_regular_grid(frame: pl.DataFrame) -> int:
    """Measure the frame's cadence, refusing anything that is not one grid.

    Returns:
        The single spacing between consecutive ``close_time`` values.

    The cadence is the SMALLEST step, and every other step must be a
    whole multiple of it. A frame with a hole in it is therefore accepted
    and its missing minutes are absent inputs -- which is the answer the
    contract wants, since a lookup that misses is null. What is refused
    is a frame that steps by something the cadence does not divide, which
    is not a source-cadence materialization of one window at all.

    Raises:
        ConfigError: If the frame holds fewer than two rows, does not
            ascend, or steps by a value the smallest step does not
            divide.

    """
    closes = frame.get_column("close_time")
    if closes.null_count():
        logger.warning("Rejecting {} null close_time value(s).", closes.null_count())
        raise ConfigError(
            f"close_time must not be null; this frame has {closes.null_count()} "
            "null value(s), and a row with no close time is a row no lookup can "
            "reach."
        )
    if closes.len() < _MINIMUM_ROWS:
        logger.warning("Rejecting a phased frame of {} row(s).", closes.len())
        raise ConfigError(
            f"A phased lookback needs at least {_MINIMUM_ROWS} rows to measure a "
            f"cadence from, got {closes.len()}."
        )
    steps = closes.diff().drop_nulls()
    spacings = sorted(steps.unique().to_list())
    if not spacings or spacings[0] < 0:
        logger.warning("Rejecting a close_time column that does not ascend.")
        raise ConfigError(
            f"A phased lookback needs an ascending close_time grid; this frame's "
            f"row spacings are {bounded_echo(spacings)}."
        )
    if spacings[0] == 0:
        logger.warning("Rejecting a frame that repeats a close_time.")
        raise ConfigError(
            "A phased lookback needs each close_time once; this frame repeats one, "
            "so an exact-equality lookup would have two answers."
        )

    # The MODAL step, not the smallest. Taking the smallest let a single
    # off-grid row redefine the cadence as its own tiny gap, at which point
    # both the emit-multiple rule and the anchor-phase rule are vacuously
    # satisfied and the caller gets a frame of nulls -- the exact answer
    # those two rules exist to refuse.
    cadence = int(steps.mode().min())  # type: ignore[arg-type]
    ragged = [spacing for spacing in spacings if spacing % cadence or spacing < cadence]
    if ragged:
        logger.warning("Rejecting row spacings that are not whole steps: {}", ragged)
        raise ConfigError(
            f"Every close_time step must be a whole multiple of the {cadence}s "
            f"cadence; this frame steps by {bounded_echo(ragged)}, so it is not a "
            "source-cadence materialization of one window."
        )
    return cadence


def _require_window_span(frame: pl.DataFrame, window_seconds: int) -> None:
    """Refuse a frame whose rows do not span the window they are said to.

    Raises:
        ConfigError: If any row's ``close_time - open_time`` is not ``W``.
            Reading a 5m frame as though it were a 3m one would phase
            correctly and mean nothing.

    """
    spans = (
        (frame.get_column("close_time") - frame.get_column("open_time"))
        .unique()
        .to_list()
    )
    if spans != [window_seconds]:
        logger.warning(
            "Rejecting a frame spanning {} rather than the stated window.", spans
        )
        raise ConfigError(
            f"Every row must span the stated window of {window_seconds}s; this "
            f"frame spans {bounded_echo(sorted(spans))}."
        )


def _require_cadence_divides_window(cadence_seconds: int, window_seconds: int) -> None:
    """Refuse a frame whose cadence cannot reach the phases of its own window.

    Every phase is read at ``t - kW`` by exact equality, and ``t`` is a
    row of the frame. Where the cadence does not divide ``W`` that
    address falls between two rows for every ``k >= 1``, so every phase
    but the zeroth misses, the all-or-nothing mask nulls the tick, and
    the caller is handed a column that is null end to end -- the same
    silent answer :func:`_require_ticks_on_the_grid` refuses an
    out-of-phase anchor for, arrived at from the other side.

    Refused whatever the lookback, though a lookback of 1 reads only
    phase zero and cannot expose it: a recipe validated at 1 and then
    raised to 2 would go from a wholly correct column to a wholly null
    one, and the frame was the wrong frame at both. The emit rule cannot
    stand in for this one either, since a cadence may divide ``E``
    exactly and still not divide ``W``.

    Raises:
        ConfigError: If ``W`` is not a whole multiple of the cadence.

    """
    remainder = window_seconds % cadence_seconds
    if not remainder:
        return
    logger.warning(
        "Rejecting a {}s cadence that does not divide a {}s window.",
        cadence_seconds,
        window_seconds,
    )
    accepted = [
        candidate
        for candidate in (
            window_seconds - remainder,
            window_seconds - remainder + cadence_seconds,
        )
        if candidate > 0
    ]
    raise ConfigError(
        f"The frame's {cadence_seconds}s cadence must divide the window of "
        f"{window_seconds}s and leaves {remainder}s over, so every phase but the "
        f"newest would be read at a time this frame has no row for. Materialize "
        f"the frame at a cadence that divides {window_seconds}s, or state a window "
        f"this cadence divides: {bounded_echo(accepted)}."
    )


def _require_ticks_on_the_grid(
    frame: pl.DataFrame, ticks: tuple[int, ...], cadence_seconds: int
) -> None:
    """Refuse an anchor whose grid never lands on a row of this frame.

    Every lookup is an exact equality on ``close_time``, so a tick off
    the frame's own phase matches nothing and the whole result comes back
    null -- a silent answer of "no data" to a question that was really
    "this anchor does not belong to this frame". ``E`` is already a whole
    multiple of the cadence, so every tick shares one residue and one
    comparison settles it.

    Raises:
        ConfigError: If the emit grid is out of phase with the frame.

    """
    if not ticks:
        return
    frame_phase = int(frame.get_column("close_time")[0]) % cadence_seconds
    tick_phase = ticks[0] % cadence_seconds
    if tick_phase != frame_phase:
        logger.warning(
            "Rejecting an emit grid at phase {}s against a frame at phase {}s.",
            tick_phase,
            frame_phase,
        )
        raise ConfigError(
            f"The emit grid sits at {tick_phase}s past each {cadence_seconds}s step "
            f"and the frame at {frame_phase}s, so no emit tick is ever a row of it; "
            "every lookup would miss and every output would be null."
        )


def _emit_ticks(
    frame: pl.DataFrame, emit_seconds: int, anchor_seconds: int
) -> tuple[int, ...]:
    """List the emit ticks inside the frame's own range, ascending."""
    closes = frame.get_column("close_time")
    lowest = int(closes.min())  # type: ignore[arg-type]
    highest = int(closes.max())  # type: ignore[arg-type]
    offset = (lowest - anchor_seconds) % emit_seconds
    first = lowest if offset == 0 else lowest + emit_seconds - offset
    return tuple(range(first, highest + 1, emit_seconds))


_MINIMUM_ROWS = 2

#: The same cap a schedule carries, for the same reason: a count nobody
#: bounded is a query nobody bounded. The published lookback ladder tops
#: out at 56, so this is far above anything a recipe states.
MAX_LOOKBACK = 512
