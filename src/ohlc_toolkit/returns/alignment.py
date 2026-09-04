"""Finding each row's counterpart row by exact close time.

Both return directions ask the same question of a window frame: for the
row whose information became available at ``t``, which row's close sits
exactly one horizon away -- at ``t - H`` looking back, at ``t + H``
looking forward? This module answers it, once, for both, so that a rule
one direction enforced and the other did not cannot exist.

Why the counterpart is found by time and never by a row shift
--------------------------------------------------------------

The cheap way to reach back one horizon is to divide the horizon by the
emit cadence and step that many ROWS. It is exact on a frame with no
gaps, and it is wrong on every frame with one. A window frame is
routinely gappy on purpose: the aggregator emits a total grid, and a
quality policy
(:func:`~ohlc_toolkit.windows.quality.apply_quality_policy`) drops
under-covered rows out of the middle of it. After a single dropped row,
``k`` rows back is no longer ``k`` cadences back, and every row from the
gap onward silently reports the return of some other interval. Nothing
about the output says so: the values are finite, plausible, and
misaligned, and they stay that way for the rest of the frame.

So the counterpart is located by an exact equality join on
``close_time``. A row whose counterpart timestamp is not in the frame
joins to nothing and its return is null. There is deliberately no
nearest-match, no as-of fallback, and no forward fill: each of those
answers a question nobody asked ("the closest close we happen to have")
with a number indistinguishable from the one that was asked for.

The join is on equality alone, so it does not care what order the frame
arrives in, and it is run with the left frame's order preserved, so the
caller gets their rows back in the order they handed them over. Sorting
is neither required nor performed.

close_time has to be a key
--------------------------

Two rows sharing a close time make the join fan out: rows are duplicated,
the output comes back longer than the input, and the surplus rows look
exactly like data. There is no defensible tie-break to pick between two
closes claiming the same instant, so this module does not invent one --
it refuses the frame. A null close time is refused for a related reason:
such a row has neither a counterpart to find nor an availability to
state, and a null join key silently matches nothing, which is
indistinguishable from a counterpart that was genuinely absent.

Both are refused at this module's boundary and in its own words. Left to
polars, the fan-out surfaces -- when it surfaces at all -- as an
arithmetic error about series of different lengths, which is a true
statement about the last step rather than about the frame.

Overflow is not an abstraction
------------------------------

The counterpart timestamp is ``close_time +/- H`` in Int64, and polars
wraps Int64 arithmetic silently rather than raising: adding 100 to a
close time near the top of the range yields a large NEGATIVE one. A
wrapped key does not merely fail to match, it can match some other row,
pairing a close with a counterpart nobody asked for. A horizon too large
to hold as an Int64 at all fails differently: on the Series arithmetic
this module performs, polars raises a bare ``OverflowError`` ("Python
int too large to convert to C long") -- a foreign exception, at shift
time, about a C type. The guard below refuses both cases up front, in
this package's own words, before any arithmetic is attempted.

So the shift is checked against the Int64 range before it is performed,
using the frame's own smallest and largest close times and exact Python
integers. The check is on the shift this call actually performs, not on
the horizon alone: a close time near the top of the range has room to
look back and none to look forward, and refusing both would be refusing
arithmetic that was going to be exact.
"""

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import (
    ConfigError,
    Duration,
    validate_cadence,
    validate_horizon_duration,
)

logger = get_logger(__name__)

# The two columns every return primitive reads, by the names and in the
# kinds :func:`~ohlc_toolkit.windows.engine.compute_windows` emits them.
# Nothing else is read, so nothing else is required: a caller who has
# projected a window frame down to what this step consults is not doing
# anything wrong.
CLOSE_TIME_COLUMN = "close_time"
CLOSE_COLUMN = "close"

# The exact dtype required of each, keyed by column name in the order
# they are reported. Int64 is the width the aggregator emits and the only
# one accepted: a narrower close_time wraps as soon as a horizon is added
# to it, and a UInt64 cannot be widened safely near the top of its range
# (a strict cast raises, a lenient one yields the null this step exists
# to distinguish from a missing counterpart). A close time that is not an
# integer has no exact equality to join on at all.
_REQUIRED_DTYPES: dict[str, pl.DataType] = {
    CLOSE_TIME_COLUMN: pl.Int64(),
    CLOSE_COLUMN: pl.Float64(),
}

# The bounds of the Int64 column close times are held in.
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1

# Rejected input is echoed into logs and error messages; cap how much, so
# one pathological dtype -- a struct with a thousand fields renders as a
# thousand field names -- cannot produce an unbounded log line. Eighty
# characters is enough to name any ordinary dtype while keeping one log
# line readable.
_MAX_ECHOED_CHARS = 80

# Column names used only inside the counterpart join, never returned.
_COUNTERPART_KEY = "__counterpart_close_time"
_COUNTERPART_CLOSE = "__counterpart_close"


def _bounded(value: object) -> str:
    """Render a value for a message, truncated with a note when oversized.

    Args:
        value: The rejected input to echo back.

    Returns:
        A representation never longer than the cap plus a length note.

    """
    text = str(value)
    if len(text) <= _MAX_ECHOED_CHARS:
        return text
    return f"{text[:_MAX_ECHOED_CHARS]}... ({len(text)} chars total)"


def resolve_horizon(horizon: Duration | str, cadence: Duration | str) -> Duration:
    """Check a horizon against the emit cadence the caller states.

    A horizon that is not a whole multiple of the cadence names an
    instant that lies between two emit ticks. Every such counterpart is
    absent by construction, so the whole column would come back null --
    a silent, total, and entirely explicable failure that is far better
    reported as the configuration error it is.

    Args:
        horizon: The horizon ``H``, as a
            :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string. Strictly positive.
        cadence: The cadence ``E`` the frame's rows are emitted at, in
            the same two spellings. Strictly positive.

    Returns:
        The validated horizon.

    Raises:
        ConfigError: If either value cannot be coerced to a Duration, if
            either is zero, or if the horizon is not a whole multiple of
            the cadence.

    """
    horizon_duration = validate_horizon_duration(horizon)
    cadence_duration = validate_cadence(cadence)

    horizon_seconds = horizon_duration.total_seconds
    cadence_seconds = cadence_duration.total_seconds
    if horizon_seconds % cadence_seconds != 0:
        logger.warning(
            "Rejecting horizon of {}s: not a whole multiple of the {}s emit cadence.",
            horizon_seconds,
            cadence_seconds,
        )
        raise ConfigError(
            f"A return horizon must be a whole multiple of the emit cadence, got "
            f"{horizon_seconds}s over a {cadence_seconds}s cadence: no row's "
            f"counterpart would ever land on an emitted close time."
        )
    return horizon_duration


def _require_columns(frame: pl.DataFrame) -> None:
    """Check that both read columns exist, in exactly the required kind.

    Raises:
        ConfigError: If either column is absent, or carries a dtype other
            than the one :data:`_REQUIRED_DTYPES` names for it.

    """
    missing = [name for name in _REQUIRED_DTYPES if name not in frame.columns]
    if missing:
        logger.warning("Rejecting frame missing return column(s): {}", missing)
        raise ConfigError(
            f"Return primitives require column(s) {missing}; apply them to a "
            "window frame such as compute_windows produces."
        )

    for name, required in _REQUIRED_DTYPES.items():
        actual = frame.schema[name]
        if actual != required:
            logger.warning(
                "Rejecting {} of dtype {}; {} is required.",
                name,
                _bounded(actual),
                required,
            )
            raise ConfigError(
                f"{name} must be a {required} column, got {_bounded(actual)}; "
                "apply return primitives to an engine-produced window frame."
            )


def _require_close_time_key(frame: pl.DataFrame) -> None:
    """Check that ``close_time`` can serve as the join's key.

    Raises:
        ConfigError: If any close time is null, or if any close time
            appears on more than one row.

    """
    close_time = frame.get_column(CLOSE_TIME_COLUMN)

    null_count = close_time.null_count()
    if null_count:
        logger.warning(
            "Rejecting frame stating no close time on {} row(s).", null_count
        )
        raise ConfigError(
            f"close_time must not be null: {null_count} row(s) state none, so they "
            "have neither a counterpart to find nor an availability to state."
        )

    duplicated = close_time.is_duplicated()
    duplicate_count = int(duplicated.sum())
    if duplicate_count:
        first_duplicate = close_time[int(duplicated.arg_true()[0])]
        logger.warning(
            "Rejecting frame repeating {} close time(s); the first is {}.",
            duplicate_count,
            first_duplicate,
        )
        raise ConfigError(
            f"close_time must be unique: {duplicate_count} row(s) repeat a close "
            f"time, the first at {first_duplicate}. A counterpart join over a "
            "repeated key multiplies rows instead of choosing between them."
        )


def _require_representable_shift(frame: pl.DataFrame, *, offset_seconds: int) -> None:
    """Check that shifting every close time stays inside the Int64 range.

    Evaluated in exact Python integers over the frame's own extremes,
    because the arithmetic being checked is the arithmetic that wraps.
    An empty frame has no close time to shift and nothing to refuse.

    Raises:
        ConfigError: If the offset itself is not a representable Int64,
            or if applying it to the smallest or largest close time in
            ``frame`` would leave the range.

    """
    if not _INT64_MIN <= offset_seconds <= _INT64_MAX:
        logger.warning(
            "Rejecting a horizon of {}s: outside the Int64 close-time range.",
            abs(offset_seconds),
        )
        raise ConfigError(
            f"A horizon of {abs(offset_seconds)}s is outside the Int64 range a "
            "close_time is held in, so no counterpart close time could be formed."
        )

    if frame.height == 0:
        return

    close_time = frame.get_column(CLOSE_TIME_COLUMN)
    extremes = (int(close_time.min()), int(close_time.max()))  # type: ignore[arg-type]
    for extreme in extremes:
        shifted = extreme + offset_seconds
        if not _INT64_MIN <= shifted <= _INT64_MAX:
            logger.warning(
                "Rejecting a shift of {}s: close time {} would move to {}.",
                offset_seconds,
                extreme,
                shifted,
            )
            raise ConfigError(
                f"Shifting close_time {extreme} by {offset_seconds}s leaves the "
                f"Int64 range: {shifted} is not a representable close time."
            )


def require_alignable_frame(frame: pl.DataFrame, *, offset_seconds: int) -> None:
    """Refuse, up front, every frame this module cannot align by close time.

    Runs shortest-first so each rule can fire on its own and report the
    most specific reason: the columns have to exist before their dtypes
    can be read, the dtypes have to be right before a null count or a
    duplicate means anything, and both have to hold before the frame's
    own extremes can be shifted.

    Args:
        frame: The frame a return is about to be composed onto.
        offset_seconds: The signed shift the caller's horizon implies for
            this direction.

    Raises:
        ConfigError: If a required column is absent or carries the wrong
            dtype, if ``close_time`` holds a null or a duplicate, or if
            the shift would leave the Int64 range.

    """
    _require_columns(frame)
    _require_close_time_key(frame)
    _require_representable_shift(frame, offset_seconds=offset_seconds)


def shifted_close_times(frame: pl.DataFrame, *, offset_seconds: int) -> pl.Series:
    """Return every row's close time moved by ``offset_seconds``.

    Args:
        frame: A frame carrying an ``Int64`` ``close_time`` column.
        offset_seconds: The signed offset to apply.

    Returns:
        One shifted close time per row, in row order, as ``Int64``.

    """
    return frame.get_column(CLOSE_TIME_COLUMN) + offset_seconds


def counterpart_closes(frame: pl.DataFrame, *, offset_seconds: int) -> pl.Series:
    """Look up each row's close at ``close_time + offset_seconds``.

    An exact equality join, never a shift, a nearest match, or a fill --
    see this module's docstring for why. The left side's row order is
    preserved, so the returned series lines up with ``frame`` positionally
    whatever order its rows arrived in.

    Args:
        frame: A frame carrying ``close_time`` and ``close``.
        offset_seconds: How far to look, negative to look back and
            positive to look forward.

    Returns:
        One close per row of ``frame``, in row order. A row whose
        counterpart close time is absent from ``frame`` yields null.

    """
    lookup = frame.select(
        pl.col(CLOSE_TIME_COLUMN).alias(_COUNTERPART_KEY),
        pl.col(CLOSE_COLUMN).alias(_COUNTERPART_CLOSE),
    )
    keys = shifted_close_times(frame, offset_seconds=offset_seconds).rename(
        _COUNTERPART_KEY
    )
    joined = pl.DataFrame([keys]).join(
        lookup, on=_COUNTERPART_KEY, how="left", maintain_order="left"
    )
    return joined.get_column(_COUNTERPART_CLOSE)
