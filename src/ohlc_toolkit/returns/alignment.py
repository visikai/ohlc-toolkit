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

# Column names used only inside the counterpart join, never returned.
_COUNTERPART_KEY = "__counterpart_close_time"
_COUNTERPART_CLOSE = "__counterpart_close"


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
