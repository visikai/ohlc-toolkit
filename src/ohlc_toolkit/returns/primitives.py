"""Returns over window frames, composed as columns onto a frame.

These are mechanisms, not a recipe. There is no default horizon, no
horizon grid, no label, no barrier, and no position sizing anywhere in
this module: a caller states one horizon and one formula per call, and
gets back a new frame carrying one more column than it handed over.
Composing several is composing several calls.

Backward is the causal one
--------------------------

A backward return at the row whose information became available at
``close_time = t`` relates that row's close to the close of the row at
exactly ``t - H``. Both closes were known by ``t``, so the value is a
feature: it may be read at ``t`` without knowing anything that had not
happened yet. That is a property of the definition, not of the
implementation, and it survives a gap because the counterpart is found
by close time rather than by counting rows -- see
:mod:`ohlc_toolkit.returns.alignment`.

The two formulas
----------------

:attr:`ReturnMethod.SIMPLE` is ``numerator / denominator - 1`` and
:attr:`ReturnMethod.LOG` is ``ln(numerator / denominator)``, over the
pair ``(close(t), close(t - H))``. Neither is a default: ``method`` is a
required argument, because a column of numbers whose formula has to be
guessed from its magnitude is worse than no column. The name of the
column records the choice too, so the formula survives being written to
disk and read back by somebody else.

The log return is taken of the ratio rather than as a difference of two
logarithms. ``ln(a) - ln(b)`` subtracts two numbers that are nearly equal
whenever the price barely moved, which is most of the time, and loses
most of its significant digits doing so; the ratio of two nearby doubles
is computed to within half an ulp and stays that way through ``ln``.
"""

from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.returns.alignment import (
    CLOSE_COLUMN,
    counterpart_closes,
    resolve_horizon,
)
from ohlc_toolkit.temporal import ConfigError, Duration, validate_horizon_duration

logger = get_logger(__name__)


@unique
class ReturnMethod(Enum):
    """Which formula relates the two closes a return is taken over.

    Recorded explicitly on every call and written into the output column
    name, so the formula behind a column of numbers is never something a
    reader has to infer.

    Attributes:
        SIMPLE: ``numerator / denominator - 1``.
        LOG: ``ln(numerator / denominator)``.

    """

    SIMPLE = "simple"
    LOG = "log"


def _require_method(method: ReturnMethod) -> ReturnMethod:
    """Reject anything that is not a :class:`ReturnMethod` member.

    Args:
        method: The candidate method, of any type.

    Returns:
        ``method`` unchanged.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`.

    """
    if not isinstance(method, ReturnMethod):
        logger.warning("Rejecting non-ReturnMethod method: {!r}", method)
        raise ConfigError(f"method must be a ReturnMethod, got {type(method).__name__}")
    return method


def backward_return_column(method: ReturnMethod, horizon: Duration | str) -> str:
    """Name the column :func:`add_backward_returns` writes.

    The name carries the direction, the formula, and the horizon in its
    canonical spelling, so two spellings of one horizon name one column
    and two horizons never name the same one.

    Args:
        method: The formula the column holds.
        horizon: The horizon ``H``, as a
            :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string.

    Returns:
        The column name, for example ``"backward_return_simple_1m"``.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, or if
            the horizon is not a strictly positive duration.

    """
    _require_method(method)
    return f"backward_return_{method.value}_{validate_horizon_duration(horizon)}"


def _return_values(
    numerator: pl.Series, denominator: pl.Series, method: ReturnMethod
) -> pl.Series:
    """Apply one of the two formulas elementwise to two aligned closes.

    Args:
        numerator: The later close of each pair, positionally aligned
            with ``denominator``.
        denominator: The earlier close of each pair.
        method: Which formula to apply.

    Returns:
        One ``Float64`` value per row.

    """
    ratio = numerator / denominator
    if method is ReturnMethod.SIMPLE:
        return ratio - 1.0
    return ratio.log()


def add_backward_returns(
    frame: pl.DataFrame,
    *,
    horizon: Duration | str,
    cadence: Duration | str,
    method: ReturnMethod,
) -> pl.DataFrame:
    """Add the causal return over ``H`` to a window frame.

    The value at the row closing at ``t`` relates that row's close to the
    close of the row at exactly ``t - H``, located by close-time equality
    rather than by stepping back a number of rows. A row whose
    counterpart close time is absent from ``frame`` gets a null: no
    nearest match, no as-of fallback, no fill.

    Both closes were known by ``t``, so the column is a feature: it may be
    read at ``t`` without knowing anything later.

    Never mutates ``frame``, never sorts it, and never reads or alters any
    column other than ``close_time`` and ``close``.

    Args:
        frame: A window frame such as
            :func:`~ohlc_toolkit.windows.engine.compute_windows` produces,
            carrying at least an ``Int64`` ``close_time`` and a
            ``Float64`` ``close``.
        horizon: The horizon ``H``, as a
            :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string. Strictly positive, and a whole multiple of
            ``cadence``.
        cadence: The cadence the frame's rows are emitted at, in the same
            two spellings.
        method: Which formula to apply. Required: there is no default.

    Returns:
        A new frame: ``frame``'s columns unchanged and in their original
        order, followed by the column :func:`backward_return_column`
        names.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, or if
            the horizon or cadence fails
            :func:`~ohlc_toolkit.returns.alignment.resolve_horizon`.

    """
    _require_method(method)
    resolved = resolve_horizon(horizon, cadence)
    column = backward_return_column(method, resolved)

    counterpart = counterpart_closes(frame, offset_seconds=-resolved.total_seconds)
    values = _return_values(frame.get_column(CLOSE_COLUMN), counterpart, method)

    logger.debug("Adding {!r} over {} row(s).", column, frame.height)
    return frame.with_columns(values.rename(column))
