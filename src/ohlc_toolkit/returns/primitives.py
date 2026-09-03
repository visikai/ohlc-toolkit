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

Forward is not, and says so twice
---------------------------------

A forward return at the same row describes the interval ``[t, t + H]``:
it relates the close at ``t + H`` to the close at ``t``. Its value is not
available at ``t``. It is available at ``t + H``, and consuming it as
though it were available earlier is the most productive way there is to
build a model that cannot exist -- so this module makes that mistake
loud rather than possible, in two independent ways:

- the value column is named with a ``forward_`` prefix, which travels
  with it into a column list, a correlation matrix, a parquet file, and
  a feature-importance plot;
- every forward value carries its own ``available_at`` column, holding
  ``t + H`` in the same dtype as ``close_time``, so the instant the value
  may first be used is data rather than folklore. It is a sidecar of that
  specific value column, named by suffixing it, so two horizons or two
  formulas on one frame each keep their own.

``available_at`` is total: every row states one, including the rows whose
return came back null because the counterpart close time was absent.
Nulling the availability of a null value would lose the one fact still
standing -- that this row's ``H``-ahead outcome, whatever it turns out to
be, belongs to ``t + H`` -- and would make a null in that column mean two
different things. Availability is a property of the horizon and the row's
own close time, both always present; whether a value was found there is a
separate question, answered by a separate column.

The two directions are one relation read from opposite ends, and that is
checkable rather than merely claimed: the forward return at ``t`` over
``H`` and the backward return at ``t + H`` over ``H`` are the same
expression over the same two closes, so wherever both rows exist the two
values agree bit for bit.

The two formulas
----------------

:attr:`ReturnMethod.SIMPLE` is ``numerator / denominator - 1`` and
:attr:`ReturnMethod.LOG` is ``ln(numerator / denominator)``, over the
pair ``(close(t), close(t - H))`` looking back and ``(close(t + H),
close(t))`` looking forward. Neither is a default: ``method`` is a
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
    shifted_close_times,
)
from ohlc_toolkit.temporal import ConfigError, Duration, validate_horizon_duration

logger = get_logger(__name__)

# Appended to a forward value column's name to name its availability
# sidecar. Deriving one name from the other is what pairs them: two
# horizons, or two formulas over one horizon, each keep their own
# availability column instead of contending for a shared one.
_AVAILABLE_AT_SUFFIX = "_available_at"


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


def _return_column(
    direction: str, method: ReturnMethod, horizon: Duration | str
) -> str:
    """Assemble a return column name from its direction, formula, and horizon.

    The horizon is spelled canonically, so two spellings of one horizon
    name one column and two horizons never name the same one.

    Args:
        direction: ``"backward"`` or ``"forward"``.
        method: The formula the column holds.
        horizon: The horizon ``H``.

    Returns:
        The column name.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, or if
            the horizon is not a strictly positive duration.

    """
    _require_method(method)
    return f"{direction}_return_{method.value}_{validate_horizon_duration(horizon)}"


def backward_return_column(method: ReturnMethod, horizon: Duration | str) -> str:
    """Name the column :func:`add_backward_returns` writes.

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
    return _return_column("backward", method, horizon)


def forward_return_column(method: ReturnMethod, horizon: Duration | str) -> str:
    """Name the value column :func:`add_forward_returns` writes.

    The ``forward_`` prefix is load-bearing rather than decorative: it is
    what tells a reader, a column list, or a feature-importance plot that
    this is not a feature.

    Args:
        method: The formula the column holds.
        horizon: The horizon ``H``, as a
            :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string.

    Returns:
        The column name, for example ``"forward_return_simple_1m"``.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, or if
            the horizon is not a strictly positive duration.

    """
    return _return_column("forward", method, horizon)


def forward_available_at_column(method: ReturnMethod, horizon: Duration | str) -> str:
    """Name the availability column that accompanies a forward value column.

    Args:
        method: The formula the accompanied value column holds.
        horizon: The horizon ``H``, as a
            :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string.

    Returns:
        The value column's name with :data:`_AVAILABLE_AT_SUFFIX`
        appended, for example
        ``"forward_return_simple_1m_available_at"``.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, or if
            the horizon is not a strictly positive duration.

    """
    return f"{forward_return_column(method, horizon)}{_AVAILABLE_AT_SUFFIX}"


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


def add_forward_returns(
    frame: pl.DataFrame,
    *,
    horizon: Duration | str,
    cadence: Duration | str,
    method: ReturnMethod,
) -> pl.DataFrame:
    """Add the return over ``[t, t + H]``, and the instant it arrives, to a frame.

    The value at the row closing at ``t`` relates the close of the row at
    exactly ``t + H`` to that row's own close, located by close-time
    equality rather than by stepping forward a number of rows. A row whose
    counterpart close time is absent from ``frame`` gets a null.

    THE VALUE IS NOT AVAILABLE AT ``t``. It is available at ``t + H``,
    which the second column states for every row, including rows whose
    value is null. Read this module's docstring before consuming either
    column: a forward return used as a feature is a model that cannot
    exist.

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
        order, followed by the column :func:`forward_return_column` names
        and then the one :func:`forward_available_at_column` names.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, or if
            the horizon or cadence fails
            :func:`~ohlc_toolkit.returns.alignment.resolve_horizon`.

    """
    _require_method(method)
    resolved = resolve_horizon(horizon, cadence)
    value_column = forward_return_column(method, resolved)
    available_at_column = forward_available_at_column(method, resolved)

    counterpart = counterpart_closes(frame, offset_seconds=resolved.total_seconds)
    values = _return_values(counterpart, frame.get_column(CLOSE_COLUMN), method)
    available_at = shifted_close_times(frame, offset_seconds=resolved.total_seconds)

    logger.debug(
        "Adding {!r} and {!r} over {} row(s).",
        value_column,
        available_at_column,
        frame.height,
    )
    return frame.with_columns(
        values.rename(value_column), available_at.rename(available_at_column)
    )
