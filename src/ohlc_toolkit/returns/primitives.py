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
loud rather than silent, in two independent ways:

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

:attr:`ReturnMethod.SIMPLE` is ``(numerator - denominator) /
denominator`` and :attr:`ReturnMethod.LOG` is ``log1p`` of that same
quotient -- the log return is literally the ``log1p`` of the simple
return -- over the pair ``(close(t), close(t - H))`` looking back and
``(close(t + H), close(t))`` looking forward. Neither is a default:
``method`` is a required argument, because a column of numbers whose
formula has to be guessed from its magnitude is worse than no column.
The name of the column records the choice too, so the formula survives
being written to disk and read back by somebody else.

The spelling matters most where returns are smallest, which is most of
the time. Two closes one cadence apart are usually nearly equal, so
their difference is EXACT (the subtraction of two doubles within a
factor of two of each other is itself a double), and the quotient then
carries about half an ulp. Every spelling that rounds the RATIO first --
``a / b - 1``, ``ln(a / b)``, or ``ln(a) - ln(b)`` -- parks a half-ulp
of error next to ``1``, and for a return of size ``x`` the subtraction
or the logarithm amplifies that error by ``1 / x``: about six
significant digits gone at ``x`` near ``1e-6``, measured at roughly
``1.7e-11`` relative error against a 60-digit reference where
``log1p((a - b) / b)`` measures near ``1e-17``. The tests pin the
emitted value to within ``1e-15`` of such a reference, which only the
difference-quotient spelling achieves.

Every emitted value is a finite float or null
---------------------------------------------

Real closes produce ordinary returns. A window frame's closes are not
guaranteed to be ordinary: the aggregator emits a null close for a window
holding no source candle, and nothing upstream of here promises a close
is positive, non-zero, or of a sane magnitude. Left alone, polars returns
``inf``, ``-inf``, or ``NaN`` from a division or a logarithm, and each of
those travels through a downstream fit as a number rather than as an
absence -- silently, and with no error to notice.

So the last thing every path here does is ask whether the value it
computed is finite, and emit null when it is not. One guard, applied for
one stated reason -- there is no real number to report -- rather than a
list of special cases to keep in step:

- a zero denominator: ``x / 0`` is ``+/-inf`` and ``0 / 0`` is ``NaN``;
- a null close on either side, which polars already propagates as null;
- a quotient that overflows to infinity, which two closes far enough
  apart in magnitude will do;
- a simple return at or below ``-1`` under ``LOG`` -- a non-positive
  price ratio, stated the other way -- where ``log1p(-1)`` is ``-inf``
  and anything below it is not real at all, so ``NaN``.

What is NOT nulled is anything the formula produced that happens to be a
real number. A close that fell to zero has a simple return of exactly
``-1``. A close so dwarfed by its counterpart that the quotient rounds
to ``-1`` has the same. A simple return
over a NEGATIVE denominator is finite too, and is reported: this step has
no opinion about the sign of a price, and forming one belongs upstream in
:func:`~ohlc_toolkit.source.validation.validate_source_frame`, where a
negative price is a finding about the data rather than a shrug about one
row.
"""

from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.returns.alignment import (
    CLOSE_COLUMN,
    counterpart_closes,
    require_alignable_frame,
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

# Column names used only inside the one-expression return computation,
# never returned. Both closes are renamed onto them because the two
# series arrive carrying whatever names their source columns had, and one
# of the two directions would otherwise hand over two columns called the
# same thing.
_NUMERATOR = "__numerator_close"
_DENOMINATOR = "__denominator_close"


@unique
class ReturnMethod(Enum):
    """Which formula relates the two closes a return is taken over.

    Recorded explicitly on every call and written into the output column
    name, so the formula behind a column of numbers is never something a
    reader has to infer.

    Attributes:
        SIMPLE: ``(numerator - denominator) / denominator``.
        LOG: ``log1p`` of the simple return -- ``ln`` of the price
            ratio, computed without rounding the ratio first.

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


def _require_absent_columns(frame: pl.DataFrame, columns: tuple[str, ...]) -> None:
    """Refuse to write over a column the frame already carries.

    Overwriting is never the intent here and would be undetectable after
    the fact: the replaced column keeps its name, its dtype, and its
    plausibility, having lost whatever the caller put there. The two
    horizons and the two formulas already name distinct columns, so the
    only way to reach this is to repeat a call or to have named a column
    the same thing by hand -- both of which are better reported than
    absorbed.

    Args:
        frame: The frame about to be written to.
        columns: The column names this call would add.

    Raises:
        ConfigError: If ``frame`` already carries any of ``columns``.

    """
    present = [name for name in columns if name in frame.columns]
    if present:
        logger.warning("Rejecting frame that already carries column(s): {}", present)
        raise ConfigError(
            f"The frame already carries the column(s) {present}; adding them again "
            "would overwrite values this call did not compute."
        )


def _return_values(
    numerator: pl.Series, denominator: pl.Series, method: ReturnMethod
) -> pl.Series:
    """Apply one of the two formulas elementwise, nulling what is not finite.

    The finiteness guard is the last step for every method, so a value
    this module cannot state as a real number is stated as an absence
    instead of as ``inf`` or ``NaN``. See the module docstring for which
    inputs reach it and which finite-but-surprising ones deliberately do
    not.

    Args:
        numerator: The later close of each pair, positionally aligned
            with ``denominator``.
        denominator: The earlier close of each pair.
        method: Which formula to apply.

    Returns:
        One ``Float64`` value per row: finite, or null.

    """
    simple = (pl.col(_NUMERATOR) - pl.col(_DENOMINATOR)) / pl.col(_DENOMINATOR)
    value = simple if method is ReturnMethod.SIMPLE else simple.log1p()
    pair = pl.DataFrame(
        [numerator.rename(_NUMERATOR), denominator.rename(_DENOMINATOR)]
    )
    return pair.select(
        pl.when(value.is_finite()).then(value).otherwise(None).cast(pl.Float64)
    ).to_series()


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
            two spellings. Trusted from the caller: it gates which
            horizons are accepted and is never verified against the
            frame's actual row spacing.
        method: Which formula to apply. Required: there is no default.

    Returns:
        A new frame: ``frame``'s columns unchanged and in their original
        order, followed by the column :func:`backward_return_column`
        names.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, if the
            horizon or cadence fails
            :func:`~ohlc_toolkit.returns.alignment.resolve_horizon`, if
            ``frame`` fails
            :func:`~ohlc_toolkit.returns.alignment.require_alignable_frame`,
            or if ``frame`` already carries a column this call would
            write.

    """
    _require_method(method)
    resolved = resolve_horizon(horizon, cadence)
    column = backward_return_column(method, resolved)
    require_alignable_frame(frame, offset_seconds=-resolved.total_seconds)
    _require_absent_columns(frame, (column,))

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
            two spellings. Trusted from the caller: it gates which
            horizons are accepted and is never verified against the
            frame's actual row spacing.
        method: Which formula to apply. Required: there is no default.

    Returns:
        A new frame: ``frame``'s columns unchanged and in their original
        order, followed by the column :func:`forward_return_column` names
        and then the one :func:`forward_available_at_column` names.

    Raises:
        ConfigError: If ``method`` is not a :class:`ReturnMethod`, if the
            horizon or cadence fails
            :func:`~ohlc_toolkit.returns.alignment.resolve_horizon`, if
            ``frame`` fails
            :func:`~ohlc_toolkit.returns.alignment.require_alignable_frame`,
            or if ``frame`` already carries a column this call would
            write.

    """
    _require_method(method)
    resolved = resolve_horizon(horizon, cadence)
    value_column = forward_return_column(method, resolved)
    available_at_column = forward_available_at_column(method, resolved)
    require_alignable_frame(frame, offset_seconds=resolved.total_seconds)
    _require_absent_columns(frame, (value_column, available_at_column))

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
