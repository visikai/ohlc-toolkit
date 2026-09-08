"""A pure decomposition of an Int64 Unix-seconds clock, and its cyclic columns.

Three scalar-or-expression functions decompose ``t`` -- an Int64 count of
seconds since the Unix epoch -- into where it falls inside a day and
inside a week:

- :func:`second_of_day` -- seconds since the most recent UTC midnight.
- :func:`second_of_week` -- seconds since the most recent Monday
  00:00 UTC.
- :func:`day_of_week` -- which day that is, Monday ``0`` through Sunday
  ``6``.

The clock is UTC BY CONSTRUCTION. There is no timezone parameter to get
wrong because there is no timezone: ``t`` is a plain count of seconds,
and every definition here is arithmetic on that count. Nothing in this
module reads a calendar, a locale, a venue's trading hours, or a
holiday -- ``ohlc_toolkit.temporal.calendar`` knows what second of the
week a timestamp falls in and nothing about what that means to any
market. A caller that wants a session boundary, a holiday, or a
non-UTC trading day composes that on top; it is not this module's
concern.

Unix time 0 is a Thursday, so a day-of-week that counted straight from
the epoch would put Monday at ``3`` instead of ``0``. The offset baked
into :func:`second_of_week` corrects for exactly that, once, so every
caller gets Monday at ``0`` without restating the shift themselves.

:func:`add_calendar_columns` turns ``second_of_day`` and
``second_of_week`` into four columns a model can read directly: each
pair is a point on the unit circle, so the boundary where a raw count
would jump -- day 86399 back to 0, week second 604799 back to 0 -- is
not a discontinuity here. The two values nearest that boundary sit next
to each other on the circle exactly as they sit next to each other in
time.
"""

import math
from typing import overload

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal.columns import require_absent_columns
from ohlc_toolkit.temporal.echo import bounded_echo
from ohlc_toolkit.temporal.errors import ConfigError

logger = get_logger(__name__)

# Exact whole seconds -- a day and a week, the two cycles this module
# names. Neither accounts for a leap second: Unix time does not either.
SECONDS_PER_DAY = 86_400
SECONDS_PER_WEEK = 604_800

# Unix time 0 falls on a Thursday. Adding three days before reducing mod
# a week rotates that Thursday to where a Monday belongs -- second 0 --
# so second_of_week and day_of_week read Monday 00:00 UTC as the start
# of the cycle without either caller or module carrying a lookup table.
_THURSDAY_TO_MONDAY_OFFSET = 3 * SECONDS_PER_DAY

# The default input column, and the four columns add_calendar_columns
# writes. The default matches the column name every other stage in this
# package reads the clock from.
DEFAULT_SECONDS_COLUMN = "close_time"
CAL_MOD_SIN = "cal_mod_sin"
CAL_MOD_COS = "cal_mod_cos"
CAL_DOW_SIN = "cal_dow_sin"
CAL_DOW_COS = "cal_dow_cos"


@overload
def second_of_day(t: int) -> int: ...
@overload
def second_of_day(t: pl.Expr) -> pl.Expr: ...
def second_of_day(t: int | pl.Expr) -> int | pl.Expr:
    """Seconds since the most recent UTC midnight.

    ``t`` is a plain Int64 count of seconds; there is no timezone to
    supply because the clock is UTC by construction. Correct for any
    ``t``, negative or positive, because Python's ``%`` and polars'
    integer ``%`` both floor rather than truncate.

    Args:
        t: A Unix-seconds instant, as a Python ``int`` or a polars
            ``Expr`` over an integer column.

    Returns:
        ``t mod 86400``, in ``[0, 86400)``, in the same form ``t`` was
        given in.

    """
    return t % SECONDS_PER_DAY


@overload
def second_of_week(t: int) -> int: ...
@overload
def second_of_week(t: pl.Expr) -> pl.Expr: ...
def second_of_week(t: int | pl.Expr) -> int | pl.Expr:
    """Seconds since the most recent Monday 00:00 UTC.

    ``t`` is a plain Int64 count of seconds; there is no timezone to
    supply because the clock is UTC by construction.

    Args:
        t: A Unix-seconds instant, as a Python ``int`` or a polars
            ``Expr`` over an integer column.

    Returns:
        ``(t + 3 * 86400) mod 604800``, in ``[0, 604800)``, zero at a
        Monday 00:00 UTC, in the same form ``t`` was given in.

    """
    return (t + _THURSDAY_TO_MONDAY_OFFSET) % SECONDS_PER_WEEK


@overload
def day_of_week(t: int) -> int: ...
@overload
def day_of_week(t: pl.Expr) -> pl.Expr: ...
def day_of_week(t: int | pl.Expr) -> int | pl.Expr:
    """Which day of the week ``t`` falls on, Monday ``0`` through Sunday ``6``.

    ``t`` is a plain Int64 count of seconds; there is no timezone to
    supply because the clock is UTC by construction.

    Args:
        t: A Unix-seconds instant, as a Python ``int`` or a polars
            ``Expr`` over an integer column.

    Returns:
        ``second_of_week(t) // 86400``, in ``[0, 6]``, in the same form
        ``t`` was given in.

    """
    return second_of_week(t) // SECONDS_PER_DAY  # type: ignore[operator, return-value]


def _require_seconds_column(frame: pl.DataFrame, column: str) -> None:
    """Check that ``column`` exists in ``frame`` and carries an integer dtype.

    A ``Datetime`` column is refused rather than converted: this module
    knows no timezone, so silently reading a ``Datetime``'s underlying
    integer would silently pick one, and picking one is exactly the
    decision this module exists to never make.

    Args:
        frame: The frame to check.
        column: The column that must carry Unix seconds.

    Raises:
        ConfigError: If ``column`` is absent from ``frame``, or is
            present but not of an integer dtype.

    """
    if column not in frame.columns:
        logger.warning("Rejecting a frame missing column {}.", bounded_echo(column))
        raise ConfigError(
            f"{bounded_echo(column)} is not a column of this frame; pass the name "
            "of an Int64 Unix-seconds column, or add one before calling this."
        )
    dtype = frame.schema[column]
    if not dtype.is_integer():
        logger.warning("Rejecting non-integer {}: {}", column, bounded_echo(dtype))
        raise ConfigError(
            f"{column} must be an integer column of Unix seconds, got "
            f"{bounded_echo(dtype)}; a Datetime column is never converted here "
            "-- convert it to Int64 Unix seconds before calling this, choosing "
            "the timezone yourself."
        )


def add_calendar_columns(
    frame: pl.DataFrame, column: str = DEFAULT_SECONDS_COLUMN
) -> pl.DataFrame:
    """Add four cyclic UTC calendar columns derived from an Int64 seconds column.

    ``column`` is read as a plain count of Unix seconds; the clock is UTC
    by construction and this function knows no venue, no timezone, and no
    holiday. Two pairs are added, each a point on the unit circle rather
    than a raw count, so the wraparound at midnight or at the week
    boundary is not a discontinuity for whatever reads them next:

    - ``cal_mod_sin``, ``cal_mod_cos`` -- the second of the UTC day, as
      ``sin`` and ``cos`` of ``2*pi * second_of_day(t) / 86400``.
    - ``cal_dow_sin``, ``cal_dow_cos`` -- the second of the UTC week
      (zero at Monday 00:00 UTC), as ``sin`` and ``cos`` of
      ``2*pi * second_of_week(t) / 604800``.

    Each pair satisfies ``sin**2 + cos**2 == 1`` to float tolerance on
    every row, and both pairs are invariant under adding any integer
    multiple of their own cycle (86400 or 604800 seconds) to ``t``.

    Never mutates ``frame``, and never reads or alters any column other
    than ``column``.

    Args:
        frame: A frame carrying an Int64 (or other integer-dtype) column
            of Unix seconds.
        column: The name of that column. Defaults to ``"close_time"``.

    Returns:
        A new frame: ``frame``'s columns unchanged and in their original
        order, followed by ``cal_mod_sin``, ``cal_mod_cos``,
        ``cal_dow_sin`` and ``cal_dow_cos``.

    Raises:
        ConfigError: If ``column`` is absent from ``frame``, is present
            but not of an integer dtype, or if ``frame`` already carries
            any of the four output columns.

    """
    _require_seconds_column(frame, column)
    require_absent_columns(
        frame,
        (CAL_MOD_SIN, CAL_MOD_COS, CAL_DOW_SIN, CAL_DOW_COS),
        remedy="adding them again would overwrite values this call did not compute.",
    )

    tau = 2.0 * math.pi
    mod_angle = second_of_day(pl.col(column)).cast(pl.Float64) * (tau / SECONDS_PER_DAY)
    dow_angle = second_of_week(pl.col(column)).cast(pl.Float64) * (
        tau / SECONDS_PER_WEEK
    )

    logger.debug(
        "Adding {!r}, {!r}, {!r} and {!r} over {} row(s).",
        CAL_MOD_SIN,
        CAL_MOD_COS,
        CAL_DOW_SIN,
        CAL_DOW_COS,
        frame.height,
    )
    return frame.with_columns(
        mod_angle.sin().alias(CAL_MOD_SIN),
        mod_angle.cos().alias(CAL_MOD_COS),
        dow_angle.sin().alias(CAL_DOW_SIN),
        dow_angle.cos().alias(CAL_DOW_COS),
    )
