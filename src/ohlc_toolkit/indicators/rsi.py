"""Cutler's RSI: momentum as the share of movement that was upward.

The formula is ``100 - 100 / (1 + G / D)``, where ``G`` and ``D`` are the
SIMPLE means of the positive and the negative close-to-close changes over
the ``P`` changes that ``P + 1`` phased windows hold. Every change is
counted in both means, as a zero when it has the other sign, so the two
always average over ``P``.

Cutler's and not Wilder's, recorded here so that nobody has to rediscover
it: Wilder's smoothing is a recursion with infinite memory, so its value
depends on where the series was seeded and two artifacts built from
different history starts disagree on the same window. Cutler's reads
exactly ``P`` changes, which is what makes a phased window's value a
function of that window's inputs and nothing else. Wilder's may be added
later as a separately named indicator carrying its own warm-up and
restart rule; it is never substituted for this one.

The three boundary conventions, which exist so that a reading is never
undefined and never non-finite:

* only rises (``D = 0``, ``G > 0``) reads ``100``;
* only falls (``G = 0``, ``D > 0``) reads ``0``;
* every one of the ``P`` changes exactly zero reads ``50``, the symmetric
  limit, because the inputs are all present and the momentum is genuinely
  neutral. That is not the same as a window with too little trading in
  it, which the harness has already turned into a null input.
"""

from dataclasses import dataclass
from typing import ClassVar, Final

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.indicators.frames import PhasedLookback

# `_validated_period` is this package's one period validator, shared
# inside the package rather than published: a second public name for the
# same check is a 2.0 commitment that no caller has asked for.
from ohlc_toolkit.indicators.identity import NormalizationClass, _validated_period
from ohlc_toolkit.indicators.primitives import (
    indicator_identity,
    require_phased_inputs,
)
from ohlc_toolkit.temporal import DataValidationError

logger = get_logger(__name__)

# The reading a run of unchanged closes takes: the symmetric limit of the
# formula, approached from either side. Not published: the convention is
# documented above and observable in the values, and no caller has asked
# to compare against it by name.
_NEUTRAL_RSI: Final = 50.0

_FULL_SCALE: Final = 100.0
_CLOSE: Final = "close"


@dataclass(frozen=True, slots=True)
class CutlersRSI:
    """The relative strength index over ``P`` simple close-to-close changes.

    Bounded by construction: every value is a share of a non-negative
    total, so the range is `[0, 100]` whatever the prices were, and no
    sample is needed to make two of them comparable.
    """

    #: Class attributes rather than fields: a primitive's name and class
    #: are what it IS, and an instance that could be constructed with a
    #: different name would be a different indicator wearing this one's
    #: arithmetic.
    name: ClassVar[str] = "rsi"
    normalization: ClassVar[NormalizationClass] = (
        NormalizationClass.BOUNDED_BY_CONSTRUCTION
    )

    def lookback(self, period: int) -> int:
        """``P + 1`` windows, because ``P`` changes need one more close.

        Args:
            period: The period ``P``.

        Returns:
            The lookback count ``L``.

        Raises:
            ConfigError: If the period is not a strictly positive int.

        """
        return _validated_period(period) + 1

    def values(self, phased: PhasedLookback, *, period: int) -> pl.Series:
        """Compute one reading per tick from the tick's ``P + 1`` closes.

        Args:
            phased: The harness output, resolved at this primitive's own
                lookback.
            period: The period ``P``.

        Returns:
            One `Float64` value per tick in `[0, 100]`, null exactly where
            the tick's inputs were, named as the identity record derives
            it.

        Raises:
            ConfigError: If the period is unusable, or if the frame was
                not assembled for this lookback.
            DataValidationError: If any reading is non-finite, which
                finite closes cannot produce.

        """
        require_phased_inputs(self, phased, period=period, fields=(_CLOSE,))
        identity = indicator_identity(self, phased, period=period)
        values = phased.frame.select(
            _reading(period).alias(identity.column_name)
        ).to_series()
        _require_finite(values, identity.column_name)
        return values


def _reading(period: int) -> pl.Expr:
    """Build the reading in the closed form its conventions fall out of.

    ``100 - 100 / (1 + G / D)`` is ``100 * G / (G + D)`` for any ``G`` and
    ``D`` that are not both zero -- the same number by algebra, but it
    divides by a sum of non-negative means rather than by ``D``, so the
    two one-sided conventions are arithmetic rather than branches: a run
    with no falls has ``D = 0`` and reads ``100 * G / G``. Only the
    both-zero case is genuinely undefined, and it is the one branch
    below. The spelled-out form would divide by zero at ``D = 0`` and
    manufacture an infinity for the next expression to turn into a NaN.

    The parenthesis around ``gain / movement`` is load-bearing and was
    put there by a failing property test. Scaling first,
    ``100 * gain / movement``, rounds the product before it divides:
    with ``D = 0`` and ``G = 256842.5 / 3`` it returns
    ``100.00000000000001``, which is outside the range this indicator
    claims to be bounded to. Dividing first gives exactly ``1.0``
    whenever the two are equal, and a quotient that can never exceed one
    otherwise, so the bound holds by construction rather than by luck.

    Args:
        period: The period ``P``, the divisor of both means.

    Returns:
        The expression, over a `close` list column of ``P + 1`` closes.

    """
    # `list.reverse` first: the harness orders each list newest FIRST, and
    # a difference taken in that order is the negative of the change.
    changes = pl.col(_CLOSE).list.reverse().list.eval(pl.element().diff().drop_nulls())
    gain = changes.list.eval(pl.element().clip(lower_bound=0.0)).list.sum() / period
    loss = changes.list.eval((-pl.element()).clip(lower_bound=0.0)).list.sum() / period
    movement = gain + loss
    return (
        pl.when(
            pl.col(_CLOSE).list.drop_nulls().list.len() != pl.col(_CLOSE).list.len()
        )
        .then(None)
        .when(movement == 0.0)
        .then(pl.lit(_NEUTRAL_RSI))
        .otherwise(_FULL_SCALE * (gain / movement))
        .cast(pl.Float64)
    )


def _require_finite(values: pl.Series, column: str) -> None:
    """Refuse a non-finite reading rather than writing one into a frame.

    Unreachable from finite closes, which is the point: the source layer
    already refuses non-finite prices, so a NaN or an infinity here means
    a caller assembled the lookback by hand out of values no reader would
    have accepted. Derived frames do not manufacture what the source
    refuses.

    Args:
        values: The computed column.
        column: Its name, for the refusal.

    Raises:
        DataValidationError: If any value is not finite.

    """
    present = values.drop_nulls()
    if present.is_finite().all():
        return
    logger.error("Refusing a non-finite reading in {}.", column)
    raise DataValidationError(
        f"{column} holds {(~present.is_finite()).sum()} non-finite value(s); "
        "the closes they were computed from are not finite."
    )
