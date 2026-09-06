"""Cutler's RSI: momentum as the share of movement that was upward.

The formula is ``100 - 100 / (1 + G / D)``, where ``G`` and ``D`` are the
SIMPLE means of the positive and the negative close-to-close changes over
the ``P`` changes that ``P + 1`` phased windows hold. Every change is
counted in both means, as a zero when it has the other sign, so the two
always average over ``P``.

That common divisor cancels: ``(Sg/P) / ((Sg/P) + (Sd/P))`` is
``Sg / (Sg + Sd)``. The implementation therefore sums and never divides
by ``P``, and the difference between a mean and a total is stated here
rather than written as an operation no test could reach -- dividing both
sides by ``P`` changes no reading, so an implementation that dropped it
by accident would look identical to one that meant to. What the period
really controls is the NUMBER OF CHANGES, and that is fixed by the
lookback the harness was resolved with.

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

The two one-sided conventions are the exact values at the exact
boundaries, and they are NOT symmetric just beyond them. Float64 rounding
makes ``100.0`` reachable from a nonzero ``D`` -- ``G = 1.0`` with
``D = 1.11e-16`` reads exactly ``100.0`` -- while the low end keeps its
resolution: ``G = 5e-324`` with ``D = 1.0`` reads ``4.94e-322`` rather
than ``0.0``. A reading of exactly ``100.0`` therefore means "no falls,
or falls too small to register against the rises"; a reading of exactly
``0.0`` means no rises at all.
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
    has_missing_input,
    indicator_identity,
    require_finite_columns,
    require_phased_inputs,
)

logger = get_logger(__name__)

# The reading a run of unchanged closes takes: the symmetric limit of the
# formula, approached from either side. Not published: the convention is
# documented above and observable in the values, and no caller has asked
# to compare against it by name.
_NEUTRAL_RSI: Final = 50.0

_FULL_SCALE: Final = 100.0
_CLOSE: Final = "close"

# The decomposition's own column names, prefixed so that they cannot
# collide with a phased field the frame already carries.
_HOLE: Final = "_incomplete"
_UP: Final = "_up"
_DOWN: Final = "_down"


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
            DataValidationError: If either change total is non-finite,
                which finite closes whose differences do not overflow
                cannot produce.

        """
        require_phased_inputs(self, phased, period=period, fields=(_CLOSE,))
        identity = indicator_identity(self, phased, period=period)
        parts = phased.frame.select(_decomposed())
        # On the TOTALS, not on the reading: a finite up total over an
        # infinite down total is `0.0`, in range and indistinguishable
        # from the only-falls convention.
        require_finite_columns(parts, (_UP, _DOWN), computing=identity.column_name)
        return parts.select(_reading().alias(identity.column_name)).to_series()


def _decomposed() -> list[pl.Expr]:
    """Split each tick's changes into its upward and its downward total.

    Kept as columns rather than folded into one expression so that the
    finiteness check below can look at them. An infinity here does not
    survive into the reading: a finite ``up`` over an infinite ``down``
    is ``0.0``, which is in range, finite, and indistinguishable from the
    only-falls convention.

    Returns:
        The hole flag and the two totals, as named expressions over a
        `close` list column.

    """
    # `list.reverse` first: the harness orders each list newest FIRST, and
    # a difference taken in that order is the negative of the change.
    changes = pl.col(_CLOSE).list.reverse().list.eval(pl.element().diff().drop_nulls())
    return [
        has_missing_input(_CLOSE).alias(_HOLE),
        changes.list.eval(pl.element().clip(lower_bound=0.0)).list.sum().alias(_UP),
        changes.list.eval((-pl.element()).clip(lower_bound=0.0))
        .list.sum()
        .alias(_DOWN),
    ]


def _reading() -> pl.Expr:
    """Build the reading in the closed form its conventions fall out of.

    ``100 - 100 / (1 + G / D)`` is ``100 * (G / (G + D))`` for any ``G``
    and ``D`` that are not both zero -- the same number by algebra, but it
    divides by a sum of non-negative totals rather than by ``D``, so the
    two one-sided conventions are arithmetic rather than branches: a run
    with no falls has ``D = 0`` and reads ``100 * (G / G)``. Only the
    both-zero case is genuinely undefined, and it is the one branch
    below. The spelled-out form would divide by zero at ``D = 0`` and
    manufacture an infinity for the next expression to turn into a NaN.

    The parenthesis around ``G / (G + D)`` is load-bearing and was put
    there by a failing property test. Scaling first,
    ``100 * G / (G + D)``, rounds the product before it divides: with
    ``D = 0`` and ``G = 256842.5`` it returns ``100.00000000000001``,
    which is outside the range this indicator claims to be bounded to.
    Dividing first gives exactly ``1.0`` whenever the two are equal --
    correctly-rounded addition is monotonic, so ``fl(G + D) >= G`` and the
    quotient can never exceed one -- and the bound holds by construction
    rather than by luck.

    Returns:
        The expression, over the columns :func:`_decomposed` produces.

    """
    movement = pl.col(_UP) + pl.col(_DOWN)
    return (
        pl.when(pl.col(_HOLE))
        .then(None)
        .when(movement == 0.0)
        .then(pl.lit(_NEUTRAL_RSI))
        .otherwise(_FULL_SCALE * (pl.col(_UP) / movement))
        .cast(pl.Float64)
    )
