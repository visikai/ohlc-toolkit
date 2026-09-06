"""Relative range: how far a window travelled, as a fraction of its price.

The true range of a window is
``max(high - low, |high - prev_close|, |low - prev_close|)``, where
``prev_close`` is the close of the PREVIOUS phased window -- the window
one ``W`` earlier, not the previous row of any frame. Averaged over the
``P`` windows that have a predecessor and divided by the current window's
close, it is a scale-free measure of movement: stationarized, because a
series running from $5 to $100k+ makes any price-unit quantity
incomparable with itself a year later.

The period exists so that every indicator has one and the identity scheme
stays uniform. ``P = 1`` is the single-window true range and remains
expressible. Dividing by the close rather than taking a log of high over
low is what lets this compose with the other ratios in the slice.
"""

from dataclasses import dataclass
from typing import ClassVar, Final

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.indicators.frames import PhasedLookback
from ohlc_toolkit.indicators.identity import NormalizationClass, _validated_period
from ohlc_toolkit.indicators.primitives import (
    has_missing_input,
    indicator_identity,
    require_finite_columns,
    require_phased_inputs,
    require_positive_inputs,
)

logger = get_logger(__name__)

_HIGH: Final = "high"
_LOW: Final = "low"
_CLOSE: Final = "close"
_FIELDS: Final = (_HIGH, _LOW, _CLOSE)

_HOLE: Final = "_incomplete"
_RANGE: Final = "_mean_true_range"
_PRICE: Final = "_close"

_PRICE_CONTRACT: Final = (
    "a windowed candle's close is a traded price, so a present window's "
    "close is strictly positive."
)


@dataclass(frozen=True, slots=True)
class RelativeRange:
    """Mean true range over ``P`` windows, as a fraction of the current close.

    Stationarized: a ratio of two prices in the same units, so the scale
    divides out and two readings a decade apart are comparable without
    anything having been estimated from a sample.
    """

    name: ClassVar[str] = "relrange"
    normalization: ClassVar[NormalizationClass] = NormalizationClass.STATIONARIZED

    def lookback(self, period: int) -> int:
        """``P + 1`` windows, because the oldest one is only a predecessor.

        Args:
            period: The period ``P``.

        Returns:
            The lookback count ``L``.

        Raises:
            ConfigError: If the period is not a strictly positive int.

        """
        return _validated_period(period) + 1

    def values(self, phased: PhasedLookback, *, period: int) -> pl.Series:
        """Compute one reading per tick from the tick's ``P + 1`` windows.

        Args:
            phased: The harness output, resolved at this primitive's own
                lookback.
            period: The period ``P``.

        Returns:
            One `Float64` value per tick, never negative, null exactly
            where the tick's inputs were, named as the identity record
            derives it.

        Raises:
            ConfigError: If the period is unusable, or if the frame was
                not assembled for this lookback.
            DataValidationError: If a present close is not positive, or
                if an intermediate is non-finite.

        """
        require_phased_inputs(self, phased, period=period, fields=_FIELDS)
        require_positive_inputs(phased, fields=(_CLOSE,), reason=_PRICE_CONTRACT)
        identity = indicator_identity(self, phased, period=period)
        parts = phased.frame.select(_decomposed(period))
        require_finite_columns(parts, (_RANGE, _PRICE), computing=identity.column_name)
        return parts.select(_reading().alias(identity.column_name)).to_series()


def _decomposed(period: int) -> list[pl.Expr]:
    """Split each tick into its mean true range and its current close.

    The true range needs three fields at one phase and a fourth value
    from the phase behind it, and polars has no elementwise maximum
    between two list columns. The three slices are therefore PACKED into
    one list -- ``P`` highs, then ``P`` lows, then the ``P`` predecessor
    closes -- so that ``list.eval`` sees them as three aligned series and
    the maximum is an ordinary horizontal one.

    The identity used is ``max(high, prev) - min(low, prev)``, which is
    the three-candidate definition exactly rather than approximately.
    Given ``high >= low``: if ``prev`` lies between them the answer is
    ``high - low`` and the other two candidates are smaller; if ``prev``
    is above ``high`` it is ``prev - low``; if below ``low`` it is
    ``high - prev``. Subtraction is monotone, so the rounded values order
    the same way the exact ones do and no candidate can win by rounding.
    The tempting arithmetic-only form,
    ``clip(high - prev, 0) + clip(prev - low, 0)``, is NOT exact: where
    ``prev`` lies inside the bar it computes ``(high - prev) + (prev -
    low)``, which can differ from ``high - low`` in the last place.

    Args:
        period: The period ``P``, the number of windows with a
            predecessor.

    Returns:
        The hole flag, the mean true range, and the current close.

    """
    previous_close = (
        pl.col(_CLOSE).list.eval(pl.element().shift(-1)).list.slice(0, period)
    )
    packed = pl.concat_list(
        pl.col(_HIGH).list.slice(0, period),
        pl.col(_LOW).list.slice(0, period),
        previous_close,
    )
    highs = pl.element().slice(0, period)
    lows = pl.element().slice(period, period)
    previous = pl.element().slice(2 * period, period)
    true_range = packed.list.eval(
        pl.max_horizontal(highs, previous) - pl.min_horizontal(lows, previous)
    )
    return [
        has_missing_input(*_FIELDS).alias(_HOLE),
        true_range.list.mean().alias(_RANGE),
        pl.col(_CLOSE).list.first().alias(_PRICE),
    ]


def _reading() -> pl.Expr:
    """Build the reading from the columns :func:`_decomposed` produces.

    Returns:
        The expression.

    """
    return (
        pl.when(pl.col(_HOLE))
        .then(None)
        .otherwise(pl.col(_RANGE) / pl.col(_PRICE))
        .cast(pl.Float64)
    )
