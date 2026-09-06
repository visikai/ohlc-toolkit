"""Price to moving average: where the close sits against its own recent mean.

The natural log of the current window's close over the simple mean close
of the ``P`` phased windows ending at the tick, the current one INCLUDED.
Positive means the price is above its recent average, negative below, and
zero exactly on it.

This replaces a moving-average log SLOPE, which was shown to be nearly
proportional to the ``P x W`` backward log return this package already
computes -- two names for one measurement. Where the mean MOVED and where
the price SITS relative to it are different facts, and the second is not
a linear function of one return.

``P`` identical closes read exactly ``0.0``. At ``P = 1`` the mean is the
current close itself, so every reading is identically zero: the
degenerate case that is the reason a lookback slice starts above one.
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

_CLOSE: Final = "close"
_HOLE: Final = "_incomplete"
_CURRENT: Final = "_current"
_MEAN: Final = "_mean"

_PRICE_CONTRACT: Final = (
    "a windowed candle's close is a traded price, so a present window's "
    "close is strictly positive."
)


@dataclass(frozen=True, slots=True)
class PriceToMovingAverage:
    """Log of the current close over the mean close of ``P`` windows.

    Stationarized: a ratio of two prices in the same units, logged.
    """

    name: ClassVar[str] = "mapos"
    normalization: ClassVar[NormalizationClass] = NormalizationClass.STATIONARIZED

    def lookback(self, period: int) -> int:
        """``P``: the mean includes the current window, so no extra one.

        Args:
            period: The period ``P``.

        Returns:
            The lookback count ``L``.

        Raises:
            ConfigError: If the period is not a strictly positive int.

        """
        return _validated_period(period)

    def values(self, phased: PhasedLookback, *, period: int) -> pl.Series:
        """Compute one reading per tick from the tick's ``P`` closes.

        Args:
            phased: The harness output, resolved at this primitive's own
                lookback.
            period: The period ``P``.

        Returns:
            One `Float64` value per tick, null exactly where the tick's
            inputs were, named as the identity record derives it.

        Raises:
            ConfigError: If the period is unusable, or if the frame was
                not assembled for this lookback.
            DataValidationError: If a present close is not positive, or
                if an intermediate is non-finite.

        """
        require_phased_inputs(self, phased, period=period, fields=(_CLOSE,))
        require_positive_inputs(phased, fields=(_CLOSE,), reason=_PRICE_CONTRACT)
        identity = indicator_identity(self, phased, period=period)
        parts = phased.frame.select(_decomposed())
        require_finite_columns(parts, (_CURRENT, _MEAN), computing=identity.column_name)
        return parts.select(_reading().alias(identity.column_name)).to_series()


def _decomposed() -> list[pl.Expr]:
    """Split each tick into its current close and the mean of all ``P``.

    No slice: the current window is IN its own mean, which is what makes
    ``P`` identical closes read exactly zero rather than reading the gap
    between the newest close and the ``P - 1`` behind it.

    Returns:
        The hole flag, the current close, and the mean close.

    """
    return [
        has_missing_input(_CLOSE).alias(_HOLE),
        pl.col(_CLOSE).list.first().alias(_CURRENT),
        pl.col(_CLOSE).list.mean().alias(_MEAN),
    ]


def _reading() -> pl.Expr:
    """Build the reading from the columns :func:`_decomposed` produces.

    Returns:
        The expression.

    """
    return (
        pl.when(pl.col(_HOLE))
        .then(None)
        .otherwise((pl.col(_CURRENT) / pl.col(_MEAN)).log())
        .cast(pl.Float64)
    )
