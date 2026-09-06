"""Log volume ratio: this window's volume against its recent baseline.

The natural log of the current window's volume over the MEDIAN volume of
the ``P`` windows before it. The current window is excluded from its own
baseline, so a spike cannot shrink the ratio it is supposed to produce.

The median rather than the mean because Bitcoin volume is heavy-tailed
and a single outlying window would drag a mean baseline toward itself,
flattening exactly the events the indicator exists to show. The log
rather than the plain ratio because the plain ratio lives on the same
heavy right tail -- doubling and halving are the same size of event and
should be the same distance from zero -- and because it then composes
with the other log ratios in the slice.

The baseline uses the indicator's own ``P``, so no fourth field appears
in the identity and the column name still says everything that varies.

Division by zero cannot arise from a well-formed frame: a present window
has traded seconds above the recipe's threshold and therefore volume
above zero. A zero reaching this primitive is a contract violation
upstream, and it is refused loudly rather than turned into a null, a
negative infinity, or a guess.
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

_VOLUME: Final = "volume"
_HOLE: Final = "_incomplete"
_CURRENT: Final = "_current"
_BASELINE: Final = "_baseline"

_VOLUME_CONTRACT: Final = (
    "a present window has traded seconds at or above the recipe's threshold "
    "and therefore a volume above zero, so a zero or negative one is a "
    "violation upstream rather than something to read through."
)


@dataclass(frozen=True, slots=True)
class LogVolumeRatio:
    """Log of the current volume over the median of the ``P`` before it.

    Stationarized: a ratio of two volumes in the same units, logged, so
    the scale divides out and nothing is estimated from a sample.
    """

    name: ClassVar[str] = "logvolratio"
    normalization: ClassVar[NormalizationClass] = NormalizationClass.STATIONARIZED

    def lookback(self, period: int) -> int:
        """``P + 1``: the current window plus the ``P`` its baseline is.

        Args:
            period: The period ``P``.

        Returns:
            The lookback count ``L``.

        Raises:
            ConfigError: If the period is not a strictly positive int.

        """
        return _validated_period(period) + 1

    def values(self, phased: PhasedLookback, *, period: int) -> pl.Series:
        """Compute one reading per tick from the tick's ``P + 1`` volumes.

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
            DataValidationError: If a present volume is not positive, or
                if an intermediate is non-finite.

        """
        require_phased_inputs(self, phased, period=period, fields=(_VOLUME,))
        require_positive_inputs(phased, fields=(_VOLUME,), reason=_VOLUME_CONTRACT)
        identity = indicator_identity(self, phased, period=period)
        parts = phased.frame.select(_decomposed())
        require_finite_columns(
            parts, (_CURRENT, _BASELINE), computing=identity.column_name
        )
        return parts.select(_reading().alias(identity.column_name)).to_series()


def _decomposed() -> list[pl.Expr]:
    """Split each tick into its current volume and its baseline.

    ``list.slice(1)`` is where the exclusion lives: the harness orders
    each list newest first, so dropping element zero drops the current
    window and leaves exactly the ``P`` before it.

    Returns:
        The hole flag, the current volume, and the median baseline.

    """
    return [
        has_missing_input(_VOLUME).alias(_HOLE),
        pl.col(_VOLUME).list.first().alias(_CURRENT),
        pl.col(_VOLUME).list.slice(1).list.median().alias(_BASELINE),
    ]


def _reading() -> pl.Expr:
    """Build the reading from the columns :func:`_decomposed` produces.

    Returns:
        The expression.

    """
    return (
        pl.when(pl.col(_HOLE))
        .then(None)
        .otherwise((pl.col(_CURRENT) / pl.col(_BASELINE)).log())
        .cast(pl.Float64)
    )
