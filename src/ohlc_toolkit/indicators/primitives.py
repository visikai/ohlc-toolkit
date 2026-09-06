"""What an input indicator IS, and the one path that writes one.

A primitive is the smallest thing this package computes from a phased
lookback: it takes an integer period ``P``, reads the ``L`` phased windows
the harness assembled for each tick, and emits one column. The protocol
below fixes what every primitive must declare so that the recipe layer
can read those declarations off any of them without knowing which it
holds -- its short name, the lookback count its period implies, the
normalization class its values already have, and the column of values
itself.

Two things are deliberately NOT the primitive's to state. It does not
report effective history: that is ``L * W``, and ``W`` belongs to the
grid the harness resolved, so a primitive multiplying it out would be a
second place for the same number to be computed differently. And it does
not name its own column: the name is derived from
:class:`~ohlc_toolkit.indicators.identity.FeatureIdentity`, which is what
keeps the name and the manifest from disagreeing.
"""

from collections.abc import Collection
from typing import Protocol

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.indicators.frames import PhasedLookback
from ohlc_toolkit.indicators.identity import (
    FeatureFamily,
    FeatureIdentity,
    NormalizationClass,
)
from ohlc_toolkit.temporal import ConfigError, Duration, require_absent_columns
from ohlc_toolkit.temporal.echo import bounded_echo

logger = get_logger(__name__)


class IndicatorPrimitive(Protocol):
    """The four things every input indicator declares about itself.

    A protocol rather than a base class: a primitive is a bundle of
    declarations plus one computation, it inherits no state, and the
    recipe layer only ever reads it structurally. Nothing here is
    registered anywhere -- a caller holds the primitive it wants.
    """

    @property
    def name(self) -> str:
        """The short indicator name, as it appears in a column name."""
        ...

    @property
    def normalization(self) -> NormalizationClass:
        """What kind of comparability the values already have."""
        ...

    def lookback(self, period: int) -> int:
        """How many phased windows ``L`` a period ``P`` consumes.

        A function rather than an offset constant, because the relation
        is the indicator's own: Cutler's RSI needs ``P + 1`` windows to
        take ``P`` differences, while a price-to-moving-average reading
        needs exactly ``P``.

        Args:
            period: The period ``P``.

        Returns:
            The lookback count ``L``.

        Raises:
            ConfigError: If the period is unusable.

        """
        ...

    def values(self, phased: PhasedLookback, *, period: int) -> pl.Series:
        """Compute the column, named as the identity record derives it.

        Args:
            phased: The harness output, whose grid must have been
                resolved with this primitive's own lookback.
            period: The period ``P``.

        Returns:
            One ``Float64`` value per tick, null where the tick's inputs
            were, named ``{indicator}_{family}{period}_w{window}``.

        Raises:
            ConfigError: If the period is unusable, or if the grid was
                resolved with a different lookback.

        """
        ...


def indicator_identity(
    primitive: IndicatorPrimitive, phased: PhasedLookback, *, period: int
) -> FeatureIdentity:
    """Build the identity record a primitive's column is named from.

    The family is phased for every primitive this package ships, because
    the phased lookback is the only harness there is. The dense family
    exists in the record so that shipping it renames nothing; it does not
    exist here.

    Args:
        primitive: The primitive.
        phased: The harness output, which carries the window ``W``.
        period: The period ``P``.

    Returns:
        The record, from which the column name is derived.

    Raises:
        ConfigError: If any field is unusable.

    """
    return FeatureIdentity(
        indicator=primitive.name,
        family=FeatureFamily.PHASED,
        period=period,
        window=Duration(phased.grid.window_seconds),
        normalization=primitive.normalization,
    )


def require_phased_inputs(
    primitive: IndicatorPrimitive,
    phased: PhasedLookback,
    *,
    period: int,
    fields: Collection[str],
) -> int:
    """Check the frame really carries what the primitive is about to read.

    Three ways harness output can be wrong for a primitive, refused
    together because a primitive that checked one and assumed the other
    two would compute a number rather than refuse:

    1. The grid was resolved for a different lookback. A frame assembled
       at ``L = 14`` handed to an indicator that needs 15 would take
       fourteen differences and divide by fifteen.
    2. A field the primitive reads is absent.
    3. A list is not ``L`` long. The harness only ever emits a complete
       list or a null one, but a :class:`PhasedLookback` is a plain
       record a caller can build, and a short list would be averaged over
       the length the primitive expected rather than the length it got.

    Args:
        primitive: The primitive about to read the frame.
        phased: The harness output.
        period: The period ``P``.
        fields: The phased fields the primitive reads.

    Returns:
        The lookback the grid and the primitive agree on.

    Raises:
        ConfigError: For any of the three.

    """
    wanted = primitive.lookback(period)
    if phased.grid.lookback != wanted:
        logger.warning(
            "Rejecting harness output resolved at lookback {} for {} at period {}.",
            phased.grid.lookback,
            bounded_echo(primitive.name),
            period,
        )
        raise ConfigError(
            f"{bounded_echo(primitive.name)} at period {period} reads {wanted} "
            f"phased window(s); this frame was resolved for "
            f"{phased.grid.lookback}. Re-run the lookback with lookback={wanted}."
        )
    missing = [field for field in fields if field not in phased.frame.columns]
    if missing:
        echoed = ", ".join(bounded_echo(field) for field in missing)
        logger.warning("Rejecting harness output missing field(s): {}", echoed)
        raise ConfigError(
            f"{bounded_echo(primitive.name)} reads column(s) {echoed}, which this "
            f"frame does not carry."
        )
    ragged = [
        field
        for field, wrong in phased.frame.select(
            (pl.col(field).list.len() != wanted).any().alias(field) for field in fields
        )
        .row(0, named=True)
        .items()
        if wrong
    ]
    if ragged:
        echoed = ", ".join(bounded_echo(field) for field in ragged)
        logger.warning(
            "Rejecting phased list(s) that are not {} long: {}", wanted, echoed
        )
        raise ConfigError(
            f"Column(s) {echoed} carry list(s) that are not {wanted} long; every "
            f"phased list is the lookback's length or null."
        )
    return wanted


def add_indicator(
    phased: PhasedLookback, primitive: IndicatorPrimitive, *, period: int
) -> pl.DataFrame:
    """Append one primitive's column to the harness frame it was computed from.

    The frame keeps its list columns: a second primitive reading the same
    lookback appends beside the first, and the collision guard refuses the
    same one twice rather than overwriting a column a caller may already
    have written.

    Args:
        phased: The harness output.
        primitive: The primitive to compute.
        period: The period ``P``.

    Returns:
        The harness frame with one column appended.

    Raises:
        ConfigError: If the frame already carries the column, if the grid
            was resolved for a different lookback, or if any identity
            field is unusable.

    """
    values = primitive.values(phased, period=period)
    require_absent_columns(
        phased.frame,
        (values.name,),
        remedy="compute this indicator once, or write it to a frame of its own.",
    )
    return phased.frame.with_columns(values)
