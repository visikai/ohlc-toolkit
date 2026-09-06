"""What a feature is called, and the two counts it has to report.

A field that VARIES within a frame goes in the column name; a field
CONSTANT across the artifact goes in the manifest. That single rule is
why `rsi_p14_w21m` says the indicator, the family, the period and the
window and says nothing about the history start, the traded threshold or
the schedule ids -- those are the same for every column in the artifact,
so repeating them in each name would be noise that could disagree with
the manifest.

The name is DERIVED from the identity rather than passed beside it, so a
column and the record describing it cannot drift: there is nowhere for a
second spelling to live.

One exception to the varies/constant rule is made for stability. The
family character is in the name even though a given artifact carries one
family, so that the day a dense family ships, no phased column is
renamed. A rename is a break for every consumer holding a stored frame,
and paying one character now is cheaper than paying that later.
"""

import re
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import ConfigError, Duration, coerce_duration
from ohlc_toolkit.temporal.echo import bounded_echo

logger = get_logger(__name__)

#: The name the derived-output contract bans outright. A feature column
#: carrying it would reintroduce the ambiguity `close_time` exists to
#: remove, so it is refused wherever it appears rather than only as a
#: whole name.
BANNED_NAME_PART = "timestamp"

#: An indicator's own name: lowercase letters and digits, starting with a
#: letter. No underscore, because the underscore is what separates the
#: three parts of a column name and a name containing one could not be
#: parsed back.
_INDICATOR_PATTERN = re.compile(r"[a-z][a-z0-9]*")

_COLUMN_PATTERN = re.compile(
    r"(?P<indicator>[a-z][a-z0-9]*)_(?P<family>[a-z])(?P<period>[0-9]+)_w(?P<window>.+)"
)


@unique
class FeatureFamily(Enum):
    """How a feature reads its inputs.

    Attributes:
        PHASED: The `L` non-overlapping windows ending at `t`, `t - W`,
            `t - 2W`, and so on.
        DENSE: Every window ending at or before `t`. Nothing implements
            this yet; it is representable so that shipping it renames no
            existing column.

    """

    PHASED = "p"
    DENSE = "d"


@unique
class NormalizationClass(Enum):
    """What kind of comparability a feature's values already have.

    A primitive declares its class rather than a caller guessing it,
    because the answer follows from how the number is constructed and not
    from what it looks like in one sample.

    Attributes:
        BOUNDED_BY_CONSTRUCTION: The value lies in a fixed range whatever
            the input -- a ratio of a part to its whole, an index on
            `[0, 100]`.
        STATIONARIZED: The value is a difference or a log ratio, so its
            scale does not follow the price level.
        EMPIRICALLY_NORMALIZED: The value is comparable only after being
            scaled by something measured from the data.

    """

    BOUNDED_BY_CONSTRUCTION = "bounded_by_construction"
    STATIONARIZED = "stationarized"
    EMPIRICALLY_NORMALIZED = "empirically_normalized"


@dataclass(frozen=True)
class FeatureIdentity:
    """Everything that varies between the feature columns of one artifact.

    Attributes:
        indicator: The indicator's own name, lowercase alphanumeric.
        family: How it reads its inputs.
        period: The lookback count `L`.
        window: The window duration `W`.
        normalization: What kind of comparability the values already
            have. Carried by the record and NOT by the column name: it
            does not vary between the columns of one feature, and the
            rule is that the name holds what varies. A primitive declares
            it, because the answer follows from how the number is built.

    """

    indicator: str
    family: FeatureFamily
    period: int
    window: Duration
    normalization: NormalizationClass

    def __post_init__(self) -> None:
        """Check every field, however this was constructed.

        Raises:
            ConfigError: If the indicator name is not lowercase
                alphanumeric starting with a letter, if it carries the
                banned name part, if the family is not a
                :class:`FeatureFamily`, if the period is not a strictly
                positive int, if the window is not a strictly positive
                duration, or if the normalization is not a
                :class:`NormalizationClass`.

        """
        object.__setattr__(self, "indicator", _validated_indicator(self.indicator))
        if not isinstance(self.family, FeatureFamily):
            logger.warning(
                "Rejecting non-FeatureFamily family: {}", type(self.family).__name__
            )
            raise ConfigError(
                f"family must be a FeatureFamily, got {type(self.family).__name__}"
            )
        object.__setattr__(self, "period", _validated_period(self.period))
        object.__setattr__(self, "window", _validated_window(self.window))
        if not isinstance(self.normalization, NormalizationClass):
            logger.warning(
                "Rejecting non-NormalizationClass normalization: {}",
                type(self.normalization).__name__,
            )
            raise ConfigError(
                f"normalization must be a NormalizationClass, got "
                f"{type(self.normalization).__name__}"
            )

    @property
    def column_name(self) -> str:
        """The column this feature writes, derived rather than declared.

        `{indicator}_{family}{period}_w{window}`, with the window spelled
        by `Duration.__str__` -- the same canonical form the parquet
        filenames, the manifest fields and the return-column horizons
        already use, so one duration has one spelling everywhere.
        """
        return f"{self.indicator}_{self.family.value}{self.period}_w{self.window}"

    @classmethod
    def parse(cls, column: str, *, normalization: NormalizationClass) -> Self:
        """Recover the identity a column name was derived from.

        The normalization class is an ARGUMENT, not something recovered:
        the name does not carry it, by design, so a reader parsing a
        stored column supplies it from the manifest that recorded it.
        Defaulting it here would invent the one field the name cannot
        vouch for.

        Args:
            column: A column name as :attr:`column_name` produces.
            normalization: The class the manifest recorded for this
                feature.

        Returns:
            The identity it names.

        Raises:
            ConfigError: If the name does not have the three parts, names
                no known family, carries a non-integer period, spells a
                window that does not parse, or carries the banned name
                part anywhere in it.

        """
        _require_no_banned_part(column)
        match = _COLUMN_PATTERN.fullmatch(column)
        if match is None:
            logger.warning(
                "Rejecting unparsable feature column: {}", bounded_echo(column)
            )
            raise ConfigError(
                f"{bounded_echo(column)} is not a feature column name; the shape is "
                "indicator_familyperiod_wwindow, as in rsi_p14_w21m."
            )
        try:
            family = FeatureFamily(match["family"])
        except ValueError as error:
            logger.warning(
                "Rejecting unknown feature family: {}", bounded_echo(match["family"])
            )
            raise ConfigError(
                f"{bounded_echo(match['family'])} names no feature family; the "
                f"families are {[member.value for member in FeatureFamily]}."
            ) from error
        return cls(
            indicator=match["indicator"],
            family=family,
            period=int(match["period"]),
            window=_parsed_window(match["window"]),
            normalization=normalization,
        )


def _parsed_window(spelling: str) -> Duration:
    """Parse the window part of a column name, refusing what will not parse.

    ``coerce_duration`` raises :class:`ConfigError` for the shapes it
    knows about, and a bare ``ValueError`` for one it does not: a numeric
    component of more than 4300 digits trips CPython's integer-parsing
    limit inside it. This entry point takes a column name off a stored
    artifact and documents ``ConfigError``, so the leak is closed here
    rather than left for a caller to discover.

    Raises:
        ConfigError: For any window spelling this cannot turn into a
            duration.

    """
    try:
        return coerce_duration(spelling)
    except ValueError as error:
        # ConfigError does not derive from ValueError, so the taxonomy's
        # own refusals pass through here untouched and only the leak is
        # converted.
        logger.warning("Rejecting an unparsable window: {}", bounded_echo(spelling))
        raise ConfigError(
            f"{bounded_echo(spelling)} is not a duration this can parse."
        ) from error


def effective_history(period: int, window: Duration | str) -> Duration:
    """How far back one tick's inputs reach: ``L * W``.

    The phased windows do not overlap, so the span really is the product.

    Args:
        period: The lookback count `L`.
        window: The window duration `W`.

    Returns:
        The span one value is computed from.

    Raises:
        ConfigError: If either argument is unusable.

    """
    return Duration(_validated_period(period) * _validated_window(window).total_seconds)


def effective_n(
    history_range: Duration | str, window: Duration | str, period: int
) -> int:
    """Count the INDEPENDENT BLOCKS a range holds.

    Non-overlapping windows within the range, divided by the lookback --
    so it counts how many times a feature's whole input span fits in the
    data, not how many rows there are.

    This is NOT an effective sample size. A statistical effective N
    accounts for the autocorrelation between blocks and would be a
    smaller, model-dependent number; this is a count of blocks and
    nothing more. The two are recorded under different names because
    reading one as the other would overstate how much independent
    evidence a feature has.

    Args:
        history_range: How much history the recipe covers.
        window: The window duration `W`.
        period: The lookback count `L`.

    Returns:
        The block count, which is zero when the range holds fewer than
        one whole span.

    Raises:
        ConfigError: If any argument is unusable.

    """
    span = _validated_window(history_range, label="history_range").total_seconds
    window_seconds = _validated_window(window).total_seconds
    return span // window_seconds // _validated_period(period)


def _validated_indicator(value: object) -> str:
    """Return an indicator name, rejecting anything unusable."""
    if not isinstance(value, str):
        logger.warning("Rejecting non-str indicator: {}", type(value).__name__)
        raise ConfigError(f"indicator must be a str, got {type(value).__name__}")
    _require_no_banned_part(value)
    if _INDICATOR_PATTERN.fullmatch(value) is None:
        logger.warning("Rejecting malformed indicator: {}", bounded_echo(value))
        raise ConfigError(
            f"An indicator name must be lowercase letters and digits starting with "
            f"a letter, got {bounded_echo(value)}; an underscore in particular "
            "would make the column name unparsable."
        )
    return value


def _require_no_banned_part(value: str) -> None:
    """Refuse a name carrying the banned part anywhere in it.

    Raises:
        ConfigError: If it does.

    """
    if BANNED_NAME_PART in value:
        logger.warning(
            "Rejecting a name carrying the banned part: {}", bounded_echo(value)
        )
        raise ConfigError(
            f"{bounded_echo(value)} carries {BANNED_NAME_PART!r}, which derived "
            "outputs never use: close_time is the time key, and the ambiguity is "
            "the whole reason for that."
        )


def _validated_period(value: object) -> int:
    """Return a period count, rejecting anything that is not a positive int."""
    if isinstance(value, bool) or not isinstance(value, int):
        logger.warning("Rejecting non-int period: {}", type(value).__name__)
        raise ConfigError(f"period must be an int, got {type(value).__name__}")
    if value <= 0:
        logger.warning("Rejecting non-positive period: {}", value)
        raise ConfigError(f"period must be strictly positive, got {value}")
    return value


def _validated_window(value: Duration | str, *, label: str = "window") -> Duration:
    """Coerce a duration and refuse a zero one, saying which one it was.

    The label is not decoration. :func:`effective_n` takes two durations,
    and without it both ``("0s", "3m", 3)`` and ``("1d", "0s", 3)``
    produced the same message naming neither.
    """
    duration = coerce_duration(value)
    if duration.total_seconds == 0:
        logger.warning("Rejecting a zero {}.", label)
        raise ConfigError(f"{label} must be strictly positive, got 0s.")
    return duration
