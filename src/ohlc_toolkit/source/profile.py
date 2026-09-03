"""Source profiles describing how a provider's raw frame maps to candles."""

from dataclasses import dataclass
from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import (
    ConfigError,
    Duration,
    coerce_duration,
    validate_cadence,
)

logger = get_logger(__name__)


@unique
class Availability(Enum):
    """When a candle becomes usable relative to its own interval.

    Only one rule exists today: a candle becomes usable once its close
    time has passed. Modelling this as an enum, rather than a boolean
    flag, means a future rule (for example, availability after some fixed
    publication lag past close) is a new member, not a repurposed flag.
    """

    CLOSE_TIME = "close_time"


@unique
class ColumnKind(Enum):
    """The numeric kind expected for a raw source column."""

    INTEGER = "integer"
    FLOATING = "floating"


def column_kind_matches(kind: ColumnKind, dtype: pl.DataType) -> bool:
    """Report whether a polars dtype satisfies a declared column kind.

    Args:
        kind: The numeric kind a source profile declares for a column.
        dtype: The actual polars dtype observed in a raw frame.

    Returns:
        True if ``dtype`` is a numeric dtype of the requested kind.

    """
    if kind is ColumnKind.INTEGER:
        return dtype.is_integer()
    return dtype.is_float()


@dataclass(frozen=True)
class SourceProfile:
    """Describes how a provider's raw frame maps to finalized candles.

    A profile is a declaration, not a transform: it never touches data
    itself. It is consumed by :mod:`ohlc_toolkit.source.validation` and
    :mod:`ohlc_toolkit.source.reader` to check and read a provider's raw
    frame consistently.

    Attributes:
        name: The source identifier, e.g. ``"bitstamp-btcusd-1m"``.
        timestamp_column: The raw column holding each candle's Unix-second
            interval open.
        availability: The rule governing when a candle becomes usable.
        raw_schema: The required raw columns, mapped to their expected
            numeric kind. Must include ``timestamp_column``.
        cadence: The source's fixed candle duration. The constructor
            accepts a :class:`~ohlc_toolkit.temporal.Duration` or a
            compact duration string; either way, the stored value is
            always a strictly positive ``Duration`` (see
            :func:`~ohlc_toolkit.temporal.validate_cadence`).

    """

    name: str
    timestamp_column: str
    availability: Availability
    raw_schema: dict[str, ColumnKind]
    # Declared as the constructor boundary type (Duration | str); always
    # normalized to a Duration in __post_init__. Read through
    # coerce_duration() at use sites so the narrower runtime type is
    # visible to the type checker without re-declaring this field.
    cadence: Duration | str

    def __post_init__(self) -> None:
        """Normalize the cadence and validate cross-field invariants.

        Raises:
            ConfigError: If ``cadence`` cannot be resolved to a strictly
                positive Duration, ``name`` or ``timestamp_column`` is
                empty, ``raw_schema`` is empty, or ``timestamp_column`` is
                not itself a key of ``raw_schema``.

        """
        object.__setattr__(self, "cadence", validate_cadence(self.cadence))

        if not self.name:
            logger.warning("Rejecting source profile with an empty name.")
            raise ConfigError("Source profile name must not be empty.")
        if not self.timestamp_column:
            logger.warning(
                "Rejecting source profile {!r} with an empty timestamp column.",
                self.name,
            )
            raise ConfigError("Source profile timestamp_column must not be empty.")
        if not self.raw_schema:
            logger.warning(
                "Rejecting source profile {!r} with an empty raw schema.", self.name
            )
            raise ConfigError(
                "Source profile raw_schema must declare at least one column."
            )
        if self.timestamp_column not in self.raw_schema:
            logger.warning(
                "Rejecting source profile {!r}: timestamp column {!r} is not a "
                "declared raw column.",
                self.name,
                self.timestamp_column,
            )
            raise ConfigError(
                f"timestamp_column {self.timestamp_column!r} must be a key of raw_schema."
            )

    def derive_interval_bounds(self, raw_frame: pl.DataFrame) -> pl.DataFrame:
        """Derive each row's half-open candle interval from its timestamp.

        Args:
            raw_frame: A raw source frame containing at least
                ``self.timestamp_column``, holding each candle's
                Unix-second interval open.

        Returns:
            A new two-column polars DataFrame with int64 columns
            ``open_time`` and ``close_time``, one row per input row, such
            that each candle spans the half-open range
            ``[open_time, close_time)``.

        """
        cadence_seconds = coerce_duration(self.cadence).total_seconds
        open_time = pl.col(self.timestamp_column).cast(pl.Int64)
        return raw_frame.select(
            open_time.alias("open_time"),
            (open_time + cadence_seconds).alias("close_time"),
        )


# The published Bitstamp minute-data files are six plain columns: a
# Unix-second timestamp plus open/high/low/close/volume. Reading the
# timestamp as each candle's interval OPEN (rather than its close) is an
# evidence-backed interpretation of the published files, not a documented
# guarantee from the publisher: consecutive rows are exactly 60 seconds
# apart, and the earliest published row's timestamp lands on a plain
# round minute boundary consistent with an opening tick, not a closing one.
BITSTAMP_BTCUSD_1M = SourceProfile(
    name="bitstamp-btcusd-1m",
    timestamp_column="timestamp",
    availability=Availability.CLOSE_TIME,
    raw_schema={
        "timestamp": ColumnKind.INTEGER,
        "open": ColumnKind.FLOATING,
        "high": ColumnKind.FLOATING,
        "low": ColumnKind.FLOATING,
        "close": ColumnKind.FLOATING,
        "volume": ColumnKind.FLOATING,
    },
    cadence=Duration.parse("1m"),
)
