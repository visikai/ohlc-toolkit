"""Source profiles describing how a provider's raw frame maps to candles."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum, unique
from types import MappingProxyType
from typing import Self

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

    def matches(self, dtype: pl.DataType) -> bool:
        """Report whether a polars dtype satisfies this declared column kind.

        Args:
            dtype: The actual polars dtype observed in a raw frame.

        Returns:
            True if ``dtype`` is a numeric dtype of this kind.

        """
        if self is ColumnKind.INTEGER:
            return dtype.is_integer()
        return dtype.is_float()


def _zero_duration() -> Duration:
    """Return the zero Duration used as the default declared phase.

    A module-level factory, rather than a bare ``Duration(0)`` default:
    dataclass defaults must not be mutable-looking call expressions
    (Ruff RUF009 flags a direct call as a field default), so
    ``dataclasses.field(default_factory=...)`` is used instead.
    """
    return Duration(0)


@dataclass(frozen=True)
class SourceProfile:
    """Describes how a provider's raw frame maps to finalized candles.

    A profile is a declaration, not a transform: it never touches data
    itself. It is consumed by :mod:`ohlc_toolkit.source.validation` and
    :mod:`ohlc_toolkit.source.reader` to check and read a provider's raw
    frame consistently.

    Construction: :meth:`create` is the documented boundary constructor,
    accepting ``Duration | str`` for ``cadence`` and ``phase`` and
    coercing both via the temporal helpers. Direct construction (calling
    ``SourceProfile(...)``) expects ``cadence`` and ``phase`` to already
    be ``Duration`` instances; every internal read of these fields uses
    them directly, with no boundary coercion left at the use site.

    Attributes:
        name: The source identifier, e.g. ``"bitstamp-btcusd-1m"``.
        timestamp_column: The raw column holding each candle's Unix-second
            interval open. Must be declared ``ColumnKind.INTEGER`` in
            ``raw_schema``: a floating timestamp column can truncate
            sub-second data silently on cast, which is exactly the kind
            of silent corruption this package refuses to allow.
        availability: The rule governing when a candle becomes usable.
        raw_schema: The required raw columns, mapped to their expected
            numeric kind. Must include ``timestamp_column``. Stored as a
            read-only mapping (a defensive copy of whatever was passed
            in), so mutating the caller's original dict after
            construction never leaks into the profile. Because this
            field is a mapping, instances of this class are not
            hashable.
        cadence: The source's fixed candle duration, always a strictly
            positive ``Duration`` (see
            :func:`~ohlc_toolkit.temporal.validate_cadence`).
        phase: The declared offset of the source's timestamp grid from
            the plain epoch grid: every timestamp must satisfy
            ``timestamp % cadence == phase``. Declared, never inferred
            from data — a uniformly shifted frame is corruption, not a
            new convention. Zero (the default) means round cadence
            boundaries. Must be strictly smaller than ``cadence``.

    """

    name: str
    timestamp_column: str
    availability: Availability
    raw_schema: Mapping[str, ColumnKind]
    cadence: Duration
    phase: Duration = field(default_factory=_zero_duration)

    @classmethod
    def create(  # noqa: PLR0913 - one keyword per declared profile field
        cls,
        *,
        name: str,
        timestamp_column: str,
        availability: Availability,
        raw_schema: Mapping[str, ColumnKind],
        cadence: Duration | str,
        phase: Duration | str = "0s",
    ) -> Self:
        """Build a profile from the public, string-accepting boundary types.

        This is the documented entry point for constructing a
        :class:`SourceProfile`: ``cadence`` and ``phase`` accept either a
        :class:`~ohlc_toolkit.temporal.Duration` or a compact duration
        string, coerced here, once, before the frozen dataclass itself is
        built.

        Args:
            name: The source identifier, e.g. ``"bitstamp-btcusd-1m"``.
            timestamp_column: The raw column holding each candle's
                Unix-second interval open.
            availability: The rule governing when a candle becomes usable.
            raw_schema: The required raw columns, mapped to their
                expected numeric kind.
            cadence: The source's fixed candle duration, as a Duration or
                a compact duration string.
            phase: The declared grid-phase offset, as a Duration or a
                compact duration string. Defaults to zero.

        Returns:
            The constructed, validated profile.

        Raises:
            ConfigError: If any invariant checked by ``__post_init__``
                fails.

        """
        return cls(
            name=name,
            timestamp_column=timestamp_column,
            availability=availability,
            raw_schema=raw_schema,
            cadence=coerce_duration(cadence),
            phase=coerce_duration(phase),
        )

    def __post_init__(self) -> None:
        """Normalize fields and validate cross-field invariants.

        Raises:
            ConfigError: If ``cadence`` cannot be resolved to a strictly
                positive Duration, ``phase`` cannot be resolved to a
                Duration strictly smaller than ``cadence``, ``name`` or
                ``timestamp_column`` is empty, ``raw_schema`` is empty,
                ``timestamp_column`` is not itself a key of
                ``raw_schema``, or ``timestamp_column`` is not declared
                ``ColumnKind.INTEGER``.

        """
        # Defensive normalization: `.create()` is the documented boundary
        # and already coerces, but direct construction (used internally,
        # e.g. by BITSTAMP_BTCUSD_1M below) may still pass either type,
        # and validate_cadence's positivity check must run regardless.
        object.__setattr__(self, "cadence", validate_cadence(self.cadence))
        object.__setattr__(self, "phase", coerce_duration(self.phase))
        object.__setattr__(self, "raw_schema", MappingProxyType(dict(self.raw_schema)))

        cadence_seconds = self.cadence.total_seconds
        phase_seconds = self.phase.total_seconds
        if phase_seconds >= cadence_seconds:
            logger.warning(
                "Rejecting source profile {!r}: phase {}s is not smaller than "
                "cadence {}s.",
                self.name,
                phase_seconds,
                cadence_seconds,
            )
            raise ConfigError(
                f"Source profile phase must be strictly smaller than the "
                f"cadence, got {phase_seconds}s >= {cadence_seconds}s."
            )

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
        if self.raw_schema[self.timestamp_column] is not ColumnKind.INTEGER:
            logger.warning(
                "Rejecting source profile {!r}: timestamp column {!r} must be "
                "declared ColumnKind.INTEGER, not {!r}.",
                self.name,
                self.timestamp_column,
                self.raw_schema[self.timestamp_column],
            )
            raise ConfigError(
                f"timestamp_column {self.timestamp_column!r} must be declared "
                f"ColumnKind.INTEGER, not "
                f"{self.raw_schema[self.timestamp_column]!r}: a floating "
                "timestamp column can silently truncate sub-second data."
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
        cadence_seconds = self.cadence.total_seconds
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
    # Published timestamps land on round minute boundaries.
    phase=Duration(0),
)
