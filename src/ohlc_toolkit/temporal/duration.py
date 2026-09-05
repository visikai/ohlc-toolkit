"""A first-class, exact-second Duration value type and its compact grammar."""

import re
from dataclasses import dataclass
from typing import Self

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal.echo import bounded_echo
from ohlc_toolkit.temporal.errors import ConfigError

logger = get_logger(__name__)

# Canonical unit seconds, largest first. This order drives both the
# required ordering when parsing and the greedy decomposition when
# formatting, so it is defined once and reused for both directions.
_UNIT_SECONDS: dict[str, int] = {
    "w": 604800,
    "d": 86400,
    "h": 3600,
    "m": 60,
    "s": 1,
}
_UNIT_RANK: dict[str, int] = {unit: rank for rank, unit in enumerate(_UNIT_SECONDS)}

# A duration string is one or more `<digits><unit>` components with no
# separators, signs, or whitespace anywhere. Units are matched
# case-sensitively so that, e.g., "3M" (a calendar month, not supported)
# is rejected rather than silently read as "3m" (three minutes).
# Digits are [0-9], NOT \d: \d matches every Unicode decimal digit (and
# int() converts them), which would admit inputs the ASCII canonical
# formatter can never emit — and that therefore can never round-trip.
_GRAMMAR_PATTERN = re.compile(r"^(?:[0-9]+[wdhms])+$")
_COMPONENT_PATTERN = re.compile(r"([0-9]+)([wdhms])")


@dataclass(frozen=True, order=True)
class Duration:
    """An immutable, exact duration measured in whole seconds.

    ``Duration`` is the internal canonical representation for every
    duration, cadence, and offset in this package: all arithmetic is exact
    integer seconds, so there is no accumulated floating-point drift and
    no ambiguity about the unit in scope.

    Attributes:
        total_seconds: The duration's length in whole seconds. Always a
            non-negative ``int``.

    """

    total_seconds: int

    def __post_init__(self) -> None:
        """Reject any value that is not a non-negative integer of seconds.

        Raises:
            ConfigError: If ``total_seconds`` is not an ``int`` (``bool``
                is rejected too, even though it is an ``int`` subtype), or
                is negative.

        """
        if isinstance(self.total_seconds, bool) or not isinstance(
            self.total_seconds, int
        ):
            logger.warning(
                "Rejecting non-integer duration seconds: {}",
                type(self.total_seconds).__name__,
            )
            raise ConfigError(
                "Duration seconds must be an int, got "
                f"{type(self.total_seconds).__name__}"
            )
        if self.total_seconds < 0:
            logger.warning(
                "Rejecting negative duration seconds: {}", self.total_seconds
            )
            raise ConfigError(
                f"Duration seconds must be non-negative, got {self.total_seconds}"
            )

    def __str__(self) -> str:
        """Render the canonical compact form, e.g. ``5400`` -> ``'1h30m'``."""
        return _format_seconds(self.total_seconds)

    def to_polars_index_count(self) -> str:
        r"""Format this duration as a polars index-count duration string.

        The result is ``f"{total_seconds}i"``: the polars syntax for a
        window measured in a plain integer count of index steps, for use
        with ``group_by_dynamic``/``rolling`` on an int64 unix-second
        column. Never use a temporal-unit string such as ``"1d"`` or
        ``"1w"`` for that kind of column: those units are calendar-aware
        in polars (variable month length, daylight saving time) and would
        silently compute the wrong window size against a plain integer
        second count.

        Returns:
            The index-count duration string. Always fullmatches ``\d+i``.

        """
        return f"{self.total_seconds}i"

    @classmethod
    def parse(cls, text: str) -> Self:
        """Parse a compact duration string into a Duration.

        Args:
            text: One or more ``<digits><unit>`` components with units
                strictly descending in the order w > d > h > m > s, each
                unit used at most once, and no separators, signs,
                decimals, or whitespace.

        Returns:
            The parsed Duration.

        Raises:
            ConfigError: If ``text`` is not a ``str``, does not match the
                grammar, or repeats/misorders a unit.

        """
        if not isinstance(text, str):
            logger.warning(
                "Rejecting non-string duration input: {}", type(text).__name__
            )
            raise ConfigError(f"Duration text must be a str, got {type(text).__name__}")

        if not _GRAMMAR_PATTERN.fullmatch(text):
            quoted = bounded_echo(text)
            logger.warning("Rejecting malformed duration string: {}", quoted)
            raise ConfigError(f"Invalid duration string: {quoted}")

        total_seconds = 0
        previous_rank = -1
        for amount, unit in _COMPONENT_PATTERN.findall(text):
            rank = _UNIT_RANK[unit]
            if rank <= previous_rank:
                quoted = bounded_echo(text)
                logger.warning(
                    "Rejecting out-of-order or duplicate duration unit in: {}",
                    quoted,
                )
                raise ConfigError(
                    "Duration units must be strictly descending "
                    f"(w>d>h>m>s), got: {quoted}"
                )
            previous_rank = rank
            total_seconds += int(amount) * _UNIT_SECONDS[unit]

        return cls(total_seconds)


def _format_seconds(total_seconds: int) -> str:
    """Greedily decompose seconds into the canonical unit string.

    Args:
        total_seconds: A non-negative number of seconds.

    Returns:
        The canonical compact form, largest unit first, with zero-valued
        components omitted. ``0`` formats as ``"0s"``.

    """
    if total_seconds == 0:
        return "0s"

    remaining = total_seconds
    parts = []
    for unit, unit_seconds in _UNIT_SECONDS.items():
        value, remaining = divmod(remaining, unit_seconds)
        if value:
            parts.append(f"{value}{unit}")
    return "".join(parts)


def coerce_duration(value: Duration | str) -> Duration:
    """Coerce a boundary ``Duration | str`` value into a ``Duration``.

    Public entry points accept ``Duration | str``; once past the
    boundary, internals should hold and pass around ``Duration`` only.

    Args:
        value: Either an existing Duration, or a string in the grammar
            accepted by :meth:`Duration.parse`.

    Returns:
        The coerced Duration.

    Raises:
        ConfigError: If ``value`` is a string that fails to parse, or is
            neither a Duration nor a string.

    """
    if isinstance(value, Duration):
        return value
    if isinstance(value, str):
        return Duration.parse(value)

    logger.warning(
        "Rejecting duration value of unsupported type: {}", type(value).__name__
    )
    raise ConfigError(f"Expected Duration or str, got {type(value).__name__}")


def _validate_strictly_positive(value: Duration | str, *, label: str) -> Duration:
    """Coerce ``value`` and reject a zero result.

    Args:
        value: A Duration, or a compact duration string.
        label: A human-readable name for the value, used in log messages
            and the error message (e.g. ``"window duration"``).

    Returns:
        The coerced, strictly positive Duration.

    Raises:
        ConfigError: If ``value`` cannot be coerced to a Duration, or
            coerces to the zero duration.

    """
    duration = coerce_duration(value)
    if duration.total_seconds == 0:
        logger.warning("Rejecting zero {}.", label)
        raise ConfigError(f"{label} must be strictly positive, got 0s.")
    return duration


def validate_window_duration(value: Duration | str) -> Duration:
    """Coerce and validate a window duration.

    Window durations are strictly positive: a zero-length window carries
    no data. Zero remains representable in general (for example, as an
    anchor offset elsewhere); it is only rejected here, at this use site.

    Args:
        value: A Duration, or a compact duration string.

    Returns:
        The validated, strictly positive Duration.

    Raises:
        ConfigError: If ``value`` cannot be coerced to a Duration, or
            coerces to the zero duration.

    """
    return _validate_strictly_positive(value, label="Window duration")


def validate_horizon_duration(value: Duration | str) -> Duration:
    """Coerce and validate a return horizon.

    Horizons are strictly positive: a zero horizon relates every close to
    itself, which is a constant rather than a return. Zero remains
    representable in general (for example, as an anchor offset
    elsewhere); it is only rejected here, at this use site.

    Args:
        value: A Duration, or a compact duration string.

    Returns:
        The validated, strictly positive Duration.

    Raises:
        ConfigError: If ``value`` cannot be coerced to a Duration, or
            coerces to the zero duration.

    """
    return _validate_strictly_positive(value, label="Horizon duration")


def validate_cadence(value: Duration | str) -> Duration:
    """Coerce and validate a cadence.

    Cadences are strictly positive: a zero-length step never advances.
    Zero remains representable in general (for example, as an anchor
    offset elsewhere); it is only rejected here, at this use site.

    Args:
        value: A Duration, or a compact duration string.

    Returns:
        The validated, strictly positive Duration.

    Raises:
        ConfigError: If ``value`` cannot be coerced to a Duration, or
            coerces to the zero duration.

    """
    return _validate_strictly_positive(value, label="Cadence")
