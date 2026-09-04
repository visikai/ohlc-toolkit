"""Temporal primitives: Duration, the exception taxonomy, and the bounded echo.

The top-level package imports this one, so ``ohlc_toolkit.temporal`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.temporal.X``, or
import them from here.
"""

from ohlc_toolkit.temporal.duration import (
    Duration,
    coerce_duration,
    validate_cadence,
    validate_horizon_duration,
    validate_window_duration,
)
from ohlc_toolkit.temporal.echo import MAX_ECHO_CHARS, bounded_echo
from ohlc_toolkit.temporal.errors import (
    ConfigError,
    CoverageError,
    DataValidationError,
)

__all__ = [
    "MAX_ECHO_CHARS",
    "ConfigError",
    "CoverageError",
    "DataValidationError",
    "Duration",
    "bounded_echo",
    "coerce_duration",
    "validate_cadence",
    "validate_horizon_duration",
    "validate_window_duration",
]
