"""Temporal primitives: Duration, the exception taxonomy, and the bounded echo.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.temporal`` directly.
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
