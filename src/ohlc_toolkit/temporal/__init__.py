"""Temporal primitives: an exact-second Duration and its exception taxonomy.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.temporal`` directly.
"""

from ohlc_toolkit.temporal.duration import (
    Duration,
    coerce_duration,
    validate_cadence,
    validate_window_duration,
)
from ohlc_toolkit.temporal.errors import (
    ConfigError,
    CoverageError,
    DataValidationError,
)

__all__ = [
    "ConfigError",
    "CoverageError",
    "DataValidationError",
    "Duration",
    "coerce_duration",
    "validate_cadence",
    "validate_window_duration",
]
