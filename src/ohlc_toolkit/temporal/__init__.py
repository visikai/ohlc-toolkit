"""Cross-cutting primitives every other subpackage needs.

Duration, the exception taxonomy, the bounded echo, and the guard against
overwriting a column.

The last two are here for the same reason the taxonomy is: every
subpackage needs them, so a home inside any one of them would be imported
upward or copied. That is why this package is named for what it started
as rather than for what it now holds.

One cost, recorded rather than discovered: ``columns`` imports polars, so
importing ``ohlc_toolkit.temporal`` now pulls polars in. It was one of
three subpackages that did not.

The top-level package imports this one, so ``ohlc_toolkit.temporal`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.temporal.X``, or
import them from here.
"""

from ohlc_toolkit.temporal.columns import require_absent_columns
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
    "require_absent_columns",
    "validate_cadence",
    "validate_horizon_duration",
    "validate_window_duration",
]
