"""Window-scale schedules and emit-cadence rules, with recorded identity.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.schedules`` directly.

Everything here is a mechanism. The package ships no default schedule,
no default coefficient, and no default bounds: a caller states what they
want and gets back a resolved schedule that records exactly what it was
asked for.
"""

from ohlc_toolkit.schedules.generators import (
    MAX_RESOLVED_WINDOWS,
    DedupRule,
    ExplicitSpec,
    GeneratorKind,
    GeneratorSpec,
    LogSpacedSpec,
    MetallicRecurrenceSpec,
    RoundingRule,
    WindowSchedule,
    explicit,
    log_spaced,
    metallic_recurrence,
)

__all__ = [
    "MAX_RESOLVED_WINDOWS",
    "DedupRule",
    "ExplicitSpec",
    "GeneratorKind",
    "GeneratorSpec",
    "LogSpacedSpec",
    "MetallicRecurrenceSpec",
    "RoundingRule",
    "WindowSchedule",
    "explicit",
    "log_spaced",
    "metallic_recurrence",
]
