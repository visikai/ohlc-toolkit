"""Window-scale schedules and emit-cadence rules, with recorded identity.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.schedules`` directly.

Everything here is a mechanism. The package ships no default schedule,
no default coefficient, no default bounds, and no default emit divisor:
a caller states what they want and gets back a resolved schedule or
cadence rule that records exactly what it was asked for.
"""

from ohlc_toolkit.schedules.cadence import (
    CadenceKind,
    CadenceRule,
    CadenceSpec,
    ExplicitPairsSpec,
    WindowEmitPair,
    WOverKSpec,
    explicit_pairs,
    w_over_k,
)
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
    "CadenceKind",
    "CadenceRule",
    "CadenceSpec",
    "DedupRule",
    "ExplicitPairsSpec",
    "ExplicitSpec",
    "GeneratorKind",
    "GeneratorSpec",
    "LogSpacedSpec",
    "MetallicRecurrenceSpec",
    "RoundingRule",
    "WOverKSpec",
    "WindowEmitPair",
    "WindowSchedule",
    "explicit",
    "explicit_pairs",
    "log_spaced",
    "metallic_recurrence",
    "w_over_k",
]
