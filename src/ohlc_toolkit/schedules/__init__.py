"""Window-scale schedules and emit-cadence rules, with recorded identity.

The top-level package imports this one, so ``ohlc_toolkit.schedules`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.schedules.X``, or
import them from here.

Everything here is a mechanism. The package ships no default schedule,
no default coefficient, no default bounds, and no default emit divisor:
a caller states what they want and gets back a resolved schedule or
cadence rule that records exactly what it was asked for.

:mod:`ohlc_toolkit.schedules.registry` holds named schedules, which are
data: nothing here consults it unless a caller asks for a name.
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
from ohlc_toolkit.schedules.registry import (
    METALLIC_LEGACY_2025,
    named_schedule,
    named_schedule_names,
)

__all__ = [
    "MAX_RESOLVED_WINDOWS",
    "METALLIC_LEGACY_2025",
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
    "named_schedule",
    "named_schedule_names",
    "w_over_k",
]
