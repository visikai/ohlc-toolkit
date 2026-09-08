"""Window-scale schedules and emit-cadence rules, with recorded identity.

The top-level package imports this one, so ``ohlc_toolkit.schedules`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.schedules.X``, or
import them from here.

Three kinds of schedule live here, and identity is per kind: a
:class:`WindowSchedule` names the scales a frame is built at, a
:class:`LookbackSchedule` names counts of whatever period a frame carries,
and a :class:`HorizonSchedule` names the distances a target reaches ahead.
Each records its members under its own key, so no two ever share a
``schedule_id`` and each reader refuses the others' payloads.

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
from ohlc_toolkit.schedules.horizon import (
    HorizonSchedule,
    explicit_horizons,
    log_spaced_horizons,
    metallic_horizons,
)
from ohlc_toolkit.schedules.lookback import (
    ExplicitLookbackSpec,
    LogSpacedLookbackSpec,
    LookbackSchedule,
    LookbackSpec,
    MetallicLookbackSpec,
    explicit_lookback,
    log_spaced_lookback,
    metallic_lookback,
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
    "ExplicitLookbackSpec",
    "ExplicitPairsSpec",
    "ExplicitSpec",
    "GeneratorKind",
    "GeneratorSpec",
    "HorizonSchedule",
    "LogSpacedLookbackSpec",
    "LogSpacedSpec",
    "LookbackSchedule",
    "LookbackSpec",
    "MetallicLookbackSpec",
    "MetallicRecurrenceSpec",
    "RoundingRule",
    "WOverKSpec",
    "WindowEmitPair",
    "WindowSchedule",
    "explicit",
    "explicit_horizons",
    "explicit_lookback",
    "explicit_pairs",
    "log_spaced",
    "log_spaced_horizons",
    "log_spaced_lookback",
    "metallic_horizons",
    "metallic_lookback",
    "metallic_recurrence",
    "named_schedule",
    "named_schedule_names",
    "w_over_k",
]
