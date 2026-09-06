"""Phased lookback machinery, and the indicators built on it.

An indicator in this package reads a windowed-candle frame materialized
at SOURCE cadence and emits on the window's own ``E`` grid. What sits
between the two is the phased lookback: at each emit tick, the ``L``
non-overlapping windows of duration ``W`` ending at ``t``, ``t - W``,
``t - 2W``, and so on.

The phase set is anchored at the emit tick, not at the window's anchor,
and ``{t - kW}`` lies on the ``E`` grid only when ``E`` divides ``W`` --
which under the cadence rules this package resolves, it frequently does
not. Reading from the source-cadence materialization is what makes every
``t - kW`` a legal lookup rather than a near miss.
"""

from ohlc_toolkit.indicators.frames import (
    MAX_LOOKBACK,
    PHASED_COLUMNS,
    REQUIRED_COLUMNS,
    PhasedGrid,
    PhasedLookback,
)
from ohlc_toolkit.indicators.identity import (
    BANNED_NAME_PART,
    FeatureFamily,
    FeatureIdentity,
    NormalizationClass,
    effective_history,
    effective_n,
)
from ohlc_toolkit.indicators.phased import phased_lookback
from ohlc_toolkit.indicators.reference import phased_lookback_reference

# `resolve_phased_grid` is deliberately NOT here. Both entry points call
# it and neither caller does; exporting it would enlarge the published
# contract with machinery that has no stated user, and a 2.0 name cannot
# be withdrawn without another major.
__all__ = [
    "BANNED_NAME_PART",
    "MAX_LOOKBACK",
    "PHASED_COLUMNS",
    "REQUIRED_COLUMNS",
    "FeatureFamily",
    "FeatureIdentity",
    "NormalizationClass",
    "PhasedGrid",
    "PhasedLookback",
    "effective_history",
    "effective_n",
    "phased_lookback",
    "phased_lookback_reference",
]
