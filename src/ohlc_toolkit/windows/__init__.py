"""Windowed candle aggregation: one contract, two implementations.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.windows`` directly.

:func:`~ohlc_toolkit.windows.engine.compute_windows` is the one to call.
It is Polars-native and linear in rows plus emit ticks.

:func:`~ohlc_toolkit.windows.reference.compute_reference_windows` is a
correctness oracle: the plainest possible reading of the window contract,
quadratic on purpose, meant to be checked against by eye and tested
against by the engine. It is not the fast path, and it is not deprecated
either -- it is what the fast path is measured by.

Both resolve their schedules through
:mod:`ohlc_toolkit.windows.resolution`, so a configuration one refuses is
refused by the other in the same words.

:mod:`ohlc_toolkit.windows.quality` is a separate, later step: a
recipe-recordable quality policy composed AFTER either implementation's
output, never inside it.
"""

from ohlc_toolkit.windows.engine import compute_windows
from ohlc_toolkit.windows.quality import (
    GateMode,
    QualityMode,
    QualityReport,
    WindowCoverageError,
    WindowQualityPolicy,
    apply_quality_policy,
)
from ohlc_toolkit.windows.reference import compute_reference_windows
from ohlc_toolkit.windows.resolution import (
    ExplicitRange,
    Materialization,
    MaterializationRule,
)

__all__ = [
    "ExplicitRange",
    "GateMode",
    "Materialization",
    "MaterializationRule",
    "QualityMode",
    "QualityReport",
    "WindowCoverageError",
    "WindowQualityPolicy",
    "apply_quality_policy",
    "compute_reference_windows",
    "compute_windows",
]
