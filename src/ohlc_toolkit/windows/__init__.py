"""Windowed candle aggregation: one contract, two implementations.

The top-level package imports this one, so ``ohlc_toolkit.windows`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.windows.X``, or
import them from here.

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

:mod:`ohlc_toolkit.windows.annotations` is another later step: it joins a
sparse interval sidecar onto that output as opaque flags with overlap
accounting, and reads nothing but the two window bounds.
"""

from ohlc_toolkit.windows.annotations import (
    AnnotationColumns,
    AnnotationValidationError,
    annotate_windows,
    read_annotations,
)
from ohlc_toolkit.windows.engine import compute_windows
from ohlc_toolkit.windows.quality import (
    GateMode,
    QualityMode,
    QualityPolicyResult,
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
    "AnnotationColumns",
    "AnnotationValidationError",
    "ExplicitRange",
    "GateMode",
    "Materialization",
    "MaterializationRule",
    "QualityMode",
    "QualityPolicyResult",
    "QualityReport",
    "WindowCoverageError",
    "WindowQualityPolicy",
    "annotate_windows",
    "apply_quality_policy",
    "compute_reference_windows",
    "compute_windows",
    "read_annotations",
]
