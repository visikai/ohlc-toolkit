"""Windowed candle aggregation: the brute-force reference implementation.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.windows`` directly.

What lives here is a correctness oracle: the plainest possible reading of
the window contract, meant to be checked against by eye and property-
tested against by faster implementations. It is not the fast path.
"""

from ohlc_toolkit.windows.reference import compute_reference_windows
from ohlc_toolkit.windows.resolution import (
    ExplicitRange,
    Materialization,
    MaterializationRule,
)

__all__ = [
    "ExplicitRange",
    "Materialization",
    "MaterializationRule",
    "compute_reference_windows",
]
