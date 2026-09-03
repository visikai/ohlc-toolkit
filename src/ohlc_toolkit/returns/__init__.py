"""Return primitives over window frames.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.returns`` directly.

:func:`~ohlc_toolkit.returns.primitives.add_backward_returns` composes a
causal return onto a window frame, as a later, independent step over the
output of :func:`~ohlc_toolkit.windows.engine.compute_windows`, the same
way :func:`~ohlc_toolkit.windows.quality.apply_quality_policy` does. It
never feeds back into the aggregator and reads only ``close_time`` and
``close``.

:mod:`ohlc_toolkit.returns.alignment` holds the rule that step and its
refusals rest on: a counterpart row is found by exact ``close_time``
equality, never by shifting a number of rows.
"""

from ohlc_toolkit.returns.primitives import (
    ReturnMethod,
    add_backward_returns,
    backward_return_column,
)

__all__ = [
    "ReturnMethod",
    "add_backward_returns",
    "backward_return_column",
]
