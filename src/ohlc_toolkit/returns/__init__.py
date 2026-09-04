"""Return primitives over window frames.

The top-level package imports this one, so ``ohlc_toolkit.returns`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.returns.X``, or
import them from here.

:func:`~ohlc_toolkit.returns.primitives.add_backward_returns` and
:func:`~ohlc_toolkit.returns.primitives.add_forward_returns` compose
returns onto a window frame, as a later, independent step over the output
of :func:`~ohlc_toolkit.windows.engine.compute_windows`, the same way
:func:`~ohlc_toolkit.windows.quality.apply_quality_policy` does. Neither
feeds back into the aggregator, and both read only ``close_time`` and
``close``.

The backward one is the causal feature. The forward one is not, and both
its column naming and the ``available_at`` column beside it exist to keep
that from being forgotten -- read
:mod:`ohlc_toolkit.returns.primitives`'s docstring before consuming it.

:mod:`ohlc_toolkit.returns.alignment` holds the rule both rest on: a
counterpart row is found by exact ``close_time`` equality, never by
shifting a number of rows.
"""

from ohlc_toolkit.returns.primitives import (
    ReturnMethod,
    add_backward_returns,
    add_forward_returns,
    backward_return_column,
    forward_available_at_column,
    forward_return_column,
)

__all__ = [
    "ReturnMethod",
    "add_backward_returns",
    "add_forward_returns",
    "backward_return_column",
    "forward_available_at_column",
    "forward_return_column",
]
