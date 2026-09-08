"""Return primitives over window frames.

The top-level package imports this one, so ``ohlc_toolkit.returns`` is
reachable from a bare ``import ohlc_toolkit``. The names below are NOT
flattened into that namespace: spell them ``ohlc_toolkit.returns.X``, or
import them from here.

:func:`~ohlc_toolkit.returns.primitives.add_backward_returns`,
:func:`~ohlc_toolkit.returns.primitives.add_forward_returns` and
:func:`~ohlc_toolkit.returns.primitives.add_forward_excursions` compose
columns onto a window frame, as a later, independent step over the output
of :func:`~ohlc_toolkit.windows.engine.compute_windows`, the same way
:func:`~ohlc_toolkit.windows.quality.apply_quality_policy` does. None
feeds back into the aggregator. The two returns read only ``close_time``
and ``close``; the excursions read ``high`` and ``low`` as well, because
an extremum over an interval is a fact about the bars inside it and not
about their closes.

The backward return is the causal feature. The forward return and both
excursions are not, and their column naming and the ``available_at``
column beside each exist to keep that from being forgotten -- read
:mod:`ohlc_toolkit.returns.primitives`'s docstring before consuming them.

:mod:`ohlc_toolkit.returns.alignment` holds the rules they rest on. A
return's counterpart row is found by exact ``close_time`` equality, never
by shifting a number of rows. An excursion reads the rows that follow a
position, so it is positional where a return is not, and it requires the
frame to be a hole-free grid at exactly the stated cadence rather than
reading through a gap.
"""

from ohlc_toolkit.returns.primitives import (
    ReturnMethod,
    add_backward_returns,
    add_forward_excursions,
    add_forward_returns,
    backward_return_column,
    forward_available_at_column,
    forward_excursion_available_at_column,
    forward_mae_column,
    forward_mfe_column,
    forward_return_column,
)

__all__ = [
    "ReturnMethod",
    "add_backward_returns",
    "add_forward_excursions",
    "add_forward_returns",
    "backward_return_column",
    "forward_available_at_column",
    "forward_excursion_available_at_column",
    "forward_mae_column",
    "forward_mfe_column",
    "forward_return_column",
]
