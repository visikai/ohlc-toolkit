"""A Polars-native toolkit for OHLC market data.

The public surface is six subpackages, and this module imports each one
so that a bare ``import ohlc_toolkit`` reaches all of them:

- :mod:`ohlc_toolkit.temporal` -- the ``Duration`` value type, its
  compact grammar, and the error taxonomy everything else raises.
- :mod:`ohlc_toolkit.source` -- source profiles, raw-frame validation,
  and a reader that never silently repairs its input.
- :mod:`ohlc_toolkit.windows` -- window aggregation, with a fast engine
  and an independent brute-force oracle it is checked against.
- :mod:`ohlc_toolkit.schedules` -- window-scale schedules and emit
  cadence rules, each recording the request that produced it.
- :mod:`ohlc_toolkit.returns` -- backward and forward returns, with the
  instant a forward value becomes available carried beside it.
- :mod:`ohlc_toolkit.snapshot` -- fetching a published dataset release,
  refusing any byte that does not match its manifest.

Names are NOT flattened into this namespace. ``ohlc_toolkit.Duration``
does not exist and is not meant to: the surface is roughly ninety public
names, and the subpackage that owns each one is the most useful thing a
call site can say about it. ``windows.ExplicitRange`` and
``schedules.ExplicitSpec`` are different things, ``compute_windows`` and
``compute_reference_windows`` sit on opposite sides of a fast-path /
oracle divide, and ``returns.add_forward_returns`` is a stage that runs
after aggregation rather than inside it. A flat namespace would spend all
of that to save one dotted component.

So spell it ``ohlc_toolkit.windows.compute_windows``, or import the
subpackage (``from ohlc_toolkit import windows``), or import the name
from where it lives (``from ohlc_toolkit.windows import
compute_windows``). All three work; none of them needs a second import
line to make the first one resolve.

This is 1.0, and 1.0 is a clean break. Every 0.4 name -- ``read_ohlc_csv``,
``transform_ohlc``, ``DatasetDownloader``, ``parse_timeframe``,
``format_timeframe``, ``validate_timeframe``,
``validate_timeframe_format``, ``calculate_percentage_return`` -- is
gone, with no alias and no deprecation shim. 0.4.x remains installable
from PyPI for code that wants it.
"""

from ohlc_toolkit import returns, schedules, snapshot, source, temporal, windows

__all__ = [
    "returns",
    "schedules",
    "snapshot",
    "source",
    "temporal",
    "windows",
]
