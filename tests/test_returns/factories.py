"""Hand-written frames for the return-primitive tests.

Everything here is a literal. Nothing in this module computes a return,
imports the module under test, or derives a close from another close: the
frames below are the inputs, and every expected value lives in the test
module beside the arithmetic that produced it.

Why these particular closes
---------------------------

``128``, ``160``, ``320``, ``80``, ``100`` and ``25`` are chosen so that
every ratio the tests take between them -- one cadence apart, two
cadences apart, in either direction -- is a dyadic rational: ``160/128``
is exactly ``1.25``, ``320/128`` exactly ``2.5``, ``80/320`` exactly
``0.25``. Each is representable in an IEEE-754 double with no rounding at
all, and subtracting one from a dyadic rational in ``[0.25, 2.5]`` is
exact too. The expected simple returns are therefore exact literals that
a correct implementation must reproduce bit for bit, with no tolerance
anywhere.

Why the gap is where it is
--------------------------

:data:`GAPPED_CLOSE_TIMES` omits the fourth tick, so the frame's rows are
one cadence apart across most of it and two cadences apart once. That one
missing tick is what separates a time-based counterpart lookup from a
row-shift: over a two-cadence horizon, the row after the gap has a
counterpart that is present (two cadences back lands on the last tick
before the gap) while the row after that does not, and a shift by two
ROWS gets both of them wrong. See ``test_backward.py``.
"""

from collections.abc import Sequence

import polars as pl

# A real-looking Unix second that is a whole multiple of a minute, so the
# fixture close times read like emit ticks off a 1m grid rather than like
# small integers that might collide with a horizon or a row index.
TIME_BASE = 1_700_000_040

# The cadence every fixture frame is emitted at, in both the spellings a
# caller may pass.
CADENCE = "1m"
CADENCE_SECONDS = 60

# A frame with one tick missing: ticks 0, 1, 2, 4, 5 of a 1m grid.
GAPPED_OFFSETS = (0, 60, 120, 240, 300)
GAPPED_CLOSES = (128.0, 160.0, 320.0, 80.0, 100.0)

# The same shape with no gap at all: six consecutive 1m ticks.
GAP_FREE_OFFSETS = (0, 60, 120, 180, 240, 300)
GAP_FREE_CLOSES = (128.0, 160.0, 320.0, 80.0, 100.0, 25.0)


def return_frame(
    offsets: Sequence[int], closes: Sequence[float | None]
) -> pl.DataFrame:
    """Build a minimal window frame carrying only the two columns read.

    The dtypes are written out literally, and they are exactly the ones
    :func:`~ohlc_toolkit.windows.engine.compute_windows` emits for these
    two columns: an ``Int64`` ``close_time`` of Unix seconds and a
    ``Float64`` ``close``.

    Args:
        offsets: Close-time offsets from :data:`TIME_BASE`, in seconds.
        closes: One close per offset. ``None`` is a window that reported
            no close, which the engine emits for a window holding no
            source candle.

    Returns:
        A two-column polars DataFrame in exactly the given row order.

    """
    return pl.DataFrame(
        [
            pl.Series(
                "close_time", [TIME_BASE + offset for offset in offsets], dtype=pl.Int64
            ),
            pl.Series("close", list(closes), dtype=pl.Float64),
        ]
    )


def gapped_frame() -> pl.DataFrame:
    """Return the five-row fixture whose fourth 1m tick is missing."""
    return return_frame(GAPPED_OFFSETS, GAPPED_CLOSES)


def gap_free_frame() -> pl.DataFrame:
    """Return the six-row fixture with every 1m tick present."""
    return return_frame(GAP_FREE_OFFSETS, GAP_FREE_CLOSES)
