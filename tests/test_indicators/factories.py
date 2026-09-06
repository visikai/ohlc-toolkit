"""Hand-built harness output, so a primitive's inputs are exactly stated.

The real harness is exercised where it matters -- the golden that writes
a derived column to parquet, and the wiring test that runs a primitive
over `phased_lookback` output. Everywhere else a test needs to say "these
four closes, newest first" and read one number back, and building that
through the engine would hide the inputs behind three layers that have
their own tests.
"""

from collections.abc import Sequence

import polars as pl

from ohlc_toolkit.indicators import PHASED_COLUMNS, PhasedGrid, PhasedLookback

#: A base on the minute grid and on every window grid used below.
BASE = 1_700_000_000 - 1_700_000_000 % (7 * 24 * 3600)

#: What the fields a primitive does not read are filled with. A constant
#: rather than a range: a test that started passing because `high` moved
#: would be measuring the wrong column.
_FILLER = {
    "open": 100.0,
    "high": 101.0,
    "low": 99.0,
    "close": 100.0,
    "volume": 5.0,
    "src_count": 1,
    "coverage_seconds": 60,
    "traded_seconds": 60,
}


def phased_from_closes(
    closes: Sequence[Sequence[float | None] | None],
    *,
    lookback: int,
    window_seconds: int = 180,
    emit_seconds: int = 60,
) -> PhasedLookback:
    """Assemble harness output whose `close` lists are exactly as given.

    Args:
        closes: One list of closes per tick, NEWEST FIRST as the harness
            orders them, or `None` for a tick whose inputs were missing.
        lookback: The `L` the grid records.
        window_seconds: The window `W`.
        emit_seconds: The emit cadence `E`.

    Returns:
        The record a primitive reads.

    """
    ticks = tuple(BASE + index * emit_seconds for index in range(len(closes)))
    columns: dict[str, pl.Series] = {
        "close_time": pl.Series("close_time", list(ticks), dtype=pl.Int64)
    }
    for field, dtype in PHASED_COLUMNS.items():
        values = (
            list(closes)
            if field == "close"
            else [
                None if row is None else [_FILLER[field]] * len(row) for row in closes
            ]
        )
        columns[field] = pl.Series(field, values, dtype=pl.List(dtype))
    return PhasedLookback(
        frame=pl.DataFrame(columns),
        grid=PhasedGrid(
            window_seconds=window_seconds,
            emit_seconds=emit_seconds,
            cadence_seconds=60,
            lookback=lookback,
            min_traded_seconds=0,
            ticks=ticks,
        ),
    )


def rising(count: int, *, step: float = 1.0, start: float = 100.0) -> list[float]:
    """`count` closes, newest first, each `step` above the one before it."""
    return [start + step * (count - 1 - index) for index in range(count)]
