"""Hand-built source frames, profiles, and expectations for window tests.

Everything here is written by hand or built from hand-written literals.
Nothing in this module calls the window oracle: the expected frames these
helpers assemble are the specification the oracle is measured against, so
deriving them from the oracle itself would make every golden test
circular.
"""

from collections.abc import Sequence

import polars as pl

from ohlc_toolkit.source.profile import Availability, ColumnKind, SourceProfile
from ohlc_toolkit.temporal import Duration

# Column layout of a raw source frame: a Unix-second interval-open
# timestamp followed by the five OHLCV price/volume columns.
RAW_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")

# (timestamp, open, high, low, close, volume)
SourceRow = tuple[int, float, float, float, float, float]

# (open_time, close_time, open, high, low, close, volume, src_count,
#  coverage_seconds) -- the nine output columns, in order. The five
# price/volume entries are None exactly when a window included no
# candles.
WindowRow = tuple[
    int,
    int,
    float | None,
    float | None,
    float | None,
    float | None,
    float | None,
    int,
    int,
]


def profile_for(cadence_seconds: int, *, phase_seconds: int = 0) -> SourceProfile:
    """Build a minimal OHLCV source profile for one cadence and phase.

    Args:
        cadence_seconds: The source's fixed candle duration in seconds.
        phase_seconds: The declared offset of the source timestamp grid
            from the plain epoch grid. Must be smaller than the cadence.

    Returns:
        A profile declaring an integer ``timestamp`` column plus the five
        floating OHLCV columns.

    """
    return SourceProfile(
        name=f"window-test-{cadence_seconds}s",
        timestamp_column="timestamp",
        availability=Availability.CLOSE_TIME,
        raw_schema={
            "timestamp": ColumnKind.INTEGER,
            "open": ColumnKind.FLOATING,
            "high": ColumnKind.FLOATING,
            "low": ColumnKind.FLOATING,
            "close": ColumnKind.FLOATING,
            "volume": ColumnKind.FLOATING,
        },
        cadence=Duration(cadence_seconds),
        phase=Duration(phase_seconds),
    )


def frame_from_rows(rows: Sequence[SourceRow]) -> pl.DataFrame:
    """Build a raw source frame from hand-written row tuples.

    Row order is preserved exactly as given: these frames are the oracle's
    input, and the oracle must never depend on, nor repair, their order.

    Args:
        rows: ``(timestamp, open, high, low, close, volume)`` tuples.

    Returns:
        A polars DataFrame with an int64 ``timestamp`` column and five
        float64 OHLCV columns.

    """
    return pl.DataFrame(
        [
            pl.Series("timestamp", [row[0] for row in rows], dtype=pl.Int64),
            pl.Series("open", [row[1] for row in rows], dtype=pl.Float64),
            pl.Series("high", [row[2] for row in rows], dtype=pl.Float64),
            pl.Series("low", [row[3] for row in rows], dtype=pl.Float64),
            pl.Series("close", [row[4] for row in rows], dtype=pl.Float64),
            pl.Series("volume", [row[5] for row in rows], dtype=pl.Float64),
        ]
    )


def expected_frame(rows: Sequence[WindowRow]) -> pl.DataFrame:
    """Build the expected window output from hand-written row tuples.

    The column names, dtypes, and order below are written out literally,
    by hand, so that comparing against this frame also pins the output
    schema.

    Args:
        rows: One :data:`WindowRow` per expected output row, in the
            expected order.

    Returns:
        A nine-column polars DataFrame with the exact expected schema.

    """
    return pl.DataFrame(
        [
            pl.Series("open_time", [row[0] for row in rows], dtype=pl.Int64),
            pl.Series("close_time", [row[1] for row in rows], dtype=pl.Int64),
            pl.Series("open", [row[2] for row in rows], dtype=pl.Float64),
            pl.Series("high", [row[3] for row in rows], dtype=pl.Float64),
            pl.Series("low", [row[4] for row in rows], dtype=pl.Float64),
            pl.Series("close", [row[5] for row in rows], dtype=pl.Float64),
            pl.Series("volume", [row[6] for row in rows], dtype=pl.Float64),
            pl.Series("src_count", [row[7] for row in rows], dtype=pl.UInt32),
            pl.Series("coverage_seconds", [row[8] for row in rows], dtype=pl.Int64),
        ]
    )
