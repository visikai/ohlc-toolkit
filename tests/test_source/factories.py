"""Synthetic source-frame builders shared across ``test_source`` modules."""

import polars as pl

# Column order matches the public Bitstamp minute-data layout: a Unix-second
# interval-open timestamp followed by the five OHLCV price/volume columns.
RAW_COLUMNS = ("timestamp", "open", "high", "low", "close", "volume")


def build_clean_frame(*, start: int, cadence_seconds: int, length: int) -> pl.DataFrame:
    """Build a complete, on-grid, strictly increasing raw source frame.

    Args:
        start: The first row's Unix-second interval-open timestamp.
        cadence_seconds: The fixed spacing between successive opens.
        length: The number of rows to generate.

    Returns:
        A polars DataFrame with an int64 ``timestamp`` column and five
        float64 OHLCV columns, one row per cadence step starting at
        ``start``.

    """
    timestamps = [start + i * cadence_seconds for i in range(length)]
    prices = [float(100 + i) for i in range(length)]
    return pl.DataFrame(
        {
            "timestamp": pl.Series(timestamps, dtype=pl.Int64),
            "open": pl.Series(prices, dtype=pl.Float64),
            "high": pl.Series(prices, dtype=pl.Float64),
            "low": pl.Series(prices, dtype=pl.Float64),
            "close": pl.Series(prices, dtype=pl.Float64),
            "volume": pl.Series([1.0] * length, dtype=pl.Float64),
        }
    )
