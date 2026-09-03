"""Committed data fixtures used by the window tests.

Real-data fixtures, and which one to reach for
----------------------------------------------

``tests/test_data/bitstamp_btcusd_1min_14d.csv.gz`` is the PRIMARY
real-data fixture for window work. It holds 20160 rows: a complete,
gap-free one-minute grid over the half-open Unix-second range
``[1786924800, 1788134400)``, which is 2026-08-17 00:00:00 UTC up to
2026-08-31 00:00:00 UTC. It is a slice of the recently published public
Bitstamp BTC/USD minute-data history, with that source's own six columns
unchanged. Fourteen days of minutes is enough to carry a window measured
in thousands of source candles, and complete enough to pass strict source
validation, so it can exercise a schedule-scale run end to end.

``tests/test_data/real_world_data.csv`` (1439 rows, a little under one
day) is SMOKE-ONLY from here on. A window longer than a day cannot even
be materialized over it, so it can show that a call runs but never that
aggregation is right at scale. It stays exactly where it is for the
legacy tests that already read it; new window work should use the 14-day
slice above.
"""

from pathlib import Path

import polars as pl

from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M, ValidationMode, read_source_csv

# tests/test_windows/fixtures.py -> tests/test_data/...
REAL_SLICE_PATH = (
    Path(__file__).parents[1] / "test_data" / "bitstamp_btcusd_1min_14d.csv.gz"
)

# The slice's own shape, restated here so a test can assert it rather than
# discover it.
REAL_SLICE_ROW_COUNT = 20_160
REAL_SLICE_START = 1_786_924_800
REAL_SLICE_END = 1_788_134_400
REAL_SLICE_CADENCE_SECONDS = 60


def load_real_slice() -> pl.DataFrame:
    """Read the committed 14-day real slice, validating it strictly.

    Strict validation is not incidental here: the window oracle documents
    that its input has already been validated, so a fixture that quietly
    developed a gap or a duplicate must fail loudly at the read rather
    than silently change what the windows over it mean.

    Returns:
        The raw source frame, exactly as committed.

    Raises:
        SourceValidationError: If the committed slice ever stops being a
            complete, on-phase, strictly increasing minute grid.

    """
    return read_source_csv(
        REAL_SLICE_PATH, BITSTAMP_BTCUSD_1M, mode=ValidationMode.STRICT
    )
