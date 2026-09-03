# Test data

Committed input fixtures for the test suite. Nothing here is generated at
test time; nothing here is a program's output.

## `bitstamp_btcusd_1min_14d.csv.gz`

A 14-day slice of the published public Bitstamp BTC/USD one-minute
history, with that source's own six columns (`timestamp`, `open`, `high`,
`low`, `close`, `volume`) unchanged. 20160 rows: a complete, gap-free
minute grid over the half-open Unix-second range `[1734998400,
1736208000)`, which is 2024-12-24 00:00:00 UTC up to 2025-01-07 00:00:00
UTC. Each `timestamp` is its candle's interval open.

This is the primary real-data fixture for windowed-aggregation tests: long
enough to carry a window measured in thousands of source candles, and
complete enough to pass strict source validation.

## `real_world_data.csv`

1439 rows of minute data, a little under one day. Smoke-only: a window
longer than a day cannot be materialized over it. Kept for the tests that
already read it.

## `test_bad_data.csv`, `test_csv_no_header.csv`, `test_csv_w_header.csv`

Tiny hand-written CSVs for the reader's parsing and failure paths.
