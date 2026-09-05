# OHLC Toolkit

[![PyPI](https://img.shields.io/pypi/v/ohlc-toolkit)](https://pypi.org/project/ohlc-toolkit/)
[![Python](https://img.shields.io/pypi/pyversions/ohlc-toolkit.svg)](https://pypi.org/project/ohlc-toolkit/)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/0db6f73fe9bb4e8a8591055a6ea284f2)](https://app.codacy.com/gh/visikai/ohlc-toolkit/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_grade)
[![Codacy Badge](https://app.codacy.com/project/badge/Coverage/0db6f73fe9bb4e8a8591055a6ea284f2)](https://app.codacy.com/gh/visikai/ohlc-toolkit/dashboard?utm_source=gh&utm_medium=referral&utm_content=&utm_campaign=Badge_coverage)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

A Polars-native toolkit for reading, validating, and aggregating OHLC
(Open, High, Low, Close) market data.

Each stage is a separate, composable step: fetch a published dataset,
read and validate it, aggregate windows over it, add returns, apply a
quality policy. Nothing sorts, fills, interpolates, or de-duplicates your
data on the way past. When the data is not what you expected, you get a
value describing that — or an exception — rather than a tidied frame and
a log line.

## Installation

```bash
pip install ohlc-toolkit
```

Python 3.11 or newer. Four runtime dependencies: `polars`, `requests`,
`loguru`, `orjson`.

## 1.0 is a break

1.0 is a rewrite, not an upgrade. Every name 0.4 exported is gone:

| 0.4 | 1.0 |
| --- | --- |
| `read_ohlc_csv` | `source.read_source_csv` |
| `transform_ohlc` | `windows.compute_windows` |
| `DatasetDownloader` | `snapshot.fetch_snapshot` |
| `parse_timeframe`, `format_timeframe` | `temporal.Duration` |
| `validate_timeframe`, `validate_timeframe_format` | `temporal.validate_window_duration`, `temporal.validate_cadence` |
| `calculate_percentage_return` | `returns.add_forward_returns` |

There are no aliases, no deprecation warnings, and no migration path. The
replacements are not renames — they take different arguments, return
different types, and mean different things. A shim would have had to
guess which of those differences you wanted, and guessing about a price
series is how a wrong number reaches a model.

pandas is no longer a dependency of any kind.

0.4.x stays on PyPI and keeps working. If you depend on it, pin it:

```bash
pip install "ohlc-toolkit<1"
```

## Quickstart

Against the real published BTC/USD minute history
([ff137/bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)).
The release is about 260 MB across three assets; it is fetched once and
re-used on later runs. The toolkit logs each fetch and verification step
as it runs; those log lines are omitted from the pasted output below.

```python
from ohlc_toolkit.returns import (
    ReturnMethod,
    add_backward_returns,
    add_forward_returns,
)
from ohlc_toolkit.snapshot import (
    BITSTAMP_BTCUSD_1M_REPOSITORY,
    SnapshotRelease,
    fetch_snapshot,
    read_snapshot_frame,
    verify_snapshot_continuity,
)
from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M
from ohlc_toolkit.windows import compute_windows

# Fetch the published minute history. Nothing lands until its SHA-256 and
# size match what the release's own manifest declared.
release = SnapshotRelease(
    repository=BITSTAMP_BTCUSD_1M_REPOSITORY,
    tag="bitstamp-btcusd-1m-2026-08",
)
result = fetch_snapshot(release, "data/bitstamp")
print("snapshot identity:", result.manifest_sha256)

# Read it under strict validation, then check the frame against what the
# manifest said it would be.
frame = read_snapshot_frame(result)
report = verify_snapshot_continuity(frame, result.manifest)
print("rows:", report.rows_checked, "| seam mismatches:", report.seam_mismatches)

# One-hour windows, emitted every fifteen minutes.
hourly = compute_windows(
    frame,
    BITSTAMP_BTCUSD_1M,
    window="1h",
    emit_every="15m",
    materialization="skip_warmup",
)
print(hourly.tail(3))

# The causal four-hour log return, and the non-causal one beside the
# instant it becomes readable.
labelled = add_forward_returns(
    add_backward_returns(
        hourly, horizon="4h", cadence="15m", method=ReturnMethod.LOG
    ),
    horizon="4h",
    cadence="15m",
    method=ReturnMethod.LOG,
)
print(
    labelled.select(
        "close_time",
        "backward_return_log_4h",
        "forward_return_log_4h",
        "forward_return_log_4h_available_at",
    ).tail(3)
)
```

```text
snapshot identity: 96e96cc32b313e4985a3d2d105e40ee528f8243bd2d1146a38f1b600f0bd3de1
rows: 7714079 | seam mismatches: ()
shape: (3, 9)
┌────────────┬────────────┬──────────┬──────────┬───┬──────────┬───────────┬───────────┬───────────┐
│ open_time  ┆ close_time ┆ open     ┆ high     ┆ … ┆ close    ┆ volume    ┆ src_count ┆ coverage_ │
│ ---        ┆ ---        ┆ ---      ┆ ---      ┆   ┆ ---      ┆ ---       ┆ ---       ┆ seconds   │
│ i64        ┆ i64        ┆ f64      ┆ f64      ┆   ┆ f64      ┆ f64       ┆ u32       ┆ ---       │
│            ┆            ┆          ┆          ┆   ┆          ┆           ┆           ┆ i64       │
╞════════════╪════════════╪══════════╪══════════╪═══╪══════════╪═══════════╪═══════════╪═══════════╡
│ 1788215400 ┆ 1788219000 ┆ 78734.71 ┆ 78742.28 ┆ … ┆ 78554.24 ┆ 37.490404 ┆ 60        ┆ 3600      │
│ 1788216300 ┆ 1788219900 ┆ 78642.61 ┆ 78642.61 ┆ … ┆ 78539.91 ┆ 34.778154 ┆ 60        ┆ 3600      │
│ 1788217200 ┆ 1788220800 ┆ 78572.31 ┆ 78576.84 ┆ … ┆ 78571.17 ┆ 36.012591 ┆ 60        ┆ 3600      │
└────────────┴────────────┴──────────┴──────────┴───┴──────────┴───────────┴───────────┴───────────┘
shape: (3, 4)
┌────────────┬────────────────────────┬───────────────────────┬─────────────────────────────────┐
│ close_time ┆ backward_return_log_4h ┆ forward_return_log_4h ┆ forward_return_log_4h_availabl… │
│ ---        ┆ ---                    ┆ ---                   ┆ ---                             │
│ i64        ┆ f64                    ┆ f64                   ┆ i64                             │
╞════════════╪════════════════════════╪═══════════════════════╪═════════════════════════════════╡
│ 1788219000 ┆ -0.006804              ┆ null                  ┆ 1788233400                      │
│ 1788219900 ┆ -0.005415              ┆ null                  ┆ 1788234300                      │
│ 1788220800 ┆ -0.003928              ┆ null                  ┆ 1788235200                      │
└────────────┴────────────────────────┴───────────────────────┴─────────────────────────────────┘
```

Those three forward returns are null because their counterparts lie past
the end of the data — and their `available_at` is stated anyway, because
when a value would arrive is a property of the horizon, not of whether it
happened to be found.

Timings from that run, on a 16-core desktop: 2.8 s to read and strictly
validate all 7,714,079 rows, 0.8 s to aggregate the 514,268 windows, and
under 0.1 s for both return columns. polars uses the whole thread pool.

## What's in it

`import ohlc_toolkit` reaches all six subpackages. Names are not
flattened into the top level — spell them `ohlc_toolkit.windows.X`, or
import from the subpackage.

### `temporal` — durations that carry their unit

`Duration` holds exact whole seconds and parses one compact grammar:
components in strictly descending order, each unit at most once, no
separators or signs.

```python
from ohlc_toolkit.temporal import Duration

Duration.parse("1h15m").total_seconds == 4500
str(Duration(4500)) == "1h15m"
```

Anywhere a duration is accepted, a `Duration` or its string spelling both
work; a bare integer whose unit you have to infer does not. `ConfigError`,
`DataValidationError` and `CoverageError` are the package's error
taxonomy, and every message that quotes untrusted input goes through one
bounded echo.

### `source` — reads that report instead of repair

A `SourceProfile` states a source's cadence, phase, timestamp column, and
raw column kinds. `read_source_csv` reads against one without sorting,
filling, dropping or de-duplicating anything, and `validate_source_frame`
returns findings as data:

```python
import polars as pl

from ohlc_toolkit.source import (
    BITSTAMP_BTCUSD_1M,
    ValidationMode,
    validate_source_frame,
)

frame = pl.DataFrame(
    {
        "timestamp": [1786924800, 1786924860, 1786924980],  # 1786924920 missing
        "open": [1.0, 2.0, 3.0],
        "high": [1.0, 2.0, 3.0],
        "low": [1.0, 2.0, 3.0],
        "close": [1.0, 2.0, 3.0],
        "volume": [1.0, 1.0, 1.0],
    }
)
report = validate_source_frame(frame, BITSTAMP_BTCUSD_1M, mode=ValidationMode.REPORT)
for finding in report.findings:
    print(finding)
```

```text
Finding(kind=<FindingKind.GAP: 'gap'>, message='1 missing candle(s) expected in [1786924920, 1786924980)', count=1, sample_timestamps=(1786924920, 1786924980))
```

The eight finding kinds are schema, nulls, non-finite values, non-increasing
timestamps, overlapping intervals, off-phase timestamps, irregular intervals,
and gaps. A non-finite value is a NaN or an infinity in a declared price or
volume column; it is kept distinct from a null because a null is an absent
cell and a NaN is a present cell that is not a number, and neither is coerced
into the other. `ValidationMode.STRICT` raises `SourceValidationError` on any
of them instead.

### `windows` — aggregation with an independent oracle

`compute_windows` emits one row per tick of an epoch-anchored emit grid,
each row aggregating the candles whose intervals fall inside the window
ending at that tick. Membership is decided by close time, never by
counting rows, so a gap in the source changes the window's reported
coverage rather than silently changing what it spans. Nine columns come
back: `open_time`, `close_time`, OHLCV, `src_count`, `coverage_seconds`.

`compute_reference_windows` computes the same thing the plainest possible
way — quadratic, on purpose. It is the specification; the engine is what
you run. The suite holds the two to the same rows, in the same order,
with the same dtypes, across a synthetic matrix, property tests,
committed goldens, and a real 14-day slice — exactly equal on every
integer column and every selected price, and within a tolerance derived
from the oracle's own fold on `volume`, the one column either
implementation sums. Both resolve schedules through the same module, so a
configuration one refuses is refused by the other in the same words.

`apply_quality_policy` is a separate, later step over the output, never
inside it. `PASS_THROUGH` records a deliberate no-op, `FILTER` drops rows
below a coverage threshold, `GATE` raises (or reports) without dropping.
Its threshold is exact rational arithmetic, not a float product.

`annotate_windows` is another later step: it joins a sparse sidecar of
half-open `[start, end)` intervals -- an outage log, a maintenance
calendar -- onto the output as two appended columns, the distinct flags
overlapping each window and the seconds of the window inside the union of
those intervals. Overlap is half-open on both sides, so an interval that
ends exactly at a window's open touches nothing; flags are opaque strings
the transform reports and never interprets; no OHLCV or coverage value is
read or changed. `read_annotations` reads such a sidecar from CSV, typed
and checked, in file order and under an optional row cap, with the
Bitstamp provenance file's column names as defaults; a broken sidecar
raises `AnnotationValidationError`, a misconfigured call `ConfigError`.

### `schedules` — schedules that record what they are

Generators (`log_spaced`, `metallic_recurrence`, `explicit`) produce a
`WindowSchedule`; cadence rules (`w_over_k`, `explicit_pairs`) produce a
`CadenceRule`. There is no default schedule, coefficient, bound or
divisor anywhere: a caller states what it wants and gets back something
that records exactly what was asked for, named by a content hash over
that record.

```python
from ohlc_toolkit.schedules import WindowSchedule, log_spaced

schedule = log_spaced(count=5, minimum="15m", maximum="1d", grain="15m")
print([str(window) for window in schedule.windows])
print(schedule.schedule_id)
print(WindowSchedule.from_dict(schedule.to_dict()) == schedule)
```

```text
['15m', '45m', '2h30m', '7h45m', '1d']
0abc992caafb27a7ce8ba1ba5edca1118366b9ec84190744e2a641c0972d3e07
True
```

A payload whose recorded `schedule_id` does not match its content is
refused rather than repaired, so a schedule read back from disk is the
one that was written.

### `returns` — causal and non-causal, told apart

`add_backward_returns` relates a row's close to the close exactly `H`
earlier; both were known at the row's own close time, so it is a feature.
`add_forward_returns` relates the close `H` later to the row's own, and
is not. The distinction is carried in the data twice — the `forward_`
prefix travels with the column name into any file or plot, and every
forward value has an `available_at` column stating the instant it may
first be read.

Counterparts are located by exact close-time equality, never by shifting
rows, so a gap yields a null instead of a wrong pairing. `method` is
required — `ReturnMethod.SIMPLE` or `ReturnMethod.LOG` — and is recorded
in the column name. Any value that is not a finite float comes back null.

### `snapshot` — fail-closed fetching

A release is named by repository and immutable tag; asset URLs are
composed from that identity rather than discovered, so there is no API
call, no token, and no listing step to disagree with the manifest.
Downloads land on a temporary name and are renamed into place only after
size and SHA-256 both match what the manifest declared. An already
present file is reused only when its digest matches, and a mismatch is
refused rather than overwritten unless you pass
`ExistingAssetPolicy.REPLACE`.

`fetch_snapshot` also returns the snapshot identity — the digest over the
manifest bytes — and accepts it back as `expected_manifest_sha256`, which
refuses both a wholesale manifest swap and a release re-cut under the
same tag.

## Development

```bash
git clone https://github.com/visikai/ohlc-toolkit.git
cd ohlc-toolkit

uv sync --all-groups

uv run pytest
uv run mypy .
uv run ruff check .
```

`pytest` deselects the network-marked tests by default; run them with
`uv run pytest -m network` to fetch and verify the real published
release. `benchmarks/window_engine.py` measures the window engine over a
full minute history against independently computed references.

## Support

If you need any help or have any questions, please feel free to open an
issue or contact me directly.

We hope this repo makes your life easier! If it does, please give us a
star! ⭐
