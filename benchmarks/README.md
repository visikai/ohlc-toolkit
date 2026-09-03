# Benchmarks

`window_engine.py` runs `ohlc_toolkit.windows.compute_windows` over a full
published minute history and reports both what it costs and whether the
`volume` column's guarantees hold at that size.

The unit suite checks the engine against the brute-force oracle in
`ohlc_toolkit.windows.reference`, but the oracle is quadratic, so those
comparisons top out at a few thousand candles. Floating-point summation
error does not appear at that size. This is the other half of the story.

```bash
mise exec -- uv run python benchmarks/window_engine.py path/to/history.csv.gz
```

The dataset path is an argument — nothing in this directory knows or
assumes where the data lives. The file is read through
`ohlc_toolkit.source.read_source_csv` under the published
`BITSTAMP_BTCUSD_1M` profile in STRICT mode, so anything that is not a
complete, on-phase, strictly increasing minute grid is refused at the read
rather than quietly benchmarked. The script exits non-zero if any window
reports a negative volume, reports a non-zero total for a window whose
candles are all zero, or fails to reproduce itself, so it is usable as a
check and not only as a report.

## What it verifies, and against what

Every reference below is computed inside the script, independently of the
engine, and none of them is quadratic:

| Column | Reference |
|---|---|
| `min volume` | Nothing. Reported directly; summing non-negative volumes must never produce a negative. |
| `zero wins` / `bad zeros` | Two binary searches over the open and close times of the candles that *do* carry volume. A window holds only zeros exactly when no such candle falls inside it. No arithmetic, so the count is exact. |
| `worst dev` | `math.fsum` re-summing each sampled window's own candles. `fsum` sums exactly and rounds once, so it is the value any other summation of the same addends is approximating. |
| `repeats` / `digest` | A SHA-256 of the rechunked output, from two independent runs. |

The sample behind `worst dev` is seeded and stratified: the leading and
trailing windows, the largest reported volumes, the smallest strictly
positive ones, and a random draw over the rest.

## Recorded run

### Environment

| | |
|---|---|
| CPU | AMD Ryzen 7 7800X3D, 8 cores / 16 threads |
| Memory | 61 GiB |
| OS | Pop!\_OS 24.04 LTS, Linux 7.0.11, glibc 2.39 |
| Python | CPython 3.14.7 |
| polars | 1.44.1 |

### Dataset

The published Bitstamp BTC/USD one-minute history
(`btcusd_bitstamp_1min_2012-2025.csv.gz` from the public
[bitstamp-btcusd-minute-data](https://github.com/ff137/bitstamp-btcusd-minute-data)
dataset repository).

| | |
|---|---|
| Rows | 6,847,200 |
| Span | `[1325376060, 1736208060)` Unix seconds |
| | 2012-01-01T00:01:00Z up to 2025-01-07T00:01:00Z |
| Cadence | 60s, phase 0s |
| Shape | A complete, gap-free grid: the span is exactly 6,847,200 minutes, so every window recorded below is fully covered |

### Results

Emit cadence 1m, anchor 0s, `skip_warmup` materialization, 700 windows
re-summed with `math.fsum` per row. Run at `POLARS_MAX_THREADS=8`.

| window | rows | seconds | min volume | zero-only windows | reporting non-zero | worst dev vs `fsum` | digest |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1m | 6,847,200 | 5.02 | 0.0 | 1,305,607 | 0 | 0.000e+00 | `97a249e5cdc436b8` |
| 3m | 6,847,198 | 4.92 | 0.0 | 709,686 | 0 | 2.219e-16 | `569d2a52a2d62d36` |
| 8m | 6,847,193 | 5.07 | 0.0 | 486,339 | 0 | 3.931e-16 | `62183a33d5b3adc7` |
| 21m | 6,847,180 | 5.12 | 0.0 | 352,889 | 0 | 5.598e-16 | `e270dadf30425443` |
| 56m | 6,847,145 | 5.18 | 0.0 | 220,515 | 0 | 1.012e-15 | `c89b94d1b0d97d16` |
| 146m | 6,847,055 | 5.12 | 0.0 | 114,093 | 0 | 3.595e-16 | `f8b88c8d10215d60` |
| 380m | 6,846,821 | 5.61 | 0.0 | 43,013 | 0 | 4.387e-16 | `c58a378691c2587f` |
| 993m | 6,846,208 | 5.72 | 0.0 | 10,396 | 0 | 3.268e-16 | `b8d9f6226fa92859` |
| 2590m | 6,844,611 | 6.58 | 0.0 | 3,883 | 0 | 3.121e-16 | `6edac10b637477be` |
| 6758m | 6,840,443 | 8.83 | 95.15474131 | 0 | 0 | 2.499e-16 | `78ff0af0b2bbf2e2` |
| 17632m | 6,829,569 | 13.44 | 486.87267862 | 0 | 0 | 3.472e-16 | `147d480b00757891` |

- **Total engine time:** 70.60 s for all eleven windows.
- **Peak RSS:** 2,385 MiB (2,357 MiB single-threaded).
- **Smallest volume anywhere:** `0.0`. Never negative, in any window.
- **Zero-only windows reporting non-zero:** 0, out of 3,246,421 such
  windows across the ladder. (The previous sliding sum got 1,294,807 of
  them wrong — see "The cost of summing per window" below.)
- **Worst deviation from `math.fsum`:** 1.012e-15 relative, at the 56m
  window — about 9 units in the last place. Every other window is inside
  6e-16.

### Determinism

Two independent runs per window, in the same process, hashed with:

```python
buffer = io.BytesIO()
frame.rechunk().write_ipc(buffer, compression="uncompressed")
hashlib.sha256(buffer.getvalue()).hexdigest()
```

The `.rechunk()` is load-bearing and easy to leave out. Chunk boundaries
are part of the IPC byte stream, and polars is free to split the same
values differently between runs, so hashing without rechunking first
compares memory layout as much as data and reports differences that are
not there.

**Verdict: identical.** Every digest above reproduced across the two
in-process passes, and every one of the eleven digests is byte-identical
between a run at `POLARS_MAX_THREADS=8` and a run at
`POLARS_MAX_THREADS=1` — separate processes, so this is a cross-process
comparison, not just a repeat.

Single-threaded totals for reference: 74.09 s of engine time, same
digests, same deviations.

### The cost of summing per window

`volume` is summed over each window's own contiguous slice rather than
with a running total that adds the entering candle and subtracts the
leaving one. That is what makes the column a function of the candles the
window contains, and it is not free. The same ladder, over the same
dataset, with the previous sliding sum:

| window | sliding sum | per-window sum | ratio |
|---:|---:|---:|---:|
| 1m | 2.47 s | 5.02 s | 2.0x |
| 993m | 2.52 s | 5.72 s | 2.3x |
| 6758m | 2.54 s | 8.83 s | 3.5x |
| 17632m | 2.55 s | 13.44 s | 5.3x |
| **total** | **27.84 s** | **70.60 s** | **2.5x** |

The old profile was flat in the window length; this one is not, because
re-summing every window costs O(sum of window lengths) rather than
O(rows). Over a complete grid that is `ticks * W / d`, so the 17632-minute
window does about 1.2e11 additions. It still finishes in thirteen seconds
because each window's slice is contiguous, already in cache from the
window before it, and summed by a vectorized polars reduction.

What the extra 43 seconds buys is measurable on the same run. Over this
ladder the previous sliding sum produced:

| | sliding sum | per-window sum |
|---|---:|---:|
| zero-only windows reporting non-zero | 1,294,807 of 3,246,421 | 0 |
| rows with a negative volume | 655,862 | 0 |
| window lengths emitting a negative at all | 6 of 11 | 0 of 11 |
| worst negative volume | -4.234834705130197e-12 | — |

and, less visibly but more seriously, a `volume` that changed value when
more history was prepended in front of the same window: not a function of
the window's contents at all, and not answerable by recomputing only the
tail after an append.

## Checking this directory

The scripts here are linted and formatted like the rest of the
repository — `[tool.ruff] include` in `pyproject.toml` lists
`benchmarks/*.py`.

```bash
mise run fmt:check   # read-only: reports without touching files
mise run lint        # NOT read-only: runs `ruff check --fix` and `ruff format`
```

Use `fmt:check` when you want to know whether the tree is clean. `lint`
will make it clean, which is a different thing.
