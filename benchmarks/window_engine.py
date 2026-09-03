"""Measure and verify the window engine over a real minute history.

What this is for
----------------

:func:`ohlc_toolkit.windows.compute_windows` is checked for correctness
against the brute-force oracle in
:mod:`ohlc_toolkit.windows.reference`, but the oracle is quadratic, so
those checks can only run over a few thousand candles. This script is the
other half: it runs the engine over a full published minute history --
millions of rows -- and reports both what it costs and whether the
volume column's guarantees survive at that size.

For each window it records the wall time, the smallest volume emitted,
how many windows whose candles are all zero came back non-zero, and the
worst deviation from :func:`math.fsum` over a sampled set of windows.
Every one of those references is computed here, independently of the
engine, and none of them is quadratic:

- Which windows hold nothing but zero-volume candles is decided with two
  binary searches over the close times of the candles that DO carry
  volume. No arithmetic is involved, so the answer is exact.
- ``math.fsum`` sums exactly and rounds once, so it is the value any
  other summation of the same addends is approximating.

It also checks determinism, by running each window twice and comparing a
hash of the serialized output. Note the ``.rechunk()`` in
:func:`frame_digest`: polars is free to lay the same values out in
different chunks from one run to the next, and those chunk boundaries are
visible in the serialized bytes, so hashing without rechunking first
compares memory layout rather than data and reports differences that are
not there.

Running it
----------

The dataset path is an argument; nothing here knows or assumes where the
data lives::

    mise exec -- uv run python benchmarks/window_engine.py path/to/history.csv.gz

The file is read through
:func:`ohlc_toolkit.source.read_source_csv` under the published
:data:`~ohlc_toolkit.source.BITSTAMP_BTCUSD_1M` profile, in STRICT mode,
so a file that is not a complete, on-phase, strictly increasing minute
grid is refused at the read rather than quietly benchmarked.

``benchmarks/README.md`` holds a recorded run: environment, dataset
identity, and the numbers this script printed.
"""

import argparse
import hashlib
import io
import math
import random
import resource
import time
from dataclasses import dataclass

import polars as pl

from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M, ValidationMode, read_source_csv
from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.windows import MaterializationRule, compute_windows

# A geometric ladder of window lengths, each roughly 2.6 times the last,
# spanning one source candle up to a little over twelve days. It is a
# stress profile rather than a recommendation: it covers the single-candle
# window, the mid-range, and a window holding tens of thousands of
# candles, which is where a per-window summation is most expensive.
DEFAULT_WINDOW_MINUTES = (1, 3, 8, 21, 56, 146, 380, 993, 2590, 6758, 17632)

# How many windows are re-summed with math.fsum per window length. A few
# hundred is enough to characterize the deviation and cheap enough that
# the sampling does not dominate the measurement.
DEFAULT_SAMPLE_SIZE = 500

DEFAULT_SAMPLE_SEED = 20_260_903

_SECONDS_PER_MINUTE = 60

# How much of each SHA-256 digest to print. Sixteen hex characters is far
# more than enough to tell two runs apart by eye and keeps the table
# readable.
_DIGEST_PREFIX_LENGTH = 16


@dataclass(frozen=True)
class WindowMeasurement:
    """Everything one window length produced.

    Attributes:
        window_minutes: The window length, in whole source candles.
        emitted_rows: How many windows came back.
        seconds: Wall time of the first (timed) run.
        minimum_volume: The smallest volume emitted, over non-null rows.
        zero_volume_windows: How many emitted windows hold only
            zero-volume candles.
        zero_volume_windows_reporting_non_zero: How many of those did not
            come back as exactly ``0.0``. Must be zero.
        worst_relative_deviation: The largest relative gap from
            ``math.fsum`` over the sampled windows.
        sampled_windows: How many windows were re-summed exactly.
        digest: A hash of the rechunked output, for the determinism check.
        repeat_digest: The same hash from a second, independent run.

    """

    window_minutes: int
    emitted_rows: int
    seconds: float
    minimum_volume: float | None
    zero_volume_windows: int
    zero_volume_windows_reporting_non_zero: int
    worst_relative_deviation: float
    sampled_windows: int
    digest: str
    repeat_digest: str

    @property
    def is_deterministic(self) -> bool:
        """Report whether the two runs serialized to the same bytes."""
        return self.digest == self.repeat_digest


def frame_digest(frame: pl.DataFrame) -> str:
    """Hash a frame's contents, independently of how it is chunked.

    Args:
        frame: The frame to hash.

    Returns:
        A hex SHA-256 digest of the frame written as uncompressed IPC.

    """
    # The rechunk is load-bearing: chunk boundaries are part of the IPC
    # byte stream, and polars may split the same values differently
    # between runs or thread counts. Without it this hash compares memory
    # layout as much as data.
    buffer = io.BytesIO()
    frame.rechunk().write_ipc(buffer, compression="uncompressed")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def peak_memory_mib() -> float:
    """Return this process's peak resident set size, in MiB."""
    # ru_maxrss is in kibibytes on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def window_bounds(
    open_times: pl.Series, close_times: pl.Series, ticks: pl.Series, window_seconds: int
) -> tuple[pl.Series, pl.Series]:
    """Locate each window's candles as a half-open slice, by binary search.

    This restates the window rule -- ``open_time >= t - W`` and
    ``close_time <= t`` -- rather than asking the engine where it thought
    the window was, so the references built on it stay independent of the
    thing being measured.

    Args:
        open_times: Candle open times, ascending.
        close_times: Candle close times, in the same order.
        ticks: The emit times to locate windows for.
        window_seconds: The window duration ``W``.

    Returns:
        The first contained row and one past the last, per tick.

    """
    lower = open_times.search_sorted(ticks - window_seconds, side="left")
    upper = close_times.search_sorted(ticks, side="right")
    return lower.cast(pl.Int64), upper.cast(pl.Int64)


def count_zero_volume_windows(
    frame: pl.DataFrame,
    profile: SourceProfile,
    result: pl.DataFrame,
    window_seconds: int,
) -> tuple[int, int]:
    """Count the all-zero windows, and how many of them came back non-zero.

    A window holds only zero-volume candles exactly when it contains no
    candle whose volume is non-zero, which is the same window rule applied
    to a filtered set of candles. Two binary searches answer it for every
    tick at once, with no arithmetic anywhere, so the count is exact.

    Args:
        frame: The raw source frame.
        profile: The profile describing it.
        result: The engine's output.
        window_seconds: The window duration ``W``.

    Returns:
        How many emitted windows hold only zero volumes, and how many of
        those the engine did not report as exactly ``0.0``.

    """
    bounds = profile.derive_interval_bounds(frame)
    carries_volume = frame.get_column("volume") != 0.0
    lower, upper = window_bounds(
        bounds.get_column("open_time").filter(carries_volume),
        bounds.get_column("close_time").filter(carries_volume),
        result.get_column("close_time"),
        window_seconds,
    )

    # Only windows that actually held candles are in scope: a window with
    # no candles at all reports a null volume, not a zero one.
    holds_only_zeros = ((upper - lower) == 0) & (result.get_column("src_count") > 0)
    reported = result.get_column("volume").filter(holds_only_zeros)
    return int(holds_only_zeros.sum()), int((reported != 0.0).sum())


def worst_deviation_from_exact(  # noqa: PLR0913 - one argument per input series
    frame: pl.DataFrame,
    profile: SourceProfile,
    result: pl.DataFrame,
    *,
    window_seconds: int,
    sample_size: int,
    seed: int,
) -> tuple[float, int]:
    """Re-sum sampled windows with :func:`math.fsum` and report the worst gap.

    The sample is seeded and stratified: the leading and trailing windows,
    the largest and the smallest strictly positive reported volumes, and a
    random draw over the rest.

    Args:
        frame: The raw source frame.
        profile: The profile describing it.
        result: The engine's output.
        window_seconds: The window duration ``W``.
        sample_size: How many windows to draw at random.
        seed: The seed for that draw.

    Returns:
        The worst relative deviation observed, and how many windows were
        checked.

    """
    bounds = profile.derive_interval_bounds(frame)
    lower, upper = window_bounds(
        bounds.get_column("open_time"),
        bounds.get_column("close_time"),
        result.get_column("close_time"),
        window_seconds,
    )
    volumes = frame.get_column("volume").cast(pl.Float64).rechunk()
    reported = result.get_column("volume")

    positions = sample_positions(result, sample_size=sample_size, seed=seed)
    worst = 0.0
    for position in positions:
        start = int(lower[position])
        length = int(upper[position]) - start
        if length == 0:
            continue
        exact = math.fsum(volumes.slice(start, length).to_list())
        actual = reported[position]
        if exact == 0.0:
            # No relative gap to take: any non-zero here is pure residue,
            # so the absolute value is the honest number to report.
            worst = max(worst, abs(actual))
        else:
            worst = max(worst, abs(actual - exact) / abs(exact))
    return worst, len(positions)


def sample_positions(result: pl.DataFrame, *, sample_size: int, seed: int) -> list[int]:
    """Choose which emitted windows to re-sum exactly.

    Args:
        result: The engine's output.
        sample_size: How many windows to draw at random.
        seed: The seed for that draw.

    Returns:
        Sorted, de-duplicated row positions into ``result``.

    """
    height = result.height
    if height == 0:
        return []

    stratum = max(1, sample_size // 10)
    indexed = result.with_row_index("position").select("position", "volume")
    largest = indexed.sort("volume", descending=True, nulls_last=True).head(stratum)
    smallest = indexed.filter(pl.col("volume") > 0.0).sort("volume").head(stratum)

    rng = random.Random(seed)
    positions = {
        *range(min(stratum, height)),
        *range(max(0, height - stratum), height),
        *(int(value) for value in largest.get_column("position")),
        *(int(value) for value in smallest.get_column("position")),
        *(rng.randrange(height) for _ in range(sample_size)),
    }
    return sorted(positions)


def measure_window(  # noqa: PLR0913 - one argument per measurement knob
    frame: pl.DataFrame,
    profile: SourceProfile,
    *,
    window_minutes: int,
    emit_every: str,
    anchor: str,
    sample_size: int,
    seed: int,
) -> WindowMeasurement:
    """Run, time, and verify one window length.

    Args:
        frame: The raw source frame.
        profile: The profile describing it.
        window_minutes: The window length in whole minutes.
        emit_every: The emit cadence.
        anchor: The emit-grid anchor.
        sample_size: How many windows to re-sum exactly.
        seed: The seed for the random part of that sample.

    Returns:
        The measurement for this window length.

    """
    window = f"{window_minutes}m"
    arguments = {
        "window": window,
        "emit_every": emit_every,
        "anchor": anchor,
        "materialization": MaterializationRule.SKIP_WARMUP,
    }

    started = time.perf_counter()
    result = compute_windows(frame, profile, **arguments)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started

    repeat = compute_windows(frame, profile, **arguments)  # type: ignore[arg-type]

    window_seconds = window_minutes * _SECONDS_PER_MINUTE
    zero_windows, zero_windows_wrong = count_zero_volume_windows(
        frame, profile, result, window_seconds
    )
    worst, sampled = worst_deviation_from_exact(
        frame,
        profile,
        result,
        window_seconds=window_seconds,
        sample_size=sample_size,
        seed=seed,
    )
    minimum = result.get_column("volume").min()

    return WindowMeasurement(
        window_minutes=window_minutes,
        emitted_rows=result.height,
        seconds=elapsed,
        minimum_volume=None if minimum is None else float(minimum),  # type: ignore[arg-type]
        zero_volume_windows=zero_windows,
        zero_volume_windows_reporting_non_zero=zero_windows_wrong,
        worst_relative_deviation=worst,
        sampled_windows=sampled,
        digest=frame_digest(result),
        repeat_digest=frame_digest(repeat),
    )


def describe_dataset(frame: pl.DataFrame, profile: SourceProfile) -> str:
    """Summarize the dataset by shape and span, never by where it came from."""
    timestamps = frame.get_column(profile.timestamp_column)
    cadence_seconds = profile.cadence.total_seconds
    first_open = int(timestamps.min())  # type: ignore[arg-type]
    last_close = int(timestamps.max()) + cadence_seconds  # type: ignore[arg-type]
    return (
        f"{frame.height} rows, [{first_open}, {last_close}) Unix seconds, "
        f"{cadence_seconds}s cadence"
    )


def print_report(measurements: list[WindowMeasurement], total_seconds: float) -> None:
    """Print the results table and the totals beneath it.

    The digest column is printed, not just compared: two runs at different
    ``POLARS_MAX_THREADS`` settings are separate processes, so the only
    way to compare them is to read the same digests out of both reports.
    """
    header = (
        f"{'window':>10} {'rows':>10} {'seconds':>9} {'min volume':>14} "
        f"{'zero wins':>10} {'bad zeros':>10} {'worst dev':>11} {'sampled':>8} "
        f"{'repeats':>8} {'digest':>{_DIGEST_PREFIX_LENGTH}}"
    )
    print(header)
    print("-" * len(header))
    for measurement in measurements:
        print(
            f"{measurement.window_minutes:>9}m "
            f"{measurement.emitted_rows:>10} "
            f"{measurement.seconds:>9.2f} "
            f"{measurement.minimum_volume!r:>14} "
            f"{measurement.zero_volume_windows:>10} "
            f"{measurement.zero_volume_windows_reporting_non_zero:>10} "
            f"{measurement.worst_relative_deviation:>11.3e} "
            f"{measurement.sampled_windows:>8} "
            f"{'yes' if measurement.is_deterministic else 'NO':>8} "
            f"{measurement.digest[:_DIGEST_PREFIX_LENGTH]}"
        )
    print("-" * len(header))
    print(f"total engine time: {total_seconds:.2f} s")
    print(f"peak RSS: {peak_memory_mib():.0f} MiB")


def parse_arguments() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", help="path to a source CSV the profile describes")
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=list(DEFAULT_WINDOW_MINUTES),
        metavar="MINUTES",
        help="window lengths in whole minutes",
    )
    parser.add_argument("--emit-every", default="1m", help="emit cadence")
    parser.add_argument("--anchor", default="0s", help="emit-grid anchor offset")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="windows re-summed with math.fsum, per window length",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SAMPLE_SEED, help="sampling seed"
    )
    return parser.parse_args()


def main() -> int:
    """Read the dataset, measure every window, and print the report.

    Returns:
        ``0`` when every window kept every guarantee, ``1`` otherwise, so
        the script is usable as a check and not only as a report.

    """
    arguments = parse_arguments()
    profile = BITSTAMP_BTCUSD_1M

    started = time.perf_counter()
    frame = read_source_csv(arguments.dataset, profile, mode=ValidationMode.STRICT)
    print(f"dataset: {describe_dataset(frame, profile)}")
    print(f"read in {time.perf_counter() - started:.2f} s")
    print(f"emit every {arguments.emit_every}, anchor {arguments.anchor}\n")

    measurements = []
    total_seconds = 0.0
    for window_minutes in arguments.windows:
        measurement = measure_window(
            frame,
            profile,
            window_minutes=window_minutes,
            emit_every=arguments.emit_every,
            anchor=arguments.anchor,
            sample_size=arguments.sample_size,
            seed=arguments.seed,
        )
        measurements.append(measurement)
        total_seconds += measurement.seconds

    print_report(measurements, total_seconds)

    failures = [
        measurement
        for measurement in measurements
        if measurement.zero_volume_windows_reporting_non_zero > 0
        or (measurement.minimum_volume is not None and measurement.minimum_volume < 0.0)
        or not measurement.is_deterministic
    ]
    for measurement in failures:
        print(f"FAILED: {measurement.window_minutes}m")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
