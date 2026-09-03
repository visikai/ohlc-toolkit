"""A Polars-native batch engine for windowed candle aggregation.

This module computes exactly what
:func:`~ohlc_toolkit.windows.reference.compute_reference_windows` computes
-- same rule, same schema, same total emit grid, same refusals -- without
paying its O(rows x ticks) cost. The oracle is the specification; this is
the implementation meant to be run. Where the two could disagree, the
oracle is right by definition.

Why the included candles are a contiguous slice
-----------------------------------------------

A profile derives every candle's interval as ``[ts, ts + d)``
(:meth:`~ohlc_toolkit.source.profile.SourceProfile.derive_interval_bounds`),
so ``close_time`` is ``open_time`` plus the SAME constant on every row.
Sorting by ``open_time`` therefore sorts by ``close_time`` too, and each
of the two membership inequalities cuts that sorted frame at a single
point:

- ``open_time >= t - W`` cuts off a prefix,
- ``close_time <= t`` cuts off a suffix.

The candles in the window are the rows between the two cuts. Two binary
searches locate them, one per inequality, for every emit tick at once.

Nothing in that argument assumes anything about the DATA. A frame whose
rows are unsorted, duplicated, off the declared phase, overlapping, or
riddled with gaps still sorts, and the two cuts still land in the right
places. What it does rest on is that all rows share one duration, which
is a property of how the interval is derived rather than of what the
provider published.

Both inequalities, every time
-----------------------------

Over a frame that has passed strict source validation the lower
inequality is redundant: the upper one alone would select the same
candles. It is still evaluated. The engine is not permitted to be correct
only for the inputs a validator would have accepted, and the synthetic
family of boundary-straddling candles exists precisely to punish an
implementation that drops it.

The three aggregates that are not slice endpoints
-------------------------------------------------

``open`` and ``close`` are read off the ends of the slice, and
``src_count`` and ``coverage_seconds`` are its length. ``high``, ``low``
and ``volume`` are not: they have to look at every candle in the window.
Restating membership on ``close_time`` alone -- legal because
``close_time`` is ``open_time + d``, so a two-sided bound on either is a
two-sided bound on the other -- gives ``close_time`` in
``[t - W + d, t]``, and that is exactly the window polars' rolling-by-key
aggregations compute when they reach back ``W - d + 1`` seconds from each
row.

Those aggregations run over a frame holding one row per source candle AND
one row per emit tick, merged on that key. Tick rows carry the identity
element of each aggregate (negative infinity for a maximum, positive
infinity for a minimum, zero for a sum), so a tick can never change the
answer for the ticks around it, and the tick rows are where the answers
are read back out. Every column is then masked to null wherever the
window held no candle, because an absent observation is not an observed
zero.

Aligned emission is not a mode
------------------------------

``E == W`` runs the same code as every other schedule. A bucketed fast
path -- aggregating each emit bucket once and combining buckets -- was
considered and not taken: it pays off only in the aligned case, it cannot
express ``E < W`` without rebuilding the rolling machinery anyway, and an
unproven shortcut in the one place callers most often look is a bad
trade. Correctness against the oracle outranks a constant factor here.

Cost
----

One O(rows log rows) sort, skipped when the frame already arrives in
ascending open-time order; then O(rows + ticks) for the merge and the
rolling passes, and O(ticks log rows) for the binary searches. Memory is
O(rows + ticks): the engine materializes one row per candle and one row
per emit tick, and nothing that scales with the calendar span between
them.

Floating point
--------------

``volume`` is the only output that is neither an integer nor a value
copied straight out of the input. The oracle folds each window's volumes
left to right in the frame's own row order; this engine sums them with a
vectorized rolling sum. Both are correct sums of the same addends and can
differ only in the order the rounding happens: measured against the
oracle over real minute volumes the two agree to within about 1e-13
relative, and they agree bit for bit wherever the addends and their
partial sums are exactly representable. Neither number is more true than
the other -- the oracle's own fold carries comparable rounding -- but the
engine's is the one that will move if polars changes how it sums, so a
caller comparing outputs across versions should compare volumes with a
tolerance. Every other column is exact.
"""

from dataclasses import dataclass

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.windows.resolution import (
    OHLCV_COLUMNS,
    ExplicitRange,
    Materialization,
    ResolvedSchedule,
    coerce_materialization,
    count_ticks,
    first_tick_at_or_after,
    last_tick_at_or_before,
    require_source_columns,
    resolve_schedule,
)

logger = get_logger(__name__)

# The engine holds one row per source candle and one row per emit tick at
# once, so an enormous grid is an enormous allocation. This warns rather
# than raises -- a long history at a fine emit cadence is a legitimate
# thing to ask for -- but the cost is never silent.
_MAX_UNWARNED_TICKS = 20_000_000

# Identity elements, carried by the emit-tick rows of the merged frame so
# that a tick never perturbs the aggregate of a neighbouring tick. They
# are not markers: an empty window is recognised by its candle count, not
# by finding one of these values, so a source that genuinely reports an
# infinite price is reported back rather than mistaken for emptiness.
_NEUTRAL_HIGH = float("-inf")
_NEUTRAL_LOW = float("inf")
_NEUTRAL_VOLUME = 0.0

# Column names used only inside the merged candle/tick frame.
_KEY_COLUMN = "key"
_IS_TICK_COLUMN = "is_tick"


@dataclass(frozen=True)
class _SortedCandles:
    """Every source candle, ascending by open time, ties in frame order.

    Ties keep the frame's original row order because the contract's
    tie-breaks are stated in terms of it: with two candles sharing an open
    time, the window's ``open`` and ``close`` come from whichever the
    provider published first.

    Attributes:
        open_time: Each candle's interval open, ascending.
        close_time: Each candle's interval close, in the same row order.
        open: Opening prices.
        high: Highest prices.
        low: Lowest prices.
        close: Closing prices.
        volume: Traded volumes.
        cadence_seconds: The profile's cadence ``d``, which is every
            candle's duration.

    """

    open_time: pl.Series
    close_time: pl.Series
    open: pl.Series
    high: pl.Series
    low: pl.Series
    close: pl.Series
    volume: pl.Series
    cadence_seconds: int

    @property
    def count(self) -> int:
        """Return how many candles the source frame held."""
        return self.open_time.len()


def compute_windows(  # noqa: PLR0913 - one keyword per schedule knob
    frame: pl.DataFrame,
    profile: SourceProfile,
    *,
    window: Duration | str,
    emit_every: Duration | str,
    anchor: Duration | str = "0s",
    materialization: Materialization,
) -> pl.DataFrame:
    """Aggregate a raw source frame into windows, vectorized.

    This is the production counterpart to
    :func:`~ohlc_toolkit.windows.reference.compute_reference_windows`. The
    parameter list, the window rule, the emit grid, the materialization
    semantics, the nine output columns and every resolution-time refusal
    are the same; only the cost differs. Read that function's docstring
    for the contract -- it is the normative one, and this module is tested
    against it rather than restating it.

    Precondition: ``frame`` should already have passed strict validation
    (:func:`ohlc_toolkit.source.validation.validate_source_frame`). Like
    the oracle, this function does not re-validate row data and will not
    detect a gap, a duplicate, an off-phase timestamp, or a null price. It
    enforces only its own resolution-time rules on the schedule.

    The frame is never mutated, sorted in place, de-duplicated, or
    repaired. The engine sorts a copy of the columns it reads when the
    input does not already arrive in ascending open-time order.

    Args:
        frame: The raw source frame, exactly as the provider published it.
        profile: The profile describing ``frame``'s timestamp column,
            cadence ``d``, and grid phase ``p``.
        window: The window duration ``W``.
        emit_every: The emit cadence ``E``. Emit ticks are the instants
            ``t`` with ``(t - anchor) mod E == 0``.
        anchor: The emit-grid anchor offset, normalized internally to
            ``anchor mod E``. Defaults to no offset.
        materialization: Either an
            :class:`~ohlc_toolkit.windows.resolution.ExplicitRange` of
            Unix seconds, or
            :attr:`~ohlc_toolkit.windows.resolution.MaterializationRule.SKIP_WARMUP`
            (or its name, ``"skip_warmup"``).

    Returns:
        A nine-column window frame -- ``open_time``, ``close_time``,
        ``open``, ``high``, ``low``, ``close``, ``volume``, ``src_count``,
        ``coverage_seconds`` -- with one row per emit tick, ordered by
        ascending ``close_time``.

    Raises:
        ConfigError: Under exactly the conditions the oracle raises it:
            an unresolvable schedule, a profile or frame missing a needed
            column, an unsupported materialization argument, or
            ``SKIP_WARMUP`` finding no fully covered tick.

    """
    schedule = resolve_schedule(
        profile, window=window, emit_every=emit_every, anchor=anchor
    )
    candles = _extract_candles(frame, profile)
    ticks = _resolve_ticks(candles, schedule, materialization)
    logger.debug(
        "Aggregating {} candle(s) into {} window(s) of {}s.",
        candles.count,
        ticks.len(),
        schedule.window.total_seconds,
    )
    return _build_output_frame(candles, ticks, schedule)


def _extract_candles(frame: pl.DataFrame, profile: SourceProfile) -> _SortedCandles:
    """Read the columns the aggregation needs, in ascending open-time order.

    The sort is skipped when the frame already arrives ordered, which is
    the normal case for a validated source and saves the largest single
    allocation the engine makes. When it does run it is stable, so candles
    sharing an open time keep the order the provider published them in --
    the order the contract's tie-breaks are stated in.
    """
    require_source_columns(frame, profile)

    bounds = profile.derive_interval_bounds(frame)
    prepared = pl.DataFrame(
        [
            bounds.get_column("open_time"),
            bounds.get_column("close_time"),
            *(frame.get_column(name).cast(pl.Float64) for name in OHLCV_COLUMNS),
        ]
    )
    if not prepared.get_column("open_time").is_sorted():
        logger.debug("Source frame is not in ascending open-time order; sorting.")
        prepared = prepared.sort("open_time", maintain_order=True)

    return _SortedCandles(
        open_time=prepared.get_column("open_time"),
        close_time=prepared.get_column("close_time"),
        open=prepared.get_column("open"),
        high=prepared.get_column("high"),
        low=prepared.get_column("low"),
        close=prepared.get_column("close"),
        volume=prepared.get_column("volume"),
        cadence_seconds=profile.cadence.total_seconds,
    )


def _resolve_ticks(
    candles: _SortedCandles,
    schedule: ResolvedSchedule,
    materialization: Materialization,
) -> pl.Series:
    """Resolve the emit ticks for whichever materialization was requested."""
    resolved = coerce_materialization(materialization)
    if isinstance(resolved, ExplicitRange):
        return _grid_ticks(resolved.start, resolved.end, schedule)
    return _skip_warmup_ticks(candles, schedule)


def _grid_ticks(start: int, end: int, schedule: ResolvedSchedule) -> pl.Series:
    """Materialize the emit ticks in the half-open range ``[start, end)``."""
    emit_seconds = schedule.emit_every.total_seconds
    if end <= start:
        return pl.Series("close_time", [], dtype=pl.Int64)

    first_tick = first_tick_at_or_after(start, schedule)
    tick_count = count_ticks(first_tick, end, emit_seconds)
    if tick_count > _MAX_UNWARNED_TICKS:
        logger.warning(
            "Materializing {} emit ticks over [{}, {}) at a {}s cadence; this "
            "engine holds one row per tick and one per source candle at once.",
            tick_count,
            start,
            end,
            emit_seconds,
        )
    return pl.int_range(
        first_tick, end, emit_seconds, dtype=pl.Int64, eager=True
    ).rename("close_time")


def _skip_warmup_ticks(
    candles: _SortedCandles, schedule: ResolvedSchedule
) -> pl.Series:
    """Derive the materialization range from the data's own coverage.

    The semantics are the oracle's, stated in its
    :func:`~ohlc_toolkit.windows.reference.compute_reference_windows`
    docstring: start at the first emit tick whose window is fully covered,
    end one past the last emit tick at or before the source's greatest
    close time, and fail closed when no tick is ever fully covered.

    Full coverage is tested as a candle count rather than a sum of
    durations. Coverage is the sum of ``close_time - open_time`` over the
    included candles, every candle's interval is exactly one cadence long,
    and resolution has already established that the cadence divides the
    window -- so ``coverage == W`` is the same statement as
    ``src_count == W / d``, checked without accumulating anything.

    Raises:
        ConfigError: If the frame holds no candles, or if no candidate
            tick is fully covered.

    """
    window_seconds = schedule.window.total_seconds

    if candles.count == 0:
        logger.warning("Cannot skip warmup: the source frame holds no candles.")
        raise ConfigError(
            "Cannot resolve a skip_warmup range from an empty source frame: "
            "there is no coverage to measure."
        )

    earliest_open = int(candles.open_time.min())  # type: ignore[arg-type]
    latest_close = int(candles.close_time.max())  # type: ignore[arg-type]
    last_tick = last_tick_at_or_before(latest_close, schedule)
    first_candidate = first_tick_at_or_after(earliest_open + window_seconds, schedule)

    candidates = _grid_ticks(first_candidate, last_tick + 1, schedule)
    lower, upper = _window_bounds(candles, candidates, schedule)
    fully_covered = (upper - lower) == window_seconds // candles.cadence_seconds

    positions = fully_covered.arg_true()
    if positions.is_empty():
        logger.warning(
            "Cannot skip warmup: no emit tick in [{}, {}] is fully covered by a "
            "{}s window over {} candle(s).",
            first_candidate,
            last_tick,
            window_seconds,
            candles.count,
        )
        raise ConfigError(
            f"No emit tick is fully covered by a {window_seconds}s window over this "
            f"source: skip_warmup has no start tick to offer."
        )

    first_position = int(positions[0])
    logger.debug(
        "Skipping warmup: first fully covered tick is {}, last tick is {}.",
        candidates[first_position],
        last_tick,
    )
    return candidates.slice(first_position)


def _window_bounds(
    candles: _SortedCandles, ticks: pl.Series, schedule: ResolvedSchedule
) -> tuple[pl.Series, pl.Series]:
    """Locate each window's candles as a half-open slice of the sorted frame.

    One binary search per membership inequality, both evaluated for every
    tick: ``lower`` is the first row whose open time reaches the window
    open, ``upper`` is one past the last row whose close time has arrived
    by the emit tick. ``upper`` is never below ``lower`` because ``W`` is
    never below ``d``, so the slice is always well formed, and an empty
    window is simply the two cuts landing on the same row.
    """
    window_seconds = schedule.window.total_seconds
    lower = candles.open_time.search_sorted(ticks - window_seconds, side="left")
    upper = candles.close_time.search_sorted(ticks, side="right")
    return lower.cast(pl.Int64), upper.cast(pl.Int64)


def _boundary_prices(
    candles: _SortedCandles, lower: pl.Series, upper: pl.Series
) -> tuple[pl.Series, pl.Series]:
    """Read each window's opening and closing price off the ends of its slice.

    ``open`` belongs to the included candle with the smallest open time,
    and ``lower`` already points at it. ``close`` belongs to the one with
    the largest open time -- and when several candles share that open
    time, to whichever of them the provider published first, which is not
    the last row of the slice but the first row of that tied run. A third
    binary search finds where the run starts. The whole tied run is either
    inside the window or outside it (they share an open time, so they
    share both verdicts), so that start is never outside the slice.

    Indices are clamped into range and the results are read for every
    tick, including ticks whose window is empty; those values are
    discarded by the caller's emptiness mask rather than by branching per
    row.
    """
    if candles.count == 0:
        empty = pl.Series("open", [None] * lower.len(), dtype=pl.Float64)
        return empty, empty.rename("close")

    highest_position = (upper - 1).clip(0, candles.count - 1)
    highest_open_time = candles.open_time.gather(highest_position)
    tie_start = candles.open_time.search_sorted(highest_open_time, side="left")

    return (
        candles.open.gather(lower.clip(0, candles.count - 1)),
        candles.close.gather(tie_start),
    )


def _merge_candles_and_ticks(candles: _SortedCandles, ticks: pl.Series) -> pl.DataFrame:
    """Interleave the source candles and the emit ticks on one sorted key.

    Both inputs are already ascending, so this is a linear merge rather
    than a second sort. The key is ``close_time`` for a candle and the
    tick itself for a tick, which is the axis the rolling window below is
    expressed on.
    """
    candle_rows = pl.DataFrame(
        [
            candles.close_time.rename(_KEY_COLUMN),
            pl.repeat(False, candles.count, dtype=pl.Boolean, eager=True).rename(
                _IS_TICK_COLUMN
            ),
            candles.high.rename("high"),
            candles.low.rename("low"),
            candles.volume.rename("volume"),
        ]
    )
    tick_count = ticks.len()
    tick_rows = pl.DataFrame(
        [
            ticks.rename(_KEY_COLUMN),
            pl.repeat(True, tick_count, dtype=pl.Boolean, eager=True).rename(
                _IS_TICK_COLUMN
            ),
            pl.repeat(_NEUTRAL_HIGH, tick_count, dtype=pl.Float64, eager=True).rename(
                "high"
            ),
            pl.repeat(_NEUTRAL_LOW, tick_count, dtype=pl.Float64, eager=True).rename(
                "low"
            ),
            pl.repeat(_NEUTRAL_VOLUME, tick_count, dtype=pl.Float64, eager=True).rename(
                "volume"
            ),
        ]
    )
    return candle_rows.merge_sorted(tick_rows, key=_KEY_COLUMN)


def _rolling_aggregates(
    candles: _SortedCandles, ticks: pl.Series, schedule: ResolvedSchedule
) -> pl.DataFrame:
    """Compute high, low and volume for every emit tick in three passes.

    Membership restated on the merge key: a candle belongs to the window
    closing at ``t`` exactly when its ``close_time`` lies in
    ``[t - W + d, t]``. Over an integer key that is a rolling window
    reaching back ``W - d + 1`` seconds and closed on the right, which is
    positive even when ``W == d`` -- the single-candle window -- so no
    special case is needed for it.
    """
    window_size = f"{schedule.window.total_seconds - candles.cadence_seconds + 1}i"
    merged = _merge_candles_and_ticks(candles, ticks)
    return (
        merged.with_columns(
            pl.col("high").rolling_max_by(
                _KEY_COLUMN, window_size=window_size, closed="right"
            ),
            pl.col("low").rolling_min_by(
                _KEY_COLUMN, window_size=window_size, closed="right"
            ),
            pl.col("volume").rolling_sum_by(
                _KEY_COLUMN, window_size=window_size, closed="right"
            ),
        )
        .filter(pl.col(_IS_TICK_COLUMN))
        .select("high", "low", "volume")
    )


def _build_output_frame(
    candles: _SortedCandles, ticks: pl.Series, schedule: ResolvedSchedule
) -> pl.DataFrame:
    """Assemble the nine output columns, with their names, dtypes, and order.

    Every column is computed for every tick and then masked: a tick whose
    window held no candle reports null prices and a null volume, never a
    zero price and never a dropped row. Null rather than zero is the
    contract's choice -- the absence of an observation is not the same
    fact as an observed zero.
    """
    lower, upper = _window_bounds(candles, ticks, schedule)
    open_price, close_price = _boundary_prices(candles, lower, upper)
    rolling = _rolling_aggregates(candles, ticks, schedule)

    staged = pl.DataFrame(
        [
            ticks.rename("close_time"),
            (upper - lower).rename("src_count"),
            open_price.rename("open"),
            rolling.get_column("high"),
            rolling.get_column("low"),
            close_price.rename("close"),
            rolling.get_column("volume"),
        ]
    )

    has_data = pl.col("src_count") > 0
    return staged.select(
        (pl.col("close_time") - schedule.window.total_seconds)
        .cast(pl.Int64)
        .alias("open_time"),
        pl.col("close_time").cast(pl.Int64),
        *(
            pl.when(has_data)
            .then(pl.col(name))
            .otherwise(None)
            .cast(pl.Float64)
            .alias(name)
            for name in OHLCV_COLUMNS
        ),
        pl.col("src_count").cast(pl.UInt32),
        # Coverage is the sum of `close_time - open_time` over the
        # included candles. Every candle's interval is exactly one cadence
        # long, so that sum is the cadence times the count: an exact
        # integer with nothing to accumulate.
        (pl.col("src_count") * candles.cadence_seconds)
        .cast(pl.Int64)
        .alias("coverage_seconds"),
    )
