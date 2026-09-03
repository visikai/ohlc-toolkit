"""A brute-force reference implementation of windowed candle aggregation.

This module is a correctness oracle, not a production engine. Every choice
here favours being literal over being fast: window membership is written
out as the two inequalities that define it, evaluated candle by candle and
tick by tick, with no shortcut that depends on the source being uniformly
spaced.

The window inclusion rule
-------------------------

At emit time ``t``, a finalized source candle spanning the half-open
interval ``[open_time, close_time)`` belongs to the window of duration
``W`` if and only if::

    candle.open_time >= t - W  AND  candle.close_time <= t

That is exact set containment in ``[t - W, t)``. Two consequences are
deliberate:

- A candle whose ``close_time`` is exactly ``t`` IS included. It is fully
  known at ``t``, so excluding it would throw away information the caller
  legitimately has.
- A candle overlapping either boundary is excluded WHOLE. Candles are
  never split, pro-rated, or partially credited.

Both inequalities are always evaluated. A source that has passed strict
validation is uniformly spaced and on a declared phase, which would let
the lower test be skipped as redundant -- and that is precisely the
shortcut a reference implementation must not take, because a fast engine
tested against it would then inherit the assumption instead of being
checked for it.

The skip_warmup materialization rule
------------------------------------

``MaterializationRule.SKIP_WARMUP`` derives the materialization range from
the source data instead of the caller stating it outright. Its behaviour
is DEFINED, not merely whatever the implementation happens to compute:

(a) The range starts at the first emit tick whose window is fully covered
    by source data (``coverage_seconds == window_seconds``). No earlier
    tick is emitted, because its window would be honestly incomplete, and
    skipping that incomplete lead-in is the whole point of the rule.
(b) The range ends one past the last emit grid tick at or before the
    source's greatest ``close_time``. A tick later than that would reach
    for data that does not exist, so it is never emitted -- even when the
    final close time itself is not on the emit grid.
(c) If no candidate tick is ever fully covered, resolving the range is a
    configuration error, raised rather than silently returning an empty
    result: this rule fails closed on purpose. Contrast an explicit
    :class:`ExplicitRange` with ``start == end``: that is also an empty
    result, but it is legal, because the caller asked for it directly.
    ``SKIP_WARMUP`` finding nothing to start from is a different
    situation -- nobody asked for empty, and returning it anyway would
    silently hide that this schedule cannot be honestly materialized over
    this data at all.

What is NOT here
----------------

Deciding whether a schedule is legal, normalizing the anchor, and saying
which instants the emit grid holds are not part of being an oracle: they
are the same decisions for every implementation, and this package holds
more than one. They live in :mod:`ohlc_toolkit.windows.resolution`, which
this module and the fast engine both import, so a schedule refused by one
is refused by the other with the same words. What stays here is the part
that has to be literal: the membership rule and the aggregation.

Cost
----

Resolution is O(rows + ticks); aggregation is O(rows x ticks), because
every source candle is tested against every emit tick. That is the point:
the cost buys an implementation that can be read against the
specification line by line.
"""

from collections.abc import Sequence
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

# Explicit ceilings on the brute-force work, with a reason: this oracle is
# quadratic, and a caller who asks for a huge grid deserves to be told
# before waiting. These warn rather than raise -- refusing a large but
# legitimate verification run would make the oracle useless exactly where
# it is most wanted -- but they are logged so the cost is never silent.
_MAX_UNWARNED_TICKS = 1_000_000
_MAX_UNWARNED_CANDLE_TICK_PAIRS = 100_000_000


@dataclass(frozen=True)
class _Candle:
    """One finalized source candle, with its interval already derived."""

    open_time: int
    close_time: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class _WindowRow:
    """One emitted window. The five price fields are null together."""

    open_time: int
    close_time: int
    open: float | None
    high: float | None
    low: float | None
    close: float | None
    volume: float | None
    src_count: int
    coverage_seconds: int


def compute_reference_windows(  # noqa: PLR0913 - one keyword per schedule knob
    frame: pl.DataFrame,
    profile: SourceProfile,
    *,
    window: Duration | str,
    emit_every: Duration | str,
    anchor: Duration | str = "0s",
    materialization: Materialization,
) -> pl.DataFrame:
    """Aggregate a raw source frame into windows, the slow and obvious way.

    Precondition: ``frame`` should already have passed strict validation
    (:func:`ohlc_toolkit.source.validation.validate_source_frame`). This
    function does not re-validate row data -- it will not detect a gap, a
    duplicate, an off-phase timestamp, or a null price -- because
    re-implementing those checks here would mean a second, divergent
    definition of a valid frame. It does enforce its own resolution-time
    rules on the schedule, listed under Raises below.

    The frame is never mutated, never sorted, never de-duplicated, and
    never repaired.

    Window membership is the rule documented in this module: a candle is
    included when ``open_time >= t - W`` and ``close_time <= t``, and is
    otherwise excluded whole.

    The output has exactly nine columns, in this order:

    - ``open_time`` (Int64): ``t - W``.
    - ``close_time`` (Int64): ``t``, the emit time. This is the canonical
      sort key, and rows come back ascending by it.
    - ``open`` (Float64): the ``open`` of the earliest included candle,
      by ``open_time``.
    - ``high`` (Float64): the greatest included ``high``.
    - ``low`` (Float64): the smallest included ``low``.
    - ``close`` (Float64): the ``close`` of the latest included candle,
      by ``open_time``.
    - ``volume`` (Float64): the sum of the included ``volume``.
    - ``src_count`` (UInt32): how many candles were included.
    - ``coverage_seconds`` (Int64): the sum, over the included candles,
      of ``close_time - open_time``.

    A tick that includes no candle still emits its row: all five
    price/volume columns are null, ``src_count`` is 0, and
    ``coverage_seconds`` is 0. The null volume is deliberate -- the
    absence of source data is a different observation from a real
    zero-volume candle, and writing 0.0 would erase that difference.

    Aligned windows are not a separate mode: they are ``emit_every`` equal
    to ``window``, with whatever anchor the caller recorded.

    Args:
        frame: The raw source frame, exactly as the provider published it.
        profile: The profile describing ``frame``'s timestamp column,
            cadence ``d``, and grid phase ``p``.
        window: The window duration ``W``.
        emit_every: The emit cadence ``E``. Emit ticks are the instants
            ``t`` with ``(t - anchor) mod E == 0``.
        anchor: The emit-grid anchor offset, normalized internally to
            ``anchor mod E`` so that equal grids compare equal. Defaults
            to no offset.
        materialization: Either an :class:`ExplicitRange` of Unix seconds,
            or :attr:`MaterializationRule.SKIP_WARMUP` (or its name,
            ``"skip_warmup"``). An explicit range emits every grid tick
            ``t`` with ``start <= t < end``, whether or not any data
            exists there -- including an empty range (``start == end``),
            which is legal because the caller stated it. ``SKIP_WARMUP``
            is a defined policy, not an inferred one: it starts at the
            first grid tick whose window is fully covered
            (``coverage_seconds == W``) and ends one past the last grid
            tick at or before the source's greatest ``close_time``. Unlike
            an explicit empty range, ``SKIP_WARMUP`` finding no fully
            covered tick is a configuration error -- it fails closed
            rather than silently returning zero rows. See the module
            docstring's "skip_warmup materialization rule" section.

    Returns:
        The window frame described above, one row per emit tick, ordered
        by ascending ``close_time``.

    Raises:
        ConfigError: If ``window`` or ``emit_every`` is not a whole
            multiple of the source cadence ``d``; if either is shorter
            than ``d``; if the resolved emit grid does not land on the
            source close-time grid (every tick must satisfy
            ``(t - p) mod d == 0``); if the profile or the frame is
            missing a column the aggregation needs; if the
            materialization argument is not a supported value; or if
            ``SKIP_WARMUP`` finds no fully covered tick -- deliberately
            fail-closed, unlike an explicit, caller-stated empty range.

    """
    schedule = resolve_schedule(
        profile, window=window, emit_every=emit_every, anchor=anchor
    )
    candles = _extract_candles(frame, profile)
    ticks = _resolve_ticks(candles, schedule, materialization)
    _log_brute_force_cost(len(candles), len(ticks))

    window_seconds = schedule.window.total_seconds
    rows = [_compute_window_row(candles, tick, window_seconds) for tick in ticks]
    return _build_output_frame(rows)


def _extract_candles(
    frame: pl.DataFrame, profile: SourceProfile
) -> tuple[_Candle, ...]:
    """Read the frame into plain candle records, in the frame's own row order.

    The interval bounds come from the profile
    (:meth:`~ohlc_toolkit.source.profile.SourceProfile.derive_interval_bounds`),
    so the oracle and the rest of the package agree on what a row's
    interval is without restating the arithmetic.
    """
    require_source_columns(frame, profile)

    bounds = profile.derive_interval_bounds(frame)
    open_times = bounds.get_column("open_time").to_list()
    close_times = bounds.get_column("close_time").to_list()
    prices = {
        name: frame.get_column(name).cast(pl.Float64).to_list()
        for name in OHLCV_COLUMNS
    }

    return tuple(
        _Candle(
            open_time=open_time,
            close_time=close_time,
            open=open_price,
            high=high,
            low=low,
            close=close_price,
            volume=volume,
        )
        for open_time, close_time, open_price, high, low, close_price, volume in zip(
            open_times,
            close_times,
            prices["open"],
            prices["high"],
            prices["low"],
            prices["close"],
            prices["volume"],
            strict=True,
        )
    )


def _included_candles(
    candles: Sequence[_Candle], tick: int, window_seconds: int
) -> list[_Candle]:
    """Select the candles contained in ``[tick - window_seconds, tick)``.

    This is the single implementation of the inclusion rule in this
    module: both the aggregation and the warmup scan go through it, so
    there is exactly one place where membership is decided.

    Only ``open_time`` and ``close_time`` are read, and both inequalities
    are always evaluated -- see the module docstring for why the lower one
    is not skipped even though validated input makes it redundant.
    """
    window_open = tick - window_seconds
    included = []
    for candle in candles:
        if candle.open_time >= window_open and candle.close_time <= tick:
            included.append(candle)
    return included


def _coverage_seconds(included: Sequence[_Candle]) -> int:
    """Sum the durations of the included candles, in whole seconds."""
    return sum(candle.close_time - candle.open_time for candle in included)


def _compute_window_row(
    candles: Sequence[_Candle], tick: int, window_seconds: int
) -> _WindowRow:
    """Aggregate one window by walking every candle once.

    ``open`` and ``close`` are taken from the candle with the smallest and
    largest ``open_time`` respectively, compared value by value rather
    than read off the ends of the frame: a reference implementation must
    not assume its input is sorted, even when a validated input is.
    """
    included = _included_candles(candles, tick, window_seconds)
    window_open = tick - window_seconds

    if not included:
        # The grid is total, so a tick with no data still emits. Null, not
        # zero: no observation is not the same as an observed zero.
        return _WindowRow(
            open_time=window_open,
            close_time=tick,
            open=None,
            high=None,
            low=None,
            close=None,
            volume=None,
            src_count=0,
            coverage_seconds=0,
        )

    earliest = included[0]
    latest = included[0]
    high = included[0].high
    low = included[0].low
    # Volume is accumulated in the frame's row order. Floating-point
    # addition is not associative, so the order has to be fixed by
    # something; the input's own order is the one choice that needs no
    # explanation.
    volume = 0.0
    coverage = 0
    for candle in included:
        if candle.open_time < earliest.open_time:
            earliest = candle
        if candle.open_time > latest.open_time:
            latest = candle
        high = max(high, candle.high)
        low = min(low, candle.low)
        volume += candle.volume
        coverage += candle.close_time - candle.open_time

    return _WindowRow(
        open_time=window_open,
        close_time=tick,
        open=earliest.open,
        high=high,
        low=low,
        close=latest.close,
        volume=volume,
        src_count=len(included),
        coverage_seconds=coverage,
    )


def _resolve_ticks(
    candles: Sequence[_Candle],
    schedule: ResolvedSchedule,
    materialization: Materialization,
) -> tuple[int, ...]:
    """Resolve the emit ticks for whichever materialization was requested."""
    resolved = coerce_materialization(materialization)
    if isinstance(resolved, ExplicitRange):
        return _grid_ticks(resolved.start, resolved.end, schedule)
    return _skip_warmup_ticks(candles, schedule)


def _grid_ticks(start: int, end: int, schedule: ResolvedSchedule) -> tuple[int, ...]:
    """List the emit ticks in the half-open range ``[start, end)``.

    The grid is ``{t : (t - anchor) mod E == 0}``, and where that grid
    starts inside ``[start, end)`` is
    :func:`~ohlc_toolkit.windows.resolution.first_tick_at_or_after`'s
    business, not this function's: the arithmetic is shared with the fast
    engine so both materialize the same instants.
    """
    emit_seconds = schedule.emit_every.total_seconds
    if end <= start:
        return ()

    first_tick = first_tick_at_or_after(start, schedule)
    tick_count = count_ticks(first_tick, end, emit_seconds)
    if tick_count > _MAX_UNWARNED_TICKS:
        logger.warning(
            "Materializing {} emit ticks over [{}, {}) at a {}s cadence; this "
            "reference implementation is quadratic in ticks times rows.",
            tick_count,
            start,
            end,
            emit_seconds,
        )
    return tuple(range(first_tick, end, emit_seconds))


def _skip_warmup_ticks(
    candles: Sequence[_Candle], schedule: ResolvedSchedule
) -> tuple[int, ...]:
    """Derive the materialization range from the data's own coverage.

    These are DEFINED semantics of this library, not incidental behaviour
    that happens to fall out of the implementation below:

    (a) The range starts at the first emit tick whose window is fully
        covered by source data (``coverage_seconds == window_seconds``).
    (b) The range ends one past the last emit tick at or before the
        greatest source ``close_time``. Any end in ``(last_tick,
        last_tick + E]`` selects the same ticks; one past the last tick
        is the smallest of them, and the most literal reading of the
        rule.
    (c) If no candidate tick is ever fully covered, that is a
        configuration error: this function raises rather than silently
        returning an empty range. This is a deliberate fail-closed
        choice. It is not the same situation as an explicit
        :class:`ExplicitRange` with ``start == end`` -- that empty range
        is legal because the caller asked for it directly, whereas here
        nobody asked for empty, and returning it anyway would hide that
        this schedule cannot be honestly materialized over this data.

    Only ticks in ``[earliest_open + W, latest_close]`` can possibly be
    fully covered, so only those are scanned: a window ending earlier
    reaches back before the first candle, and one ending later reaches
    past the last, leaving an uncovered stretch either way. This bound
    relies on the documented precondition that the frame passed strict
    validation and so holds no overlapping candles.

    Raises:
        ConfigError: If ``candles`` is empty, or if no candidate tick is
            fully covered -- see (c) above.

    """
    window_seconds = schedule.window.total_seconds
    emit_seconds = schedule.emit_every.total_seconds

    if not candles:
        logger.warning("Cannot skip warmup: the source frame holds no candles.")
        raise ConfigError(
            "Cannot resolve a skip_warmup range from an empty source frame: "
            "there is no coverage to measure."
        )

    earliest_open = min(candle.open_time for candle in candles)
    latest_close = max(candle.close_time for candle in candles)

    last_tick = last_tick_at_or_before(latest_close, schedule)
    first_candidate = first_tick_at_or_after(earliest_open + window_seconds, schedule)

    for tick in range(first_candidate, last_tick + 1, emit_seconds):
        included = _included_candles(candles, tick, window_seconds)
        if _coverage_seconds(included) == window_seconds:
            logger.debug(
                "Skipping warmup: first fully covered tick is {}, last tick is {}.",
                tick,
                last_tick,
            )
            return _grid_ticks(tick, last_tick + 1, schedule)

    logger.warning(
        "Cannot skip warmup: no emit tick in [{}, {}] is fully covered by a "
        "{}s window over {} candle(s).",
        first_candidate,
        last_tick,
        window_seconds,
        len(candles),
    )
    raise ConfigError(
        f"No emit tick is fully covered by a {window_seconds}s window over this "
        f"source: skip_warmup has no start tick to offer."
    )


def _log_brute_force_cost(candle_count: int, tick_count: int) -> None:
    """Record, and if extreme warn about, the work this call is about to do."""
    pair_count = candle_count * tick_count
    logger.debug(
        "Testing {} candle(s) against {} emit tick(s): {} membership test(s).",
        candle_count,
        tick_count,
        pair_count,
    )
    if pair_count > _MAX_UNWARNED_CANDLE_TICK_PAIRS:
        logger.warning(
            "Reference window aggregation will run {} membership test(s) "
            "({} candles x {} ticks); this is a brute-force oracle, not a "
            "production engine.",
            pair_count,
            candle_count,
            tick_count,
        )


def _build_output_frame(rows: Sequence[_WindowRow]) -> pl.DataFrame:
    """Assemble the nine output columns, with their names, dtypes, and order.

    The columns are built as explicitly typed series rather than inferred
    from the data, so an all-null or empty result carries exactly the same
    schema as a full one.
    """
    return pl.DataFrame(
        [
            pl.Series("open_time", [row.open_time for row in rows], dtype=pl.Int64),
            pl.Series("close_time", [row.close_time for row in rows], dtype=pl.Int64),
            pl.Series("open", [row.open for row in rows], dtype=pl.Float64),
            pl.Series("high", [row.high for row in rows], dtype=pl.Float64),
            pl.Series("low", [row.low for row in rows], dtype=pl.Float64),
            pl.Series("close", [row.close for row in rows], dtype=pl.Float64),
            pl.Series("volume", [row.volume for row in rows], dtype=pl.Float64),
            pl.Series("src_count", [row.src_count for row in rows], dtype=pl.UInt32),
            pl.Series(
                "coverage_seconds",
                [row.coverage_seconds for row in rows],
                dtype=pl.Int64,
            ),
        ]
    )
