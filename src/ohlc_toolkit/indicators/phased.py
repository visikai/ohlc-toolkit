"""The phased lookback, computed with joins rather than with scans.

The production counterpart to
:func:`~ohlc_toolkit.indicators.reference.phased_lookback_reference`. That
function is the normative one -- the rule, the refusals and the reasons
are stated there -- and this module is tested against it rather than
restating it.

The only thing worth saying here that is not said there is HOW. One left
join per phase, each on ``close_time == t - kW`` by exact equality, in
the shape :mod:`ohlc_toolkit.returns.alignment` already uses for the same
reason: a shift assumes the rows either side of a gap are neighbours, and
they are not. A tick whose ``t - kW`` is absent joins to null, and null is
the answer.
"""

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.indicators.frames import (
    PHASED_COLUMNS,
    PhasedGrid,
    PhasedLookback,
    resolve_phased_grid,
)
from ohlc_toolkit.temporal import Duration

logger = get_logger(__name__)

_TICK_KEY = "close_time"


def phased_lookback(  # noqa: PLR0913 - one keyword per schedule knob
    frame: pl.DataFrame,
    *,
    window: Duration | str,
    emit_every: Duration | str,
    anchor: Duration | str = "0s",
    lookback: int,
    min_traded_seconds: int = 0,
) -> PhasedLookback:
    """Assemble the phased lookback for every tick of the emit grid.

    Args:
        frame: A windowed-candle frame at SOURCE cadence, carrying the
            ten columns schema v2 declares.
        window: The window duration ``W`` the frame was aggregated over.
        emit_every: The emit cadence ``E`` the result is emitted on.
        anchor: The emit-grid anchor offset.
        lookback: How many phased windows ``L`` each tick consumes.
        min_traded_seconds: The recipe's threshold. A window below it is
            a null input, whatever quality mode the frame was written
            under -- report mode removes nothing, so a harness that
            trusted the frame to have been filtered would consume dead
            windows.

    Returns:
        One row per tick of the ``E`` grid inside the frame's range,
        carrying ``close_time`` and one list column per phased field,
        each of length ``lookback`` and ordered newest first. Every list
        column is null at a tick whose inputs are not all present and all
        at or above the threshold.

    Raises:
        ConfigError: For any frame or schedule
            :func:`~ohlc_toolkit.indicators.frames.resolve_phased_grid`
            refuses.

    """
    grid = resolve_phased_grid(
        frame,
        window=window,
        emit_every=emit_every,
        anchor=anchor,
        lookback=lookback,
        min_traded_seconds=min_traded_seconds,
    )
    logger.debug(
        "Phased lookback: {} tick(s) x {} phase(s) at W={}s, E={}s.",
        len(grid.ticks),
        grid.lookback,
        grid.window_seconds,
        grid.emit_seconds,
    )

    ticks = pl.DataFrame([pl.Series(_TICK_KEY, list(grid.ticks), dtype=pl.Int64)])
    admitted = _admitted(frame, grid)
    joined = ticks
    for phase in range(grid.lookback):
        joined = _join_phase(joined, admitted, grid, phase)

    # One mask over EVERY phase of EVERY field, not one per column. A
    # window with no candles in it reports null prices and a real zero
    # `src_count`, so a per-column rule would null the prices and keep
    # the counts, handing an indicator a row that is half a window.
    incomplete = pl.any_horizontal(
        [
            pl.col(_phase_name(name, phase)).is_null()
            for name in PHASED_COLUMNS
            for phase in range(grid.lookback)
        ]
    )
    return PhasedLookback(
        frame=joined.select(
            pl.col(_TICK_KEY),
            *(
                _phased_column(name, grid.lookback, incomplete)
                for name in PHASED_COLUMNS
            ),
        ),
        grid=grid,
    )


def _admitted(frame: pl.DataFrame, grid: PhasedGrid) -> pl.DataFrame:
    """Drop the windows the threshold excludes, before any join sees them.

    Removing them here rather than nulling them afterwards means a
    below-threshold window and an absent one reach the join in the same
    state, so exactly one rule decides what a missing input is.
    """
    # No `is_not_null` beside the comparison: polars drops a null
    # predicate, so a null `traded_seconds` fails the filter already.
    # Spelling it out was inert code inflating a coverage figure.
    return frame.filter(pl.col("traded_seconds") >= grid.min_traded_seconds).select(
        pl.col(_TICK_KEY), *(pl.col(name) for name in PHASED_COLUMNS)
    )


def _join_phase(
    joined: pl.DataFrame, admitted: pl.DataFrame, grid: PhasedGrid, phase: int
) -> pl.DataFrame:
    """Left-join one phase's window onto every tick, by exact equality."""
    offset = phase * grid.window_seconds
    lookup = admitted.select(
        (pl.col(_TICK_KEY) + offset).alias(_TICK_KEY),
        *(pl.col(name).alias(_phase_name(name, phase)) for name in PHASED_COLUMNS),
    )
    return joined.join(lookup, on=_TICK_KEY, how="left", maintain_order="left")


def _phase_name(name: str, phase: int) -> str:
    """Name one phase's copy of one field."""
    return f"{name}__{phase}"


def _phased_column(name: str, lookback: int, incomplete: pl.Expr) -> pl.Expr:
    """Gather one field's phases into a list, or null it with the row.

    All-or-nothing rather than a list with a hole in it: any null among
    the ``L`` inputs yields a null output, and a partial list would leave
    that rule to every indicator downstream to remember. The mask is the
    row's, so every column nulls together.
    """
    phases = [pl.col(_phase_name(name, phase)) for phase in range(lookback)]
    return pl.when(incomplete).then(None).otherwise(pl.concat_list(phases)).alias(name)
