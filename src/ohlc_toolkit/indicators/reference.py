"""A brute-force phased-lookback harness, written before the fast one.

The specification is here and the engine is what you run, exactly as
:mod:`ohlc_toolkit.windows.reference` stands to
:mod:`ohlc_toolkit.windows.engine`. This module scans the frame for every
lookup it needs and assembles the answer a row at a time; it is quadratic
and says so, and the point of it is that its correctness is readable
rather than argued.

The rule, stated once so the fast path can be tested against it rather
than restating it:

- At each emit tick ``t`` of the ``E`` grid, a phased indicator consumes
  the windows of duration ``W`` ending at ``t``, ``t - W``, ``t - 2W``,
  ..., ``L`` of them. Consecutive inputs do not overlap, and the phase
  set is anchored at the EMIT TICK rather than at the window's anchor.
- ``{t - kW}`` lies on the ``E`` grid only when ``E`` divides ``W``,
  which under the accepted cadence rules it often does not. So the inputs
  are read from the window's materialization at SOURCE cadence, where
  every ``t - kW`` is a legal tick, and the result is emitted on the
  ``E`` grid.
- A window whose ``traded_seconds`` falls below the recipe's threshold is
  a NULL INPUT, whatever quality mode the frame was written under --
  report mode removes nothing, so a harness that trusted the frame to
  have been filtered would silently consume dead windows.
- Any null among the ``L`` inputs yields a null output, and lookups are
  exact equality on ``close_time``. No tolerance, no nearest match, no
  as-of, no fill.
"""

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.indicators.frames import (
    PHASED_COLUMNS,
    PhasedGrid,
    PhasedLookback,
    resolve_phased_grid,
)

logger = get_logger(__name__)


def phased_lookback_reference(  # noqa: PLR0913 - one keyword per schedule knob
    frame: pl.DataFrame,
    *,
    window: str,
    emit_every: str,
    anchor: str = "0s",
    lookback: int,
    min_traded_seconds: int = 0,
) -> PhasedLookback:
    """Assemble the phased lookback the slow and obvious way.

    Args:
        frame: A windowed-candle frame at SOURCE cadence, carrying the
            ten columns schema v2 declares.
        window: The window duration ``W`` the frame was aggregated over.
        emit_every: The emit cadence ``E`` the result is emitted on.
        anchor: The emit-grid anchor offset.
        lookback: How many phased windows ``L`` each tick consumes.
        min_traded_seconds: The recipe's threshold. A window below it is
            a null input.

    Returns:
        One row per tick of the ``E`` grid inside the frame's range,
        carrying ``close_time`` and one list column per phased field,
        each of length ``lookback`` and ordered newest first. Every list
        column is null at a tick whose inputs are not all present and all
        at or above the threshold.

    Raises:
        ConfigError: For any frame or schedule the shared resolution
            refuses; see :func:`~ohlc_toolkit.indicators.frames.resolve_phased_grid`.

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
        "Reference phased lookback: {} tick(s) x {} phase(s) = {} lookup(s).",
        len(grid.ticks),
        grid.lookback,
        len(grid.ticks) * grid.lookback,
    )
    rows: dict[str, list[list[object] | None]] = {name: [] for name in PHASED_COLUMNS}
    for tick in grid.ticks:
        assembled = _assemble(frame, grid, tick)
        for name in PHASED_COLUMNS:
            rows[name].append(None if assembled is None else assembled[name])
    return PhasedLookback(
        frame=pl.DataFrame(
            [
                pl.Series("close_time", list(grid.ticks), dtype=pl.Int64),
                *(
                    pl.Series(name, rows[name], dtype=pl.List(dtype))
                    for name, dtype in PHASED_COLUMNS.items()
                ),
            ]
        ),
        grid=grid,
    )


def _assemble(
    frame: pl.DataFrame, grid: PhasedGrid, tick: int
) -> dict[str, list[object]] | None:
    """Gather one tick's ``L`` phased windows, or report that it cannot.

    Every lookup is a scan of the whole frame for an exact ``close_time``
    match. That is the quadratic part, and it is deliberate: a reference
    implementation that indexed would be testing the index.
    """
    gathered: dict[str, list[object]] = {name: [] for name in PHASED_COLUMNS}
    window_seconds = grid.window_seconds
    for phase in range(grid.lookback):
        wanted = tick - phase * window_seconds
        matched = [
            row for row in frame.iter_rows(named=True) if row["close_time"] == wanted
        ]
        if len(matched) != 1:
            return None
        row = matched[0]
        if (
            row["traded_seconds"] is None
            or row["traded_seconds"] < grid.min_traded_seconds
        ):
            return None
        for name in PHASED_COLUMNS:
            if row[name] is None:
                return None
            gathered[name].append(row[name])
    return gathered
