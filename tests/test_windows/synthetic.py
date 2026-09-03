"""Seeded synthetic source-frame families and the committed golden matrix.

Every frame here is generated from a fixed integer seed. There is no wall
clock, no environment lookup, and no randomness outside those seeds, so
running this module a year from now on another machine produces the same
bytes it produced when the goldens below were committed.

Prices and volumes are always whole multiples of a quarter unit. Quarters
are exact in binary floating point at these magnitudes, so a window's
volume total is independent of the order the addition happens in -- which
keeps the committed CSV goldens byte-stable rather than
summation-order-stable.

Regenerate the committed goldens with::

    mise exec -- uv run python -c \
        "from tests.test_windows.synthetic import write_golden_files; \
         write_golden_files()"

That command must be a no-op on a clean tree: if it changes a file, the
oracle's behaviour changed and the diff is the review.
"""

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.windows import (
    ExplicitRange,
    MaterializationRule,
    compute_reference_windows,
)
from tests.test_windows.factories import SourceRow, frame_from_rows, profile_for

GOLDENS_DIRECTORY = Path(__file__).parent / "goldens"

# Price and volume granularity. See the module docstring: quarters are
# exactly representable, so golden bytes do not depend on summation order.
_QUARTER = 0.25

_MINUTE = 60

# A plain Unix second that lands on a round minute boundary, so the
# minute-cadence families sit at phase 0 by construction:
# 1700000040 == 60 * 28333334.
_MINUTE_BASE_OPEN = 1_700_000_040
_MINUTE_SLOT_COUNT = 40
# One past the last close of a complete minute-cadence family.
_MINUTE_SPAN_END = _MINUTE_BASE_OPEN + _MINUTE_SLOT_COUNT * _MINUTE

_SECOND_BASE_OPEN = 1_700_000_000
_SECOND_SLOT_COUNT = 30
_SECOND_SPAN_END = _SECOND_BASE_OPEN + _SECOND_SLOT_COUNT

# A source whose grid is deliberately NOT on round minute boundaries: every
# open time is 7 seconds past the minute, declared as the profile's phase.
_PHASED_PHASE_SECONDS = 7
_PHASED_BASE_OPEN = _MINUTE_BASE_OPEN + _PHASED_PHASE_SECONDS


@dataclass(frozen=True)
class SyntheticFamily:
    """One named synthetic source frame with the profile that describes it.

    Attributes:
        name: The family's registry name.
        profile: The profile declaring the frame's cadence and phase.
        frame: The raw source frame, in the row order a provider would
            publish it.

    """

    name: str
    profile: SourceProfile
    frame: pl.DataFrame


def _draw_candle(rng: random.Random) -> tuple[float, float, float, float, float]:
    """Draw one coherent OHLCV candle on the quarter-unit price grid.

    Returns:
        ``(open, high, low, close, volume)`` with ``high`` at or above both
        ends, ``low`` at or below both ends, and a volume that may legally
        be an exact ``0.0`` -- a real zero-volume candle, which the output
        contract distinguishes from the null of an empty window.

    """
    open_price = 20_000.0 + rng.randrange(-2_000, 2_000) * _QUARTER
    close_price = open_price + rng.randrange(-40, 40) * _QUARTER
    high = max(open_price, close_price) + rng.randrange(0, 40) * _QUARTER
    low = min(open_price, close_price) - rng.randrange(0, 40) * _QUARTER
    volume = rng.randrange(0, 400) * _QUARTER
    return open_price, high, low, close_price, volume


def _build_rows(  # noqa: PLR0913 - one keyword per independent grid knob
    *,
    seed: int,
    first_open: int,
    cadence_seconds: int,
    slot_count: int,
    missing_slots: frozenset[int] = frozenset(),
    extra_rows: Sequence[SourceRow] = (),
) -> tuple[SourceRow, ...]:
    """Generate one family's rows over a fixed grid of candle slots.

    Args:
        seed: The fixed integer seed for this family.
        first_open: The Unix-second open time of slot 0.
        cadence_seconds: The spacing between consecutive slots.
        slot_count: How many slots the grid spans, present or missing.
        missing_slots: Slot indices to leave out, producing gaps.
        extra_rows: Additional rows to merge in, used by the family that
            deliberately places candles off the declared grid.

    Returns:
        The rows, ordered by ascending open time.

    """
    rng = random.Random(seed)
    rows: list[SourceRow] = []
    for index in range(slot_count):
        # The candle is drawn for every slot, including a missing one, so
        # that dropping a slot punches a hole in the series instead of
        # shifting every later price. Families then differ only where they
        # mean to.
        open_price, high, low, close_price, volume = _draw_candle(rng)
        if index in missing_slots:
            continue
        rows.append(
            (
                first_open + index * cadence_seconds,
                open_price,
                high,
                low,
                close_price,
                volume,
            )
        )
    rows.extend(extra_rows)
    rows.sort(key=lambda row: row[0])
    return tuple(rows)


def _complete_grid_1m() -> SyntheticFamily:
    """Build a complete, on-phase, minute-cadence grid with no gaps."""
    return SyntheticFamily(
        name="complete_grid_1m",
        profile=profile_for(_MINUTE),
        frame=frame_from_rows(
            _build_rows(
                seed=11_000_001,
                first_open=_MINUTE_BASE_OPEN,
                cadence_seconds=_MINUTE,
                slot_count=_MINUTE_SLOT_COUNT,
            )
        ),
    )


def _single_gap_1m() -> SyntheticFamily:
    """Build a minute grid with one interior run of three missing candles."""
    return SyntheticFamily(
        name="single_gap_1m",
        profile=profile_for(_MINUTE),
        frame=frame_from_rows(
            _build_rows(
                seed=11_000_002,
                first_open=_MINUTE_BASE_OPEN,
                cadence_seconds=_MINUTE,
                slot_count=_MINUTE_SLOT_COUNT,
                missing_slots=frozenset({17, 18, 19}),
            )
        ),
    )


def _multi_gap_1m() -> SyntheticFamily:
    """Build a minute grid missing candles at its head, middle, and tail."""
    return SyntheticFamily(
        name="multi_gap_1m",
        profile=profile_for(_MINUTE),
        frame=frame_from_rows(
            _build_rows(
                seed=11_000_003,
                first_open=_MINUTE_BASE_OPEN,
                cadence_seconds=_MINUTE,
                slot_count=_MINUTE_SLOT_COUNT,
                missing_slots=frozenset({1, 2, 13, 22, 23, 24, 25, 38}),
            )
        ),
    )


def _complete_grid_1s() -> SyntheticFamily:
    """Build a complete, second-cadence grid: the finest cadence supported."""
    return SyntheticFamily(
        name="complete_grid_1s",
        profile=profile_for(1),
        frame=frame_from_rows(
            _build_rows(
                seed=11_000_004,
                first_open=_SECOND_BASE_OPEN,
                cadence_seconds=1,
                slot_count=_SECOND_SLOT_COUNT,
            )
        ),
    )


def _phased_grid_1m() -> SyntheticFamily:
    """Build a minute grid sitting 7 seconds past every minute boundary."""
    return SyntheticFamily(
        name="phased_grid_1m",
        profile=profile_for(_MINUTE, phase_seconds=_PHASED_PHASE_SECONDS),
        frame=frame_from_rows(
            _build_rows(
                seed=11_000_005,
                first_open=_PHASED_BASE_OPEN,
                cadence_seconds=_MINUTE,
                slot_count=_MINUTE_SLOT_COUNT,
            )
        ),
    )


def _straddling_1m() -> SyntheticFamily:
    """Build a minute grid polluted with candles that straddle boundaries.

    Under the strict resolution rules, an on-grid source can never produce
    a candle that crosses a window boundary: the window edges are
    themselves source grid points. The only way to exercise the whole-candle
    boundary rule is therefore a frame whose rows are NOT on the declared
    grid, which is what this family is for.

    That makes this family deliberately invalid input for strict source
    validation (its extra rows are off-phase and overlap their neighbours).
    The oracle does not re-validate row data, and it must still exclude a
    straddling candle whole rather than splitting it -- so a window here may
    legitimately report coverage above the window length, because the extra
    candles overlap the real ones.
    """
    half_minute = _MINUTE // 2
    extra_rng = random.Random(11_000_006)
    extra_rows: list[SourceRow] = []
    for slot in (5, 12, 26):
        open_price, high, low, close_price, volume = _draw_candle(extra_rng)
        extra_rows.append(
            (
                _MINUTE_BASE_OPEN + slot * _MINUTE + half_minute,
                open_price,
                high,
                low,
                close_price,
                volume,
            )
        )
    return SyntheticFamily(
        name="straddling_1m",
        profile=profile_for(_MINUTE),
        frame=frame_from_rows(
            _build_rows(
                seed=11_000_007,
                first_open=_MINUTE_BASE_OPEN,
                cadence_seconds=_MINUTE,
                slot_count=_MINUTE_SLOT_COUNT,
                extra_rows=tuple(extra_rows),
            )
        ),
    )


_FAMILY_BUILDERS: dict[str, Callable[[], SyntheticFamily]] = {
    "complete_grid_1m": _complete_grid_1m,
    "single_gap_1m": _single_gap_1m,
    "multi_gap_1m": _multi_gap_1m,
    "complete_grid_1s": _complete_grid_1s,
    "phased_grid_1m": _phased_grid_1m,
    "straddling_1m": _straddling_1m,
}

FAMILY_NAMES: tuple[str, ...] = tuple(_FAMILY_BUILDERS)


def build_family(name: str) -> SyntheticFamily:
    """Build one named synthetic family from scratch.

    Args:
        name: A key of :data:`FAMILY_NAMES`.

    Returns:
        A freshly generated family. Two calls always produce equal frames.

    """
    return _FAMILY_BUILDERS[name]()


@dataclass(frozen=True)
class GoldenCase:
    """One reviewed (family, window, emit cadence, anchor, range) combination.

    Attributes:
        label: The case name, also the golden file's stem.
        family: The synthetic family to run the oracle over.
        window: The window duration, as a compact duration string.
        emit_every: The emit cadence, as a compact duration string.
        anchor: The emit-grid anchor offset, as a compact duration string.
        materialization: Either an explicit half-open Unix-second range or
            the named warmup-skipping rule.

    """

    label: str
    family: str
    window: str
    emit_every: str
    anchor: str
    materialization: ExplicitRange | MaterializationRule

    @property
    def path(self) -> Path:
        """Return the committed golden file for this case."""
        return GOLDENS_DIRECTORY / f"{self.label}.csv"


_SKIP_WARMUP = MaterializationRule.SKIP_WARMUP

GOLDEN_CASES: tuple[GoldenCase, ...] = (
    # Aligned tiling: E == W, the case that is deliberately not a mode.
    GoldenCase(
        label="complete_1m_aligned_5m",
        family="complete_grid_1m",
        window="5m",
        emit_every="5m",
        anchor="0s",
        materialization=_SKIP_WARMUP,
    ),
    # Overlapping windows: a fresh 5m window every minute.
    GoldenCase(
        label="complete_1m_rolling_5m_every_1m",
        family="complete_grid_1m",
        window="5m",
        emit_every="1m",
        anchor="0s",
        materialization=_SKIP_WARMUP,
    ),
    # A nonzero anchor on an otherwise round grid.
    GoldenCase(
        label="complete_1m_rolling_3m_anchor_2m",
        family="complete_grid_1m",
        window="3m",
        emit_every="1m",
        anchor="2m",
        materialization=_SKIP_WARMUP,
    ),
    # An explicit range that starts inside the warmup and runs past the
    # data, so both the leading and trailing empty windows are recorded.
    GoldenCase(
        label="complete_1m_explicit_range_10m_every_5m",
        family="complete_grid_1m",
        window="10m",
        emit_every="5m",
        anchor="0s",
        materialization=ExplicitRange(
            start=_MINUTE_BASE_OPEN, end=_MINUTE_SPAN_END + 5 * _MINUTE
        ),
    ),
    # A single gap seen through a rolling window, over the whole span.
    GoldenCase(
        label="single_gap_1m_rolling_5m_every_1m",
        family="single_gap_1m",
        window="5m",
        emit_every="1m",
        anchor="0s",
        materialization=ExplicitRange(
            start=_MINUTE_BASE_OPEN, end=_MINUTE_SPAN_END + 1
        ),
    ),
    # Several gaps, including one at the head and one at the tail.
    GoldenCase(
        label="multi_gap_1m_aligned_10m",
        family="multi_gap_1m",
        window="10m",
        emit_every="10m",
        anchor="0s",
        materialization=ExplicitRange(
            start=_MINUTE_BASE_OPEN, end=_MINUTE_SPAN_END + 1
        ),
    ),
    GoldenCase(
        label="multi_gap_1m_rolling_4m_every_2m",
        family="multi_gap_1m",
        window="4m",
        emit_every="2m",
        anchor="0s",
        materialization=_SKIP_WARMUP,
    ),
    # Second-level cadence.
    GoldenCase(
        label="complete_1s_rolling_10s_every_1s",
        family="complete_grid_1s",
        window="10s",
        emit_every="1s",
        anchor="0s",
        materialization=_SKIP_WARMUP,
    ),
    GoldenCase(
        label="complete_1s_aligned_5s_anchor_3s",
        family="complete_grid_1s",
        window="5s",
        emit_every="5s",
        anchor="3s",
        materialization=ExplicitRange(
            start=_SECOND_BASE_OPEN, end=_SECOND_SPAN_END + 1
        ),
    ),
    # A phased source: the anchor has to share the source's phase.
    GoldenCase(
        label="phased_1m_rolling_4m_every_2m",
        family="phased_grid_1m",
        window="4m",
        emit_every="2m",
        anchor="7s",
        materialization=_SKIP_WARMUP,
    ),
    # Boundary-straddling candles: excluded whole at both edges.
    GoldenCase(
        label="straddling_1m_rolling_4m_every_2m",
        family="straddling_1m",
        window="4m",
        emit_every="2m",
        anchor="0s",
        materialization=ExplicitRange(
            start=_MINUTE_BASE_OPEN, end=_MINUTE_SPAN_END + 1
        ),
    ),
)


def compute_case(case: GoldenCase) -> pl.DataFrame:
    """Run the oracle over one golden case.

    Args:
        case: The case to compute.

    Returns:
        The oracle's window frame for that case.

    """
    family = build_family(case.family)
    return compute_reference_windows(
        family.frame,
        family.profile,
        window=case.window,
        emit_every=case.emit_every,
        anchor=case.anchor,
        materialization=case.materialization,
    )


def render_golden_csv(case: GoldenCase) -> str:
    """Render one golden case as the exact CSV text that belongs on disk.

    Args:
        case: The case to render.

    Returns:
        The CSV text, header included.

    """
    return compute_case(case).write_csv()


def write_golden_files() -> list[Path]:
    """Write every golden case to its committed path.

    Returns:
        The paths written, in case order.

    """
    GOLDENS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    written = []
    for case in GOLDEN_CASES:
        case.path.write_text(render_golden_csv(case))
        written.append(case.path)
    return written
