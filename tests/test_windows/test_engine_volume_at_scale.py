"""The engine's ``volume`` column, pinned at scale against exact arithmetic.

Why this suite exists
---------------------

Every other equivalence suite in this package measures the engine against
the brute-force oracle, and the oracle is quadratic, so those comparisons
can only ever run over a few thousand candles. Summation error does not
show up there. It shows up over millions of rows, where a sliding
add/subtract sum -- the shape most rolling implementations reach for --
accumulates a rounding residue that is a function of everything the series
has already added and subtracted, not of the window's own candles.

That residue has three visible consequences, and all three are things this
library must never ship:

- A window whose candles all have exactly zero volume reports a small
  non-zero total.
- A sum of non-negative volumes comes back NEGATIVE.
- The same window over the same candles changes value when more history is
  prepended in front of it, which quietly breaks both containment-only
  semantics and any "recompute only the tail" append story.

So this suite runs the real engine over a multi-million-row minute grid
built to be maximally unkind to a sliding sum, and asserts the invariants
directly. The reference is not the oracle: it is
:func:`math.fsum`, which is correctly rounded, and a rolling maximum, which
involves no arithmetic at all. Both are linear, so the check is affordable
at a scale the oracle could never reach.

The synthetic grid
------------------

A complete, gap-free minute grid whose volumes repeat with a fixed period.
Each cycle holds:

- a long stretch of candles with exactly ``0.0`` volume, longer than the
  longest window under test, so that windows containing nothing but zeros
  exist for every window length;
- short zero runs inside the active stretch, so the shortest window has
  zero-only windows too;
- isolated tiny volumes (``5e-07``) surrounded by zeros, which a residue of
  the same order would swamp;
- an active stretch whose magnitudes ramp across nine decades, so the
  running total a sliding sum carries is large exactly where the window
  contents are small.

Nothing here is random: the grid is a pure function of the row index, so it
is identical on every machine and every run. Only the sample of windows
checked against :func:`math.fsum` is drawn, and it is drawn from a fixed
seed.
"""

import math
import random
import struct
from dataclasses import dataclass

import polars as pl
import pytest

from ohlc_toolkit.windows import MaterializationRule, compute_windows
from tests.test_windows.factories import profile_for

_MINUTE = 60

# Multi-million rows: about five and a half years of minutes. Large enough
# that a sliding sum's residue is unmistakable, small enough that the whole
# suite stays in the tens of seconds.
_ROW_COUNT = 3_000_000

# 1600000020 == 60 * 26666667, so slot 0 opens on a round minute and the
# grid sits at phase 0.
_FIRST_OPEN = 1_600_000_020

# Window lengths in whole source candles, drawn from the canonical window
# set: one very short, one mid-scale, one long. The short one is where a
# residue is largest relative to the window's own total; the long one is
# where the most addends accumulate.
_WINDOW_MINUTES = (8, 993, 17_632)

# One repeat of the volume pattern, in rows.
_CYCLE_ROWS = 65_536
# Rows of the cycle that carry volume at all. The remainder is one
# unbroken stretch of exactly zero, longer than the longest window above,
# so every window length has windows holding nothing but zeros.
_ACTIVE_ROWS = 39_000
_QUIET_ROWS = _CYCLE_ROWS - _ACTIVE_ROWS

# Inside the active stretch, a shorter repeat: volume, then one isolated
# tiny volume, then a run of zeros longer than the shortest window.
_SUBCYCLE_ROWS = 1_024
_SUBCYCLE_ACTIVE_ROWS = 999
_TINY_VOLUME = 5e-07

# The active volumes span nine decades, 1e-5 up to about 1e5.
_DECADE_COUNT = 9
_LOWEST_DECADE = -4
_RAMP_PERIOD = 97
_RAMP_STEP = 0.1
# A floor the ramp's top decade clears comfortably, asserted so that a
# future edit cannot quietly flatten the magnitude spread this suite needs.
_LARGEST_VOLUME_FLOOR = 1e4

# Prices are irrelevant to this suite -- only volume is under test -- but
# they still have to be coherent, so they walk a fixed sawtooth.
_PRICE_PERIOD = 1_000
_PRICE_BASE = 20_000.0

# How many windows are checked against math.fsum, and how they are chosen:
# a seeded random draw plus four deliberate strata. The strata matter more
# than the draw -- the extremes of the reported distribution are where a
# summation defect shows first.
_SAMPLE_SEED = 20_260_903
_RANDOM_SAMPLE_SIZE = 400
_STRATUM_SIZE = 60

# How far the engine's per-window sum may sit from the correctly rounded
# one. polars reduces a Float64 slice in blocks rather than one addend at a
# time, so its error does not grow with the window: measured against
# math.fsum it stays within about ten units in the last place -- roughly
# 1e-15 relative -- whether the window holds 8 candles or 17632, both over
# this grid and over a full published minute history. The bound below is
# that observation with an order of magnitude of headroom.
#
# It is deliberately NOT the worst case a summation could have, which is
# ``(m - 1) * 2**-53`` and reaches 2e-12 at m = 17632. A summation that
# really drifted that far could not support the two exactness assertions
# in this module, so this suite should fail on it rather than make room
# for it.
_VOLUME_RELATIVE_TOLERANCE = 1e-14

# A window whose contained volumes are all zero must report exactly 0.0,
# and windows like that must actually exist or these assertions are
# vacuous. One per cycle, at a minimum, for every window length above.
_MINIMUM_ZERO_VOLUME_WINDOWS = _ROW_COUNT // _CYCLE_ROWS


@dataclass(frozen=True)
class _ScaleCase:
    """One window length, run over the shared adversarial grid.

    Attributes:
        window_minutes: The window length in whole source candles.
        volumes: The grid's volume column, ascending by open time.
        result: The engine's output for this window at a one-minute emit
            cadence.

    """

    window_minutes: int
    volumes: pl.Series
    result: pl.DataFrame

    def last_row_indices(self) -> pl.Series:
        """Return the last source row each emitted window contains.

        The grid is complete and every emitted window is fully covered, so
        the candle closing exactly at emit time ``t`` is at row
        ``(t - first_open) / d - 1``, and the window's candles are the
        ``window_minutes`` rows ending there. Deriving that from the emit
        times rather than from the engine's own bounds keeps the reference
        independent of the thing it is checking.
        """
        close_times = self.result.get_column("close_time")
        return (close_times - _FIRST_OPEN) // _MINUTE - 1

    def exact_volume(self, position: int) -> float:
        """Return the correctly rounded sum of one window's own volumes.

        Args:
            position: A row position in :attr:`result`.

        Returns:
            ``math.fsum`` over the ``window_minutes`` source volumes that
            window contains, which is the exactly-summed-then-rounded-once
            value every other summation is approximating.

        """
        last_row = int(self.last_row_indices()[position])
        first_row = last_row - self.window_minutes + 1
        return math.fsum(self.volumes.slice(first_row, self.window_minutes).to_list())


def _volume_expression() -> pl.Expr:
    """Build the volume column as a pure function of the row index.

    See the module docstring for what the pattern is for. Written as one
    polars expression rather than a Python loop so that three million rows
    cost milliseconds.
    """
    row = pl.int_range(0, _ROW_COUNT, dtype=pl.Int64)
    position_in_cycle = row % _CYCLE_ROWS
    position_in_subcycle = row % _SUBCYCLE_ROWS
    decade = (row // _CYCLE_ROWS) % _DECADE_COUNT + _LOWEST_DECADE
    ramp = (
        pl.lit(10.0).pow(decade)
        * ((row % _RAMP_PERIOD) + 1).cast(pl.Float64)
        * _RAMP_STEP
    )
    return (
        pl.when(position_in_cycle >= _ACTIVE_ROWS)
        .then(pl.lit(0.0))
        .when(position_in_subcycle > _SUBCYCLE_ACTIVE_ROWS)
        .then(pl.lit(0.0))
        .when(position_in_subcycle == _SUBCYCLE_ACTIVE_ROWS)
        .then(pl.lit(_TINY_VOLUME))
        .otherwise(ramp)
        .cast(pl.Float64)
        .alias("volume")
    )


def _build_adversarial_grid() -> pl.DataFrame:
    """Build the complete minute grid this module runs the engine over."""
    row = pl.int_range(0, _ROW_COUNT, dtype=pl.Int64)
    price = _PRICE_BASE + (row % _PRICE_PERIOD).cast(pl.Float64)
    return pl.select(
        (_FIRST_OPEN + row * _MINUTE).alias("timestamp"),
        price.alias("open"),
        (price + 5.0).alias("high"),
        (price - 5.0).alias("low"),
        (price + 1.0).alias("close"),
        _volume_expression(),
    )


@pytest.fixture(scope="module")
def adversarial_grid() -> pl.DataFrame:
    """Build the shared grid once for the whole module."""
    return _build_adversarial_grid()


@pytest.fixture(
    scope="module",
    params=_WINDOW_MINUTES,
    ids=lambda minutes: f"{minutes}m",
)
def scale_case(
    request: pytest.FixtureRequest, adversarial_grid: pl.DataFrame
) -> _ScaleCase:
    """Run the engine over the grid for one window length."""
    window_minutes = int(request.param)
    result = compute_windows(
        adversarial_grid,
        profile_for(_MINUTE),
        window=f"{window_minutes}m",
        emit_every="1m",
        materialization=MaterializationRule.SKIP_WARMUP,
    )
    return _ScaleCase(
        window_minutes=window_minutes,
        volumes=adversarial_grid.get_column("volume"),
        result=result,
    )


def _zero_volume_mask(case: _ScaleCase) -> pl.Series:
    """Mark the emitted windows whose contained volumes are all zero.

    A rolling maximum is a comparison, never an addition, so it carries no
    rounding at all: where it reports zero over a window of non-negative
    volumes, every one of those volumes is exactly zero and the only
    correct total is exactly ``0.0``.
    """
    contained_maximum = case.volumes.rolling_max(window_size=case.window_minutes)
    return contained_maximum.gather(case.last_row_indices()) == 0.0


def _sampled_positions(case: _ScaleCase) -> list[int]:
    """Choose the window positions to check against :func:`math.fsum`.

    A seeded random draw plus four strata: the leading and trailing
    windows, the largest reported volumes, and the smallest strictly
    positive ones. That last stratum is the one that catches a residue --
    a sliding sum's leftovers are exactly the smallest non-zero values in
    the column.
    """
    height = case.result.height
    indexed = case.result.with_row_index("position").select("position", "volume")

    largest = indexed.sort("volume", descending=True).head(_STRATUM_SIZE)
    smallest_positive = (
        indexed.filter(pl.col("volume") > 0.0).sort("volume").head(_STRATUM_SIZE)
    )

    rng = random.Random(_SAMPLE_SEED)
    positions = {
        *range(min(_STRATUM_SIZE, height)),
        *range(max(0, height - _STRATUM_SIZE), height),
        *(int(value) for value in largest.get_column("position")),
        *(int(value) for value in smallest_positive.get_column("position")),
        *(rng.randrange(height) for _ in range(_RANDOM_SAMPLE_SIZE)),
    }
    return sorted(positions)


def test_the_adversarial_grid_is_the_shape_this_module_claims(
    adversarial_grid: pl.DataFrame,
) -> None:
    """The fixture is what the docstring says, or every assertion below drifts."""
    volumes = adversarial_grid.get_column("volume")
    timestamps = adversarial_grid.get_column("timestamp")

    assert adversarial_grid.height == _ROW_COUNT
    assert timestamps[0] == _FIRST_OPEN
    # A complete grid: consecutive rows are exactly one cadence apart.
    assert timestamps.diff().drop_nulls().unique().to_list() == [_MINUTE]
    # The quiet stretch alone is longer than the longest window under test.
    assert _QUIET_ROWS > max(_WINDOW_MINUTES)
    assert volumes.min() == 0.0
    assert (volumes == 0.0).sum() > 0
    assert (volumes == _TINY_VOLUME).sum() > 0
    # Nine decades of magnitude, so a sliding sum's running total is large
    # where the window contents are small.
    assert volumes.max() > _LARGEST_VOLUME_FLOOR  # type: ignore[operator]
    assert volumes.filter(volumes > 0.0).min() == _TINY_VOLUME


def test_the_engine_never_reports_a_negative_volume(scale_case: _ScaleCase) -> None:
    """Summing non-negative volumes cannot produce a negative total.

    This is an invariant of the arithmetic, not a range check the engine
    applies afterwards: nothing in the output path clamps, because a clamp
    would hide exactly the defect this asserts against.
    """
    volumes = scale_case.result.get_column("volume")

    assert volumes.null_count() == 0
    assert volumes.min() >= 0.0  # type: ignore[operator]


def test_a_window_of_only_zero_volume_candles_reports_exactly_zero(
    scale_case: _ScaleCase,
) -> None:
    """Zero in, exactly zero out -- not a residue that rounds to zero."""
    zero_windows = _zero_volume_mask(scale_case)
    reported = scale_case.result.get_column("volume").filter(zero_windows)

    assert zero_windows.sum() >= _MINIMUM_ZERO_VOLUME_WINDOWS
    assert (reported == 0.0).all()


def test_the_engine_agrees_with_an_exact_sum_over_sampled_windows(
    scale_case: _ScaleCase,
) -> None:
    """Sampled windows match :func:`math.fsum` over their own candles.

    ``math.fsum`` sums exactly and rounds once, so it is the value every
    other summation of the same addends is approximating. Agreement here is
    the positive statement behind the two invariants above: the engine is
    not merely non-negative and zero-preserving, it is summing the right
    candles.
    """
    # Every emitted window is fully covered, which is what lets the
    # reference locate a window's candles from its emit time alone.
    assert scale_case.result.get_column("src_count").unique().to_list() == [
        scale_case.window_minutes
    ]

    reported = scale_case.result.get_column("volume")
    positions = _sampled_positions(scale_case)
    assert len(positions) >= _RANDOM_SAMPLE_SIZE

    for position in positions:
        exact = scale_case.exact_volume(position)
        actual = reported[position]
        if exact == 0.0:
            assert actual == 0.0, f"window {position} should be exactly zero"
        else:
            assert abs(actual - exact) <= _VOLUME_RELATIVE_TOLERANCE * exact, (
                f"window {position}: {actual!r} vs exact {exact!r}"
            )


_LAYOUT_ROW_COUNT = 4097
_LAYOUT_LARGE_VOLUME = 1.0
_LAYOUT_RESIDUE_VOLUME = 1e-16
_LAYOUT_CHUNK_ROWS = (7, 64, 999)
_LAYOUT_WINDOW_MINUTES = (2048, _LAYOUT_ROW_COUNT)


def _build_layout_grid() -> pl.DataFrame:
    """Build a grid whose volumes make chunked summation observable.

    One large volume followed by a long tail of values far below its last
    representable bit. Adding the tail to the running total loses every
    addend individually, but adding the tail to itself first does not, so
    the total depends on how the addends were grouped -- which is exactly
    what a physical chunk boundary decides.
    """
    row = pl.int_range(0, _LAYOUT_ROW_COUNT, dtype=pl.Int64)
    volume = (
        pl.when(row == 0)
        .then(pl.lit(_LAYOUT_LARGE_VOLUME))
        .otherwise(pl.lit(_LAYOUT_RESIDUE_VOLUME))
        .cast(pl.Float64)
    )
    price = pl.lit(_PRICE_BASE).cast(pl.Float64)
    return pl.select(
        (_FIRST_OPEN + row * _MINUTE).alias("timestamp"),
        price.alias("open"),
        (price + 5.0).alias("high"),
        (price - 5.0).alias("low"),
        (price + 1.0).alias("close"),
        volume.alias("volume"),
    )


def _carried_in_chunks(frame: pl.DataFrame, rows_per_chunk: int) -> pl.DataFrame:
    """Return the same rows carried in fixed-size physical chunks."""
    pieces = [
        frame.slice(offset, rows_per_chunk)
        for offset in range(0, frame.height, rows_per_chunk)
    ]
    return pl.concat(pieces, rechunk=False)


def _volume_bit_patterns(frame: pl.DataFrame) -> list[bytes | None]:
    """Return each volume as raw IEEE-754 bits, so ``-0.0 != 0.0``."""
    return [
        None if value is None else struct.pack(">d", value)
        for value in frame.get_column("volume").to_list()
    ]


@pytest.mark.parametrize("window_minutes", _LAYOUT_WINDOW_MINUTES)
@pytest.mark.parametrize("rows_per_chunk", _LAYOUT_CHUNK_ROWS)
def test_volume_ignores_the_callers_chunk_layout(
    window_minutes: int, rows_per_chunk: int
) -> None:
    """Identical candles must total identically however they are stored.

    Polars sums a column chunk by chunk, so the same values carried in a
    different number of physical chunks can total to a different float.
    A caller does not choose that layout deliberately: reading the
    published history yields hundreds of chunks, and concatenating or
    stacking frames yields as many pieces as were joined. If the layout
    reached the summation, ``volume`` would be a function of how the
    frame was built rather than of the candles the window contains --
    the same containment-only violation as a sliding sum, arriving
    through a different door.

    The engine forecloses that by rechunking once when it extracts the
    candle columns. This test is what holds that line: it fails if the
    rechunk is ever dropped as redundant.
    """
    grid = _build_layout_grid()
    window = f"{window_minutes}m"

    reference = compute_windows(
        grid.rechunk(),
        profile_for(_MINUTE),
        window=window,
        emit_every="1m",
        materialization=MaterializationRule.SKIP_WARMUP,
    )
    assert reference.height >= 1

    chunked = _carried_in_chunks(grid, rows_per_chunk)
    assert chunked.n_chunks() > 1, "the layout under test must be chunked"

    actual = compute_windows(
        chunked,
        profile_for(_MINUTE),
        window=window,
        emit_every="1m",
        materialization=MaterializationRule.SKIP_WARMUP,
    )

    assert _volume_bit_patterns(actual) == _volume_bit_patterns(reference), (
        f"volume changed when the same {_LAYOUT_ROW_COUNT} candles were "
        f"carried in {chunked.n_chunks()} chunks instead of one"
    )
    assert actual.equals(reference)


if __name__ == "__main__":
    pytest.main([__file__])
