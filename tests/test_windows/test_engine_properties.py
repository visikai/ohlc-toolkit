"""Property-based equivalence between the window engine and the oracle.

The oracle is the specification, so these suites do not restate the
window rule a third time: they draw a source frame and a schedule at
random and assert that the two implementations produce the same thing.
What the strategy below is FOR is reaching the shapes a hand-written
matrix does not: frames that are mostly holes, frames whose rows are not
in timestamp order, frames carrying candles that straddle window
boundaries, and duplicate open times that force the oracle's
first-in-row-order tie-break to be reproduced rather than guessed.

Every drawn value lands on a quarter-unit grid, so every partial sum of
volumes is exactly representable and the two summation orders agree bit
for bit. That is what lets these comparisons be exact rather than
approximate.

The oracle is brute force -- O(rows x ticks) -- so the strategy keeps
frames to at most a couple of dozen candles and the example count is
capped. These suites are here to find disagreements in kind, not to
benchmark.
"""

from dataclasses import dataclass
from datetime import timedelta

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.temporal import ConfigError
from ohlc_toolkit.windows import (
    ExplicitRange,
    Materialization,
    MaterializationRule,
    compute_reference_windows,
    compute_windows,
)
from tests.test_windows.factories import SourceRow, frame_from_rows, profile_for

# A small fixed set of cadences rather than an unbounded search space: the
# engine is cadence-agnostic, so these exercise second-level,
# minute-level, and coarse-multiple arithmetic without inflating run time.
_CADENCE_CHOICES = (1, 5, 60, 300)

# Price and volume granularity. Quarters are exact in binary floating
# point at these magnitudes, so a window's volume total does not depend on
# the order the addition happens in.
_QUARTER = 0.25

# The oracle costs rows x ticks per example, so both are bounded and the
# example count is kept well under hypothesis' default.
_SETTINGS = settings(max_examples=40, deadline=timedelta(seconds=10))


@dataclass(frozen=True)
class _Scenario:
    """One randomly drawn source frame plus a resolvable schedule over it."""

    profile: SourceProfile
    frame: pl.DataFrame
    window: str
    emit_every: str
    anchor: str
    materialization: Materialization


def _draw_prices(data: st.DataObject, label: str) -> tuple[float, ...]:
    """Draw one coherent OHLCV candle on the quarter-unit grid."""
    open_price = 20_000.0 + data.draw(st.integers(-40, 40), label=f"{label}_open")
    close_price = open_price + _QUARTER * data.draw(
        st.integers(-40, 40), label=f"{label}_close"
    )
    high = max(open_price, close_price) + _QUARTER * data.draw(
        st.integers(0, 40), label=f"{label}_high"
    )
    low = min(open_price, close_price) - _QUARTER * data.draw(
        st.integers(0, 40), label=f"{label}_low"
    )
    volume = _QUARTER * data.draw(st.integers(0, 400), label=f"{label}_volume")
    return open_price, high, low, close_price, volume


def _draw_rows(
    data: st.DataObject, *, first_open: int, cadence: int, present_weight: int
) -> tuple[SourceRow, ...]:
    """Draw the source rows: a grid with holes, optional straddlers, shuffled.

    Args:
        data: The hypothesis data object to draw from.
        first_open: The Unix-second open time of slot 0.
        cadence: The spacing between consecutive slots.
        present_weight: Percent chance that any one slot is present, so a
            caller can ask for a gap-heavy frame.

    Returns:
        The rows, in a drawn (not necessarily ascending) order.

    """
    slot_count = data.draw(st.integers(min_value=1, max_value=24), label="slot_count")
    present = data.draw(
        st.lists(
            st.integers(min_value=1, max_value=100),
            min_size=slot_count,
            max_size=slot_count,
        ).filter(lambda drawn: any(roll <= present_weight for roll in drawn)),
        label="present_rolls",
    )

    rows = [
        (first_open + index * cadence, *_draw_prices(data, f"slot{index}"))
        for index, roll in enumerate(present)
        if roll <= present_weight
    ]

    # Candles off the declared grid: the only way a source can produce a
    # candle that straddles a window boundary, and the only way duplicate
    # open times arise. Both are invalid input that the oracle still
    # defines an answer for, so the engine has to reproduce it.
    if cadence > 1:
        straddler_offsets = data.draw(
            st.lists(
                st.tuples(
                    st.integers(min_value=0, max_value=slot_count - 1),
                    st.integers(min_value=1, max_value=cadence - 1),
                ),
                max_size=3,
            ),
            label="straddlers",
        )
        rows.extend(
            (first_open + slot * cadence + offset, *_draw_prices(data, f"extra{index}"))
            for index, (slot, offset) in enumerate(straddler_offsets)
        )

    # Row order is drawn, not sorted: the contract says the frame is never
    # sorted or repaired, so both implementations must read it as given.
    return tuple(data.draw(st.permutations(rows), label="row_order"))


def _draw_scenario(
    data: st.DataObject,
    *,
    align_emit_to_window: bool = False,
    present_weight: int = 80,
) -> _Scenario:
    """Draw a source frame and a schedule that satisfies every strict rule.

    The declared phase is drawn first and the grid is built from it, so the
    frame is always on the phase the profile declares. Window, emit
    cadence, and anchor are all drawn as whole multiples of the cadence
    (the anchor offset by the phase), which is exactly the region the
    strict resolution rules admit.

    Args:
        data: The hypothesis data object to draw from.
        align_emit_to_window: When True, force ``E == W``, the aligned
            tiling case.
        present_weight: Percent chance that any one source slot is
            present.

    Returns:
        The drawn scenario.

    """
    cadence = data.draw(st.sampled_from(_CADENCE_CHOICES), label="cadence_seconds")
    phase = data.draw(
        st.integers(min_value=0, max_value=cadence - 1), label="phase_seconds"
    )
    # Start several cadence steps up the grid so a drawn materialization
    # range can reach below the data and stay non-negative.
    grid_index = data.draw(st.integers(min_value=8, max_value=40), label="grid_index")
    first_open = phase + cadence * grid_index

    rows = _draw_rows(
        data, first_open=first_open, cadence=cadence, present_weight=present_weight
    )

    window_multiple = data.draw(
        st.integers(min_value=1, max_value=6), label="window_multiple"
    )
    emit_multiple = (
        window_multiple
        if align_emit_to_window
        else data.draw(st.integers(min_value=1, max_value=6), label="emit_multiple")
    )
    anchor_multiple = data.draw(
        st.integers(min_value=0, max_value=6), label="anchor_multiple"
    )

    last_close = max(row[0] for row in rows) + cadence
    lead = data.draw(st.integers(min_value=0, max_value=4), label="lead_steps")
    trail = data.draw(st.integers(min_value=0, max_value=4), label="trail_steps")
    use_skip_warmup = data.draw(st.booleans(), label="use_skip_warmup")
    materialization: Materialization = (
        MaterializationRule.SKIP_WARMUP
        if use_skip_warmup
        else ExplicitRange(
            start=first_open - lead * cadence, end=last_close + trail * cadence + 1
        )
    )

    return _Scenario(
        profile=profile_for(cadence, phase_seconds=phase),
        frame=frame_from_rows(rows),
        window=f"{window_multiple * cadence}s",
        emit_every=f"{emit_multiple * cadence}s",
        anchor=f"{phase + anchor_multiple * cadence}s",
        materialization=materialization,
    )


def _outcome(
    scenario: _Scenario, *, use_engine: bool
) -> tuple[str | None, pl.DataFrame | None]:
    """Run one implementation, returning either its error text or its frame."""
    compute = compute_windows if use_engine else compute_reference_windows
    try:
        frame = compute(
            scenario.frame,
            scenario.profile,
            window=scenario.window,
            emit_every=scenario.emit_every,
            anchor=scenario.anchor,
            materialization=scenario.materialization,
        )
    except ConfigError as error:
        return str(error), None
    return None, frame


def _assert_engine_matches_oracle(scenario: _Scenario) -> None:
    """Assert the two implementations agree, error text included."""
    reference_error, reference_frame = _outcome(scenario, use_engine=False)
    engine_error, engine_frame = _outcome(scenario, use_engine=True)

    assert engine_error == reference_error
    if reference_frame is None:
        return
    assert engine_frame is not None
    assert_frame_equal(
        engine_frame,
        reference_frame,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


@_SETTINGS
@given(data=st.data())
def test_the_engine_matches_the_oracle_on_random_frames_and_schedules(
    data: st.DataObject,
) -> None:
    """Random frame, random legal schedule, identical output."""
    _assert_engine_matches_oracle(_draw_scenario(data))


@_SETTINGS
@given(data=st.data())
def test_the_engine_matches_the_oracle_on_gap_heavy_frames(
    data: st.DataObject,
) -> None:
    """Mostly holes: the region where a sliding index shortcut would drift.

    With only one slot in five present, the number of candles inside a
    window varies wildly from tick to tick, and consecutive windows can
    share no candles at all. An engine that assumed a fixed candle count
    per window would pass the dense cases and fail here.
    """
    _assert_engine_matches_oracle(_draw_scenario(data, present_weight=20))


@_SETTINGS
@given(data=st.data())
def test_aligned_emission_is_the_rolling_definition_at_equal_window_and_cadence(
    data: st.DataObject,
) -> None:
    """E == W is not a mode: it is the rolling definition, and must stay so.

    Aligned tiling is the case an implementation is most tempted to route
    through a bucketing shortcut. This suite pins it to the same answer
    the general rolling definition gives, drawn over the same frames.
    """
    _assert_engine_matches_oracle(_draw_scenario(data, align_emit_to_window=True))


if __name__ == "__main__":
    pytest.main([__file__])
