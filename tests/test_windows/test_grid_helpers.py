"""The shared emit-grid arithmetic, against answers computed by hand.

The oracle and the engine materialize their emit ticks very differently --
a Python tuple of integers on one side, a polars series on the other --
but both ask :mod:`ohlc_toolkit.windows.resolution` WHICH instants those
ticks are. All three helpers below are shared, so an error in one of them
moves both implementations by exactly the same amount and every
engine-versus-oracle comparison in this package goes on passing.
Equivalence structurally cannot see these. They need answers of their own.

So every expectation here is worked out by hand from the definitions, and
each helper is exercised over the cases where off-by-one arithmetic hides:
a bound that lands exactly on a tick and one that misses by a second, a
zero anchor and a non-zero one, negative Unix seconds either side of the
epoch, present-day timestamps, and ranges that hold no ticks at all.

The emit grid is ``{t : (t - anchor) mod E == 0}``. Only ``emit_every``
and ``anchor`` take part, which is why the schedules built below leave
``window`` at an arbitrary value: a helper that started consulting it
would be reading something these definitions do not mention.
"""

import pytest

from ohlc_toolkit.temporal import Duration
from ohlc_toolkit.windows.resolution import (
    ResolvedSchedule,
    count_ticks,
    first_tick_at_or_after,
    last_tick_at_or_before,
)

# An arbitrary window length, present only because ResolvedSchedule carries
# one. Nothing under test reads it.
_UNUSED_WINDOW_SECONDS = 3_600

_HOUR = 3_600

# 2025-01-01T00:00:00Z == 3600 * 482136, so it is on the hourly grid.
_NEW_YEAR_2025 = 1_735_689_600

# Rows in the published Bitstamp minute history at the time of writing,
# used here only as a realistically large tick count.
_LARGE_TICK_COUNT = 6_847_200


def _schedule(emit_seconds: int, anchor_seconds: int = 0) -> ResolvedSchedule:
    """Build a schedule carrying one emit cadence and anchor.

    Args:
        emit_seconds: The emit cadence ``E``.
        anchor_seconds: The grid anchor, already reduced into ``[0, E)`` as
            resolution would have left it.

    Returns:
        A resolved schedule. Built directly rather than through
        :func:`~ohlc_toolkit.windows.resolution.resolve_schedule`, because
        the point is to test the grid arithmetic on its own, without a
        profile deciding which combinations are legal.

    """
    return ResolvedSchedule(
        window=Duration(_UNUSED_WINDOW_SECONDS),
        emit_every=Duration(emit_seconds),
        anchor=Duration(anchor_seconds),
    )


@pytest.mark.parametrize(
    ("emit_seconds", "anchor_seconds", "bound", "expected"),
    [
        # A minute grid anchored at the epoch.
        pytest.param(60, 0, 0, 0, id="epoch_is_its_own_tick"),
        pytest.param(60, 0, 1, 60, id="one_second_past_a_tick_rounds_up"),
        pytest.param(60, 0, 59, 60, id="one_second_before_a_tick"),
        pytest.param(60, 0, 60, 60, id="exactly_on_the_next_tick"),
        # Negative Unix seconds: the grid runs backwards through the epoch
        # just as it runs forwards from it.
        pytest.param(60, 0, -1, 0, id="just_before_the_epoch_rounds_up_to_it"),
        pytest.param(60, 0, -60, -60, id="negative_bound_on_a_tick"),
        pytest.param(60, 0, -61, -60, id="negative_bound_one_second_early"),
        # The same grid shifted seven seconds. Ticks are ..., -53, 7, 67.
        pytest.param(60, 7, 0, 7, id="anchored_grid_skips_the_epoch"),
        pytest.param(60, 7, 7, 7, id="anchored_bound_on_a_tick"),
        pytest.param(60, 7, 8, 67, id="anchored_bound_one_second_late"),
        pytest.param(60, 7, -53, -53, id="anchored_negative_bound_on_a_tick"),
        pytest.param(60, 7, -54, -53, id="anchored_negative_bound_one_second_early"),
        # A one-second cadence: every whole second is a tick.
        pytest.param(1, 0, 12_345, 12_345, id="unit_cadence_admits_every_second"),
        pytest.param(1, 0, -7, -7, id="unit_cadence_negative"),
        # Present-day magnitudes.
        pytest.param(
            _HOUR, 0, _NEW_YEAR_2025, _NEW_YEAR_2025, id="large_bound_on_a_tick"
        ),
        pytest.param(
            _HOUR,
            0,
            _NEW_YEAR_2025 + 1,
            _NEW_YEAR_2025 + _HOUR,
            id="large_bound_one_second_late",
        ),
    ],
)
def test_the_first_tick_at_or_after_a_bound(
    emit_seconds: int, anchor_seconds: int, bound: int, expected: int
) -> None:
    """The smallest tick that is not before the bound, bound included."""
    schedule = _schedule(emit_seconds, anchor_seconds)

    result = first_tick_at_or_after(bound, schedule)

    assert result == expected
    # Restating the definition: on the grid, and no earlier tick qualifies.
    assert (result - anchor_seconds) % emit_seconds == 0
    assert result >= bound
    assert result - emit_seconds < bound


@pytest.mark.parametrize(
    ("emit_seconds", "anchor_seconds", "bound", "expected"),
    [
        pytest.param(60, 0, 0, 0, id="epoch_is_its_own_tick"),
        pytest.param(60, 0, 59, 0, id="one_second_before_a_tick_rounds_down"),
        pytest.param(60, 0, 60, 60, id="exactly_on_a_tick"),
        pytest.param(60, 0, 61, 60, id="one_second_past_a_tick"),
        pytest.param(60, 0, -1, -60, id="just_before_the_epoch_rounds_down"),
        pytest.param(60, 0, -60, -60, id="negative_bound_on_a_tick"),
        pytest.param(60, 0, -61, -120, id="negative_bound_one_second_early"),
        pytest.param(60, 7, 7, 7, id="anchored_bound_on_a_tick"),
        pytest.param(60, 7, 6, -53, id="anchored_bound_falls_back_past_the_epoch"),
        pytest.param(60, 7, 66, 7, id="anchored_bound_one_second_before_the_next"),
        pytest.param(60, 7, 67, 67, id="anchored_bound_on_the_next_tick"),
        pytest.param(1, 0, -5, -5, id="unit_cadence_negative"),
        pytest.param(
            _HOUR, 0, _NEW_YEAR_2025, _NEW_YEAR_2025, id="large_bound_on_a_tick"
        ),
        pytest.param(
            _HOUR,
            0,
            _NEW_YEAR_2025 - 1,
            _NEW_YEAR_2025 - _HOUR,
            id="large_bound_one_second_early",
        ),
    ],
)
def test_the_last_tick_at_or_before_a_bound(
    emit_seconds: int, anchor_seconds: int, bound: int, expected: int
) -> None:
    """The greatest tick that is not after the bound, bound included."""
    schedule = _schedule(emit_seconds, anchor_seconds)

    result = last_tick_at_or_before(bound, schedule)

    assert result == expected
    assert (result - anchor_seconds) % emit_seconds == 0
    assert result <= bound
    assert result + emit_seconds > bound


@pytest.mark.parametrize(
    ("emit_seconds", "anchor_seconds", "bound"),
    [
        pytest.param(60, 0, 0, id="epoch"),
        pytest.param(60, 0, -600, id="negative"),
        pytest.param(60, 7, 67, id="anchored"),
        pytest.param(_HOUR, 0, _NEW_YEAR_2025, id="large"),
        pytest.param(1, 0, 12_345, id="unit_cadence"),
    ],
)
def test_a_bound_already_on_the_grid_is_its_own_neighbour_in_both_directions(
    emit_seconds: int, anchor_seconds: int, bound: int
) -> None:
    """On a tick, "at or after" and "at or before" are both that tick.

    The two helpers are separate functions with separate arithmetic, and
    this is the one input where they have to agree. An off-by-one in
    either would show up here as a whole emit step of disagreement.
    """
    schedule = _schedule(emit_seconds, anchor_seconds)

    assert first_tick_at_or_after(bound, schedule) == bound
    assert last_tick_at_or_before(bound, schedule) == bound


@pytest.mark.parametrize(
    ("first_tick", "end", "emit_seconds", "expected"),
    [
        pytest.param(0, 0, 60, 0, id="empty_range_at_the_first_tick"),
        pytest.param(0, 1, 60, 1, id="one_second_of_range_still_holds_its_tick"),
        pytest.param(0, 60, 60, 1, id="range_ending_on_the_next_tick_excludes_it"),
        pytest.param(0, 61, 60, 2, id="one_second_past_the_next_tick_includes_it"),
        pytest.param(0, 600, 60, 10, id="ten_whole_steps"),
        # The range end may precede the first tick -- skip_warmup can hand
        # this shape over when the data ends before the grid resumes. The
        # count is zero, never negative.
        pytest.param(60, 0, 60, 0, id="end_before_the_first_tick"),
        pytest.param(60, 60, 60, 0, id="end_equal_to_the_first_tick"),
        pytest.param(60, 59, 60, 0, id="end_one_second_before_the_first_tick"),
        # Negative Unix seconds, and a range straddling the epoch.
        pytest.param(-120, 0, 60, 2, id="two_ticks_below_the_epoch"),
        pytest.param(-120, -60, 60, 1, id="one_tick_below_the_epoch"),
        pytest.param(-120, 61, 60, 4, id="straddling_the_epoch"),
        # An anchored grid: ticks at 7 and 67, and 127 is out of range.
        pytest.param(7, 68, 60, 2, id="anchored_grid"),
        pytest.param(1, 0, 1, 0, id="unit_cadence_empty"),
        pytest.param(
            0,
            _LARGE_TICK_COUNT * 60,
            60,
            _LARGE_TICK_COUNT,
            id="a_full_minute_history",
        ),
    ],
)
def test_the_number_of_ticks_in_a_half_open_range(
    first_tick: int, end: int, emit_seconds: int, expected: int
) -> None:
    """Counting the ticks agrees with walking them, at every boundary."""
    result = count_ticks(first_tick, end, emit_seconds)

    assert result == expected
    # range() is the definition of the half-open walk both implementations
    # perform, so it is the honest cross-check for the closed form.
    assert result == len(range(first_tick, end, emit_seconds))


if __name__ == "__main__":
    pytest.main([__file__])
