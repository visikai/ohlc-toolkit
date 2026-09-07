"""Forward excursions: the best and worst the interval reached, not its end.

A forward return reads two closes and nothing between them. An excursion
reads the INTERIOR of ``(t, t + H]``, and that one difference is what
every test here is about. It is why a hole in the grid is refused rather
than tolerated -- a lookup over a hole yields null, an extremum over a
hole yields a confident wrong answer -- and why an absent bar inside the
interval nulls both columns instead of being skipped.

The expected values are exact literals derived by hand from dyadic
fixture prices, compared with no tolerance, as in
:mod:`tests.test_returns.test_forward`.
"""

import math
import re
from collections.abc import Callable

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from ohlc_toolkit.returns import (
    ReturnMethod,
    add_forward_excursions,
    add_forward_returns,
    forward_available_at_column,
    forward_excursion_available_at_column,
    forward_mae_column,
    forward_mfe_column,
    forward_return_column,
)
from ohlc_toolkit.temporal import ConfigError
from tests.test_returns.factories import (
    CADENCE,
    CADENCE_SECONDS,
    TIME_BASE,
    excursion_frame,
)

# A six-tick 1m grid. The prices are dyadic so every ratio below is exact
# in binary and the comparisons need no tolerance.
_OFFSETS = (0, 60, 120, 180, 240, 300)
_HIGHS = (130.0, 160.0, 320.0, 96.0, 112.0, 40.0)
_LOWS = (120.0, 128.0, 144.0, 64.0, 80.0, 16.0)
_CLOSES = (128.0, 160.0, 256.0, 80.0, 96.0, 32.0)

_HORIZON = "2m"
_ROWS_PER_HORIZON = 2

# Hand-derived expectations, named so the arithmetic beside each one is
# attached to the number it produces.
_ROW0_MFE = 1.5  # 320 / 128 - 1, the high at t=120
_ROW0_MAE = 0.0  # 128 / 128 - 1, the low at t=60
_ROW3_MFE = 0.4  # 112 /  80 - 1, the high at t=240
_ROW3_MAE = -0.8  #  16 /  80 - 1, the low at t=300, the interval's last bar
# The carried bar IS the favorable extremum of row 0's interval, which is
# what makes the two readings of it give different numbers. Included as it
# stands, the highest high over (t=60, t=120) is the carried 128 and the
# mfe is 0.0. Skipped, the only high left is 64 and the mfe is -0.5. A
# fixture where the carried bar is neither extremum cannot tell those
# apart: both readings take their answer from the other bar.
_CARRIED_ROW0_MFE = 0.0  # 128 / 128 - 1, the carried bar's own high
_CARRIED_ROW0_MFE_IF_SKIPPED = -0.5  # 64 / 128 - 1, the bar past it
_CARRIED_ROW0_MAE = -0.5  # 64 / 128 - 1
_GAPPED_ROW0_MFE = -0.5  # 64 / 128 - 1: the best the interval reached
_GAPPED_ROW0_MAE = -0.875  # 16 / 128 - 1


def _fixture() -> pl.DataFrame:
    """Return the six-row total-grid fixture the literals below describe."""
    return excursion_frame(_OFFSETS, _HIGHS, _LOWS, _CLOSES)


def _oracle(
    highs: tuple[float | None, ...],
    lows: tuple[float | None, ...],
    closes: tuple[float | None, ...],
    *,
    rows: int,
    method: ReturnMethod,
) -> tuple[list[float | None], list[float | None]]:
    """Compute both excursion columns row by row in plain Python.

    Deliberately built from lists, ranges and ``max``/``min`` rather than
    from anything the implementation uses: no polars, no rolling window,
    no reversal. It is a second statement of the DEFINITION, so agreement
    between the two is evidence about the definition rather than about a
    shared expression.

    Args:
        highs: One high per row, ``None`` for an absent bar.
        lows: One low per row, on the same convention.
        closes: One close per row, on the same convention.
        rows: How many rows after ``t`` the interval spans.
        method: Which formula to apply.

    Returns:
        The maximum-favorable and maximum-adverse columns, in row order.

    """
    mfe: list[float | None] = []
    mae: list[float | None] = []
    for index, close in enumerate(closes):
        # The interval EXCLUDES the bar at t and INCLUDES the one at t+H.
        interval = range(index + 1, index + 1 + rows)
        bars = [(highs[j], lows[j]) for j in interval if j < len(highs)]
        incomplete = len(bars) < rows or any(
            high is None or low is None for high, low in bars
        )
        if incomplete or close is None:
            mfe.append(None)
            mae.append(None)
            continue
        best = max(high for high, _ in bars if high is not None)
        worst = min(low for _, low in bars if low is not None)
        mfe.append(_stated(best, close, method))
        mae.append(_stated(worst, close, method))
    return mfe, mae


def _stated(extremum: float, close: float, method: ReturnMethod) -> float | None:
    """Return the value the definition states, or ``None`` where it states none.

    Derived from the arithmetic failing to produce a finite number, not
    from a list of named cases: a zero close comes back ``None`` because
    dividing by it is not finite, not because ``close == 0`` was written
    down. An oracle that names the implementation's special cases is
    drifting toward restating the implementation.

    Args:
        extremum: The highest high or lowest low over the interval.
        close: The close at ``t``.
        method: Which formula to apply.

    Returns:
        The finite value, or ``None``.

    """
    try:
        ratio = extremum / close
        value = ratio - 1 if method is ReturnMethod.SIMPLE else math.log(ratio)
    except (ZeroDivisionError, ValueError):
        return None
    return value if math.isfinite(value) else None


@pytest.mark.parametrize("method", list(ReturnMethod))
def test_both_excursions_agree_with_a_brute_force_oracle(
    method: ReturnMethod,
) -> None:
    """Match a definition restated in plain Python, including both endpoints."""
    out = add_forward_excursions(
        _fixture(), horizon=_HORIZON, cadence=CADENCE, method=method
    )
    expected_mfe, expected_mae = _oracle(
        _HIGHS, _LOWS, _CLOSES, rows=_ROWS_PER_HORIZON, method=method
    )

    actual_mfe = out.get_column(forward_mfe_column(method, _HORIZON)).to_list()
    actual_mae = out.get_column(forward_mae_column(method, _HORIZON)).to_list()
    assert actual_mfe == pytest.approx(expected_mfe)
    assert actual_mae == pytest.approx(expected_mae)


def test_the_interval_excludes_the_bar_at_t_and_includes_the_bar_at_t_plus_h() -> None:
    """Read neither the starting bar's own extremes nor one bar too far.

    The fixture is built so both boundary mistakes are visible: the bar at
    ``t`` carries the highest high in the whole frame, and the bar one
    step past ``t + H`` carries the lowest low. An implementation that
    included either would report a different number here, not merely a
    differently-rounded one.
    """
    # Row 0: close 128, interval is the bars at t=60 and t=120.
    # highs (160, 320) -> 320 / 128 - 1 = 1.5
    # lows  (128, 144) -> 128 / 128 - 1 = 0.0
    # The bar at t=0 has high 130 and low 120: both are ignored, and the
    # low of 120 would have made the adverse excursion negative.
    out = add_forward_excursions(
        _fixture(), horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    mfe = out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))
    mae = out.get_column(forward_mae_column(ReturnMethod.SIMPLE, _HORIZON))

    assert mfe[0] == _ROW0_MFE
    assert mae[0] == _ROW0_MAE

    # Row 3: close 80, interval is the bars at t=240 and t=300.
    # highs (112, 40) -> 112 / 80 - 1 = 0.4
    # lows  (80,  16) ->  16 / 80 - 1 = -0.8, the last bar's low, which is
    # inside the interval precisely because the interval includes t + H.
    assert mfe[3] == pytest.approx(_ROW3_MFE)
    assert mae[3] == _ROW3_MAE


def test_the_twin_states_t_plus_h_on_every_row_including_null_ones() -> None:
    """State availability on rows whose excursions could not be computed."""
    out = add_forward_excursions(
        _fixture(), horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.LOG
    )
    twin = out.get_column(
        forward_excursion_available_at_column(ReturnMethod.LOG, _HORIZON)
    )

    assert twin.null_count() == 0
    assert twin.to_list() == [
        TIME_BASE + offset + _ROWS_PER_HORIZON * CADENCE_SECONDS for offset in _OFFSETS
    ]
    # The last two rows have no complete interval, and still say when the
    # value they do not have would have arrived.
    assert out.get_column(forward_mfe_column(ReturnMethod.LOG, _HORIZON))[-1] is None


def test_the_excursion_twin_does_not_collide_with_the_return_twin() -> None:
    """Let the return and both excursions sit on one frame over one horizon."""
    frame = add_forward_returns(
        _fixture(), horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    out = add_forward_excursions(
        frame, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )

    return_twin = forward_available_at_column(ReturnMethod.SIMPLE, _HORIZON)
    excursion_twin = forward_excursion_available_at_column(
        ReturnMethod.SIMPLE, _HORIZON
    )
    assert return_twin != excursion_twin
    assert {return_twin, excursion_twin} <= set(out.columns)
    # Same interval, so the same instant: distinct columns, equal values.
    assert out.get_column(return_twin).to_list() == (
        out.get_column(excursion_twin).to_list()
    )


def test_an_untraded_bar_inside_the_interval_is_read_as_it_stands() -> None:
    """Include a carried bar rather than skipping or nulling it."""
    # A bar with volume 0 carries the previous close into every price, so
    # the bar at t=60 states 128.0 throughout -- exactly row 0's close,
    # which is the claim being pinned. It is also the HIGHEST high in row
    # 0's interval, and that is what makes this fixture able to fail: an
    # implementation that skipped carried bars would report -0.5 here
    # instead of 0.0, where a carried bar sitting between the extremes
    # would give 0.25 under both readings and prove nothing.
    carried = excursion_frame(
        _OFFSETS,
        (130.0, 128.0, 64.0, 96.0, 112.0, 40.0),
        (120.0, 128.0, 64.0, 64.0, 80.0, 16.0),
        (128.0, 128.0, 64.0, 80.0, 96.0, 32.0),
    )
    out = add_forward_excursions(
        carried, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    mfe = out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))
    mae = out.get_column(forward_mae_column(ReturnMethod.SIMPLE, _HORIZON))

    # Row 0's interval is (t=60, t=120): highs (128, 64), lows (128, 64).
    assert mfe[0] == _CARRIED_ROW0_MFE
    assert mfe[0] != _CARRIED_ROW0_MFE_IF_SKIPPED, (
        "the carried bar was left out of the interval: its high is the "
        "favorable extremum here, so skipping it changes the answer"
    )
    assert mae[0] == _CARRIED_ROW0_MAE
    assert mfe.null_count() == mae.null_count() == _ROWS_PER_HORIZON


def test_an_absent_bar_inside_the_interval_nulls_both_excursions() -> None:
    """Refuse the whole interval when a bar inside it states no price."""
    absent = excursion_frame(
        _OFFSETS,
        (130.0, 160.0, None, 96.0, 112.0, 40.0),
        (120.0, 128.0, None, 64.0, 80.0, 16.0),
        (128.0, 160.0, None, 80.0, 96.0, 32.0),
    )
    out = add_forward_excursions(
        absent, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    mfe = out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))
    mae = out.get_column(forward_mae_column(ReturnMethod.SIMPLE, _HORIZON))

    # Rows 0 and 1 have the absent bar at t=120 inside their intervals.
    assert mfe[0] is None
    assert mae[0] is None
    assert mfe[1] is None
    assert mae[1] is None
    # Row 2 is the absent bar itself: its own close is null, so it has
    # nothing to measure FROM either.
    assert mfe[2] is None
    # Row 3's interval is entirely after the absent bar and is unaffected.
    assert mfe[3] == pytest.approx(_ROW3_MFE)
    assert mae[3] == _ROW3_MAE


def test_a_bar_stating_only_one_of_its_prices_nulls_both_excursions() -> None:
    """Refuse the interval for a HALF-absent bar, not only a wholly absent one.

    A wholly absent bar nulls both columns whether or not anything masks
    it, because a rolling extremum over a window containing a null is
    null anyway. This is the case that tells the two apart: a bar whose
    high is missing while its low is present. Without the mask the
    adverse excursion is computed from the lows regardless, quietly
    reporting a minimum over an interval it has just been told it cannot
    fully see, while the favorable one goes null -- two columns over one
    interval disagreeing about whether that interval is knowable.
    """
    half_absent = excursion_frame(
        _OFFSETS,
        (130.0, 160.0, None, 96.0, 112.0, 40.0),
        (120.0, 128.0, 144.0, 64.0, 80.0, 16.0),
        (128.0, 160.0, 256.0, 80.0, 96.0, 32.0),
    )
    out = add_forward_excursions(
        half_absent, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    mfe = out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))
    mae = out.get_column(forward_mae_column(ReturnMethod.SIMPLE, _HORIZON))

    # Rows 0 and 1 both have the half-absent bar at t=120 in their interval.
    assert mfe[0] is None
    assert mae[0] is None, (
        "the adverse excursion was stated over an interval containing a bar "
        "that reported no high: both columns read the same interval, so "
        "either both can be stated or neither can."
    )
    assert mfe[1] is None
    assert mae[1] is None


def test_a_nan_price_is_unusable_like_an_absent_one() -> None:
    """Keep the two columns agreeing about which intervals are knowable.

    ``NaN`` is not null, and the mask read nullness alone. A ``NaN`` high
    made the favorable column null through the arithmetic while the
    adverse column STATED a number taken from the other bars in the same
    interval -- two columns over one interval disagreeing about whether
    it can be stated at all, which is what this module exists to prevent.

    The aggregator emits null and never ``NaN``, so such a frame is
    outside the stated contract. It is not outside the API: the extremum
    guard accepts any ``Float64`` from any caller, and this package is
    published, so nothing between a caller and here enforces it.

    The assertion is over the SET of rows rather than the one row the
    disagreement was found on: the property is that the two columns are
    null in the same places, not that they happen to agree at row 0.
    """
    tainted = excursion_frame(
        _OFFSETS,
        (_HIGHS[0], float("nan"), *_HIGHS[2:]),
        _LOWS,
        _CLOSES,
    )
    out = add_forward_excursions(
        tainted, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    mfe = out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))
    mae = out.get_column(forward_mae_column(ReturnMethod.SIMPLE, _HORIZON))

    assert mfe.is_null().to_list() == mae.is_null().to_list()
    # Only row 0's interval (rows 1 and 2) reads the tainted bar at t=60;
    # row 1's interval starts past it. Rows 4 and 5 run off the end.
    assert mfe.is_null().to_list() == [True, False, False, False, True, True]


def test_a_frame_out_of_grid_order_is_refused_rather_than_read_positionally() -> None:
    """Refuse a shuffled frame: an extremum is positional and a return is not.

    The returns find a counterpart by close-time equality and hand any row
    order back unchanged, and the shared battery in test_alignment.py pins
    that for them. An excursion reads the ROWS that follow each position,
    so order is part of its input, and a frame out of grid order is caught
    by the total-grid rule -- consecutive steps are no longer the cadence
    -- rather than read as though the rows had been sorted first.
    """
    shuffled = _fixture()[[3, 0, 5, 2, 1, 4]]
    with pytest.raises(ConfigError, match=r"total .*grid"):
        add_forward_excursions(
            shuffled, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
        )


def test_an_empty_frame_gains_the_three_columns_with_their_dtypes() -> None:
    """No rows is not an error; the schema still says what was asked for."""
    out = add_forward_excursions(
        _fixture().clear(), horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.LOG
    )
    assert out.height == 0
    assert out.schema[forward_mfe_column(ReturnMethod.LOG, _HORIZON)] == pl.Float64()
    assert out.schema[forward_mae_column(ReturnMethod.LOG, _HORIZON)] == pl.Float64()
    assert out.schema[
        forward_excursion_available_at_column(ReturnMethod.LOG, _HORIZON)
    ] == (pl.Int64())


def test_a_single_row_has_no_interval_and_states_only_its_availability() -> None:
    """One row has nothing after it: both excursions null, the twin stated."""
    out = add_forward_excursions(
        _fixture().head(1),
        horizon=_HORIZON,
        cadence=CADENCE,
        method=ReturnMethod.SIMPLE,
    )
    assert out.get_column(
        forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON)
    ).to_list() == [None]
    assert out.get_column(
        forward_mae_column(ReturnMethod.SIMPLE, _HORIZON)
    ).to_list() == [None]
    assert out.get_column(
        forward_excursion_available_at_column(ReturnMethod.SIMPLE, _HORIZON)
    ).to_list() == [TIME_BASE + _ROWS_PER_HORIZON * CADENCE_SECONDS]


def test_a_frame_missing_a_row_is_refused_rather_than_read_through() -> None:
    """Refuse a hole, because an extremum over a hole is wrong, not null."""
    gapped = excursion_frame(
        (0, 60, 180, 240),
        (130.0, 160.0, 96.0, 112.0),
        (120.0, 128.0, 64.0, 80.0),
        (128.0, 160.0, 80.0, 96.0),
    )
    with pytest.raises(ConfigError) as refusal:
        add_forward_excursions(
            gapped, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
        )

    message = str(refusal.value)
    assert "120s" in message  # the offending step
    assert str(TIME_BASE + 60) in message  # and where it starts
    assert str(TIME_BASE + 180) in message


def test_a_gap_below_the_close_is_reported_rather_than_clamped() -> None:
    """Report a negative favorable excursion when the interval never rises.

    ``mfe >= 0 >= mae`` is the TYPICAL sign, not an invariant: price can gap down between the close at ``t`` and the
    first trade after it, putting every high in the interval below the
    close it is measured from. Clamping at zero would report a profit
    that no one could have taken.
    """
    gapped_down = excursion_frame(
        _OFFSETS,
        (130.0, 64.0, 32.0, 96.0, 112.0, 40.0),
        (120.0, 32.0, 16.0, 64.0, 80.0, 16.0),
        (128.0, 64.0, 32.0, 80.0, 96.0, 32.0),
    )
    out = add_forward_excursions(
        gapped_down, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    mfe = out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))
    mae = out.get_column(forward_mae_column(ReturnMethod.SIMPLE, _HORIZON))

    # Row 0: close 128, interval highs (64, 32). The best the interval ever
    # reached is 64, which is half the close: -0.5, not 0.
    assert mfe[0] == _GAPPED_ROW0_MFE
    assert mae[0] == _GAPPED_ROW0_MAE
    assert mfe[0] < 0


def test_the_columns_are_float64_and_refuse_to_overwrite() -> None:
    """Emit Float64 and refuse a frame already carrying either column."""
    out = add_forward_excursions(
        _fixture(), horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    for name in (
        forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON),
        forward_mae_column(ReturnMethod.SIMPLE, _HORIZON),
    ):
        assert out.schema[name] == pl.Float64()

    with pytest.raises(ConfigError, match="already carries"):
        add_forward_excursions(
            out, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
        )


@pytest.mark.parametrize(
    "column_for",
    [forward_mfe_column, forward_mae_column, forward_excursion_available_at_column],
    ids=["mfe", "mae", "available_at"],
)
def test_each_column_this_call_writes_is_refused_on_its_own(
    column_for: Callable[[ReturnMethod, str], str],
) -> None:
    """Make the refusal name the column, so one guard cannot cover three.

    Re-feeding the whole output frame, as the test above does, proves only
    that SOMETHING collided. The mfe column collides first and the other
    two are never reached, so a guard narrowed to two of the three -- or
    to one -- leaves that test green. Each column is planted by itself
    here and the refusal has to name the one that was planted.
    """
    column = column_for(ReturnMethod.SIMPLE, _HORIZON)
    frame = _fixture().with_columns(pl.lit(0.0).alias(column))
    with pytest.raises(ConfigError, match=re.escape(column)):
        add_forward_excursions(
            frame, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
        )


def test_a_zero_close_gives_null_rather_than_an_infinity() -> None:
    """Null a value this module cannot state as a real number."""
    zeroed = excursion_frame(
        _OFFSETS,
        _HIGHS,
        _LOWS,
        (0.0, 160.0, 256.0, 80.0, 96.0, 32.0),
    )
    out = add_forward_excursions(
        zeroed, horizon=_HORIZON, cadence=CADENCE, method=ReturnMethod.SIMPLE
    )
    assert out.get_column(forward_mfe_column(ReturnMethod.SIMPLE, _HORIZON))[0] is None


@st.composite
def _coherent_grids(draw: st.DrawFn) -> tuple[pl.DataFrame, int]:
    """Draw a total grid of coherent bars, some absent, with price gaps.

    Coherent means ``low <= close <= high`` on every traded bar. The
    ordering identity below is a statement about bars, not about arbitrary
    triples of floats: the toolkit does not require a frame's high to be
    its bar's actual maximum, so a generator free to emit ``high < close``
    would refute an identity that real data satisfies.
    """
    length = draw(st.integers(min_value=3, max_value=24))
    rows = draw(st.integers(min_value=1, max_value=4))
    highs: list[float | None] = []
    lows: list[float | None] = []
    closes: list[float | None] = []
    for _ in range(length):
        if draw(st.booleans()) and draw(st.integers(0, 4)) == 0:
            highs.append(None)
            lows.append(None)
            closes.append(None)
            continue
        # A wide range so consecutive bars gap rather than drift.
        close = draw(st.floats(min_value=0.5, max_value=5_000.0, allow_nan=False))
        low = draw(st.floats(min_value=0.25, max_value=close, allow_nan=False))
        high = draw(st.floats(min_value=close, max_value=20_000.0, allow_nan=False))
        highs.append(high)
        lows.append(low)
        closes.append(close)
    offsets = tuple(index * CADENCE_SECONDS for index in range(length))
    return excursion_frame(offsets, highs, lows, closes), rows


@settings(max_examples=150, deadline=None)
@given(grid=_coherent_grids(), method=st.sampled_from(list(ReturnMethod)))
def test_the_adverse_return_and_favorable_excursions_stay_ordered(
    grid: tuple[pl.DataFrame, int], method: ReturnMethod
) -> None:
    """Keep ``mae <= forward_return <= mfe`` wherever all three are stated.

    The return's counterpart close is the close of the bar at ``t + H``,
    which lies inside the interval the extremes are taken over, so the
    ordering is a consequence of the definitions rather than of the
    arithmetic. A violation would mean one of the three columns is
    reading a different interval than the other two.
    """
    frame, rows = grid
    horizon = f"{rows * CADENCE_SECONDS}s"
    out = add_forward_returns(frame, horizon=horizon, cadence=CADENCE, method=method)
    out = add_forward_excursions(out, horizon=horizon, cadence=CADENCE, method=method)

    mfe = pl.col(forward_mfe_column(method, horizon))
    mae = pl.col(forward_mae_column(method, horizon))
    value = pl.col(forward_return_column(method, horizon))

    stated = out.filter(mfe.is_not_null() & mae.is_not_null() & value.is_not_null())
    violations = stated.filter((mae > value) | (value > mfe))
    assert violations.height == 0, (
        f"{violations.height} row(s) break mae <= return <= mfe: {violations}"
    )

    # The two twins describe the same instant for the same horizon.
    assert out.get_column(forward_available_at_column(method, horizon)).to_list() == (
        out.get_column(forward_excursion_available_at_column(method, horizon)).to_list()
    )


@settings(max_examples=100, deadline=None)
@given(grid=_coherent_grids())
def test_the_excursions_agree_with_the_oracle_on_generated_grids(
    grid: tuple[pl.DataFrame, int],
) -> None:
    """Agree with the plain-Python definition on frames nobody hand-picked."""
    frame, rows = grid
    horizon = f"{rows * CADENCE_SECONDS}s"
    method = ReturnMethod.SIMPLE
    out = add_forward_excursions(frame, horizon=horizon, cadence=CADENCE, method=method)
    expected_mfe, expected_mae = _oracle(
        tuple(frame.get_column("high").to_list()),
        tuple(frame.get_column("low").to_list()),
        tuple(frame.get_column("close").to_list()),
        rows=rows,
        method=method,
    )

    assert out.get_column(forward_mfe_column(method, horizon)).to_list() == (
        pytest.approx(expected_mfe, nan_ok=False)
    )
    assert out.get_column(forward_mae_column(method, horizon)).to_list() == (
        pytest.approx(expected_mae, nan_ok=False)
    )
