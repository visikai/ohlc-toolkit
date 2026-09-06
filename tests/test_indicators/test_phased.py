"""The phased lookback harness, against its oracle and against the rule.

The load-bearing test here is the one at `W = 2h26m, E = 3m`. Everything
else could be satisfied by a harness that read its inputs from the
window's own `E`-cadence frame; that case is where the two differ, and
where reading from the `E` frame silently returns nothing.
"""

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from ohlc_toolkit.indicators import phased_lookback, phased_lookback_reference
from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from tests.test_windows.factories import frame_from_rows, profile_for
from tests.test_windows.fixtures import load_real_slice
from tests.test_windows.synthetic import FAMILY_NAMES, build_family

_MINUTE = 60

# The lookback every test below uses unless it says otherwise: enough
# phases that an off-by-one in the ordering is visible, few enough that a
# fixture stays readable.
_LOOKBACK = 3
# A base on the minute grid AND on every window grid used below, so the
# emit anchor and the frame's phase agree without arithmetic at the call
# site.
_BASE = 1_700_000_000 - 1_700_000_000 % (7 * 24 * 3600)


def _source_rows(count: int, *, volume: float = 5.0, start: int = _BASE) -> list[tuple]:
    """One minute per row, prices rising by 1 so a wrong phase is visible."""
    return [
        (
            start + index * _MINUTE,
            100.0 + index,
            101.0 + index,
            99.0 + index,
            100.5 + index,
            volume,
        )
        for index in range(count)
    ]


def _source_cadence_windows(
    rows: list[tuple], *, window: str, first_tick: int, last_tick: int
) -> pl.DataFrame:
    """Materialize `window` windows at SOURCE cadence, the harness's input.

    Built with the engine rather than fabricated, so the fixture is the
    artifact the harness will really be handed.
    """
    return compute_windows(
        frame_from_rows(rows),
        profile_for(_MINUTE),
        window=window,
        emit_every="1m",
        materialization=ExplicitRange(start=first_tick, end=last_tick),
    )


def _default_frame(*, window: str = "3m", count: int = 60) -> pl.DataFrame:
    rows = _source_rows(count)
    window_seconds = Duration.parse(window).total_seconds
    return _source_cadence_windows(
        rows,
        window=window,
        first_tick=_BASE + window_seconds,
        last_tick=_BASE + count * _MINUTE + 1,
    )


def test_the_output_is_one_row_per_emit_tick_with_lists_newest_first() -> None:
    """The shape, and the order inside the lists.

    Newest first is the order every indicator reads in -- `close[0]` is
    the window ending at the tick -- so it is asserted rather than left
    to whichever way the joins happened to accumulate.
    """
    frame = _default_frame()

    result = phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)

    assert result.frame.columns[0] == "close_time"
    complete = result.frame.drop_nulls()
    assert complete.height > 0
    for row in complete.iter_rows(named=True):
        assert len(row["close"]) == _LOOKBACK
        # Each phase is one window earlier, and the fixture rises by 1.0 a
        # minute, so three minutes back is three lower.
        assert row["close"][0] - row["close"][1] == pytest.approx(3.0)
        assert row["close"][1] - row["close"][2] == pytest.approx(3.0)


@pytest.mark.parametrize("family", FAMILY_NAMES)
def test_the_harness_agrees_with_the_oracle_on_every_synthetic_family(
    family: str,
) -> None:
    """Same rows, same dtypes, same bits -- or the same refusal."""
    built = build_family(family)
    cadence = built.profile.cadence.total_seconds
    window = f"{_LOOKBACK * cadence}s"
    closes = built.frame.get_column("timestamp")
    # The emit grid has to sit on the source grid, and two of these
    # families are deliberately off-phase. The anchor carries that phase
    # rather than the test avoiding the families that have one.
    anchor = f"{(int(closes.min()) + cadence) % (_LOOKBACK * cadence)}s"  # type: ignore[arg-type]
    first = int(closes.min()) + _LOOKBACK * cadence  # type: ignore[arg-type]
    last = int(closes.max()) + cadence + 1  # type: ignore[arg-type]
    frame = compute_windows(
        built.frame,
        built.profile,
        window=window,
        emit_every=f"{cadence}s",
        anchor=f"{int(closes.min()) % cadence}s",  # type: ignore[arg-type]
        materialization=ExplicitRange(start=first, end=last),
    )

    fast = phased_lookback(
        frame, window=window, emit_every=window, anchor=anchor, lookback=_LOOKBACK
    )
    slow = phased_lookback_reference(
        frame, window=window, emit_every=window, anchor=anchor, lookback=_LOOKBACK
    )

    assert_frame_equal(
        fast.frame,
        slow.frame,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )
    assert fast.effective_history == slow.effective_history


def test_a_removed_interior_row_nulls_exactly_the_ticks_that_referenced_it() -> None:
    """Exact equality, and nothing near it.

    A shift or an as-of join would carry the neighbouring row across the
    hole and report a number. The lookup misses, and a miss is null.
    """
    frame = _default_frame()
    window_seconds = 180
    # A row the emit grid actually reads. The frame steps by a minute and
    # the grid by three, so two rows in three are referenced by nothing
    # and removing one of those would prove nothing.
    victim = next(
        close
        for close in frame.get_column("close_time").to_list()
        if close % window_seconds == 0
        and close - 2 * window_seconds >= frame.get_column("close_time").to_list()[0]
    )
    holed = frame.filter(pl.col("close_time") != victim)

    whole = phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)
    without = phased_lookback(holed, window="3m", emit_every="3m", lookback=_LOOKBACK)

    lost = {
        int(row["close_time"])
        for row in without.frame.iter_rows(named=True)
        if row["close"] is None
    } - {
        int(row["close_time"])
        for row in whole.frame.iter_rows(named=True)
        if row["close"] is None
    }
    # Exactly the three ticks whose phase set names the removed row.
    assert lost == {victim, victim + 180, victim + 360}


def test_every_phase_is_a_real_row_where_the_emit_grid_cannot_reach() -> None:
    """The case the source-cadence rule exists for.

    `E = 3m` does not divide `W = 2h26m`, so `t - kW` is not on the emit
    grid. Reading the phases from the window's own `E`-cadence frame would
    find nothing at all; reading from the source-cadence frame finds every
    one of them. This is the positive control: it fails if the harness is
    ever pointed at the `E` frame.
    """
    window_seconds = Duration.parse("2h26m").total_seconds
    count = _LOOKBACK * window_seconds // _MINUTE + 10
    rows = _source_rows(count)
    frame = _source_cadence_windows(
        rows,
        window="2h26m",
        first_tick=_BASE + window_seconds,
        last_tick=_BASE + count * _MINUTE + 1,
    )

    result = phased_lookback(frame, window="2h26m", emit_every="3m", lookback=_LOOKBACK)
    complete = result.frame.drop_nulls()

    assert complete.height > 0
    assert window_seconds % Duration.parse("3m").total_seconds != 0
    source_closes = set(frame.get_column("close_time").to_list())
    for tick in complete.get_column("close_time").to_list():
        for phase in range(_LOOKBACK):
            assert tick - phase * window_seconds in source_closes
    # And the emit grid really cannot reach them: one window back from a
    # tick is not itself a tick.
    a_tick = int(complete.get_column("close_time")[0])
    assert (a_tick - window_seconds) % Duration.parse("3m").total_seconds != (
        a_tick % Duration.parse("3m").total_seconds
    )


def _threshold_frame(traded: list[int]) -> pl.DataFrame:
    """Build a fully covered source-cadence frame with per-row traded seconds."""
    frame = _default_frame(count=len(traded) + 3)
    return frame.head(len(traded)).with_columns(
        pl.Series("traded_seconds", traded, dtype=pl.Int64)
    )


class TestRefusals:
    """Everything a phased lookback refuses at resolution time.

    None of these is a warning. A phase error does not surface as one
    wrong number in one column; it surfaces as every indicator built on
    the frame being wrong together, and by then the artifact is written.
    """

    def test_a_schema_v1_frame_is_named_as_one(self) -> None:
        """The shape every artifact written before schema v2 has."""
        frame = _default_frame().drop("traded_seconds")

        with pytest.raises(ConfigError, match="schema v1"):
            phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)

    def test_a_frame_spanning_a_different_window_is_refused(self) -> None:
        """Reading a 5m frame as a 3m one phases perfectly and means nothing."""
        frame = _default_frame(window="5m")

        with pytest.raises(ConfigError, match="span the stated window"):
            phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)

    def test_an_emit_cadence_off_the_source_grid_is_refused(self) -> None:
        """A tick between two rows can never be looked up."""
        frame = _default_frame()

        with pytest.raises(ConfigError, match="whole multiple"):
            phased_lookback(frame, window="3m", emit_every="90s", lookback=_LOOKBACK)

    def test_an_anchor_off_the_frame_s_phase_is_refused(self) -> None:
        """Otherwise every lookup misses and the answer is all null.

        Silence is the wrong answer to "this anchor does not belong to
        this frame", and it is indistinguishable from "there is no data".
        """
        frame = _default_frame().with_columns(
            pl.col("close_time") + 7, pl.col("open_time") + 7
        )

        with pytest.raises(ConfigError, match="no emit tick"):
            phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)

    def test_a_ragged_grid_is_refused_but_a_hole_is_not(self) -> None:
        """A hole is absent data; a ragged step is the wrong frame.

        The cadence is the smallest step, so a frame missing a minute
        still steps by whole minutes and its missing inputs are null. A
        frame stepping by 90s where the smallest step is 60s is not a
        source-cadence materialization of anything.
        """
        frame = _default_frame()
        holed = frame.filter(
            pl.col("close_time") != int(frame.get_column("close_time")[9])
        )

        assert (
            phased_lookback(
                holed, window="3m", emit_every="3m", lookback=_LOOKBACK
            ).frame.height
            > 0
        )

        ragged = frame.head(4).with_columns(
            pl.Series(
                "close_time",
                [
                    int(frame.get_column("close_time")[0]) + offset
                    for offset in (0, 60, 150, 210)
                ],
                dtype=pl.Int64,
            )
        )
        with pytest.raises(ConfigError, match="whole multiple"):
            phased_lookback(ragged, window="3m", emit_every="3m", lookback=_LOOKBACK)

    @pytest.mark.parametrize(
        ("lookback", "match"),
        [
            (0, r"strictly positive"),
            (-1, r"strictly positive"),
            (True, r"must be an int"),
            (1.0, r"must be an int"),
        ],
    )
    def test_an_unusable_lookback_is_refused(
        self, lookback: object, match: str
    ) -> None:
        """A lookback of ``True`` is nobody's intention and would mean one."""
        with pytest.raises(ConfigError, match=match):
            phased_lookback(
                _default_frame(),
                window="3m",
                emit_every="3m",
                lookback=lookback,  # type: ignore[arg-type]
            )

    def test_a_negative_threshold_is_refused(self) -> None:
        """Whole, non-negative seconds or nothing."""
        with pytest.raises(ConfigError, match="must not be negative"):
            phased_lookback(
                _default_frame(),
                window="3m",
                emit_every="3m",
                lookback=_LOOKBACK,
                min_traded_seconds=-1,
            )


class TestNullPropagation:
    """A below-threshold window is a null input, in both directions."""

    def test_one_below_threshold_window_nulls_exactly_the_ticks_that_use_it(
        self,
    ) -> None:
        """`L` ticks lose their output, spaced one window apart."""
        traded = [180] * 40
        frame = _threshold_frame(traded)
        window_seconds = 180
        victim = next(
            close
            for close in frame.get_column("close_time").to_list()
            if close % window_seconds == 0
            and close - 2 * window_seconds
            >= frame.get_column("close_time").to_list()[0]
        )
        starved = frame.with_columns(
            pl.when(pl.col("close_time") == victim)
            .then(0)
            .otherwise(pl.col("traded_seconds"))
            .alias("traded_seconds")
        )

        whole = phased_lookback(
            frame,
            window="3m",
            emit_every="3m",
            lookback=_LOOKBACK,
            min_traded_seconds=1,
        )
        starved_result = phased_lookback(
            starved,
            window="3m",
            emit_every="3m",
            lookback=_LOOKBACK,
            min_traded_seconds=1,
        )

        lost = _newly_null(whole.frame, starved_result.frame)
        assert lost == {victim + phase * window_seconds for phase in range(_LOOKBACK)}

    def test_a_window_exactly_at_the_threshold_is_admitted(self) -> None:
        """At the bar is not below it, and the boundary is where that shows."""
        traded = [180] * 40
        frame = _threshold_frame(traded)
        exact = frame.with_columns(pl.lit(7, dtype=pl.Int64).alias("traded_seconds"))

        result = phased_lookback(
            exact,
            window="3m",
            emit_every="3m",
            lookback=_LOOKBACK,
            min_traded_seconds=7,
        )

        assert result.frame.drop_nulls().height > 0

    def test_the_threshold_applies_to_a_report_mode_frame_too(self) -> None:
        """Report mode removes nothing, so the harness cannot trust the frame.

        A frame written under `GateMode.REPORT` carries every row it
        measured, below-threshold ones included. A harness that assumed
        filtering had already happened would consume dead windows and
        report numbers for them.
        """
        traded = [180] * 40
        reported = _threshold_frame(traded).with_columns(
            pl.lit(0, dtype=pl.Int64).alias("traded_seconds")
        )

        result = phased_lookback(
            reported,
            window="3m",
            emit_every="3m",
            lookback=_LOOKBACK,
            min_traded_seconds=1,
        )

        assert result.frame.height > 0
        assert result.frame.drop_nulls().height == 0


def _newly_null(whole: pl.DataFrame, after: pl.DataFrame) -> set[int]:
    """List the ticks that lost their output between two results."""

    def nulls(frame: pl.DataFrame) -> set[int]:
        return {
            int(row["close_time"])
            for row in frame.iter_rows(named=True)
            if row["close"] is None
        }

    return nulls(after) - nulls(whole)


def test_the_effective_history_is_reported_rather_than_left_to_callers() -> None:
    """`L * W`, from the harness, for both shapes the recipes use.

    Two callers multiplying it themselves are two chances to multiply by
    the emit cadence instead, which is the same number whenever `E`
    divides `W` and silently wrong when it does not.
    """
    frame = _default_frame(count=120)

    for lookback, expected in ((_LOOKBACK, "9m"), (_LOOKBACK + 1, "12m")):
        result = phased_lookback(frame, window="3m", emit_every="3m", lookback=lookback)
        assert result.effective_history == Duration.parse(expected)
        assert result.grid.effective_history == Duration.parse(expected)


def test_nothing_after_a_tick_can_change_that_tick() -> None:
    """Causality, on a window whose phases the emit grid cannot reach.

    Perturbing every source row that closes after `t` leaves every output
    at or before `t` bit-identical. The window is `2h26m` against a `3m`
    grid, so the phases are read from the source cadence -- which is
    exactly where a lookahead would be easiest to introduce.
    """
    window_seconds = Duration.parse("2h26m").total_seconds
    count = _LOOKBACK * window_seconds // _MINUTE + 40
    frame = _source_cadence_windows(
        _source_rows(count),
        window="2h26m",
        first_tick=_BASE + window_seconds,
        last_tick=_BASE + count * _MINUTE + 1,
    )
    cut = int(frame.get_column("close_time")[frame.height // 2])
    perturbed = frame.with_columns(
        pl.when(pl.col("close_time") > cut)
        .then(pl.col("close") * 1000.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    before = phased_lookback(
        frame, window="2h26m", emit_every="3m", lookback=_LOOKBACK
    ).frame.filter(pl.col("close_time") <= cut)
    after = phased_lookback(
        perturbed, window="2h26m", emit_every="3m", lookback=_LOOKBACK
    ).frame.filter(pl.col("close_time") <= cut)

    assert before.height > 0
    assert_frame_equal(before, after, check_exact=True)


@settings(max_examples=40, deadline=None)
@given(
    window_minutes=st.integers(min_value=2, max_value=6),
    lookback=st.integers(min_value=1, max_value=4),
)
def test_where_the_emit_grid_divides_the_window_the_two_frames_agree(
    window_minutes: int, lookback: int
) -> None:
    """The case both materializations can answer, and they must agree.

    When `E` divides `W` -- here `E == W`, the aligned case -- every
    `t - kW` is itself an emit tick, so the window's own `E`-cadence
    frame holds exactly the rows the phasing reads from the
    source-cadence one. Where the two CAN both answer, they answer the
    same; the test above covers where only one can.
    """
    window = f"{window_minutes}m"
    window_seconds = window_minutes * _MINUTE
    count = window_minutes * (lookback + 4)
    rows = _source_rows(count)
    first = _BASE + window_seconds
    last = _BASE + count * _MINUTE + 1

    at_source = _source_cadence_windows(
        rows, window=window, first_tick=first, last_tick=last
    )
    at_emit = compute_windows(
        frame_from_rows(rows),
        profile_for(_MINUTE),
        window=window,
        emit_every=window,
        materialization=ExplicitRange(start=first, end=last),
    )

    phased = phased_lookback(
        at_source, window=window, emit_every=window, lookback=lookback
    ).frame

    by_close = {int(row["close_time"]): row for row in at_emit.iter_rows(named=True)}
    for row in phased.drop_nulls().iter_rows(named=True):
        tick = int(row["close_time"])
        for phase in range(lookback):
            counterpart = by_close[tick - phase * window_seconds]
            assert row["close"][phase] == counterpart["close"]
            assert row["volume"][phase] == counterpart["volume"]
            assert row["traded_seconds"][phase] == counterpart["traded_seconds"]


class TestDegenerateFrames:
    """The shapes a frame can be in that are not a frame at all."""

    @pytest.mark.parametrize("value", [True, 1.0, "0"])
    def test_a_threshold_that_is_not_an_int_is_refused(self, value: object) -> None:
        """A threshold of ``True`` would silently mean one second."""
        with pytest.raises(ConfigError, match="must be an int"):
            phased_lookback(
                _default_frame(),
                window="3m",
                emit_every="3m",
                lookback=_LOOKBACK,
                min_traded_seconds=value,  # type: ignore[arg-type]
            )

    @pytest.mark.parametrize("knob", ["window", "emit_every"])
    def test_a_zero_duration_is_refused(self, knob: str) -> None:
        """A zero window phases everything onto one instant."""
        schedule = {"window": "3m", "emit_every": "3m", knob: "0s"}
        with pytest.raises(ConfigError, match="strictly positive"):
            phased_lookback(_default_frame(), lookback=_LOOKBACK, **schedule)  # type: ignore[arg-type]

    def test_a_frame_too_short_to_measure_a_cadence_from_is_refused(self) -> None:
        """One row states no spacing, and the cadence is measured not declared."""
        with pytest.raises(ConfigError, match="at least 2 rows"):
            phased_lookback(
                _default_frame().head(1),
                window="3m",
                emit_every="3m",
                lookback=_LOOKBACK,
            )

    def test_a_close_time_column_that_does_not_ascend_is_refused(self) -> None:
        """Descending rows make every spacing negative, and no cadence at all."""
        frame = _default_frame().head(4)
        reversed_frame = frame.with_columns(
            pl.Series(
                "close_time", frame.get_column("close_time").reverse(), dtype=pl.Int64
            ),
            pl.Series(
                "open_time", frame.get_column("open_time").reverse(), dtype=pl.Int64
            ),
        )

        with pytest.raises(ConfigError, match="ascending"):
            phased_lookback(
                reversed_frame, window="3m", emit_every="3m", lookback=_LOOKBACK
            )

    def test_a_frame_shorter_than_one_emit_step_emits_nothing(self) -> None:
        """No ticks is an empty result, not a refusal.

        The phase guard has nothing to compare when the grid is empty,
        and an empty frame of the right shape is the honest answer to
        "which ticks are inside this range".
        """
        frame = _default_frame().head(2)

        result = phased_lookback(
            frame, window="3m", emit_every="1h", lookback=_LOOKBACK
        )

        assert result.frame.height == 0
        assert result.frame.columns[0] == "close_time"


def test_the_oracle_nulls_a_tick_whose_window_has_no_candles() -> None:
    """A null price is an absent input to the reference too.

    An empty window reports null prices and a real zero `src_count`, so
    the two implementations have to agree that the row is unusable rather
    than one of them keeping the counts.
    """
    frame = _default_frame(count=40)
    holed = frame.with_columns(
        pl.when(pl.col("close_time") == frame.get_column("close_time").to_list()[9])
        .then(None)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    fast = phased_lookback(holed, window="3m", emit_every="3m", lookback=_LOOKBACK)
    slow = phased_lookback_reference(
        holed, window="3m", emit_every="3m", lookback=_LOOKBACK
    )

    assert_frame_equal(fast.frame, slow.frame, check_exact=True)
    assert fast.frame.drop_nulls().height < fast.frame.height


def test_the_two_implementations_agree_about_the_threshold_too() -> None:
    """The oracle applies `min_traded` itself, and so does the fast path.

    Both have to, and for the same reason: a frame written under report
    mode carries its below-threshold rows, so a harness that trusted the
    frame would consume them. Agreeing that they are absent is what makes
    the equivalence suite mean anything for the threshold.
    """
    frame = _threshold_frame([180] * 30)
    starved = frame.with_columns(
        pl.when(pl.col("close_time") % 360 == 0)
        .then(0)
        .otherwise(pl.col("traded_seconds"))
        .alias("traded_seconds")
    )

    fast = phased_lookback(
        starved,
        window="3m",
        emit_every="3m",
        lookback=_LOOKBACK,
        min_traded_seconds=1,
    )
    slow = phased_lookback_reference(
        starved,
        window="3m",
        emit_every="3m",
        lookback=_LOOKBACK,
        min_traded_seconds=1,
    )

    assert_frame_equal(fast.frame, slow.frame, check_exact=True)
    assert fast.frame.drop_nulls().height < fast.frame.height


def test_the_harness_agrees_with_the_oracle_on_the_real_slice() -> None:
    """Twenty thousand real minutes, not a generated grid."""
    real = load_real_slice()
    closes = real.get_column("timestamp").to_list()
    first, last = closes[0], closes[-1]
    frame = compute_windows(
        real,
        BITSTAMP_BTCUSD_1M,
        window="8m",
        emit_every="1m",
        materialization=ExplicitRange(start=first + 8 * _MINUTE, end=last + 1),
    )

    fast = phased_lookback(frame, window="8m", emit_every="8m", lookback=_LOOKBACK)
    slow = phased_lookback_reference(
        frame.head(400), window="8m", emit_every="8m", lookback=_LOOKBACK
    )

    assert_frame_equal(fast.frame.head(slow.frame.height), slow.frame, check_exact=True)
    assert fast.effective_history == Duration.parse("24m")


if __name__ == "__main__":
    pytest.main([__file__])
