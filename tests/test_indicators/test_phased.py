"""The phased lookback harness, against its oracle and against the rule.

The load-bearing test here is the one at `W = 2h26m, E = 3m`. Everything
else could be satisfied by a harness that read its inputs from the
window's own `E`-cadence frame; that case is where the two differ, and
where reading from the `E` frame silently returns nothing.
"""

import polars as pl
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from ohlc_toolkit.indicators import (
    MAX_LOOKBACK,
    PHASED_COLUMNS,
    phased_lookback,
    phased_lookback_reference,
)
from ohlc_toolkit.indicators import frames as phased_frames
from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.windows import (
    ExplicitRange,
    GateMode,
    QualityMode,
    WindowQualityPolicy,
    apply_quality_policy,
    compute_windows,
)
from ohlc_toolkit.windows import quality as quality_module
from tests.test_windows.factories import frame_from_rows, profile_for
from tests.test_windows.fixtures import load_real_slice
from tests.test_windows.synthetic import FAMILY_NAMES, build_family

_MINUTE = 60

# The lookback every test below uses unless it says otherwise: enough
# phases that an off-by-one in the ordering is visible, few enough that a
# fixture stays readable.
_LOOKBACK = 3

# The emit step every tick-grid test uses, in seconds.
_THREE_MINUTES = 180
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


@pytest.mark.parametrize(("window", "emit"), [("2h26m", "3m"), ("6h20m", "7m")])
def test_every_phase_reads_the_right_value_where_the_emit_grid_cannot_reach(
    window: str, emit: str
) -> None:
    """The case the source-cadence rule exists for, checked on VALUES.

    `E` does not divide `W`, so `t - kW` is not an emit tick. Reading the
    phases from the window's own `E`-cadence frame finds nothing at all;
    reading from the source-cadence frame finds every one of them.

    The earlier version of this test asserted only that each `t - kW` was
    a close time of the fixture -- which it re-derived from the fixture --
    and never read a value. A harness reading one cadence step off the
    true row passed it. Every field is compared against the source frame's
    own row now, and the `E`-cadence frame is built here to show the
    lookup really does miss there.
    """
    window_seconds = Duration.parse(window).total_seconds
    emit_seconds = Duration.parse(emit).total_seconds
    assert window_seconds % emit_seconds != 0
    count = _LOOKBACK * window_seconds // _MINUTE + 40
    rows = _source_rows(count)
    first, last = _BASE + window_seconds, _BASE + count * _MINUTE + 1
    frame = _source_cadence_windows(
        rows, window=window, first_tick=first, last_tick=last
    )

    result = phased_lookback(frame, window=window, emit_every=emit, lookback=_LOOKBACK)
    complete = result.frame.drop_nulls()
    assert complete.height > 0

    by_close = {int(row["close_time"]): row for row in frame.iter_rows(named=True)}
    for row in complete.iter_rows(named=True):
        tick = int(row["close_time"])
        for phase in range(_LOOKBACK):
            source = by_close[tick - phase * window_seconds]
            for field in ("open", "high", "low", "close", "volume"):
                assert row[field][phase] == source[field], (field, phase)
            assert row["traded_seconds"][phase] == source["traded_seconds"]

    # And the emit-cadence materialization really cannot answer: none of
    # the phases behind the first complete tick is one of its rows.
    at_emit = compute_windows(
        frame_from_rows(rows),
        profile_for(_MINUTE),
        window=window,
        emit_every=emit,
        materialization=ExplicitRange(start=first, end=last),
    )
    emit_closes = set(at_emit.get_column("close_time").to_list())
    a_tick = int(complete.get_column("close_time")[0])
    assert a_tick in emit_closes
    assert all(
        a_tick - phase * window_seconds not in emit_closes
        for phase in range(1, _LOOKBACK)
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
        """A real `GateMode.REPORT` artifact, not a hand-zeroed column.

        Report mode returns the frame UNCHANGED with a report beside it,
        so a below-threshold row is still in the artifact a recipe reads.
        A harness that assumed filtering had happened would consume dead
        windows and report numbers for them. The frame here is put
        through the policy itself, so the test fails if report mode ever
        starts removing rows and this stops being the case it claims.
        """
        # One starved window, not a periodic set: with `L = 3` on a
        # three-minute grid, starving every sixth minute puts a dead
        # phase in EVERY tick's set, and the "some survive" half of the
        # assertion below could not hold.
        base = _threshold_frame([180] * 30)
        starved_at = base.get_column("close_time").to_list()[12]
        frame = base.with_columns(
            pl.when(pl.col("close_time") == starved_at)
            .then(0)
            .otherwise(pl.col("traded_seconds"))
            .alias("traded_seconds")
        )
        reported = apply_quality_policy(
            frame,
            WindowQualityPolicy(
                mode=QualityMode.GATE,
                gate_mode=GateMode.REPORT,
                min_traded_seconds=1,
            ),
            window="3m",
        )

        # The artifact really is unchanged: report mode removed nothing.
        assert reported.frame.height == frame.height
        assert reported.report.traded_offending_count > 0

        result = phased_lookback(
            reported.frame,
            window="3m",
            emit_every="3m",
            lookback=_LOOKBACK,
            min_traded_seconds=1,
        )

        # Some ticks survive and some do not: a harness that ignored the
        # threshold would null none, and one that nulled everything would
        # pass an assertion that only counted nulls.
        assert 0 < result.frame.drop_nulls().height < result.frame.height


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

    The earlier version cut at the frame's midpoint, which at
    `W = 2h26m, L = 3` is before any tick has all three phases -- so it
    compared 56 rows of nulls to 56 rows of nulls and passed even when
    every non-zero phase read the wrong row. Three things fix that: the
    cut is past `L * W` so the compared region carries data, the
    non-nullness is asserted rather than assumed, and the perturbation is
    shown to be VISIBLE after the cut. Without that last one the test
    cannot tell "the future did not leak" from "nothing happened".
    """
    window_seconds = Duration.parse("2h26m").total_seconds
    count = _LOOKBACK * window_seconds // _MINUTE + 200
    frame = _source_cadence_windows(
        _source_rows(count),
        window="2h26m",
        first_tick=_BASE + window_seconds,
        last_tick=_BASE + count * _MINUTE + 1,
    )
    # Past `L * W` from the frame's start, so ticks at or before the cut
    # have their whole phase set.
    closes = frame.get_column("close_time").to_list()
    cut = next(
        close for close in closes if close >= closes[0] + _LOOKBACK * window_seconds
    )
    perturbed = frame.with_columns(
        pl.when(pl.col("close_time") > cut)
        .then(pl.col("close") * 1000.0)
        .otherwise(pl.col("close"))
        .alias("close")
    )

    whole_before = phased_lookback(
        frame, window="2h26m", emit_every="3m", lookback=_LOOKBACK
    ).frame
    whole_after = phased_lookback(
        perturbed, window="2h26m", emit_every="3m", lookback=_LOOKBACK
    ).frame
    before = whole_before.filter(pl.col("close_time") <= cut)
    after = whole_after.filter(pl.col("close_time") <= cut)

    # The compared region has real numbers in it, not nulls compared to
    # nulls -- which is what made the previous version vacuous.
    assert before.drop_nulls().height > 0
    assert_frame_equal(before, after, check_exact=True)

    # And the perturbation really happened: past the cut the two differ.
    later = whole_before.filter(pl.col("close_time") > cut).drop_nulls()
    later_perturbed = whole_after.filter(pl.col("close_time") > cut).drop_nulls()
    assert later.height > 0
    assert (
        later.get_column("close").to_list()
        != later_perturbed.get_column("close").to_list()
    )


@settings(max_examples=40, deadline=None)
@given(
    window_minutes=st.integers(min_value=2, max_value=6),
    divisor=st.integers(min_value=1, max_value=6),
    lookback=st.integers(min_value=1, max_value=4),
)
def test_where_the_emit_grid_divides_the_window_the_two_frames_agree(
    window_minutes: int, divisor: int, lookback: int
) -> None:
    """The case both materializations can answer, and they must agree.

    When `E` divides `W`, every `t - kW` is itself an emit tick, so the
    window's own `E`-cadence frame holds exactly the rows the phasing
    reads from the source-cadence one. `E == W` is only one of those
    cases; the emit cadence here is drawn as any proper divisor too,
    because "divides" is what the property is about and the aligned case
    is the one where the source-cadence rule does the least work.

    Every phased field is compared, not the three that were easiest.
    """
    assume(window_minutes % divisor == 0)
    window = f"{window_minutes}m"
    emit = f"{window_minutes // divisor}m"
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
        emit_every=emit,
        materialization=ExplicitRange(start=first, end=last),
    )

    phased = phased_lookback(
        at_source, window=window, emit_every=emit, lookback=lookback
    ).frame

    by_close = {int(row["close_time"]): row for row in at_emit.iter_rows(named=True)}
    for row in phased.drop_nulls().iter_rows(named=True):
        tick = int(row["close_time"])
        for phase in range(lookback):
            counterpart = by_close[tick - phase * window_seconds]
            for field in PHASED_COLUMNS:
                assert row[field][phase] == counterpart[field], (field, phase)


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
    # Both halves: some ticks lost their output and some kept it. A
    # harness that nulled everything satisfies a bare "fewer than all"
    # and proves nothing about the threshold.
    assert 0 < fast.frame.drop_nulls().height < fast.frame.height


def test_the_two_implementations_agree_about_the_threshold_too() -> None:
    """The oracle applies `min_traded` itself, and so does the fast path.

    Both have to, and for the same reason: a frame written under report
    mode carries its below-threshold rows, so a harness that trusted the
    frame would consume them. Agreeing that they are absent is what makes
    the equivalence suite mean anything for the threshold.
    """
    frame = _threshold_frame([180] * 30)
    starved_at = frame.get_column("close_time").to_list()[12]
    starved = frame.with_columns(
        pl.when(pl.col("close_time") == starved_at)
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
    # Both halves: some ticks lost their output and some kept it. A
    # harness that nulled everything satisfies a bare "fewer than all"
    # and proves nothing about the threshold.
    assert 0 < fast.frame.drop_nulls().height < fast.frame.height


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


@pytest.mark.parametrize(
    ("window", "emit"),
    [("2h26m", "3m"), ("6h20m", "7m"), ("35m", "4m"), ("12m", "4m"), ("12m", "6m")],
)
def test_the_harness_agrees_with_the_oracle_where_the_grid_does_not_divide(
    window: str, emit: str
) -> None:
    """Equivalence in the regime this subpackage exists for.

    Every other oracle comparison passes `emit_every == window`, which is
    exactly the case where `t - kW` IS an emit tick and the source-cadence
    rule makes no difference. In the un-aligned regime the oracle was
    never consulted at all, which left one test as the whole guard against
    reading the wrong row. The last two pairs are proper divisors, where
    the two materializations must also agree.
    """
    window_seconds = Duration.parse(window).total_seconds
    count = _LOOKBACK * window_seconds // _MINUTE + 30
    first, last = _BASE + window_seconds, _BASE + count * _MINUTE + 1
    frame = _source_cadence_windows(
        _source_rows(count), window=window, first_tick=first, last_tick=last
    )

    fast = phased_lookback(frame, window=window, emit_every=emit, lookback=_LOOKBACK)
    slow = phased_lookback_reference(
        frame, window=window, emit_every=emit, lookback=_LOOKBACK
    )

    assert fast.frame.drop_nulls().height > 0
    assert_frame_equal(
        fast.frame,
        slow.frame,
        check_exact=True,
        check_dtypes=True,
        check_column_order=True,
        check_row_order=True,
    )


class TestTheEmitTickGrid:
    """The tick tuple itself, which oracle equivalence cannot see.

    Both implementations share one `_emit_ticks`, deliberately, so their
    refusals cannot drift apart -- and the price is that an error in it
    shows up in neither. Measured before this class existed: dropping the
    LAST tick passed all 1600 tests, and on the default fixture that tick
    is the frame's newest and non-null row. A live indicator's most
    valuable answer would vanish silently.
    """

    @staticmethod
    def _ticks(frame: pl.DataFrame, *, emit: str, anchor: str = "0s") -> list[int]:
        return (
            phased_lookback(
                frame, window="3m", emit_every=emit, anchor=anchor, lookback=1
            )
            .frame.get_column("close_time")
            .to_list()
        )

    def test_the_last_row_is_a_tick_when_it_sits_on_the_grid(self) -> None:
        """The endpoint case, on a frame whose last row IS an emit tick.

        This is the one that matters and the one a careless fixture
        cannot see: `range(first, highest, E)` and
        `range(first, highest + 1, E)` differ ONLY when `highest` is on
        the grid. The default fixture's last row is not, so both spellings
        agree on it and the off-by-one hides. Measured: with the frame
        trimmed to end on the grid, dropping the endpoint fails here and
        nowhere else -- and on a live frame that row is the newest and
        most valuable answer the harness has.
        """
        frame = _default_frame(count=40)
        closes = frame.get_column("close_time").to_list()
        last_on_grid = max(close for close in closes if close % _THREE_MINUTES == 0)
        trimmed = frame.filter(pl.col("close_time") <= last_on_grid)

        ticks = self._ticks(trimmed, emit="3m")

        assert ticks[-1] == last_on_grid

    def test_the_grid_is_contiguous_between_its_endpoints(self) -> None:
        """No tick is skipped in the middle, and the first is not before the start."""
        frame = _default_frame(count=40)
        closes = frame.get_column("close_time").to_list()

        ticks = self._ticks(frame, emit="3m")

        assert ticks[0] >= closes[0]
        assert ticks[0] - _THREE_MINUTES < closes[0]
        assert ticks[-1] <= closes[-1]
        assert ticks == list(range(ticks[0], ticks[-1] + 1, _THREE_MINUTES))

    def test_a_frame_starting_exactly_on_the_grid_keeps_its_first_row(self) -> None:
        """`offset == 0` is its own case, and off-by-one lives there."""
        frame = _default_frame(count=40)
        closes = frame.get_column("close_time").to_list()
        on_grid = next(close for close in closes if close % _THREE_MINUTES == 0)
        aligned = frame.filter(pl.col("close_time") >= on_grid)

        ticks = self._ticks(aligned, emit="3m")

        # The frame's first row IS a tick, so it must be kept: the offset
        # is zero here and an off-by-one would drop it.
        assert ticks[0] == on_grid

    def test_an_anchor_shifts_the_whole_grid(self) -> None:
        """The anchor is the grid's phase, and every tick carries it."""
        frame = _default_frame(count=40)

        plain = self._ticks(frame, emit="3m")
        shifted = self._ticks(frame, emit="3m", anchor="1m")

        assert all(tick % _THREE_MINUTES == 0 for tick in plain)
        assert all(tick % _THREE_MINUTES == _MINUTE for tick in shifted)


def test_a_repeated_close_time_is_refused_as_a_repeat() -> None:
    """A duplicate is not "not ascending", and the message says which.

    An exact-equality lookup against a repeated key has two answers, and
    "this grid does not ascend" sends a reader looking for a sort.
    """
    frame = _default_frame(count=10)
    doubled = pl.concat([frame, frame.head(1)]).sort("close_time")

    with pytest.raises(ConfigError, match="repeats one"):
        phased_lookback(doubled, window="3m", emit_every="3m", lookback=_LOOKBACK)


def test_a_null_close_time_is_refused_rather_than_crashing() -> None:
    """A `ConfigError`, as the public docstring promises, not a TypeError."""
    frame = _default_frame(count=10)
    holed = frame.with_columns(
        pl.when(pl.col("close_time") == frame.get_column("close_time").to_list()[3])
        .then(None)
        .otherwise(pl.col("close_time"))
        .alias("close_time")
    )

    with pytest.raises(ConfigError, match="must not be null"):
        phased_lookback(holed, window="3m", emit_every="3m", lookback=_LOOKBACK)


def test_one_inserted_off_grid_row_cannot_redefine_the_cadence() -> None:
    """The cadence is the MODAL step, not the smallest.

    Taking the smallest let a single row inserted one second off the grid
    redefine the cadence as 1s -- at which point the emit-multiple rule
    and the anchor-phase rule are both vacuously satisfied and the caller
    gets a frame of nulls, which is the answer those rules exist to
    refuse.
    """
    frame = _default_frame(count=20)
    closes = frame.get_column("close_time").to_list()
    intruder = frame.head(1).with_columns(
        pl.lit(closes[5] + 1, dtype=pl.Int64).alias("close_time"),
        pl.lit(closes[5] + 1 - 180, dtype=pl.Int64).alias("open_time"),
    )
    contaminated = pl.concat([frame, intruder]).sort("close_time")

    with pytest.raises(ConfigError, match="whole multiple"):
        phased_lookback(contaminated, window="3m", emit_every="3m", lookback=_LOOKBACK)


def test_a_lookback_beyond_the_cap_is_refused() -> None:
    """Each phase is a join, so an unbounded count is an unbounded query."""
    with pytest.raises(ConfigError, match="at most"):
        phased_lookback(
            _default_frame(), window="3m", emit_every="3m", lookback=MAX_LOOKBACK + 1
        )


def test_the_two_threshold_validators_agree_without_being_one() -> None:
    """The duplication is deliberate, so something has to compare them.

    `windows.quality` validates the threshold a POLICY states and this
    module the one a RECIPE states. They are separate numbers that share
    a grammar, so importing one into the other would couple a recipe's
    validation to a policy's and let relaxing one silently relax the
    other. A copy nothing compares is a copy that drifts, so this is the
    comparison.
    """
    for bad, match in ((-1, "negative"), ("nope", "must be an int")):
        with pytest.raises(ConfigError, match=match):
            quality_module._validated_min_traded_seconds(bad)
        with pytest.raises(ConfigError, match=match):
            phased_frames._validated_threshold(bad)


if __name__ == "__main__":
    pytest.main([__file__])
