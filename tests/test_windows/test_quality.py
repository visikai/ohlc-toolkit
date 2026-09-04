"""The window quality-policy step: pass-through, filter, and gate.

Every scenario here starts from a genuinely engine-produced window frame
-- built with :func:`~ohlc_toolkit.windows.engine.compute_windows` over
the same hand-written factories the rest of ``tests/test_windows`` uses
-- rather than a hand-crafted nine-column frame, so these tests exercise
the real output shape the policy composes after.
"""

import math
from collections.abc import Sequence
from fractions import Fraction

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from ohlc_toolkit import windows as windows_namespace
from ohlc_toolkit.temporal import MAX_ECHO_CHARS, ConfigError, CoverageError
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from ohlc_toolkit.windows import quality as quality_module
from ohlc_toolkit.windows.quality import (
    GateMode,
    QualityMode,
    QualityPolicyResult,
    QualityReport,
    WindowCoverageError,
    WindowQualityPolicy,
    apply_quality_policy,
)
from tests.test_windows.factories import SourceRow, frame_from_rows, profile_for

# A ten-second cadence over a one-minute-forty window (100s = 10 slots),
# emitting every 10s: this is fine-grained enough that "one candle short
# of full coverage" and "exactly at a fractional threshold" are both
# reachable by whole src_count values, which a 60s-cadence minute grid
# cannot express (0.9 * 300 is not a multiple of 60).
_CADENCE_SECONDS = 10
_WINDOW = "1m40s"  # 100s = 10 * _CADENCE_SECONDS
_WINDOW_SECONDS = 100
_EMIT_EVERY = "10s"

# Every fixture timestamp is offset by this base -- a real-looking Unix
# second on the 10s grid -- so that a close_time is a ten-digit number
# that cannot appear as a substring of an offending-row count or of a
# threshold. A message assertion of the form "the first offender's
# close_time is named" then means what it says, instead of passing
# because "0" happens to occur inside "100.0".
_TIME_BASE = 1_700_000_000

# A dtype wide enough that echoing it whole would swamp the message.
_PATHOLOGICAL_STRUCT_FIELDS = 1000
# A refusal is fixed prose plus ONE bounded echo, so the ceiling is derived
# from the echo cap rather than written as a round number: if that cap ever
# rises, this rises with it instead of quietly going slack.
_MAX_REFUSAL_MESSAGE_CHARS = 4 * MAX_ECHO_CHARS


def _rows(*open_times: int) -> tuple[SourceRow, ...]:
    """Build source rows at the given open times, with distinct OHLCV values.

    Each row's price/volume fields are derived from its index so that a
    filtered-out row's values are never accidentally identical to a kept
    row's, which would make an assertion pass for the wrong reason.
    """
    return tuple(
        (_TIME_BASE + open_time, 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i, float(i))
        for i, open_time in enumerate(open_times)
    )


def _full_grid_frame() -> pl.DataFrame:
    """Ten consecutive 10s candles: exactly enough for one fully covered window."""
    profile = profile_for(_CADENCE_SECONDS)
    frame = frame_from_rows(_rows(*range(0, 100, _CADENCE_SECONDS)))
    return compute_windows(
        frame,
        profile,
        window=_WINDOW,
        emit_every=_EMIT_EVERY,
        materialization="skip_warmup",
    )


def _ramping_coverage_frame() -> pl.DataFrame:
    """Build a frame whose windows ramp from empty up to fully covered.

    The source candles are a complete, gap-free 10s grid over [0, 300);
    only the emitted range is what varies coverage. Emitting from
    ``close_time = 0`` (an explicit range, not skip_warmup, which would
    refuse to start before the first fully covered tick) means the first
    ten emitted windows each reach ``window_seconds`` further back than
    there is data, so ``src_count`` ramps ``0, 1, 2, ..., 9`` before every
    later window is fully covered at ``src_count == 10``. One fixture
    therefore reaches every coverage level from zero to full without any
    source-side gap at all.
    """
    profile = profile_for(_CADENCE_SECONDS)
    frame = frame_from_rows(_rows(*range(0, 300, _CADENCE_SECONDS)))
    return compute_windows(
        frame,
        profile,
        window=_WINDOW,
        emit_every=_EMIT_EVERY,
        materialization=ExplicitRange(start=_TIME_BASE, end=_TIME_BASE + 310),
    )


def _assert_ohlcv_untouched(before: pl.DataFrame, after: pl.DataFrame) -> None:
    """Assert every row of ``after`` has the exact OHLCV values it had in ``before``.

    Matches rows by ``close_time`` rather than position, so this also
    works for a filtered (row-subset) frame.
    """
    joined = after.join(before, on="close_time", how="left", suffix="_before")
    for column in ("open", "high", "low", "close", "volume"):
        assert_frame_equal(
            joined.select(pl.col(column)),
            joined.select(pl.col(f"{column}_before").alias(column)),
        )


def _quality_frame_from_coverages(coverages: Sequence[int]) -> pl.DataFrame:
    """Build a minimal frame carrying only the columns this step reads.

    Real engine output always carries the full nine columns, but the
    filter and gate logic here reads only three of them, so a test that
    varies coverage alone does not need to fabricate plausible OHLCV
    values to stay honest about what is under test.
    """
    return pl.DataFrame(
        {
            "close_time": [_TIME_BASE + i for i in range(len(coverages))],
            "src_count": [c // 10 for c in coverages],
            "coverage_seconds": coverages,
        }
    ).with_columns(
        # Both casts matter for the empty frame: a [] column infers Null,
        # and the engine emits Int64 for these columns even at height 0.
        pl.col("close_time").cast(pl.Int64),
        pl.col("coverage_seconds").cast(pl.Int64),
    )


class TestWindowQualityPolicyIdentity:
    """Construction, equality, and serialization of the policy identity."""

    def test_default_min_coverage_and_gate_mode(self) -> None:
        """A bare mode is enough to construct a policy."""
        policy = WindowQualityPolicy(mode=QualityMode.PASS_THROUGH)
        assert policy.min_coverage == 1.0
        assert policy.gate_mode is GateMode.STRICT

    def test_same_identity_policies_compare_equal(self) -> None:
        """Two independently constructed, field-identical policies are equal."""
        first = WindowQualityPolicy(
            mode=QualityMode.FILTER, min_coverage=0.9, gate_mode=GateMode.REPORT
        )
        second = WindowQualityPolicy(
            mode=QualityMode.FILTER, min_coverage=0.9, gate_mode=GateMode.REPORT
        )
        assert first == second

    def test_different_min_coverage_compares_unequal(self) -> None:
        """A policy is not equal to one differing only in threshold."""
        first = WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.9)
        second = WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5)
        assert first != second

    @pytest.mark.parametrize(
        "min_coverage",
        [-0.01, 1.01, -1.0, 2.0, math.nan],
        ids=["just_below_0", "just_above_1", "negative", "large", "nan"],
    )
    def test_invalid_threshold_is_rejected_at_construction(
        self, min_coverage: float
    ) -> None:
        """An out-of-range or NaN threshold raises ConfigError immediately."""
        with pytest.raises(ConfigError):
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=min_coverage)

    def test_boolean_min_coverage_is_rejected(self) -> None:
        """``bool`` is an ``int`` subtype in Python, and is rejected anyway."""
        with pytest.raises(ConfigError, match="min_coverage"):
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=True)

    @pytest.mark.parametrize("min_coverage", [0.0, 1.0])
    def test_boundary_thresholds_are_accepted(self, min_coverage: float) -> None:
        """0 and 1 are valid endpoints of the closed [0, 1] range."""
        WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=min_coverage)

    def test_non_quality_mode_is_rejected(self) -> None:
        """A plain string is not accepted in place of a QualityMode member."""
        with pytest.raises(ConfigError, match="mode"):
            WindowQualityPolicy(mode="filter")  # type: ignore[arg-type]

    def test_non_gate_mode_is_rejected(self) -> None:
        """A plain string is not accepted in place of a GateMode member."""
        with pytest.raises(ConfigError, match="gate_mode"):
            WindowQualityPolicy(
                mode=QualityMode.GATE,
                gate_mode="strict",  # type: ignore[arg-type]
            )

    def test_to_dict_is_json_compatible_and_deterministic(self) -> None:
        """to_dict uses only str/float values and a fixed key order."""
        policy = WindowQualityPolicy(
            mode=QualityMode.GATE, min_coverage=0.75, gate_mode=GateMode.REPORT
        )
        assert policy.to_dict() == {
            "mode": "gate",
            "min_coverage": 0.75,
            "gate_mode": "report",
        }
        assert list(policy.to_dict().keys()) == ["mode", "min_coverage", "gate_mode"]

    def test_round_trip_through_dict_reproduces_an_equal_policy(self) -> None:
        """from_dict(to_dict(p)) == p for every field combination."""
        original = WindowQualityPolicy(
            mode=QualityMode.FILTER, min_coverage=0.42, gate_mode=GateMode.REPORT
        )
        assert WindowQualityPolicy.from_dict(original.to_dict()) == original

    def test_from_dict_rejects_a_missing_key(self) -> None:
        """A dict missing a required key is refused, not defaulted."""
        with pytest.raises(ConfigError, match="missing"):
            WindowQualityPolicy.from_dict({"mode": "filter", "min_coverage": 0.5})

    def test_from_dict_rejects_an_unknown_mode(self) -> None:
        """An unrecognized mode string is refused."""
        with pytest.raises(ConfigError, match="mode"):
            WindowQualityPolicy.from_dict(
                {"mode": "sample", "min_coverage": 0.5, "gate_mode": "strict"}
            )

    def test_from_dict_rejects_an_unknown_gate_mode(self) -> None:
        """An unrecognized gate_mode string is refused."""
        with pytest.raises(ConfigError, match="gate_mode"):
            WindowQualityPolicy.from_dict(
                {"mode": "gate", "min_coverage": 0.5, "gate_mode": "lenient"}
            )

    def test_from_dict_rejects_an_invalid_threshold(self) -> None:
        """A threshold outside [0, 1] is refused even when round-tripped."""
        with pytest.raises(ConfigError):
            WindowQualityPolicy.from_dict(
                {"mode": "filter", "min_coverage": 1.5, "gate_mode": "strict"}
            )

    @pytest.mark.parametrize(
        "min_coverage",
        ["0.5", None, [0.5]],
        ids=["numeric_string", "none", "list"],
    )
    def test_from_dict_rejects_a_non_numeric_threshold(
        self, min_coverage: object
    ) -> None:
        """A non-numeric min_coverage is refused, never coerced.

        A stored ``"0.5"`` is the realistic shape of this mistake -- JSON
        that went through a stringly-typed layer -- and quietly calling
        ``float()`` on it would let a policy identity round-trip into
        something the constructor itself would have rejected.
        """
        with pytest.raises(ConfigError, match="min_coverage"):
            WindowQualityPolicy.from_dict(
                {
                    "mode": "filter",
                    "min_coverage": min_coverage,
                    "gate_mode": "strict",
                }
            )


class TestPassThrough:
    """PASS_THROUGH returns the frame unchanged, still an explicit step."""

    def test_returns_an_identical_frame(self) -> None:
        """Every row and column survives pass-through untouched."""
        frame = _full_grid_frame()
        result = apply_quality_policy(
            frame, WindowQualityPolicy(mode=QualityMode.PASS_THROUGH), window=_WINDOW
        )
        assert isinstance(result, QualityPolicyResult)
        assert_frame_equal(result.frame, frame, check_exact=True)

    def test_does_not_mutate_the_input(self) -> None:
        """The input frame is byte-for-byte the same after the call."""
        frame = _full_grid_frame()
        before = frame.clone()
        apply_quality_policy(
            frame, WindowQualityPolicy(mode=QualityMode.PASS_THROUGH), window=_WINDOW
        )
        assert_frame_equal(frame, before, check_exact=True)

    def test_handles_an_empty_frame(self) -> None:
        """An empty frame passes through as an empty frame."""
        frame = _full_grid_frame().clear()
        result = apply_quality_policy(
            frame, WindowQualityPolicy(mode=QualityMode.PASS_THROUGH), window=_WINDOW
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.frame.height == 0


class TestFilter:
    """FILTER drops below-threshold rows and only those, in a new frame."""

    def test_drops_exactly_the_below_threshold_rows(self) -> None:
        """Only rows under the threshold are dropped; kept rows are exact."""
        frame = _ramping_coverage_frame()
        threshold = 0.9 * 100  # 100s window
        expected_close_times = [
            close_time
            for close_time, coverage in zip(
                frame.get_column("close_time").to_list(),
                frame.get_column("coverage_seconds").to_list(),
                strict=True,
            )
            if coverage >= threshold
        ]
        assert expected_close_times, "scenario must include at least one kept row"
        assert len(expected_close_times) < frame.height, (
            "scenario must include at least one dropped row"
        )

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.9),
            window=_WINDOW,
        )

        assert isinstance(result, QualityPolicyResult)
        assert result.frame.get_column("close_time").to_list() == expected_close_times

    def test_returns_a_new_frame_and_does_not_mutate_the_input(self) -> None:
        """Filtering never touches the frame it was given."""
        frame = _ramping_coverage_frame()
        before = frame.clone()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.9),
            window=_WINDOW,
        )
        assert_frame_equal(frame, before, check_exact=True)
        assert result.frame is not frame

    def test_row_order_is_preserved(self) -> None:
        """Kept rows keep their original relative order."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        close_times = result.frame.get_column("close_time").to_list()
        assert close_times == sorted(close_times)

    def test_ohlc_values_are_never_altered(self) -> None:
        """Every surviving row's OHLCV values are exactly what they were."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        _assert_ohlcv_untouched(frame, result.frame)

    def test_min_coverage_zero_keeps_every_row_including_zero_coverage(self) -> None:
        """min_coverage=0 admits even a window with no source candles."""
        frame = _ramping_coverage_frame()
        assert (frame.get_column("src_count") == 0).any(), (
            "scenario must include a zero-coverage row"
        )
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.0),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.frame.height == frame.height

    def test_min_coverage_one_keeps_only_fully_covered_rows(self) -> None:
        """min_coverage=1 is the full-coverage requirement."""
        frame = _ramping_coverage_frame()
        window_seconds = 100
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert (
            result.frame.get_column("coverage_seconds").to_list()
            == [window_seconds] * result.frame.height
        )
        assert (result.frame.get_column("coverage_seconds") < window_seconds).sum() == 0

    def test_fractional_boundary_is_pinned(self) -> None:
        """A row exactly at 0.9 * W is kept; one src_count short is dropped.

        The 100s window over a 10s cadence puts the 90%-coverage boundary
        at src_count == 9 (coverage_seconds == 90), a whole slot below
        full: this is the case a naive rounding rule most often gets
        wrong in one direction or the other.
        """
        frame = _ramping_coverage_frame()
        boundary_rows = frame.filter(pl.col("src_count").is_in([8, 9]))
        assert boundary_rows.height >= 1, "scenario must reach src_count 8 or 9"

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.9),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)

        kept_close_times = set(result.frame.get_column("close_time").to_list())
        for close_time, src_count in zip(
            frame.get_column("close_time").to_list(),
            frame.get_column("src_count").to_list(),
            strict=True,
        ):
            if src_count == 9:  # noqa: PLR2004 - the boundary itself, named above
                assert close_time in kept_close_times
            elif src_count == 8:  # noqa: PLR2004 - one slot short of the boundary
                assert close_time not in kept_close_times

    def test_empty_frame_filters_to_an_empty_frame(self) -> None:
        """Filtering an empty frame is a no-op, not an error."""
        frame = _full_grid_frame().clear()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.frame.height == 0


class TestGateStrict:
    """GATE + STRICT raises on a violation, passes the frame through otherwise."""

    def test_a_fully_covered_frame_passes_cleanly(self) -> None:
        """No violation means the frame is returned unchanged."""
        frame = _full_grid_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert_frame_equal(result.frame, frame, check_exact=True)

    def test_a_violation_raises_an_error_carrying_the_whole_report(self) -> None:
        """Strict mode attaches the findings, so no caller has to parse a message."""
        frame = _ramping_coverage_frame()
        offenders = frame.filter(pl.col("coverage_seconds") < _WINDOW_SECONDS)
        assert offenders.height > 0, "scenario must include a violation"

        with pytest.raises(WindowCoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )

        report = caught.value.report
        assert report.passed is False
        assert report.rows_checked == frame.height
        assert report.offending_count == offenders.height
        assert report.threshold_seconds == _WINDOW_SECONDS
        assert (
            report.first_offending_close_time == offenders.get_column("close_time")[0]
        )

    def test_the_raised_error_is_still_a_coverage_error(self) -> None:
        """The taxonomy's promise holds: callers may keep catching the base."""
        frame = _ramping_coverage_frame()
        with pytest.raises(CoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )
        assert isinstance(caught.value, WindowCoverageError)

    def test_the_message_names_the_real_first_offending_close_time(self) -> None:
        """The summary in the message describes the row the report points at.

        The fixture's close_times are ten-digit Unix seconds, so this
        substring check cannot pass by colliding with a count or a
        threshold elsewhere in the message.
        """
        frame = _ramping_coverage_frame()
        offenders = frame.filter(pl.col("coverage_seconds") < _WINDOW_SECONDS)
        first_offending_close_time = offenders.get_column("close_time")[0]

        with pytest.raises(WindowCoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )

        message = str(caught.value)
        assert str(first_offending_close_time) in message
        assert str(caught.value.report.first_offending_close_time) in message

    def test_first_offending_means_first_in_row_order(self) -> None:
        """No sortedness is assumed: "first" is the first offending ROW.

        The frame below is deliberately out of time order, and its first
        offending row is not the offender with the smallest close_time,
        so an implementation that sorted or took a minimum would be
        caught here.
        """
        frame = pl.DataFrame(
            {
                "close_time": [_TIME_BASE + 300, _TIME_BASE + 100, _TIME_BASE + 200],
                "src_count": [6, 3, 10],
                "coverage_seconds": [60, 30, 100],
            }
        ).with_columns(pl.col("coverage_seconds").cast(pl.Int64))

        with pytest.raises(WindowCoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )

        assert caught.value.report.first_offending_close_time == _TIME_BASE + 300

    def test_the_error_is_exported_from_the_windows_namespace(self) -> None:
        """Callers reach the error the same way they reach the policy."""
        assert windows_namespace.WindowCoverageError is WindowCoverageError
        assert "WindowCoverageError" in windows_namespace.__all__

    def test_does_not_mutate_the_input_even_when_raising(self) -> None:
        """A raised gate never leaves the input frame touched."""
        frame = _ramping_coverage_frame()
        before = frame.clone()
        with pytest.raises(CoverageError):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )
        assert_frame_equal(frame, before, check_exact=True)

    def test_empty_frame_passes_strict_vacuously(self) -> None:
        """No rows means no violation: strict mode does not raise."""
        frame = _full_grid_frame().clear()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.frame.height == 0


class TestGateReport:
    """GATE + REPORT never raises; it always returns structured findings."""

    def test_a_violation_is_reported_without_raising(self) -> None:
        """Findings are returned, not raised, in report mode."""
        frame = _ramping_coverage_frame()
        offenders = frame.filter(pl.col("coverage_seconds") < 100)  # noqa: PLR2004
        assert offenders.height > 0, "scenario must include a violation"

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )

        assert isinstance(result, QualityPolicyResult)
        assert result.report.passed is False
        assert result.report.offending_count == offenders.height
        assert result.report.rows_checked == frame.height
        assert (
            result.report.first_offending_close_time
            == offenders.get_column("close_time")[0]
        )

    def test_a_clean_frame_reports_passed_true(self) -> None:
        """No violation reports passed=True with a zero offending count."""
        frame = _full_grid_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.report.passed is True
        assert result.report.offending_count == 0
        assert result.report.first_offending_close_time is None

    def test_does_not_mutate_the_input(self) -> None:
        """Report mode is read-only over the frame it was given."""
        frame = _ramping_coverage_frame()
        before = frame.clone()
        apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )
        assert_frame_equal(frame, before, check_exact=True)

    def test_empty_frame_reports_passed_true(self) -> None:
        """An empty frame has no offenders to report."""
        frame = _full_grid_frame().clear()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.report.passed is True
        assert result.report.rows_checked == 0


class TestNamedResult:
    """Every non-raising path returns the same pair: the frame and the report."""

    @pytest.mark.parametrize(
        "policy",
        [
            WindowQualityPolicy(mode=QualityMode.PASS_THROUGH),
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
        ],
        ids=["pass_through", "filter", "gate_report", "gate_strict"],
    )
    def test_every_mode_returns_the_same_type(
        self, policy: WindowQualityPolicy
    ) -> None:
        """No caller has to discriminate a return value by isinstance."""
        frame = _full_grid_frame()  # clean, so even a strict gate returns
        result = apply_quality_policy(frame, policy, window=_WINDOW)

        assert type(result) is QualityPolicyResult
        assert isinstance(result.frame, pl.DataFrame)
        assert isinstance(result.report, QualityReport)

    def test_report_mode_no_longer_loses_the_frame(self) -> None:
        """The findings and the data they describe travel together."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )
        assert_frame_equal(result.frame, frame, check_exact=True)
        assert result.report.passed is False

    def test_a_filter_result_accounts_for_exactly_the_rows_it_dropped(self) -> None:
        """The report and the returned frame cannot disagree about the drop."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.9),
            window=_WINDOW,
        )

        report = result.report
        assert report.offending_count > 0, "scenario must drop something"
        assert report.rows_checked == frame.height
        assert result.frame.height == report.rows_checked - report.offending_count

    def test_pass_through_records_what_it_declined_to_act_on(self) -> None:
        """A recorded no-op is still a recorded measurement of the frame."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.PASS_THROUGH, min_coverage=0.9),
            window=_WINDOW,
        )

        assert_frame_equal(result.frame, frame, check_exact=True)
        assert result.report.rows_checked == frame.height
        assert result.report.threshold_seconds == 90  # noqa: PLR2004 - 0.9 * 100s
        assert result.report.offending_count > 0

    def test_the_result_is_frozen(self) -> None:
        """A returned result is a record of what happened, not a mutable box."""
        result = apply_quality_policy(
            _full_grid_frame(),
            WindowQualityPolicy(mode=QualityMode.PASS_THROUGH),
            window=_WINDOW,
        )
        with pytest.raises(AttributeError):
            result.frame = _full_grid_frame()  # type: ignore[misc]

    def test_the_result_is_exported_from_the_windows_namespace(self) -> None:
        """Callers name the return type the same way they name the policy."""
        assert windows_namespace.QualityPolicyResult is QualityPolicyResult
        assert "QualityPolicyResult" in windows_namespace.__all__


def _frame_with_a_null_coverage() -> pl.DataFrame:
    """Build a fully covered frame whose middle row states no coverage at all.

    The engine cannot produce this today -- it computes coverage from a
    count it always has -- but the gate is a fail-closed check, and a
    check that reads a column has to say what it does with a blank in
    it. The surrounding rows are fully covered, so any finding here is
    about the null and nothing else.
    """
    return pl.DataFrame(
        {
            "close_time": [_TIME_BASE, _TIME_BASE + 10, _TIME_BASE + 20],
            "src_count": [10, 0, 10],
            "coverage_seconds": [100, None, 100],
        },
        schema_overrides={"coverage_seconds": pl.Int64},
    )


class TestNullCoverage:
    """A coverage the frame declines to state is a finding, never a pass."""

    def test_report_mode_records_the_null_as_an_offence(self) -> None:
        """An unverifiable row is counted, and counted as its own kind."""
        frame = _frame_with_a_null_coverage()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )

        report = result.report
        assert report.passed is False
        assert report.offending_count == 1
        assert report.null_coverage_count == 1
        assert report.first_offending_close_time == _TIME_BASE + 10

    def test_strict_mode_raises_on_the_null(self) -> None:
        """The gate fails closed: what it cannot verify, it refuses."""
        frame = _frame_with_a_null_coverage()
        with pytest.raises(WindowCoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )
        assert caught.value.report.null_coverage_count == 1

    def test_a_null_offends_even_a_zero_threshold(self) -> None:
        """A null is unverifiable, and stays so however low the bar drops."""
        frame = _frame_with_a_null_coverage()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(
                mode=QualityMode.GATE, min_coverage=0.0, gate_mode=GateMode.REPORT
            ),
            window=_WINDOW,
        )
        assert result.report.null_coverage_count == 1
        assert result.report.passed is False

    def test_filter_drops_the_null_row(self) -> None:
        """FILTER keeps only rows it can show meet the threshold."""
        frame = _frame_with_a_null_coverage()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window=_WINDOW,
        )

        assert result.frame.get_column("close_time").to_list() == [
            _TIME_BASE,
            _TIME_BASE + 20,
        ]
        assert result.frame.get_column("coverage_seconds").null_count() == 0
        assert result.frame.height == (
            result.report.rows_checked - result.report.offending_count
        )

    def test_a_frame_without_nulls_reports_none(self) -> None:
        """The count is a measurement, not a flag that is always set."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )
        assert result.report.null_coverage_count == 0
        assert result.report.offending_count > 0


class TestBoundaryConditions:
    """Cross-mode boundary cases not tied to one specific mode's suite."""

    def test_a_frame_missing_a_required_column_is_refused(self) -> None:
        """The policy cannot evaluate coverage it cannot read."""
        frame = _full_grid_frame().drop("coverage_seconds")
        with pytest.raises(ConfigError, match="coverage_seconds"):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
                window=_WINDOW,
            )

    def test_a_frame_carrying_only_the_columns_read_is_accepted(self) -> None:
        """Nothing beyond close_time and coverage_seconds is required.

        A caller who has projected an engine frame down to what this
        step actually consults is not doing anything wrong, and must not
        be refused for dropping a column no check reads.
        """
        frame = _ramping_coverage_frame().select("close_time", "coverage_seconds")
        assert "src_count" not in frame.columns

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window=_WINDOW,
        )

        assert result.frame.columns == ["close_time", "coverage_seconds"]
        assert result.frame.height == result.report.rows_checked - (
            result.report.offending_count
        )

    def test_a_fractional_coverage_column_is_refused(self) -> None:
        """Coverage must be whole seconds: a float column has no exact verdict."""
        frame = _full_grid_frame().with_columns(
            pl.col("coverage_seconds").cast(pl.Float64)
        )
        with pytest.raises(ConfigError, match="coverage_seconds"):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
                window=_WINDOW,
            )

    def test_the_engines_own_int64_coverage_column_is_accepted(self) -> None:
        """Int64 is the one width the schema declares, and it is accepted."""
        frame = _full_grid_frame()
        assert frame.schema["coverage_seconds"] == pl.Int64

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)
        assert result.frame.height == frame.height

    @pytest.mark.parametrize(
        "dtype",
        [pl.Float64, pl.String, pl.Datetime("us"), pl.UInt32],
        ids=["Float64", "String", "Datetime", "UInt32"],
    )
    def test_a_non_int64_close_time_column_is_refused(self, dtype: pl.DataType) -> None:
        """``close_time`` is held to the same word as its sibling.

        ``coverage_seconds`` is required to be Int64 while ``close_time``
        was merely required to exist -- yet the report reads it with
        ``int(...)``, so a Float64 close time is silently truncated into
        an offender name that is NOT A ROW IN THE FRAME, and a String or
        Datetime one surfaces as a foreign TypeError only on the
        offending path, letting a clean frame with the same wrong dtype
        pass. Refusing every non-Int64 kind up front keeps both failures
        at this module's boundary, in its words.
        """
        frame = _full_grid_frame().with_columns(pl.col("close_time").cast(dtype))
        with pytest.raises(ConfigError, match="Int64"):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
                window=_WINDOW,
            )

    @pytest.mark.parametrize(
        "column", ["close_time", "coverage_seconds"], ids=["close_time", "coverage"]
    )
    def test_a_pathological_dtype_is_echoed_bounded(self, column: str) -> None:
        """Refusing a column must not quote its whole dtype back.

        The dtype is read off the caller's frame, so its size is theirs
        to choose rather than ours; a thousand-field struct renders to
        roughly fifteen thousand characters. Both columns are checked
        because they are the same guard twice, and BOTH the raised
        message and the warning line are checked because they are the
        same fix twice -- bounding only what the exception says leaves
        the log free to carry the whole shape.
        """
        fields = range(_PATHOLOGICAL_STRUCT_FIELDS)
        wide = pl.Struct({f"f{index}": pl.Int64 for index in fields})
        frame = _full_grid_frame().with_columns(
            pl.lit({f"f{index}": 1 for index in fields}, dtype=wide).alias(column)
        )
        widest_field = f"f{_PATHOLOGICAL_STRUCT_FIELDS - 1}"

        # This package keeps its own logger registry, so loguru's module
        # logger is not the one that writes here; attach to the instance
        # the module actually holds.
        logged: list[str] = []
        sink_id = quality_module.logger.add(
            logged.append, level="WARNING", format="{message}"
        )
        try:
            with pytest.raises(ConfigError) as raised:
                apply_quality_policy(
                    frame,
                    WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
                    window=_WINDOW,
                )
        finally:
            quality_module.logger.remove(sink_id)

        message = str(raised.value)
        assert len(message) < _MAX_REFUSAL_MESSAGE_CHARS
        assert widest_field not in message

        assert logged, "the refusal warns before it raises; nothing was captured"
        for line in logged:
            assert len(line) < _MAX_REFUSAL_MESSAGE_CHARS
            assert widest_field not in line

    def test_the_engines_own_int64_close_time_is_accepted(self) -> None:
        """Int64 is what the engine emits for close_time, and it passes."""
        frame = _full_grid_frame()
        assert frame.schema["close_time"] == pl.Int64

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window=_WINDOW,
        )
        assert isinstance(result, QualityPolicyResult)

    @pytest.mark.parametrize(
        "dtype",
        [pl.Int8, pl.UInt8, pl.Int16, pl.Int32, pl.UInt32, pl.UInt64],
        ids=["Int8", "UInt8", "Int16", "Int32", "UInt32", "UInt64"],
    )
    def test_a_non_int64_coverage_column_is_refused(self, dtype: pl.DataType) -> None:
        """Only Int64 will do: the width the engine emits and the schema names.

        Accepting other widths is not a kindness. A narrow column
        overflows inside polars when compared against a large
        whole-second minimum (Int8 against a one-hour window), and a
        UInt64 near the top of its range cannot be safely widened either
        -- so the failure would surface past this boundary, in polars'
        words instead of this module's.
        """
        frame = _full_grid_frame().with_columns(pl.col("coverage_seconds").cast(dtype))
        with pytest.raises(ConfigError, match="Int64"):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
                window=_WINDOW,
            )

    def test_window_is_coerced_from_a_string_like_other_public_entry_points(
        self,
    ) -> None:
        """A compact duration string is accepted, matching Duration | str."""
        frame = _full_grid_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window="1m40s",
        )
        assert isinstance(result, QualityPolicyResult)

    def test_an_invalid_window_string_is_refused(self) -> None:
        """A malformed duration string is rejected at the boundary."""
        frame = _full_grid_frame()
        with pytest.raises(ConfigError):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
                window="not-a-duration",
            )

    def test_a_zero_window_is_refused(self) -> None:
        """A zero-length window must not quietly disarm the gate.

        Every threshold of a 0s window is 0, which every row meets, so
        accepting ``window="0s"`` turns a full-coverage strict gate into
        a pass-all. The sibling engine already refuses a zero window at
        its boundary; this entry point must refuse it in the same words.
        """
        frame = _quality_frame_from_coverages([0, 0])
        with pytest.raises(ConfigError, match="strictly positive"):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(
                    mode=QualityMode.GATE,
                    min_coverage=1.0,
                    gate_mode=GateMode.STRICT,
                ),
                window="0s",
            )


# (min_coverage, window_seconds, exact threshold) triples where the
# IEEE-754 product ``min_coverage * window_seconds`` lands a hair ABOVE
# the threshold the decimal literal names -- 0.55 * 180 evaluates to
# 99.00000000000001, not 99 -- so a row sitting exactly on the intended
# threshold is dropped by a float comparison. Every such pair errs in
# this one direction: a float product is never too loose, only too
# strict.
_FLOAT_PRODUCT_OVERSHOOTS = [
    (0.55, 180, 99),
    (0.56, 100, 56),
    (0.17, 300, 51),
]


class TestExactThreshold:
    """The threshold is the one the decimal literal names, not a float product."""

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold_seconds"),
        _FLOAT_PRODUCT_OVERSHOOTS,
        ids=["0.55_of_180s", "0.56_of_100s", "0.17_of_300s"],
    )
    def test_the_chosen_pairs_really_do_overshoot_in_float(
        self, min_coverage: float, window_seconds: int, threshold_seconds: int
    ) -> None:
        """Guard the fixtures themselves: each pair must actually drift.

        If a future Python or platform ever made these products exact,
        the regression tests below would still pass while no longer
        testing anything, so the drift is asserted explicitly here.
        """
        assert Fraction(str(min_coverage)) * window_seconds == threshold_seconds
        assert min_coverage * window_seconds > threshold_seconds

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold_seconds"),
        _FLOAT_PRODUCT_OVERSHOOTS,
        ids=["0.55_of_180s", "0.56_of_100s", "0.17_of_300s"],
    )
    def test_filter_keeps_a_row_sitting_exactly_on_the_threshold(
        self, min_coverage: float, window_seconds: int, threshold_seconds: int
    ) -> None:
        """A row at exactly ``min_coverage`` of the window survives FILTER."""
        frame = _quality_frame_from_coverages(
            [threshold_seconds - 1, threshold_seconds, window_seconds]
        )

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=min_coverage),
            window=f"{window_seconds}s",
        )

        assert isinstance(result, QualityPolicyResult)
        assert result.frame.get_column("coverage_seconds").to_list() == [
            threshold_seconds,
            window_seconds,
        ]

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold_seconds"),
        _FLOAT_PRODUCT_OVERSHOOTS,
        ids=["0.55_of_180s", "0.56_of_100s", "0.17_of_300s"],
    )
    def test_a_row_sitting_exactly_on_the_threshold_passes_the_strict_gate(
        self, min_coverage: float, window_seconds: int, threshold_seconds: int
    ) -> None:
        """A row at exactly ``min_coverage`` of the window is not a violation."""
        frame = _quality_frame_from_coverages([threshold_seconds, window_seconds])

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(
                mode=QualityMode.GATE,
                min_coverage=min_coverage,
                gate_mode=GateMode.STRICT,
            ),
            window=f"{window_seconds}s",
        )

        assert isinstance(result, QualityPolicyResult)

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold_seconds"),
        _FLOAT_PRODUCT_OVERSHOOTS,
        ids=["0.55_of_180s", "0.56_of_100s", "0.17_of_300s"],
    )
    def test_the_row_one_second_short_is_still_a_violation(
        self, min_coverage: float, window_seconds: int, threshold_seconds: int
    ) -> None:
        """Exactness must not loosen the gate: one second short still fails."""
        frame = _quality_frame_from_coverages([threshold_seconds - 1])

        with pytest.raises(CoverageError):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(
                    mode=QualityMode.GATE,
                    min_coverage=min_coverage,
                    gate_mode=GateMode.STRICT,
                ),
                window=f"{window_seconds}s",
            )


# (min_coverage, window_seconds, exact threshold, least passing second)
# quadruples whose exact threshold is NOT a whole second -- 0.5 * 101 is
# 50.5 -- so no row can sit exactly on it. The integer-threshold pairs
# above cannot tell rounding up from rounding down: ceil and floor agree
# on every whole number, so a bound rounded the wrong way passes all of
# them. These pairs separate the two directions deterministically: the
# whole second just below the threshold must fail, and the least whole
# second above it must pass.
_FRACTIONAL_THRESHOLDS = [
    (0.5, 101, Fraction(101, 2), 51),
    (0.9, 101, Fraction(909, 10), 91),
    (0.55, 181, Fraction(1991, 20), 100),
    (0.17, 301, Fraction(5117, 100), 52),
]
_FRACTIONAL_IDS = ["0.5_of_101s", "0.9_of_101s", "0.55_of_181s", "0.17_of_301s"]


class TestFractionalThreshold:
    """A threshold between whole seconds rounds up, never down.

    ``coverage_seconds`` is a whole number but the exact threshold need
    not be: at ``min_coverage=0.5`` of a 101s window the threshold is
    50.5s. A row at 50s is below half coverage (50/101 is 49.5049...%)
    and must fail; a row at 51s is above it and must pass. A bound
    rounded DOWN admits the 50s row through a 50% policy -- it fails
    open -- while a bound rounded one PAST the least passing second
    rejects genuinely sufficient coverage. Both directions are pinned
    here on fixed inputs, with no randomness anywhere.
    """

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold", "least_passing"),
        _FRACTIONAL_THRESHOLDS,
        ids=_FRACTIONAL_IDS,
    )
    def test_the_chosen_thresholds_really_do_fall_between_whole_seconds(
        self,
        min_coverage: float,
        window_seconds: int,
        threshold: Fraction,
        least_passing: int,
    ) -> None:
        """Guard the fixtures themselves: each threshold must be fractional.

        A whole-number threshold is met exactly by some row, so rounding
        it up and rounding it down name the same bound and the case
        guards nothing. Only a threshold strictly between two whole
        seconds separates the two rounding directions.
        """
        assert Fraction(str(min_coverage)) * window_seconds == threshold
        assert threshold.denominator > 1
        assert math.ceil(threshold) == least_passing
        assert math.floor(threshold) == least_passing - 1

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold", "least_passing"),
        _FRACTIONAL_THRESHOLDS,
        ids=_FRACTIONAL_IDS,
    )
    def test_filter_drops_below_the_threshold_and_keeps_the_least_passing(
        self,
        min_coverage: float,
        window_seconds: int,
        threshold: Fraction,
        least_passing: int,
    ) -> None:
        """FILTER rejects the whole second below and admits the one above."""
        frame = _quality_frame_from_coverages(
            [least_passing - 1, least_passing, window_seconds]
        )

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=min_coverage),
            window=f"{window_seconds}s",
        )

        assert isinstance(result, QualityPolicyResult)
        assert result.frame.get_column("coverage_seconds").to_list() == [
            least_passing,
            window_seconds,
        ]
        assert result.report.offending_count == 1
        assert result.report.threshold_seconds == threshold

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold", "least_passing"),
        _FRACTIONAL_THRESHOLDS,
        ids=_FRACTIONAL_IDS,
    )
    def test_the_strict_gate_raises_on_the_whole_second_below_the_threshold(
        self,
        min_coverage: float,
        window_seconds: int,
        threshold: Fraction,
        least_passing: int,
    ) -> None:
        """GATE/STRICT counts the second just below a fractional threshold."""
        frame = _quality_frame_from_coverages([least_passing - 1])

        with pytest.raises(WindowCoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(
                    mode=QualityMode.GATE,
                    min_coverage=min_coverage,
                    gate_mode=GateMode.STRICT,
                ),
                window=f"{window_seconds}s",
            )

        assert caught.value.report.threshold_seconds == threshold
        assert caught.value.report.offending_count == 1

    @pytest.mark.parametrize(
        ("min_coverage", "window_seconds", "threshold", "least_passing"),
        _FRACTIONAL_THRESHOLDS,
        ids=_FRACTIONAL_IDS,
    )
    def test_the_strict_gate_passes_the_least_whole_second_above_the_threshold(
        self,
        min_coverage: float,
        window_seconds: int,
        threshold: Fraction,
        least_passing: int,
    ) -> None:
        """GATE/STRICT does not over-round: the least passing second is no violation."""
        frame = _quality_frame_from_coverages([least_passing, window_seconds])

        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(
                mode=QualityMode.GATE,
                min_coverage=min_coverage,
                gate_mode=GateMode.STRICT,
            ),
            window=f"{window_seconds}s",
        )

        assert isinstance(result, QualityPolicyResult)
        assert result.report.threshold_seconds == threshold


# --- Property-based: filter keeps exactly the rows meeting the threshold ---


@settings(max_examples=200)
@given(
    coverages=st.lists(
        st.integers(min_value=0, max_value=300), min_size=0, max_size=30
    ),
    min_coverage=st.floats(
        min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
    ),
)
def test_filter_keeps_exactly_the_rows_meeting_the_threshold(
    coverages: list[int], min_coverage: float
) -> None:
    """For any frame and fraction, FILTER's kept rows are exactly the >= ones.

    The oracle deliberately does NOT reuse the implementation's
    expression. It reads ``min_coverage`` by its decimal intent -- the
    number the caller wrote -- and multiplies it by the window as an
    exact rational, so the expected set is the mathematically correct one
    rather than whatever a double-precision product happens to land on.
    An oracle written as ``min_coverage * window_seconds`` would confirm
    the implementation instead of checking it.
    """
    window_seconds = 300
    frame = _quality_frame_from_coverages(coverages)
    threshold = Fraction(str(min_coverage)) * window_seconds

    result = apply_quality_policy(
        frame,
        WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=min_coverage),
        window=f"{window_seconds}s",
    )

    assert isinstance(result, QualityPolicyResult)
    expected = [c for c in coverages if c >= threshold]
    assert result.frame.get_column("coverage_seconds").to_list() == expected


if __name__ == "__main__":
    pytest.main([__file__])
