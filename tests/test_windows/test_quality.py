"""The window quality-policy step: pass-through, filter, and gate.

Every scenario here starts from a genuinely engine-produced window frame
-- built with :func:`~ohlc_toolkit.windows.engine.compute_windows` over
the same hand-written factories the rest of ``tests/test_windows`` uses
-- rather than a hand-crafted nine-column frame, so these tests exercise
the real output shape the policy composes after.
"""

import math
from collections.abc import Sequence

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import assert_frame_equal

from ohlc_toolkit.temporal import ConfigError, CoverageError
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from ohlc_toolkit.windows.quality import (
    GateMode,
    QualityMode,
    QualityReport,
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
_EMIT_EVERY = "10s"


def _rows(*open_times: int) -> tuple[SourceRow, ...]:
    """Build source rows at the given open times, with distinct OHLCV values.

    Each row's price/volume fields are derived from its index so that a
    filtered-out row's values are never accidentally identical to a kept
    row's, which would make an assertion pass for the wrong reason.
    """
    return tuple(
        (open_time, 100.0 + i, 110.0 + i, 90.0 + i, 105.0 + i, float(i))
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
        materialization=ExplicitRange(start=0, end=310),
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


class TestPassThrough:
    """PASS_THROUGH returns the frame unchanged, still an explicit step."""

    def test_returns_an_identical_frame(self) -> None:
        """Every row and column survives pass-through untouched."""
        frame = _full_grid_frame()
        result = apply_quality_policy(
            frame, WindowQualityPolicy(mode=QualityMode.PASS_THROUGH), window=_WINDOW
        )
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, frame, check_exact=True)

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
        assert isinstance(result, pl.DataFrame)
        assert result.height == 0


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

        assert isinstance(result, pl.DataFrame)
        assert result.get_column("close_time").to_list() == expected_close_times

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
        assert result is not frame

    def test_row_order_is_preserved(self) -> None:
        """Kept rows keep their original relative order."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
            window=_WINDOW,
        )
        assert isinstance(result, pl.DataFrame)
        close_times = result.get_column("close_time").to_list()
        assert close_times == sorted(close_times)

    def test_ohlc_values_are_never_altered(self) -> None:
        """Every surviving row's OHLCV values are exactly what they were."""
        frame = _ramping_coverage_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
            window=_WINDOW,
        )
        assert isinstance(result, pl.DataFrame)
        _assert_ohlcv_untouched(frame, result)

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
        assert isinstance(result, pl.DataFrame)
        assert result.height == frame.height

    def test_min_coverage_one_keeps_only_fully_covered_rows(self) -> None:
        """min_coverage=1 is the full-coverage requirement."""
        frame = _ramping_coverage_frame()
        window_seconds = 100
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=1.0),
            window=_WINDOW,
        )
        assert isinstance(result, pl.DataFrame)
        assert (
            result.get_column("coverage_seconds").to_list()
            == [window_seconds] * result.height
        )
        assert (result.get_column("coverage_seconds") < window_seconds).sum() == 0

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
        assert isinstance(result, pl.DataFrame)

        kept_close_times = set(result.get_column("close_time").to_list())
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
        assert isinstance(result, pl.DataFrame)
        assert result.height == 0


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
        assert isinstance(result, pl.DataFrame)
        assert_frame_equal(result, frame, check_exact=True)

    def test_a_violation_raises_coverage_error_naming_the_first_offender(self) -> None:
        """Strict mode raises, naming the first offending close_time."""
        frame = _ramping_coverage_frame()
        offenders = frame.filter(pl.col("coverage_seconds") < 100)  # noqa: PLR2004
        assert offenders.height > 0, "scenario must include a violation"
        first_offending_close_time = offenders.get_column("close_time")[0]

        with pytest.raises(CoverageError) as caught:
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.STRICT),
                window=_WINDOW,
            )

        message = str(caught.value)
        assert str(first_offending_close_time) in message
        assert str(offenders.height) in message  # bounded summary: offending count
        assert "100" in message  # bounded summary: threshold

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
        assert isinstance(result, pl.DataFrame)
        assert result.height == 0


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

        assert isinstance(result, QualityReport)
        assert result.passed is False
        assert result.offending_count == offenders.height
        assert result.rows_checked == frame.height
        assert (
            result.first_offending_close_time == offenders.get_column("close_time")[0]
        )

    def test_a_clean_frame_reports_passed_true(self) -> None:
        """No violation reports passed=True with a zero offending count."""
        frame = _full_grid_frame()
        result = apply_quality_policy(
            frame,
            WindowQualityPolicy(mode=QualityMode.GATE, gate_mode=GateMode.REPORT),
            window=_WINDOW,
        )
        assert isinstance(result, QualityReport)
        assert result.passed is True
        assert result.offending_count == 0
        assert result.first_offending_close_time is None

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
        assert isinstance(result, QualityReport)
        assert result.passed is True
        assert result.rows_checked == 0


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
        assert isinstance(result, pl.DataFrame)

    def test_an_invalid_window_string_is_refused(self) -> None:
        """A malformed duration string is rejected at the boundary."""
        frame = _full_grid_frame()
        with pytest.raises(ConfigError):
            apply_quality_policy(
                frame,
                WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=0.5),
                window="not-a-duration",
            )


# --- Property-based: filter keeps exactly the rows meeting the threshold ---


def _quality_frame_from_coverages(
    coverages: Sequence[int], window_seconds: int
) -> pl.DataFrame:
    """Build a minimal frame carrying only the columns this step reads.

    Real engine output always carries the full nine columns, but the
    filter and gate logic here reads only three of them, so a property
    test that varies coverage alone does not need to fabricate plausible
    OHLCV values to stay honest about what is under test.
    """
    return pl.DataFrame(
        {
            "close_time": list(range(len(coverages))),
            "src_count": [c // 10 for c in coverages],
            "coverage_seconds": coverages,
        }
    ).with_columns(pl.col("coverage_seconds").cast(pl.Int64))


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
    """For any frame and fraction, FILTER's kept rows are exactly the >= ones."""
    window_seconds = 300
    frame = _quality_frame_from_coverages(coverages, window_seconds)
    threshold = min_coverage * window_seconds

    result = apply_quality_policy(
        frame,
        WindowQualityPolicy(mode=QualityMode.FILTER, min_coverage=min_coverage),
        window=f"{window_seconds}s",
    )

    assert isinstance(result, pl.DataFrame)
    expected = [c for c in coverages if c >= threshold]
    assert result.get_column("coverage_seconds").to_list() == expected


if __name__ == "__main__":
    pytest.main([__file__])
