"""Property-based tests for raw source-frame validation."""

import polars as pl
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ohlc_toolkit.source.profile import Availability, ColumnKind, SourceProfile
from ohlc_toolkit.source.validation import (
    FindingKind,
    ValidationMode,
    validate_source_frame,
)
from ohlc_toolkit.temporal import Duration
from tests.test_source.factories import build_clean_frame

# A handful of representative cadences rather than an unbounded search
# space: the checks are cadence-agnostic, so a small fixed set exercises
# the arithmetic (modulo, exact-multiple) without inflating run time.
_CADENCE_CHOICES = (1, 5, 60, 300)


def _profile_for(cadence_seconds: int, phase_seconds: int = 0) -> SourceProfile:
    """Build a minimal profile for a given cadence and declared phase."""
    return SourceProfile(
        name="property-test-source",
        phase=Duration(phase_seconds),
        cadence=Duration(cadence_seconds),
        timestamp_column="timestamp",
        availability=Availability.CLOSE_TIME,
        raw_schema={
            "timestamp": ColumnKind.INTEGER,
            "open": ColumnKind.FLOATING,
        },
    )


def _drop_row(frame: pl.DataFrame, index: int) -> pl.DataFrame:
    """Return a copy of ``frame`` with the row at ``index`` removed."""
    return pl.concat([frame.slice(0, index), frame.slice(index + 1)])


def _duplicate_row(frame: pl.DataFrame, index: int) -> pl.DataFrame:
    """Return a copy of ``frame`` with the row at ``index`` repeated."""
    return pl.concat(
        [frame.slice(0, index + 1), frame.slice(index, 1), frame.slice(index + 1)]
    )


@given(
    cadence_seconds=st.sampled_from(_CADENCE_CHOICES),
    length=st.integers(min_value=1, max_value=100),
    data=st.data(),
)
def test_random_complete_grids_always_validate_clean(
    cadence_seconds: int, length: int, data: st.DataObject
) -> None:
    """Any complete grid validates clean against its declared phase.

    The declared phase is drawn FIRST, and the grid's start is built FROM
    it (``start = phase + cadence * k``), rather than deriving a declared
    phase from an already-chosen start: the latter would mirror the
    from-data phase inference this profile deliberately avoids.
    """
    phase_seconds = data.draw(st.integers(min_value=0, max_value=cadence_seconds - 1))
    grid_index = data.draw(st.integers(min_value=0, max_value=200))
    start = phase_seconds + cadence_seconds * grid_index

    frame = build_clean_frame(
        start=start, cadence_seconds=cadence_seconds, length=length
    )
    report = validate_source_frame(
        frame,
        _profile_for(cadence_seconds, phase_seconds=phase_seconds),
        mode=ValidationMode.REPORT,
    )
    assert report.passed


@given(
    cadence_seconds=st.sampled_from(_CADENCE_CHOICES),
    length=st.integers(min_value=1, max_value=100),
    data=st.data(),
)
def test_grid_at_a_different_phase_than_declared_always_reports_off_phase(
    cadence_seconds: int, length: int, data: st.DataObject
) -> None:
    """A grid consistently at phase p always fails a profile declaring q != p."""
    assume(cadence_seconds > 1)  # a phase-1 cadence has only one legal phase: 0
    declared_phase = data.draw(st.integers(min_value=0, max_value=cadence_seconds - 1))
    actual_phase = data.draw(
        st.integers(min_value=0, max_value=cadence_seconds - 1).filter(
            lambda candidate: candidate != declared_phase
        )
    )
    grid_index = data.draw(st.integers(min_value=0, max_value=200))
    start = actual_phase + cadence_seconds * grid_index

    frame = build_clean_frame(
        start=start, cadence_seconds=cadence_seconds, length=length
    )
    report = validate_source_frame(
        frame,
        _profile_for(cadence_seconds, phase_seconds=declared_phase),
        mode=ValidationMode.REPORT,
    )

    off_phase = [f for f in report.findings if f.kind is FindingKind.OFF_PHASE]
    assert len(off_phase) == 1
    assert off_phase[0].count == length


@given(
    cadence_seconds=st.sampled_from(_CADENCE_CHOICES),
    length=st.integers(min_value=3, max_value=100),
    data=st.data(),
)
def test_deleting_an_interior_row_always_yields_one_gap_of_the_right_width(
    cadence_seconds: int, length: int, data: st.DataObject
) -> None:
    """Removing one interior row always reports exactly one one-candle gap."""
    index = data.draw(st.integers(min_value=1, max_value=length - 2))
    frame = build_clean_frame(start=0, cadence_seconds=cadence_seconds, length=length)
    expected_start = frame.get_column("timestamp")[index]
    next_open = frame.get_column("timestamp")[index + 1]
    corrupted = _drop_row(frame, index)

    report = validate_source_frame(
        corrupted, _profile_for(cadence_seconds), mode=ValidationMode.REPORT
    )

    gap_findings = [f for f in report.findings if f.kind is FindingKind.GAP]
    assert len(gap_findings) == 1
    assert gap_findings[0].count == 1
    assert gap_findings[0].sample_timestamps == (expected_start, next_open)


@given(
    cadence_seconds=st.sampled_from(_CADENCE_CHOICES),
    length=st.integers(min_value=1, max_value=100),
    data=st.data(),
)
def test_duplicating_a_row_always_fails_monotonicity(
    cadence_seconds: int, length: int, data: st.DataObject
) -> None:
    """Duplicating any row always fails the strictly-increasing check."""
    index = data.draw(st.integers(min_value=0, max_value=length - 1))
    frame = build_clean_frame(start=0, cadence_seconds=cadence_seconds, length=length)
    corrupted = _duplicate_row(frame, index)

    report = validate_source_frame(
        corrupted, _profile_for(cadence_seconds), mode=ValidationMode.REPORT
    )

    non_increasing = [
        f for f in report.findings if f.kind is FindingKind.NON_INCREASING_TIMESTAMPS
    ]
    assert len(non_increasing) == 1
    assert non_increasing[0].count >= 1


if __name__ == "__main__":
    pytest.main([__file__])
