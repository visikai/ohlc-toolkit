"""Property-based tests for raw source-frame validation."""

import polars as pl
import pytest
from hypothesis import given
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


def _profile_for(cadence_seconds: int) -> SourceProfile:
    """Build a minimal profile for a given cadence, for property tests."""
    return SourceProfile(
        name="property-test-source",
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
    start=st.integers(min_value=0, max_value=10_000),
    length=st.integers(min_value=1, max_value=100),
)
def test_random_complete_grids_always_validate_clean(
    cadence_seconds: int, start: int, length: int
) -> None:
    """Any complete, evenly spaced grid validates clean, whatever its phase."""
    frame = build_clean_frame(
        start=start, cadence_seconds=cadence_seconds, length=length
    )
    report = validate_source_frame(
        frame, _profile_for(cadence_seconds), mode=ValidationMode.REPORT
    )
    assert report.passed


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
