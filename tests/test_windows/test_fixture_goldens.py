"""Committed golden frames over the seeded synthetic source families.

Each golden file is the oracle's output for one reviewed case, stored as
CSV -- not parquet -- so that a change in behaviour shows up as a diff a
human can read line by line. The tests below regenerate every case and
compare BYTES, so a reformat, a dtype change, or a shifted null is a
failure just as much as a wrong number is.
"""

import pytest

from ohlc_toolkit.source import ValidationMode, validate_source_frame
from ohlc_toolkit.source.validation import FindingKind
from tests.test_windows.synthetic import (
    FAMILY_NAMES,
    GOLDEN_CASES,
    GOLDENS_DIRECTORY,
    GoldenCase,
    build_family,
    render_golden_csv,
)


def _case_label(case: GoldenCase) -> str:
    """Name a parametrized case after its golden file."""
    return case.label


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=_case_label)
def test_the_oracle_still_reproduces_each_committed_golden(case: GoldenCase) -> None:
    """Regenerating a case matches the committed file byte for byte."""
    assert case.path.exists(), f"missing golden file {case.path}"
    assert render_golden_csv(case).encode("utf-8") == case.path.read_bytes()


@pytest.mark.parametrize("case", GOLDEN_CASES, ids=_case_label)
def test_regenerating_a_golden_twice_in_process_is_identical(
    case: GoldenCase,
) -> None:
    """Determinism: same seeds, same schedule, same bytes, twice running."""
    assert render_golden_csv(case) == render_golden_csv(case)


def test_every_committed_golden_file_belongs_to_a_case() -> None:
    """No orphaned goldens: the directory and the case matrix agree exactly."""
    committed = sorted(path.name for path in GOLDENS_DIRECTORY.glob("*.csv"))
    expected = sorted(f"{case.label}.csv" for case in GOLDEN_CASES)
    assert committed == expected


def test_the_golden_matrix_exercises_every_synthetic_family() -> None:
    """A family with no golden case is a family nothing is pinning."""
    assert {case.family for case in GOLDEN_CASES} == set(FAMILY_NAMES)


@pytest.mark.parametrize("name", FAMILY_NAMES)
def test_building_a_family_twice_yields_an_identical_frame(name: str) -> None:
    """The families are seeded, so two builds are the same data."""
    assert build_family(name).frame.equals(build_family(name).frame)


@pytest.mark.parametrize(
    "name", ["complete_grid_1m", "complete_grid_1s", "phased_grid_1m"]
)
def test_the_complete_families_pass_strict_source_validation(name: str) -> None:
    """The families that claim to be complete really are, phase included."""
    family = build_family(name)
    report = validate_source_frame(
        family.frame, family.profile, mode=ValidationMode.REPORT
    )
    assert report.passed, report.findings


def test_the_single_gap_family_has_exactly_one_three_candle_gap() -> None:
    """One run of three missing candles, reported as one finding."""
    family = build_family("single_gap_1m")
    report = validate_source_frame(
        family.frame, family.profile, mode=ValidationMode.REPORT
    )

    gaps = [finding for finding in report.findings if finding.kind is FindingKind.GAP]
    assert [finding.count for finding in gaps] == [3]
    assert report.findings == tuple(gaps)


def test_the_multi_gap_family_has_the_four_gaps_it_was_built_with() -> None:
    """Slots 1-2, 13, 22-25, and 38 are missing, in that order."""
    family = build_family("multi_gap_1m")
    report = validate_source_frame(
        family.frame, family.profile, mode=ValidationMode.REPORT
    )

    gaps = [finding for finding in report.findings if finding.kind is FindingKind.GAP]
    assert [finding.count for finding in gaps] == [2, 1, 4, 1]
    assert report.findings == tuple(gaps)


def test_the_straddling_family_is_deliberately_invalid_input() -> None:
    """Its whole point is rows off the declared grid; validation must say so.

    A source that validates cleanly can never produce a candle crossing a
    window boundary, so the only way to pin the whole-candle exclusion
    rule is input that strict validation rejects. This test records that
    the family is invalid on purpose rather than by accident.
    """
    family = build_family("straddling_1m")
    report = validate_source_frame(
        family.frame, family.profile, mode=ValidationMode.REPORT
    )

    assert not report.passed
    kinds = {finding.kind for finding in report.findings}
    assert FindingKind.OFF_PHASE in kinds
    assert FindingKind.OVERLAPPING_INTERVALS in kinds


if __name__ == "__main__":
    pytest.main([__file__])
