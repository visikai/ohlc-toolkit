"""Tests for strict and report-only raw source-frame validation."""

import time
import unittest
from dataclasses import replace

import polars as pl

from ohlc_toolkit.source import validation as validation_module
from ohlc_toolkit.source.profile import Availability, ColumnKind, SourceProfile
from ohlc_toolkit.source.validation import (
    Finding,
    FindingKind,
    SourceValidationError,
    ValidationMode,
    ValidationReport,
    validate_source_frame,
)
from ohlc_toolkit.temporal import MAX_ECHO_CHARS, DataValidationError
from tests.test_source.factories import build_clean_frame

_PROFILE = SourceProfile.create(
    name="synthetic-1m",
    cadence="1m",
    timestamp_column="timestamp",
    availability=Availability.CLOSE_TIME,
    raw_schema={
        "timestamp": ColumnKind.INTEGER,
        "open": ColumnKind.FLOATING,
        "high": ColumnKind.FLOATING,
        "low": ColumnKind.FLOATING,
        "close": ColumnKind.FLOATING,
        "volume": ColumnKind.FLOATING,
    },
)

_CADENCE_SECONDS = 60

# A dtype wide enough that echoing it whole would swamp the finding, and a
# ceiling derived from the echo cap rather than written as a round number,
# so raising that cap cannot leave this assertion slack.
_PATHOLOGICAL_STRUCT_FIELDS = 1000
# A profile name far longer than any real one, for the strict refusal's echo.
_PATHOLOGICAL_NAME_CHARS = 200_000
_MAX_FINDING_MESSAGE_CHARS = 4 * MAX_ECHO_CHARS


def _set_timestamps(frame: pl.DataFrame, timestamps: list[int | None]) -> pl.DataFrame:
    """Return a copy of ``frame`` with its timestamp column replaced."""
    return frame.with_columns(pl.Series("timestamp", timestamps, dtype=pl.Int64))


def _drop_rows(frame: pl.DataFrame, indices: set[int]) -> pl.DataFrame:
    """Return a copy of ``frame`` with the given row indices removed."""
    keep = [i for i in range(frame.height) if i not in indices]
    return frame[keep]


def _find(report: ValidationReport, kind: FindingKind) -> list[Finding]:
    """Return every finding of a given kind in a report."""
    return [finding for finding in report.findings if finding.kind is kind]


class TestCleanFrameValidation(unittest.TestCase):
    """Test cases for validating a clean, complete grid."""

    def setUp(self):
        """Build a clean ten-row grid shared by every test in this class."""
        self.frame = build_clean_frame(
            start=0, cadence_seconds=_CADENCE_SECONDS, length=10
        )

    def test_strict_mode_does_not_raise(self):
        """A clean frame passes strict validation without raising."""
        report = validate_source_frame(self.frame, _PROFILE, mode=ValidationMode.STRICT)
        self.assertTrue(report.passed)

    def test_report_mode_has_zero_findings(self):
        """A clean frame produces an empty findings tuple in report mode."""
        report = validate_source_frame(self.frame, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(report.findings, ())
        self.assertTrue(report.passed)
        self.assertEqual(report.rows_checked, 10)

    def test_report_only_never_mutates_the_input_frame(self):
        """Validating in report mode leaves the input frame untouched."""
        before = self.frame.clone()
        validate_source_frame(self.frame, _PROFILE, mode=ValidationMode.REPORT)
        self.assertTrue(self.frame.equals(before))

    def test_strict_mode_never_mutates_the_input_frame(self):
        """Validating in strict mode leaves the input frame untouched."""
        before = self.frame.clone()
        validate_source_frame(self.frame, _PROFILE, mode=ValidationMode.STRICT)
        self.assertTrue(self.frame.equals(before))


class TestEmptyAndSingleRowFrames(unittest.TestCase):
    """Test cases proving validation never crashes on degenerate input."""

    def test_empty_frame_passes_with_zero_rows_checked(self):
        """An empty frame has nothing to check and reports zero rows."""
        frame = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=0)
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(report.rows_checked, 0)
        self.assertTrue(report.passed)

    def test_single_row_frame_with_no_diffs_still_passes(self):
        """A single on-grid row has no diffs to check but still validates."""
        frame = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=1)
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(report.rows_checked, 1)
        self.assertTrue(report.passed)

    def test_single_row_off_the_declared_phase_is_reported(self):
        """Even a lone row must lie on the profile's declared grid."""
        frame = build_clean_frame(start=37, cadence_seconds=_CADENCE_SECONDS, length=1)
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.OFF_PHASE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)


class TestSchemaValidationBothModes(unittest.TestCase):
    """Test cases for the missing-column and wrong-dtype schema checks."""

    def setUp(self):
        """Build a clean grid shared by every test in this class."""
        self.frame = build_clean_frame(
            start=0, cadence_seconds=_CADENCE_SECONDS, length=5
        )

    def test_missing_column_reports_a_schema_finding(self):
        """Dropping a required column is caught as a schema finding."""
        corrupted = self.frame.drop("open")
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.SCHEMA)
        self.assertEqual(len(findings), 1)
        self.assertIn("open", findings[0].message)

    def test_missing_column_raises_in_strict_mode(self):
        """Dropping a required column raises in strict mode."""
        corrupted = self.frame.drop("open")
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_wrong_dtype_reports_a_schema_finding(self):
        """A required column with the wrong numeric kind is caught."""
        corrupted = self.frame.with_columns(pl.col("open").cast(pl.Utf8))
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.SCHEMA)
        self.assertEqual(len(findings), 1)
        self.assertIn("open", findings[0].message)

    def test_wrong_dtype_raises_in_strict_mode(self):
        """A wrong-kind column raises in strict mode."""
        corrupted = self.frame.with_columns(pl.col("open").cast(pl.Utf8))
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_wrong_dtype_on_timestamp_column_does_not_crash_other_checks(self):
        """A non-numeric timestamp column is reported, not crashed on."""
        corrupted = self.frame.with_columns(pl.col("timestamp").cast(pl.Utf8))
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(len(_find(report, FindingKind.SCHEMA)), 1)

    def test_missing_timestamp_column_does_not_crash_other_checks(self):
        """Dropping the timestamp column itself skips row-level checks safely."""
        corrupted = self.frame.drop("timestamp")
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.SCHEMA)
        self.assertEqual(len(findings), 1)
        self.assertIn("timestamp", findings[0].message)

    def test_overly_long_column_name_is_truncated_in_the_message(self):
        """An overly long declared column name is echoed truncated, not in full."""
        long_name = "x" * 80
        profile = SourceProfile.create(
            name="long-column-name-1m",
            timestamp_column="timestamp",
            availability=Availability.CLOSE_TIME,
            raw_schema={
                "timestamp": ColumnKind.INTEGER,
                long_name: ColumnKind.FLOATING,
            },
            cadence="1m",
        )
        frame = pl.DataFrame({"timestamp": pl.Series([0, 60, 120], dtype=pl.Int64)})

        report = validate_source_frame(frame, profile, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.SCHEMA)
        self.assertEqual(len(findings), 1)
        self.assertIn("...", findings[0].message)
        self.assertNotIn(long_name, findings[0].message)

    def test_a_pathological_dtype_is_echoed_bounded(self):
        """A wide struct dtype must not put its whole shape in the message.

        The echoed dtype is read off the INPUT frame, so its size is the
        caller's to choose rather than ours. A thousand-field struct
        renders to roughly fifteen thousand characters unbounded.
        """
        fields = range(_PATHOLOGICAL_STRUCT_FIELDS)
        wide = pl.Struct({f"f{index}": pl.Int64 for index in fields})
        frame = pl.DataFrame(
            {"timestamp": pl.Series([{f"f{i}": 1 for i in fields}], dtype=wide)}
        )

        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.SCHEMA)
        self.assertEqual(len(findings), 1)
        message = findings[0].message
        self.assertLess(len(message), _MAX_FINDING_MESSAGE_CHARS)
        self.assertNotIn(f"f{_PATHOLOGICAL_STRUCT_FIELDS - 1}", message)

    def test_an_ordinary_dtype_is_echoed_in_full(self):
        """Bounding the pathological case must not reword the ordinary one."""
        corrupted = self.frame.with_columns(pl.col("open").cast(pl.Utf8))

        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.SCHEMA)
        self.assertEqual(len(findings), 1)
        self.assertEqual(
            findings[0].message,
            "wrong-kind columns: ['open (expected floating, got String)']",
        )


class TestMonotonicityBothModes(unittest.TestCase):
    """Test cases for the strictly-increasing timestamp check."""

    def setUp(self):
        """Build a clean five-row grid shared by every test in this class."""
        self.frame = build_clean_frame(
            start=0, cadence_seconds=_CADENCE_SECONDS, length=5
        )

    def test_duplicate_timestamp_reports_a_finding(self):
        """A duplicated timestamp fails strict increase in report mode."""
        corrupted = _set_timestamps(self.frame, [0, 60, 60, 180, 240])
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NON_INCREASING_TIMESTAMPS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)
        self.assertIn(60, findings[0].sample_timestamps)

    def test_duplicate_timestamp_raises_in_strict_mode(self):
        """A duplicated timestamp raises in strict mode."""
        corrupted = _set_timestamps(self.frame, [0, 60, 60, 180, 240])
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_out_of_order_rows_report_a_finding(self):
        """A row that goes backward in time fails strict increase."""
        corrupted = _set_timestamps(self.frame, [0, 120, 60, 180, 240])
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NON_INCREASING_TIMESTAMPS)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)

    def test_out_of_order_rows_raise_in_strict_mode(self):
        """A row that goes backward in time raises in strict mode."""
        corrupted = _set_timestamps(self.frame, [0, 120, 60, 180, 240])
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)


class TestOffPhaseBothModes(unittest.TestCase):
    """Test cases for the cadence-grid phase check.

    The grid's phase is DECLARED by the profile (default: zero), never
    inferred from the data: a uniformly shifted grid is corruption, not a
    new convention. Shifting one interior row necessarily disturbs its
    neighbouring diffs too, so that corruption is expected to also raise
    other findings; only the presence of the phase finding itself is
    asserted here.
    """

    def setUp(self):
        """Build a clean grid with one interior row shifted off phase."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        timestamps = clean.get_column("timestamp").to_list()
        timestamps[2] += 15
        self.shifted_timestamp = timestamps[2]
        self.off_phase = _set_timestamps(clean, timestamps)

    def test_shifted_row_reports_an_off_phase_finding(self):
        """The row whose residue disagrees with the grid is reported."""
        report = validate_source_frame(
            self.off_phase, _PROFILE, mode=ValidationMode.REPORT
        )
        findings = _find(report, FindingKind.OFF_PHASE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)
        self.assertIn(self.shifted_timestamp, findings[0].sample_timestamps)

    def test_off_phase_raises_in_strict_mode(self):
        """An off-phase row raises in strict mode."""
        with self.assertRaises(DataValidationError):
            validate_source_frame(self.off_phase, _PROFILE, mode=ValidationMode.STRICT)

    def test_uniformly_shifted_grid_fails_the_declared_phase(self):
        """A consistent but shifted grid is off-grid corruption, not a pass."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        shifted = _set_timestamps(
            clean, [t + 30 for t in clean.get_column("timestamp").to_list()]
        )
        report = validate_source_frame(shifted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.OFF_PHASE)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 5)

    def test_declared_nonzero_phase_accepts_a_matching_grid(self):
        """A grid on a shifted schedule passes when the profile declares it."""
        profile = SourceProfile.create(
            name=_PROFILE.name,
            timestamp_column=_PROFILE.timestamp_column,
            availability=_PROFILE.availability,
            raw_schema=dict(_PROFILE.raw_schema),
            cadence="1m",
            phase="30s",
        )
        frame = build_clean_frame(start=30, cadence_seconds=_CADENCE_SECONDS, length=5)
        report = validate_source_frame(frame, profile, mode=ValidationMode.REPORT)
        self.assertTrue(report.passed)


class TestNullValuesBothModes(unittest.TestCase):
    """Test cases for the null-value check on every declared column."""

    def setUp(self):
        """Build a clean five-row grid shared by every test in this class."""
        self.frame = build_clean_frame(
            start=0, cadence_seconds=_CADENCE_SECONDS, length=5
        )

    def test_null_timestamp_reports_a_null_values_finding(self):
        """A null in the timestamp column is reported, not silently skipped."""
        corrupted = _set_timestamps(self.frame, [0, 60, None, 180, 240])
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NULL_VALUES)
        self.assertEqual(len(findings), 1)
        self.assertIn("timestamp", findings[0].message)

    def test_null_timestamp_raises_in_strict_mode(self):
        """A null timestamp raises in strict mode instead of validating clean."""
        corrupted = _set_timestamps(self.frame, [0, 60, None, 180, 240])
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_null_timestamp_does_not_also_report_a_false_gap(self):
        """A null hides a real gap; row-level checks must not run over it."""
        corrupted = _set_timestamps(self.frame, [0, 60, None, 180, 240])
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(_find(report, FindingKind.GAP), [])
        self.assertEqual(_find(report, FindingKind.OFF_PHASE), [])
        self.assertEqual(_find(report, FindingKind.NON_INCREASING_TIMESTAMPS), [])

    def test_null_in_a_non_timestamp_column_is_also_reported(self):
        """A null in a non-timestamp declared column (volume) is caught too."""
        corrupted = self.frame.with_columns(
            pl.Series("volume", [1.0, None, 1.0, 1.0, 1.0], dtype=pl.Float64)
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NULL_VALUES)
        self.assertEqual(len(findings), 1)
        self.assertIn("volume", findings[0].message)

    def test_null_in_a_non_timestamp_column_raises_in_strict_mode(self):
        """A null volume value raises in strict mode."""
        corrupted = self.frame.with_columns(
            pl.Series("volume", [1.0, None, 1.0, 1.0, 1.0], dtype=pl.Float64)
        )
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_null_in_a_non_timestamp_column_does_not_gate_row_level_checks(self):
        """A null confined to a non-timestamp column leaves diff checks running."""
        corrupted = self.frame.with_columns(
            pl.Series("volume", [1.0, None, 1.0, 1.0, 1.0], dtype=pl.Float64)
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        # The timestamp column itself is untouched and clean, so no
        # spurious row-level finding fires alongside the null-values one.
        self.assertEqual(len(_find(report, FindingKind.NULL_VALUES)), 1)
        self.assertEqual(_find(report, FindingKind.GAP), [])
        self.assertEqual(_find(report, FindingKind.OFF_PHASE), [])


class TestOverlapAndIrregularFindingsBehaviorally(unittest.TestCase):
    """Test cases pinning kind, count, sample, and message for two finding kinds.

    Both frames below also disturb the declared grid phase, so an
    OFF_PHASE finding is expected alongside the finding under test; only
    the specific finding under test is asserted, not the full report.
    """

    def test_overlap_detects_opens_closer_than_a_full_cadence_step(self):
        """Consecutive opens 30s apart under a 60s cadence overlap each other."""
        frame = _set_timestamps(
            build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=3),
            [0, 30, 60],
        )
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.OVERLAPPING_INTERVALS)
        self.assertEqual(len(findings), 1)
        # Both diffs (0->30 and 30->60) are 30s, under the 60s cadence, so
        # both successive opens count as overlapping.
        self.assertEqual(findings[0].count, 2)
        self.assertIn(30, findings[0].sample_timestamps)
        self.assertIn("60s", findings[0].message)

    def test_irregular_interval_detects_a_gap_that_is_not_an_exact_multiple(self):
        """A 90s gap under a 60s cadence is irregular, not a clean two-step gap."""
        frame = _set_timestamps(
            build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=3),
            [0, 90, 150],
        )
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.IRREGULAR_INTERVAL)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)
        self.assertIn(90, findings[0].sample_timestamps)
        self.assertIn("60s", findings[0].message)


class TestMonotonicityGatesDownstreamChecks(unittest.TestCase):
    """Test cases proving unsorted input skips checks that assume a rising timeline."""

    def test_unsorted_complete_grid_reports_only_non_increasing(self):
        """A complete grid that is merely unsorted must not also report a false gap."""
        frame = _set_timestamps(
            build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5),
            [60, 0, 120, 180, 240],
        )
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)

        self.assertEqual(len(_find(report, FindingKind.NON_INCREASING_TIMESTAMPS)), 1)
        self.assertEqual(_find(report, FindingKind.GAP), [])
        self.assertEqual(_find(report, FindingKind.OVERLAPPING_INTERVALS), [])
        self.assertEqual(_find(report, FindingKind.IRREGULAR_INTERVAL), [])


class TestGapDetectionBothModes(unittest.TestCase):
    """Test cases for half-open gap-interval reporting."""

    def test_single_missing_candle_reports_the_exact_interval(self):
        """Removing one interior row reports a one-candle gap."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        corrupted = _drop_rows(clean, {2})  # drops timestamp 120

        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.GAP)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)
        self.assertEqual(findings[0].sample_timestamps, (120, 180))

    def test_multi_minute_gap_reports_the_exact_interval_and_count(self):
        """Removing two consecutive rows reports a two-candle gap."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        corrupted = _drop_rows(clean, {2, 3})  # drops timestamps 120 and 180

        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)

        findings = _find(report, FindingKind.GAP)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 2)
        self.assertEqual(findings[0].sample_timestamps, (120, 240))

    def test_gap_raises_in_strict_mode(self):
        """Any detected gap raises in strict mode."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        corrupted = _drop_rows(clean, {2})
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_gap_finding_count_is_capped_for_a_pathological_frame(self):
        """A frame with far more gap runs than the cap is still bounded."""
        # 1000 is the documented cap; every other row present yields one
        # gap run per surviving diff, so 1002 rows yield 1001 gap runs.
        cap = 1000
        row_count = cap + 2
        timestamps: list[int | None] = [
            i * 2 * _CADENCE_SECONDS for i in range(row_count)
        ]
        frame = _set_timestamps(
            build_clean_frame(
                start=0, cadence_seconds=_CADENCE_SECONDS, length=row_count
            ),
            timestamps,
        )

        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)

        self.assertEqual(len(_find(report, FindingKind.GAP)), cap)


class TestSparseSpanPerformance(unittest.TestCase):
    """Regression test guarding against full-span materialization."""

    def test_two_far_apart_rows_validate_quickly_with_one_gap(self):
        """A ~10**9 second gap validates fast with an exact single finding."""
        cadence_seconds = 60
        far_apart_seconds = 600_000_000  # an exact, huge multiple of cadence
        frame = _set_timestamps(
            build_clean_frame(start=0, cadence_seconds=cadence_seconds, length=2),
            [0, far_apart_seconds],
        )

        started = time.monotonic()
        report = validate_source_frame(frame, _PROFILE, mode=ValidationMode.REPORT)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 1.0)
        findings = _find(report, FindingKind.GAP)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, far_apart_seconds // cadence_seconds - 1)
        self.assertEqual(
            findings[0].sample_timestamps, (cadence_seconds, far_apart_seconds)
        )


class TestStrictModeErrorCarriesTheReport(unittest.TestCase):
    """Test cases proving the strict-mode exception exposes its report."""

    def test_exception_exposes_the_validation_report(self):
        """The raised SourceValidationError carries the full report."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        corrupted = _drop_rows(clean, {2})

        with self.assertRaises(SourceValidationError) as ctx:
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

        report = ctx.exception.report
        self.assertIsInstance(report, ValidationReport)
        self.assertFalse(report.passed)
        self.assertEqual(len(_find(report, FindingKind.GAP)), 1)

    def test_exception_is_a_data_validation_error(self):
        """The typed exception is still catchable via the broader base class."""
        clean = build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=5)
        corrupted = _drop_rows(clean, {2})

        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)


if __name__ == "__main__":
    unittest.main()


class TestStrictRefusalEchoesTheProfileNameBounded(unittest.TestCase):
    """The strict refusal quotes the profile name, so the quote is bounded.

    Every shipped profile has a short name, but nothing caps the field, so
    the refusal treats it like any value it did not choose: through the
    same bound, in the message and in the log line alike.
    """

    def test_an_enormous_profile_name_is_echoed_bounded_on_both_exits(self):
        """Neither the raised message nor the error log carries the whole name."""
        loud = replace(_PROFILE, name="n" * _PATHOLOGICAL_NAME_CHARS)
        frame = _set_timestamps(
            build_clean_frame(start=0, cadence_seconds=_CADENCE_SECONDS, length=3),
            [0, 120, 180],
        )
        logged: list[str] = []
        sink_id = validation_module.logger.add(
            logged.append, level="ERROR", format="{message}"
        )
        try:
            with self.assertRaises(SourceValidationError) as caught:
                validate_source_frame(frame, loud, mode=ValidationMode.STRICT)
        finally:
            validation_module.logger.remove(sink_id)

        self.assertLess(len(str(caught.exception)), _MAX_FINDING_MESSAGE_CHARS)
        self.assertTrue(logged, "the refusal logs before it raises")
        self.assertLess(len(logged[-1]), _MAX_FINDING_MESSAGE_CHARS)


class TestNonFiniteValuesBothModes(unittest.TestCase):
    """Test cases for the non-finite-value check on declared floating columns.

    A NaN used to pass validation completely clean: the null check counts
    nulls, and a NaN is not a null. It is a present cell that is not a
    number, and unlike a null it propagates -- one NaN price contaminates
    every window that averages it, and announces itself nowhere.
    """

    def setUp(self):
        """Build a clean five-row grid shared by every test in this class."""
        self.frame = build_clean_frame(
            start=0, cadence_seconds=_CADENCE_SECONDS, length=5
        )

    def _with_open(self, values: list[float]) -> pl.DataFrame:
        """Replace the open column with the given values."""
        return self.frame.with_columns(pl.Series("open", values, dtype=pl.Float64))

    def test_a_nan_price_is_reported_rather_than_passing_clean(self):
        """The defect this check exists for: a NaN open used to validate clean."""
        corrupted = self._with_open([1.0, float("nan"), 3.0, 4.0, 5.0])
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NON_FINITE_VALUES)
        self.assertFalse(report.passed)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 1)
        self.assertIn("open", findings[0].message)

    def test_a_nan_price_raises_in_strict_mode(self):
        """Strict validation refuses a NaN instead of handing the frame over."""
        corrupted = self._with_open([1.0, float("nan"), 3.0, 4.0, 5.0])
        with self.assertRaises(DataValidationError):
            validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.STRICT)

    def test_both_infinities_are_reported_like_a_nan(self):
        """Infinity is the same question as NaN and gets the same answer."""
        for value in (float("inf"), float("-inf")):
            with self.subTest(value=value):
                corrupted = self._with_open([1.0, value, 3.0, 4.0, 5.0])
                report = validate_source_frame(
                    corrupted, _PROFILE, mode=ValidationMode.REPORT
                )
                findings = _find(report, FindingKind.NON_FINITE_VALUES)
                self.assertEqual(len(findings), 1)
                self.assertEqual(findings[0].count, 1)

    def test_a_non_finite_volume_is_reported_too(self):
        """The rule covers every price or volume column, not just prices."""
        corrupted = self.frame.with_columns(
            pl.Series("volume", [1.0, 1.0, float("nan"), 1.0, 1.0], dtype=pl.Float64)
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NON_FINITE_VALUES)
        self.assertEqual(len(findings), 1)
        self.assertIn("volume", findings[0].message)

    def test_a_nan_is_not_reported_as_a_null(self):
        """The two kinds stay distinct: a report conflating them misstates the data."""
        corrupted = self._with_open([1.0, float("nan"), 3.0, 4.0, 5.0])
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(_find(report, FindingKind.NULL_VALUES), [])
        self.assertEqual(len(_find(report, FindingKind.NON_FINITE_VALUES)), 1)

    def test_a_null_is_not_reported_as_non_finite(self):
        """And the other way: a null is an absent cell, not a present non-number."""
        corrupted = self._with_open([1.0, None, 3.0, 4.0, 5.0])  # type: ignore[list-item]
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(_find(report, FindingKind.NON_FINITE_VALUES), [])
        self.assertEqual(len(_find(report, FindingKind.NULL_VALUES)), 1)

    def test_every_non_finite_cell_is_counted_across_columns(self):
        """The count is cells, not rows or columns, and each column is named once."""
        corrupted = self.frame.with_columns(
            pl.Series("open", [float("nan"), 2.0, float("inf"), 4.0, 5.0]),
            pl.Series("close", [1.0, float("-inf"), 3.0, 4.0, 5.0]),
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NON_FINITE_VALUES)
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0].count, 3)
        self.assertIn("open (2)", findings[0].message)
        self.assertIn("close (1)", findings[0].message)

    def test_a_non_finite_value_does_not_gate_the_row_level_checks(self):
        """A bad price says nothing about the timeline, which is still checked."""
        corrupted = _drop_rows(self._with_open([1.0, float("nan"), 3.0, 4.0, 5.0]), {2})
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(len(_find(report, FindingKind.NON_FINITE_VALUES)), 1)
        self.assertEqual(len(_find(report, FindingKind.GAP)), 1)

    def test_an_integer_column_is_not_asked_whether_it_is_finite(self):
        """Only declared floating columns are read; an integer cannot hold a NaN."""
        report = validate_source_frame(self.frame, _PROFILE, mode=ValidationMode.REPORT)
        self.assertTrue(report.passed)

    def test_a_floating_column_that_arrived_as_an_integer_is_a_schema_finding_only(
        self,
    ):
        """An integer column cannot hold a non-finite value; the mismatch is the schema check's."""
        corrupted = self.frame.with_columns(
            pl.Series("open", [1, 2, 3, 4, 5], dtype=pl.Int64)
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(len(_find(report, FindingKind.SCHEMA)), 1)
        self.assertEqual(_find(report, FindingKind.NON_FINITE_VALUES), [])

    def test_a_declared_column_the_frame_lacks_is_not_read(self):
        """A missing column is the schema check's finding, not this one's."""
        report = validate_source_frame(
            self.frame.drop("open"), _PROFILE, mode=ValidationMode.REPORT
        )
        self.assertEqual(len(_find(report, FindingKind.SCHEMA)), 1)
        self.assertEqual(_find(report, FindingKind.NON_FINITE_VALUES), [])

    def test_the_finding_message_stays_bounded_for_a_loud_column_name(self):
        """Column names are the profile's, and nothing caps their length."""
        loud = "n" * _PATHOLOGICAL_NAME_CHARS
        profile = SourceProfile.create(
            name="synthetic-1m",
            cadence="1m",
            timestamp_column="timestamp",
            availability=Availability.CLOSE_TIME,
            raw_schema={"timestamp": ColumnKind.INTEGER, loud: ColumnKind.FLOATING},
        )
        corrupted = pl.DataFrame(
            {"timestamp": [0, 60], loud: [1.0, float("nan")]},
            schema={"timestamp": pl.Int64, loud: pl.Float64},
        )
        report = validate_source_frame(corrupted, profile, mode=ValidationMode.REPORT)
        findings = _find(report, FindingKind.NON_FINITE_VALUES)
        self.assertEqual(len(findings), 1)
        self.assertLess(len(findings[0].message), _MAX_FINDING_MESSAGE_CHARS)

    def test_a_floating_column_that_arrived_as_text_does_not_crash_the_check(self):
        """Polars raises on `is_finite` for a string column, so the dtype guard is real."""
        corrupted = self.frame.with_columns(
            pl.Series("open", ["1.0", "2.0", "3.0", "4.0", "5.0"], dtype=pl.String)
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(len(_find(report, FindingKind.SCHEMA)), 1)
        self.assertEqual(_find(report, FindingKind.NON_FINITE_VALUES), [])

    def test_a_column_declared_integer_is_not_scanned_even_when_it_arrives_as_a_float(
        self,
    ):
        """The rule is about price and volume columns, which is what the profile declares.

        A frame handing a float -- NaN and all -- to a column the profile
        declares integer has a schema problem, and that is what is
        reported. Widening this check to whatever happens to arrive as a
        float would report the same cell twice under two different names.
        """
        corrupted = self.frame.with_columns(
            pl.Series("timestamp", [0.0, 60.0, float("nan"), 180.0, 240.0])
        )
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(len(_find(report, FindingKind.SCHEMA)), 1)
        self.assertEqual(_find(report, FindingKind.NON_FINITE_VALUES), [])

    def test_a_floating_column_that_arrived_as_a_decimal_is_a_schema_finding_only(self):
        """Decimal is why the guard tests `is_float`, not `is_numeric`.

        A Decimal column reports itself numeric and still raises on
        `is_finite`, so a guard written as `is_numeric()` would let it
        through and turn a reported schema finding into an exception. It is
        the one dtype that tells the two spellings apart.
        """
        corrupted = self.frame.with_columns(
            pl.Series("open", [1, 2, 3, 4, 5], dtype=pl.Decimal(10, 2))
        )
        self.assertTrue(corrupted.get_column("open").dtype.is_numeric())
        report = validate_source_frame(corrupted, _PROFILE, mode=ValidationMode.REPORT)
        self.assertEqual(len(_find(report, FindingKind.SCHEMA)), 1)
        self.assertEqual(_find(report, FindingKind.NON_FINITE_VALUES), [])
