"""Tests for the polars-native source CSV reader."""

import gzip
import tempfile
import unittest
from pathlib import Path

import polars as pl

from ohlc_toolkit.source.profile import Availability, ColumnKind, SourceProfile
from ohlc_toolkit.source.reader import SourceReadResult, read_source_csv
from ohlc_toolkit.source.validation import FindingKind, ValidationMode
from ohlc_toolkit.temporal import DataValidationError

_PROFILE = SourceProfile.create(
    name="reader-test-1m",
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

_HEADER = "timestamp,open,high,low,close,volume"


def _write_csv(directory: Path, name: str, rows: list[str], *, gzipped: bool) -> Path:
    """Write a header plus data rows to a plain or gzip CSV file."""
    path = directory / name
    text = "\n".join([_HEADER, *rows]) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as handle:
            handle.write(text)
    else:
        path.write_text(text)
    return path


def _clean_rows(*, start: int, length: int) -> list[str]:
    """Build clean, on-grid, strictly increasing CSV data rows."""
    return [
        f"{start + i * 60},{100 + i}.0,{100 + i}.0,{100 + i}.0,{100 + i}.0,1.0"
        for i in range(length)
    ]


class TestReadingCleanFrames(unittest.TestCase):
    """Test cases for reading a clean, complete grid in both modes."""

    def setUp(self):
        """Create a temporary directory for CSV fixtures."""
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.directory = Path(self._tempdir.name)
        self.rows = _clean_rows(start=0, length=5)

    def test_strict_mode_returns_the_frame(self):
        """Strict mode returns the parsed frame directly, unmodified."""
        path = _write_csv(self.directory, "clean.csv", self.rows, gzipped=False)
        result = read_source_csv(str(path), _PROFILE, mode=ValidationMode.STRICT)
        self.assertIsInstance(result, pl.DataFrame)
        assert isinstance(result, pl.DataFrame)  # narrow for the type checker
        self.assertEqual(
            result.get_column("timestamp").to_list(), [0, 60, 120, 180, 240]
        )

    def test_report_mode_returns_frame_and_zero_findings(self):
        """Report mode returns a SourceReadResult with an empty findings tuple."""
        path = _write_csv(self.directory, "clean.csv", self.rows, gzipped=False)
        result = read_source_csv(str(path), _PROFILE, mode=ValidationMode.REPORT)
        self.assertIsInstance(result, SourceReadResult)
        assert isinstance(result, SourceReadResult)
        self.assertTrue(result.report.passed)
        self.assertEqual(result.frame.height, 5)

    def test_gzip_and_plain_csv_read_identically(self):
        """A gzipped CSV and its plain-text twin parse to the same frame."""
        plain = _write_csv(self.directory, "plain.csv", self.rows, gzipped=False)
        gzipped = _write_csv(self.directory, "gzipped.csv.gz", self.rows, gzipped=True)

        plain_frame = read_source_csv(str(plain), _PROFILE, mode=ValidationMode.STRICT)
        gzipped_frame = read_source_csv(
            str(gzipped), _PROFILE, mode=ValidationMode.STRICT
        )

        assert isinstance(plain_frame, pl.DataFrame)
        assert isinstance(gzipped_frame, pl.DataFrame)
        self.assertTrue(plain_frame.equals(gzipped_frame))

    def test_explicit_schema_is_applied(self):
        """The reader applies the profile's declared dtypes, not inference."""
        path = _write_csv(self.directory, "clean.csv", self.rows, gzipped=False)
        result = read_source_csv(str(path), _PROFILE, mode=ValidationMode.STRICT)
        assert isinstance(result, pl.DataFrame)
        self.assertEqual(result.schema["timestamp"], pl.Int64)
        self.assertEqual(result.schema["open"], pl.Float64)

    def test_accepts_a_pathlike_object_directly(self):
        """The reader accepts a PathLike, not just a str, for ``path``."""
        path = _write_csv(self.directory, "clean.csv", self.rows, gzipped=False)
        result = read_source_csv(path, _PROFILE, mode=ValidationMode.STRICT)
        assert isinstance(result, pl.DataFrame)
        self.assertEqual(result.height, 5)


class TestReadingCorruptedFrames(unittest.TestCase):
    """Test cases for reading a frame that fails validation."""

    def setUp(self):
        """Create a temporary directory for CSV fixtures."""
        self._tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self._tempdir.cleanup)
        self.directory = Path(self._tempdir.name)

    def test_strict_mode_raises_data_validation_error(self):
        """A gap in the source data raises in strict mode."""
        rows = _clean_rows(start=0, length=5)
        del rows[2]  # drop the row for timestamp 120
        path = _write_csv(self.directory, "gap.csv", rows, gzipped=False)

        with self.assertRaises(DataValidationError):
            read_source_csv(str(path), _PROFILE, mode=ValidationMode.STRICT)

    def test_report_mode_returns_findings_without_raising(self):
        """A gap in the source data is reported, not raised, in report mode."""
        rows = _clean_rows(start=0, length=5)
        del rows[2]
        path = _write_csv(self.directory, "gap.csv", rows, gzipped=False)

        result = read_source_csv(str(path), _PROFILE, mode=ValidationMode.REPORT)

        assert isinstance(result, SourceReadResult)
        self.assertFalse(result.report.passed)
        gap_findings = [f for f in result.report.findings if f.kind is FindingKind.GAP]
        self.assertEqual(len(gap_findings), 1)

    def test_unsorted_input_row_order_is_preserved(self):
        """The reader never sorts: row order in the frame matches the file."""
        rows = _clean_rows(start=0, length=5)
        rows[0], rows[1] = rows[1], rows[0]  # swap the first two data rows
        path = _write_csv(self.directory, "unsorted.csv", rows, gzipped=False)

        result = read_source_csv(str(path), _PROFILE, mode=ValidationMode.REPORT)

        assert isinstance(result, SourceReadResult)
        self.assertEqual(
            result.frame.get_column("timestamp").to_list(), [60, 0, 120, 180, 240]
        )
        non_increasing = [
            f
            for f in result.report.findings
            if f.kind is FindingKind.NON_INCREASING_TIMESTAMPS
        ]
        self.assertEqual(len(non_increasing), 1)
        # The underlying grid is complete: unsorted rows must not also be
        # misread as a gap around a timestamp that is actually present.
        gap_findings = [f for f in result.report.findings if f.kind is FindingKind.GAP]
        self.assertEqual(gap_findings, [])

    def test_empty_timestamp_field_raises_in_strict_mode(self):
        """A blank timestamp field parses as null and must not validate clean."""
        rows = _clean_rows(start=0, length=5)
        columns = rows[2].split(",")
        columns[0] = ""  # blank out the third row's timestamp field
        rows[2] = ",".join(columns)
        path = _write_csv(self.directory, "null-timestamp.csv", rows, gzipped=False)

        with self.assertRaises(DataValidationError):
            read_source_csv(str(path), _PROFILE, mode=ValidationMode.STRICT)

    def test_empty_timestamp_field_is_reported_not_silently_clean(self):
        """Report mode surfaces the null instead of returning a passing report."""
        rows = _clean_rows(start=0, length=5)
        columns = rows[2].split(",")
        columns[0] = ""  # blank out the third row's timestamp field
        rows[2] = ",".join(columns)
        path = _write_csv(self.directory, "null-timestamp.csv", rows, gzipped=False)

        result = read_source_csv(str(path), _PROFILE, mode=ValidationMode.REPORT)

        assert isinstance(result, SourceReadResult)
        self.assertFalse(result.report.passed)
        null_findings = [
            f for f in result.report.findings if f.kind is FindingKind.NULL_VALUES
        ]
        self.assertEqual(len(null_findings), 1)


class TestReaderPropagatesMissingFile(unittest.TestCase):
    """Test cases for reading a path that does not exist."""

    def test_missing_file_raises_file_not_found_error(self):
        """A nonexistent path raises FileNotFoundError, not a validation error."""
        with self.assertRaises(FileNotFoundError):
            read_source_csv(
                "/nonexistent/does-not-exist.csv", _PROFILE, mode=ValidationMode.STRICT
            )


if __name__ == "__main__":
    unittest.main()
