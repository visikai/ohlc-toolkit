"""Tests for the polars-native source CSV reader."""

import gzip
import tempfile
import unittest
from pathlib import Path

import polars as pl
import pytest

from ohlc_toolkit.source import BITSTAMP_BTCUSD_1M
from ohlc_toolkit.source import reader as reader_module
from ohlc_toolkit.source.profile import Availability, ColumnKind, SourceProfile
from ohlc_toolkit.source.reader import SourceReadResult, read_source_csv
from ohlc_toolkit.source.validation import FindingKind, ValidationMode
from ohlc_toolkit.temporal import MAX_ECHO_CHARS, DataValidationError

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


# Long enough that an echoed path would swamp a line, short enough per
# component and in total that the OS answers "not found" rather than
# "name too long", so the not-found branch is the one exercised.
_LONG_PATH_COMPONENTS = 14
_LONG_COMPONENT_CHARS = 250
# One bounded echo plus prose for the not-found line, two for the debug line.
_MAX_READER_LINE_CHARS = 6 * MAX_ECHO_CHARS
# The debug line before the open, and the not-found line after it.
_EXPECTED_READER_LINES = 2


def test_a_missing_file_with_a_long_path_is_logged_bounded(tmp_path: Path) -> None:
    """Both the debug line and the not-found line bound the path they echo."""
    long_path = tmp_path.joinpath(
        *(["p" * _LONG_COMPONENT_CHARS] * _LONG_PATH_COMPONENTS)
    )
    logged: list[str] = []
    sink_id = reader_module.logger.add(logged.append, level="DEBUG", format="{message}")
    try:
        with pytest.raises(FileNotFoundError):
            read_source_csv(long_path, BITSTAMP_BTCUSD_1M, mode=ValidationMode.REPORT)
    finally:
        reader_module.logger.remove(sink_id)

    assert len(logged) >= _EXPECTED_READER_LINES, logged
    for line in logged:
        assert len(line) < _MAX_READER_LINE_CHARS


# The three shapes a file can take when it holds no rows, and the one that
# used to abort the interpreter rather than refuse.
_EMPTY_PAYLOAD = b""
_ROW_CAP = 6
# A file that says it is a gzip and then stops, and how much of a real
# archive to keep so that opening it succeeds and reading it does not.
_GZIP_MAGIC_ONLY = b"\x1f\x8b"
_TRUNCATED_BYTES = 12


def _write_bytes(directory: Path, name: str, payload: bytes, *, gzipped: bool) -> Path:
    """Write exactly these bytes, gzipped or not, and return the path."""
    path = directory / (f"{name}.csv.gz" if gzipped else f"{name}.csv")
    path.write_bytes(gzip.compress(payload, mtime=0) if gzipped else payload)
    return path


@pytest.mark.parametrize("gzipped", [True, False], ids=["gzipped", "plain"])
@pytest.mark.parametrize("max_rows", [None, _ROW_CAP], ids=["uncapped", "capped"])
def test_a_file_holding_no_data_is_refused_in_one_catchable_class(
    tmp_path: Path, gzipped: bool, max_rows: int | None
) -> None:
    """A row cap must change what is read, never what is raised.

    Measured at polars 1.44.1: an empty gzip archive read with any
    `n_rows`, zero included, aborted with a `pyo3_runtime.PanicException`.
    That is a `BaseException` and not an `Exception`, so no caller's
    `except PolarsError` -- and not even `except Exception` -- could catch
    it, and it escaped every refusal this package documents. The same file
    with no cap raised `NoDataError`, which is catchable. All four
    combinations now raise the one class.
    """
    path = _write_bytes(tmp_path, "empty", _EMPTY_PAYLOAD, gzipped=gzipped)

    with pytest.raises(pl.exceptions.NoDataError):
        read_source_csv(path, _PROFILE, mode=ValidationMode.REPORT, max_rows=max_rows)


def test_the_refusal_for_an_empty_file_is_an_ordinary_exception(tmp_path: Path) -> None:
    """The point of the guard, stated as the assertion a panic would fail."""
    path = _write_bytes(tmp_path, "empty", _EMPTY_PAYLOAD, gzipped=True)

    try:
        read_source_csv(path, _PROFILE, mode=ValidationMode.REPORT, max_rows=_ROW_CAP)
    except Exception as error:
        assert isinstance(error, pl.exceptions.PolarsError)
    else:
        pytest.fail("an empty file must be refused, not read")


@pytest.mark.parametrize("gzipped", [True, False], ids=["gzipped", "plain"])
def test_a_header_without_rows_is_not_refused_as_empty(
    tmp_path: Path, gzipped: bool
) -> None:
    """A file naming its columns and holding no rows is a schema, not a void.

    The guard refuses a file with nothing in it at all; this one has
    something in it, and reads back as the empty frame it is.
    """
    path = _write_bytes(tmp_path, "header", f"{_HEADER}\n".encode(), gzipped=gzipped)

    result = read_source_csv(
        path, _PROFILE, mode=ValidationMode.REPORT, max_rows=_ROW_CAP
    )

    assert result.frame.height == 0
    assert result.frame.columns == _HEADER.split(",")


def test_a_gzip_archive_is_recognised_by_its_bytes_not_its_name(
    tmp_path: Path,
) -> None:
    """Compression is decided the way polars decides it: from the file itself.

    A suffix check would pass every other test in this file, because each
    of them writes gzip bytes to a `.gz` name. It would also let an empty
    archive under a `.csv` name reach the capped read and abort the
    interpreter again, which is the whole defect. This is the case that
    tells the two rules apart.
    """
    path = tmp_path / "looks_plain.csv"
    path.write_bytes(gzip.compress(_EMPTY_PAYLOAD, mtime=0))

    with pytest.raises(pl.exceptions.NoDataError):
        read_source_csv(path, _PROFILE, mode=ValidationMode.REPORT, max_rows=_ROW_CAP)


def test_a_missing_file_is_logged_whether_or_not_a_cap_is_given(
    tmp_path: Path,
) -> None:
    """The guard runs before the read, and must not swallow the read's own refusal.

    A probe that opened the file itself could raise first and leave the
    reader's log line unrun, so a capped read of a missing file would
    refuse silently while an uncapped one announced itself.
    """
    missing = tmp_path / "absent.csv"

    for max_rows in (None, _ROW_CAP):
        logged: list[str] = []
        sink_id = reader_module.logger.add(
            logged.append, level="ERROR", format="{message}"
        )
        try:
            with pytest.raises(FileNotFoundError):
                read_source_csv(
                    missing, _PROFILE, mode=ValidationMode.REPORT, max_rows=max_rows
                )
        finally:
            reader_module.logger.remove(sink_id)
        assert logged, f"a missing file must be logged, max_rows={max_rows}"
        assert logged[-1].startswith("Source file not found")


@pytest.mark.parametrize(
    ("name", "payload", "expected"),
    [
        pytest.param("magic.csv.gz", _GZIP_MAGIC_ONLY, 0, id="magic_bytes_only"),
        pytest.param("truncated.csv.gz", None, None, id="truncated_archive"),
    ],
)
def test_the_guard_leaves_an_unreadable_archive_to_the_read(
    tmp_path: Path, name: str, payload: bytes | None, expected: int | None
) -> None:
    """A probe that cannot tell says nothing, so the read reports in its own words.

    Deciding on behalf of a read it merely stumbled into is how a cap
    would start changing what is raised. Both shapes below behave exactly
    as they do with no cap at all.
    """
    if payload is None:
        payload = gzip.compress(f"{_HEADER}\n".encode(), mtime=0)[:_TRUNCATED_BYTES]
    path = tmp_path / name
    path.write_bytes(payload)

    def read(max_rows: int | None) -> object:
        try:
            return read_source_csv(
                path, _PROFILE, mode=ValidationMode.REPORT, max_rows=max_rows
            ).frame.height
        except Exception as error:
            return type(error)

    assert read(_ROW_CAP) == read(None)
    if expected is not None:
        assert read(_ROW_CAP) == expected


@pytest.mark.parametrize("gzipped", [True, False], ids=["gzipped", "plain"])
def test_a_capped_read_of_a_real_file_is_untouched_by_the_guard(
    tmp_path: Path, gzipped: bool
) -> None:
    """The guard reads one byte and gets out of the way."""
    rows = _clean_rows(start=0, length=_ROW_CAP + 4)
    path = _write_csv(tmp_path, "real", rows, gzipped=gzipped)

    result = read_source_csv(
        path, _PROFILE, mode=ValidationMode.REPORT, max_rows=_ROW_CAP
    )

    assert result.frame.height == _ROW_CAP


@pytest.mark.parametrize("gzipped", [True, False], ids=["gzipped", "plain"])
def test_the_empty_file_refusal_comes_from_here_and_names_its_path_bounded(
    tmp_path: Path, gzipped: bool
) -> None:
    """An empty file is refused here, whatever polars would do with it.

    Only the gzip shape panics today, so for a plain file polars raises the
    same class unaided and the exception alone cannot say who refused. The
    log line can: it exists only if the guard ran. The guard covers both
    shapes because a reader cannot know which shapes a given polars version
    aborts on, and the cost is one byte.
    """
    deep = tmp_path.joinpath(*["p" * _LONG_COMPONENT_CHARS] * _LONG_PATH_COMPONENTS)
    deep.mkdir(parents=True)
    path = _write_bytes(deep, "empty", _EMPTY_PAYLOAD, gzipped=gzipped)

    logged: list[str] = []
    sink_id = reader_module.logger.add(logged.append, level="ERROR", format="{message}")
    try:
        with pytest.raises(pl.exceptions.NoDataError) as raised:
            read_source_csv(
                path, _PROFILE, mode=ValidationMode.REPORT, max_rows=_ROW_CAP
            )
    finally:
        reader_module.logger.remove(sink_id)

    assert len(str(raised.value)) < _MAX_READER_LINE_CHARS
    assert logged, "this package refuses the file itself, and says so before raising"
    assert logged[-1].startswith("Source file holds no data")
    assert len(logged[-1]) < _MAX_READER_LINE_CHARS
