"""Strict or report-only validation of a raw source frame against a profile.

Every check here is a vectorized polars expression or a diff over an
already-materialized int64 timestamp column. None of them build a
Python-side set or range spanning the full timestamp span: that pattern
scales with the calendar span of the data rather than its row count, and
silently falls over on a source with a large gap. Everything below is
O(n) in the number of rows actually present.
"""

from dataclasses import dataclass
from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.source.profile import SourceProfile, column_kind_matches
from ohlc_toolkit.temporal import DataValidationError, coerce_duration

logger = get_logger(__name__)

# Caps on how much a single finding echoes back, so a pathological frame
# cannot turn one finding into an unbounded log line or report.
_MAX_SAMPLE_TIMESTAMPS = 20
_MAX_SCHEMA_COLUMN_NAMES = 20
_MAX_GAP_FINDINGS = 1000

# Fewer than two rows means there are no successive-row diffs to check.
_MIN_ROWS_FOR_DIFFS = 2


@unique
class ValidationMode(Enum):
    """How :func:`validate_source_frame` reacts to its own findings.

    STRICT raises :class:`~ohlc_toolkit.temporal.DataValidationError` when
    any finding is produced. REPORT always returns the report, however
    many findings it holds, and never raises.
    """

    STRICT = "strict"
    REPORT = "report"


@unique
class FindingKind(Enum):
    """The category of check that produced a single finding."""

    SCHEMA = "schema"
    NON_INCREASING_TIMESTAMPS = "non_increasing_timestamps"
    OVERLAPPING_INTERVALS = "overlapping_intervals"
    OFF_PHASE = "off_phase"
    IRREGULAR_INTERVAL = "irregular_interval"
    GAP = "gap"


@dataclass(frozen=True)
class Finding:
    """A single, bounded validation finding.

    Attributes:
        kind: The category of check that produced this finding.
        message: A short, human-readable, already-bounded description.
        count: The number of offending rows (or columns, for schema
            findings; or missing candles, for a gap finding).
        sample_timestamps: A capped sample of relevant Unix-second
            timestamps. For a gap finding, this is always the exact
            two-element ``(expected_start, next_open)`` boundary; for
            every other kind, it is a bounded sample of offending
            timestamps, empty when the finding is not row-scoped.

    """

    kind: FindingKind
    message: str
    count: int
    sample_timestamps: tuple[int, ...] = ()


@dataclass(frozen=True)
class ValidationReport:
    """The outcome of validating a raw source frame against a profile.

    Attributes:
        rows_checked: How many rows were present in the validated frame.
        findings: Every finding raised by the checks that ran.

    """

    rows_checked: int
    findings: tuple[Finding, ...]

    @property
    def passed(self) -> bool:
        """Report whether validation found zero issues."""
        return len(self.findings) == 0


def validate_source_frame(
    frame: pl.DataFrame, profile: SourceProfile, *, mode: ValidationMode
) -> ValidationReport:
    """Validate a raw source frame against a profile's declared shape.

    Never mutates ``frame`` and never repairs it: every finding is
    reported, never silently fixed.

    Args:
        frame: The raw source frame to validate.
        profile: The profile describing the expected columns, cadence,
            and timestamp column.
        mode: ``ValidationMode.STRICT`` raises when any finding is
            produced; ``ValidationMode.REPORT`` always returns the
            report.

    Returns:
        The validation report, in both modes.

    Raises:
        DataValidationError: In strict mode, when one or more findings
            were produced. The report is attached to the exception as
            ``.report``.

    """
    findings = _run_checks(frame, profile)
    report = ValidationReport(rows_checked=frame.height, findings=tuple(findings))

    if mode is ValidationMode.STRICT and not report.passed:
        logger.error(
            "Source frame {!r} failed strict validation with {} finding(s).",
            profile.name,
            len(report.findings),
        )
        error = DataValidationError(
            f"Source frame {profile.name!r} failed strict validation with "
            f"{len(report.findings)} finding(s)."
        )
        error.report = report  # type: ignore[attr-defined]
        raise error

    return report


def _run_checks(frame: pl.DataFrame, profile: SourceProfile) -> list[Finding]:
    """Run every check that applies to ``frame``, in a fixed order."""
    findings = list(_check_schema(frame, profile))
    if not _timestamp_column_is_usable(frame, profile):
        return findings

    timestamps = frame.get_column(profile.timestamp_column).cast(pl.Int64)
    cadence_seconds = coerce_duration(profile.cadence).total_seconds

    findings.extend(_check_phase(timestamps, cadence_seconds))
    if frame.height < _MIN_ROWS_FOR_DIFFS:
        return findings

    head = timestamps.slice(0, frame.height - 1)
    tail = timestamps.slice(1)
    diffs = tail - head

    findings.extend(_check_monotonic(tail, diffs))
    findings.extend(_check_overlap(tail, diffs, cadence_seconds))
    findings.extend(_check_irregular_interval(tail, diffs, cadence_seconds))
    findings.extend(_check_gaps(head, tail, diffs, cadence_seconds))
    return findings


def _timestamp_column_is_usable(frame: pl.DataFrame, profile: SourceProfile) -> bool:
    """Report whether the timestamp column exists with a numeric dtype.

    Row-level checks need to cast and diff this column; if it is missing
    or non-numeric, the schema check already reports the problem, and the
    row-level checks are skipped rather than crashing.
    """
    if profile.timestamp_column not in frame.columns:
        return False
    kind = profile.raw_schema[profile.timestamp_column]
    return column_kind_matches(kind, frame.schema[profile.timestamp_column])


def _check_schema(frame: pl.DataFrame, profile: SourceProfile) -> list[Finding]:
    """Check that every declared raw column is present with the right kind."""
    missing = [name for name in profile.raw_schema if name not in frame.columns]
    wrong_kind = [
        name
        for name, kind in profile.raw_schema.items()
        if name in frame.columns and not column_kind_matches(kind, frame.schema[name])
    ]
    if not missing and not wrong_kind:
        return []

    parts = []
    if missing:
        parts.append(f"missing columns: {missing[:_MAX_SCHEMA_COLUMN_NAMES]}")
    if wrong_kind:
        details = [
            f"{name} (expected {profile.raw_schema[name].value}, "
            f"got {frame.schema[name]})"
            for name in wrong_kind[:_MAX_SCHEMA_COLUMN_NAMES]
        ]
        parts.append(f"wrong-kind columns: {details}")

    return [
        Finding(
            kind=FindingKind.SCHEMA,
            message="; ".join(parts),
            count=len(missing) + len(wrong_kind),
        )
    ]


def _bounded_sample(values: pl.Series, mask: pl.Series) -> tuple[int, ...]:
    """Return a capped sample of the values selected by a boolean mask."""
    return tuple(values.filter(mask).head(_MAX_SAMPLE_TIMESTAMPS).to_list())


def _check_phase(timestamps: pl.Series, cadence_seconds: int) -> list[Finding]:
    """Check that every timestamp shares the frame's own cadence-grid phase.

    The grid's phase is established by the first row: every row on a
    complete, evenly spaced grid necessarily shares the same residue
    modulo the cadence, whatever that residue happens to be. A lone row
    trivially defines its own phase and cannot fail this check.
    """
    if timestamps.len() == 0:
        return []
    expected_phase = timestamps[0] % cadence_seconds
    mask = (timestamps % cadence_seconds) != expected_phase
    count = int(mask.sum())
    if count == 0:
        return []
    return [
        Finding(
            kind=FindingKind.OFF_PHASE,
            message=(
                f"{count} timestamp(s) do not share the grid's "
                f"{cadence_seconds}s-cadence phase"
            ),
            count=count,
            sample_timestamps=_bounded_sample(timestamps, mask),
        )
    ]


def _check_monotonic(tail: pl.Series, diffs: pl.Series) -> list[Finding]:
    """Check that every timestamp strictly increases over the previous row."""
    mask = diffs <= 0
    count = int(mask.sum())
    if count == 0:
        return []
    return [
        Finding(
            kind=FindingKind.NON_INCREASING_TIMESTAMPS,
            message=(
                f"{count} timestamp(s) do not strictly increase over the previous row"
            ),
            count=count,
            sample_timestamps=_bounded_sample(tail, mask),
        )
    ]


def _check_overlap(
    tail: pl.Series, diffs: pl.Series, cadence_seconds: int
) -> list[Finding]:
    """Check for successive opens closer together than a full cadence step."""
    mask = (diffs > 0) & (diffs < cadence_seconds)
    count = int(mask.sum())
    if count == 0:
        return []
    return [
        Finding(
            kind=FindingKind.OVERLAPPING_INTERVALS,
            message=(
                f"{count} row(s) open less than {cadence_seconds}s after the "
                "previous row's open"
            ),
            count=count,
            sample_timestamps=_bounded_sample(tail, mask),
        )
    ]


def _check_irregular_interval(
    tail: pl.Series, diffs: pl.Series, cadence_seconds: int
) -> list[Finding]:
    """Check for successive diffs that are not an exact multiple of cadence."""
    mask = (diffs > cadence_seconds) & (diffs % cadence_seconds != 0)
    count = int(mask.sum())
    if count == 0:
        return []
    return [
        Finding(
            kind=FindingKind.IRREGULAR_INTERVAL,
            message=(
                f"{count} row(s) follow the previous row by a gap that is not "
                f"an exact multiple of {cadence_seconds}s"
            ),
            count=count,
            sample_timestamps=_bounded_sample(tail, mask),
        )
    ]


def _check_gaps(
    head: pl.Series, tail: pl.Series, diffs: pl.Series, cadence_seconds: int
) -> list[Finding]:
    """Report each run of missing candles as a half-open interval finding."""
    mask = (diffs > cadence_seconds) & (diffs % cadence_seconds == 0)
    if int(mask.sum()) == 0:
        return []

    starts = head.filter(mask).to_list()
    ends = tail.filter(mask).to_list()
    if len(starts) > _MAX_GAP_FINDINGS:
        logger.warning(
            "Frame has {} gap runs; reporting only the first {}.",
            len(starts),
            _MAX_GAP_FINDINGS,
        )
        starts = starts[:_MAX_GAP_FINDINGS]
        ends = ends[:_MAX_GAP_FINDINGS]

    findings = []
    for previous_open, next_open in zip(starts, ends, strict=True):
        expected_start = previous_open + cadence_seconds
        missing_count = (next_open - expected_start) // cadence_seconds
        findings.append(
            Finding(
                kind=FindingKind.GAP,
                message=(
                    f"{missing_count} missing candle(s) expected in "
                    f"[{expected_start}, {next_open})"
                ),
                count=missing_count,
                sample_timestamps=(expected_start, next_open),
            )
        )
    return findings
