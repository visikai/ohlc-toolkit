"""Strict or report-only validation of a raw source frame against a profile.

Every check here is a vectorized polars expression or a diff over an
already-materialized int64 timestamp column. None of them build a
Python-side set or range spanning the full timestamp span: that pattern
scales with the calendar span of the data rather than its row count, and
silently falls over on a source with a large gap. Everything below is
O(n) in the number of rows actually present.

Checks run in a fixed, deliberate order:

1. Schema (missing or wrong-kind declared columns) and null-value checks
   run first and unconditionally: every later check assumes the declared
   columns exist, have the declared kind, and are free of nulls.
2. If the timestamp column itself is missing, wrong-kind, or contains any
   null, every row-level check (phase, monotonicity, overlap,
   irregular-interval, gap) is skipped outright. Running modulo or diff
   arithmetic over a null timestamp would not raise -- it would silently
   produce a null comparison result that never contributes to a count,
   which is exactly how a real gap can hide behind a blank CSV field.
3. Among the row-level checks, monotonicity goes first. When it fires,
   the frame is unsorted or contains a duplicate, so overlap,
   irregular-interval, and gap -- all of which read positional diffs as
   if the frame were laid out on a rising timeline -- are skipped too:
   over unsorted data their diffs are meaningless and can misreport a
   timestamp that is actually present as a gap boundary.
"""

from dataclasses import dataclass
from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.source.profile import SourceProfile
from ohlc_toolkit.temporal import DataValidationError

logger = get_logger(__name__)

# Caps on how much a single finding echoes back, so a pathological frame
# cannot turn one finding into an unbounded log line or report.
_MAX_SAMPLE_TIMESTAMPS = 20
_MAX_SCHEMA_COLUMN_NAMES = 20
_MAX_GAP_FINDINGS = 1000
_MAX_COLUMN_NAME_CHARS = 60

# Fewer than two rows means there are no successive-row diffs to check.
_MIN_ROWS_FOR_DIFFS = 2


@unique
class ValidationMode(Enum):
    """How :func:`validate_source_frame` reacts to its own findings.

    STRICT raises :class:`SourceValidationError` when any finding is
    produced. REPORT always returns the report, however many findings it
    holds, and never raises.
    """

    STRICT = "strict"
    REPORT = "report"


@unique
class FindingKind(Enum):
    """The category of check that produced a single finding."""

    SCHEMA = "schema"
    NULL_VALUES = "null_values"
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
        count: The number of offending rows (or columns, for schema and
            null-value findings; or missing candles, for a gap finding).
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


class SourceValidationError(DataValidationError):
    """A raw source frame failed strict validation against its profile.

    Carries the full :class:`ValidationReport` as a typed attribute
    rather than as a dynamically-set one, so callers can inspect every
    finding directly, with no runtime attribute check or ``type: ignore``
    required.

    Attributes:
        report: The validation report that triggered this error.

    """

    def __init__(self, message: str, report: ValidationReport) -> None:
        """Store the message and the report that produced it.

        Args:
            message: A short, human-readable summary of the failure.
            report: The full validation report, including every finding.

        """
        super().__init__(message)
        self.report = report


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
        SourceValidationError: In strict mode, when one or more findings
            were produced. The report is attached to the exception as
            ``.report``.

    """
    findings = _run_checks(frame, profile)
    report = ValidationReport(rows_checked=frame.height, findings=tuple(findings))

    if mode is ValidationMode.STRICT and not report.passed:
        message = (
            f"Source frame {profile.name!r} failed strict validation with "
            f"{len(report.findings)} finding(s)."
        )
        logger.error(
            "Source frame {!r} failed strict validation with {} finding(s).",
            profile.name,
            len(report.findings),
        )
        raise SourceValidationError(message, report)

    return report


def _run_checks(frame: pl.DataFrame, profile: SourceProfile) -> list[Finding]:
    """Run every check that applies to ``frame``, in the fixed order documented above."""
    findings = list(_check_schema(frame, profile))
    findings.extend(_check_null_values(frame, profile))
    if not _timestamp_column_is_usable(frame, profile):
        return findings

    timestamps = frame.get_column(profile.timestamp_column).cast(pl.Int64)
    cadence_seconds = profile.cadence.total_seconds
    phase_seconds = profile.phase.total_seconds

    findings.extend(_check_phase(timestamps, cadence_seconds, phase_seconds))
    if frame.height < _MIN_ROWS_FOR_DIFFS:
        return findings

    head = timestamps.slice(0, frame.height - 1)
    tail = timestamps.slice(1)
    diffs = tail - head

    monotonic_findings = _check_monotonic(tail, diffs)
    findings.extend(monotonic_findings)
    if monotonic_findings:
        # Diffs over unsorted or duplicated timestamps are meaningless
        # for overlap, irregular-interval, and gap purposes: see the
        # module docstring.
        return findings

    findings.extend(_check_overlap(tail, diffs, cadence_seconds))
    findings.extend(_check_irregular_interval(tail, diffs, cadence_seconds))
    findings.extend(_check_gaps(head, tail, diffs, cadence_seconds))
    return findings


def _timestamp_column_is_usable(frame: pl.DataFrame, profile: SourceProfile) -> bool:
    """Report whether the timestamp column is fit for row-level checks.

    Row-level checks need to cast and diff this column; if it is missing,
    non-numeric, or contains any null, the schema or null-value check has
    already reported the problem, and row-level checks are skipped rather
    than crashing or computing misleading arithmetic over a null.
    """
    if profile.timestamp_column not in frame.columns:
        return False
    kind = profile.raw_schema[profile.timestamp_column]
    if not kind.matches(frame.schema[profile.timestamp_column]):
        return False
    return frame.get_column(profile.timestamp_column).null_count() == 0


def _bounded_column_name(name: str) -> str:
    """Truncate an echoed column name with an ellipsis when oversized."""
    if len(name) <= _MAX_COLUMN_NAME_CHARS:
        return name
    return f"{name[:_MAX_COLUMN_NAME_CHARS]}..."


def _check_schema(frame: pl.DataFrame, profile: SourceProfile) -> list[Finding]:
    """Check that every declared raw column is present with the right kind."""
    missing = [name for name in profile.raw_schema if name not in frame.columns]
    wrong_kind = [
        name
        for name, kind in profile.raw_schema.items()
        if name in frame.columns and not kind.matches(frame.schema[name])
    ]
    if not missing and not wrong_kind:
        return []

    parts = []
    if missing:
        bounded = [
            _bounded_column_name(name) for name in missing[:_MAX_SCHEMA_COLUMN_NAMES]
        ]
        parts.append(f"missing columns: {bounded}")
    if wrong_kind:
        details = [
            f"{_bounded_column_name(name)} (expected {profile.raw_schema[name].value}, "
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


def _check_null_values(frame: pl.DataFrame, profile: SourceProfile) -> list[Finding]:
    """Check every declared column present in the frame for null values.

    Null counts are read via polars' ``null_count``, an O(1)-ish
    per-column metadata read, never a full scan proportional to the
    calendar span. A null anywhere in a declared column is corruption the
    frame must fail on: for the timestamp column in particular, a null
    can hide what is really a gap or a duplicate, so letting later
    arithmetic (modulo, diff) run over it would silently mask the real
    problem instead of surfacing it.
    """
    offending = [
        (name, frame.get_column(name).null_count())
        for name in profile.raw_schema
        if name in frame.columns
    ]
    offending = [(name, count) for name, count in offending if count > 0]
    if not offending:
        return []

    total = sum(count for _, count in offending)
    details = [
        f"{_bounded_column_name(name)} ({count})"
        for name, count in offending[:_MAX_SCHEMA_COLUMN_NAMES]
    ]
    return [
        Finding(
            kind=FindingKind.NULL_VALUES,
            message=(f"{total} null value(s) across declared column(s): {details}"),
            count=total,
        )
    ]


def _bounded_sample(values: pl.Series, mask: pl.Series) -> tuple[int, ...]:
    """Return a capped sample of the values selected by a boolean mask."""
    return tuple(values.filter(mask).head(_MAX_SAMPLE_TIMESTAMPS).to_list())


def _check_phase(
    timestamps: pl.Series, cadence_seconds: int, phase_seconds: int
) -> list[Finding]:
    """Check every timestamp against the profile's DECLARED grid phase.

    The phase is a declaration on the profile, never inferred from the
    frame itself: inferring it from the first row would let a uniformly
    shifted grid — corruption relative to the declared schedule — pass as
    a self-consistent convention, and would let a corrupt first row
    condemn every healthy one.
    """
    mask = (timestamps % cadence_seconds) != phase_seconds
    count = int(mask.sum())
    if count == 0:
        return []
    return [
        Finding(
            kind=FindingKind.OFF_PHASE,
            message=(
                f"{count} timestamp(s) are not on the declared "
                f"{cadence_seconds}s-cadence grid with phase {phase_seconds}s"
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
    total_runs = int(mask.sum())
    if total_runs == 0:
        return []

    filtered_starts = head.filter(mask)
    filtered_ends = tail.filter(mask)
    if total_runs > _MAX_GAP_FINDINGS:
        logger.warning(
            "Frame has {} gap runs; reporting only the first {}.",
            total_runs,
            _MAX_GAP_FINDINGS,
        )
        filtered_starts = filtered_starts.head(_MAX_GAP_FINDINGS)
        filtered_ends = filtered_ends.head(_MAX_GAP_FINDINGS)

    starts = filtered_starts.to_list()
    ends = filtered_ends.to_list()

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
