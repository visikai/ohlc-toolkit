"""Check a fetched history against what its manifest says it is.

A matching SHA-256 proves the bytes on disk are the bytes that were
published. It proves nothing about whether those bytes are the history
the manifest describes, so this module asks the second question:

1. Does the manifest agree with itself -- does its row count match the
   count implied by its own first and last timestamps at the source
   cadence?
2. Does the frame hold the number of rows the manifest states, opening
   and closing where it states?
3. Is the frame a complete grid at the source cadence, with no
   duplicates, no gaps, and no rows out of order?

Only the first two are implemented here. The third is
:func:`ohlc_toolkit.source.validation.validate_source_frame` run against
the same :class:`~ohlc_toolkit.source.profile.SourceProfile` the rest of
the package reads Bitstamp minute data with, so a fetched snapshot is
held to exactly the standard a local file is, by exactly the same code.

Scope, stated plainly: only the six-column history CSV is opened and
checked this deeply. The Parquet and provenance assets in the same
release are fetched and digest-verified alongside it, and then handed
over on that alone. Neither omission is an oversight, and neither is
hidden -- a caller reading either one reads unvalidated structure -- but
they are omitted for different reasons.

The Parquet carries the same rows over the same timestamps as the CSV,
but it is not a drop-in twin: as published, only its ``timestamp`` column
is an integer, and open/high/low/close/volume are UTF-8 strings holding
the CSV's decimal text verbatim. ``BITSTAMP_BTCUSD_1M`` declares those
five columns :attr:`~ohlc_toolkit.source.profile.ColumnKind.FLOATING`, so
running the same validation over the Parquet would fail on schema before
it reached a single row. Checking it properly means either a second
profile describing the string encoding or a cast this package would then
have to justify, and either is a larger decision than this fetcher should
be making on its own.

The provenance CSV is a sparse outage table -- start, end, duration,
flag, price jump, reference -- on an entirely different schema, which no
source profile here describes at all.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, unique

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.snapshot.fetcher import SnapshotFetchResult
from ohlc_toolkit.snapshot.manifest import SnapshotManifest
from ohlc_toolkit.snapshot.release import BITSTAMP_HISTORY_CSV_ASSET
from ohlc_toolkit.source import (
    BITSTAMP_BTCUSD_1M,
    SourceProfile,
    ValidationMode,
    ValidationReport,
    read_source_csv,
    validate_source_frame,
)
from ohlc_toolkit.temporal import ConfigError, DataValidationError, bounded_echo

logger = get_logger(__name__)


@unique
class SeamKind(Enum):
    """Which statement of the manifest a mismatch contradicts.

    MANIFEST_SPAN is the only member that does not involve the frame at
    all: it compares the manifest's declared row count against the count
    implied by its own first and last timestamps at the profile cadence.
    """

    MANIFEST_SPAN = "manifest_span"
    ROW_COUNT = "row_count"
    FIRST_TIMESTAMP = "first_timestamp"
    LAST_TIMESTAMP = "last_timestamp"


@dataclass(frozen=True)
class SeamMismatch:
    """One statement the fetched history did not live up to.

    Attributes:
        kind: Which statement was contradicted.
        expected: What the manifest declared.
        observed: What was actually found.

    """

    kind: SeamKind
    expected: int
    observed: int


@dataclass(frozen=True)
class ContinuityReport:
    """The outcome of checking a history against its manifest.

    Attributes:
        rows_checked: How many rows the checked frame held.
        seam_mismatches: Every manifest statement that did not hold.
        validation: The grid report from
            :func:`~ohlc_toolkit.source.validation.validate_source_frame`,
            carried whole rather than summarized, so a caller can read
            the exact gap boundaries or offending timestamps.

    """

    rows_checked: int
    seam_mismatches: tuple[SeamMismatch, ...]
    validation: ValidationReport

    @property
    def passed(self) -> bool:
        """Report whether every statement held and the grid was complete."""
        return not self.seam_mismatches and self.validation.passed


class SnapshotContinuityError(DataValidationError):
    """A fetched history contradicts its manifest, or is not a complete grid.

    Carries the full :class:`ContinuityReport` as a typed attribute, so
    callers can inspect exactly which statement failed and which rows
    caused it instead of parsing a message.

    Attributes:
        report: The continuity report that triggered this error.

    """

    def __init__(self, message: str, report: ContinuityReport) -> None:
        """Store the message and the report that produced it.

        Args:
            message: A short, human-readable summary of the failure.
            report: The full report, including every seam mismatch and
                every grid finding.

        """
        super().__init__(message)
        self.report = report


def verify_snapshot_continuity(
    frame: pl.DataFrame,
    manifest: SnapshotManifest,
    *,
    profile: SourceProfile = BITSTAMP_BTCUSD_1M,
) -> ContinuityReport:
    """Check a history frame against its manifest, refusing any disagreement.

    Args:
        frame: The history exactly as read, never sorted or repaired.
        manifest: The manifest the history was published under.
        profile: The profile describing the cadence, phase, and columns
            the history is expected to have.

    Returns:
        The report, when every statement held.

    Raises:
        SnapshotContinuityError: On any seam mismatch or grid finding.
            The report is attached to the exception as ``.report``.

    """
    validation = validate_source_frame(frame, profile, mode=ValidationMode.REPORT)
    report = _build_report(frame, validation, manifest, profile)
    _enforce(report, manifest)
    return report


# How many asset names a refusal quotes back. The bound here is on the
# COUNT rather than on each name's length: a manifest may declare any
# number of assets and nothing caps that, so bounding each name would
# leave a crowded release echoing just as much.
_MAX_ECHOED_ASSET_NAMES = 20


def _echo_asset_names(names: Sequence[str]) -> str:
    """Render what a release holds, bounded in number and in length."""
    total = len(names)
    shown = ", ".join(bounded_echo(name) for name in names[:_MAX_ECHOED_ASSET_NAMES])
    if total <= _MAX_ECHOED_ASSET_NAMES:
        return f"{total} asset(s): [{shown}]"
    return f"{total} asset(s), the first {_MAX_ECHOED_ASSET_NAMES}: [{shown}]"


def read_snapshot_frame(
    result: SnapshotFetchResult,
    *,
    asset_name: str = BITSTAMP_HISTORY_CSV_ASSET,
    profile: SourceProfile = BITSTAMP_BTCUSD_1M,
) -> pl.DataFrame:
    """Read a fetched history asset and verify its continuity before returning it.

    This is the only path from a fetch result to a frame, so there is no
    way to obtain a history from this subpackage without both its bytes
    and its continuity having been checked.

    Args:
        result: A successful fetch, whose assets are already digest-verified.
        asset_name: Which fetched asset holds the history CSV. Defaults
            to the published Bitstamp full-history asset.
        profile: The profile the CSV is read and validated against.

    Returns:
        The history frame, exactly as it is on disk.

    Raises:
        ConfigError: If ``asset_name`` is not among the fetched assets.
        SnapshotContinuityError: If the history contradicts its manifest
            or is not a complete grid.

    """
    asset = result.assets.get(asset_name)
    if asset is None:
        held = _echo_asset_names(sorted(result.assets))
        logger.error(
            "Asset {} is absent from the fetched release {!r}; it holds {}.",
            bounded_echo(asset_name),
            result.release.tag,
            held,
        )
        raise ConfigError(
            f"Asset {bounded_echo(asset_name)} is absent from the fetched "
            f"release {result.release.tag!r}, which holds {held}."
        )

    # One row past the declaration: a longer file then yields exactly one
    # extra row for the seam check below to refuse, without the file --
    # or an adversarially compressed one -- ever being fully resident.
    read = read_source_csv(
        asset.path,
        profile,
        mode=ValidationMode.REPORT,
        max_rows=result.manifest.row_count + 1,
    )
    report = _build_report(read.frame, read.report, result.manifest, profile)
    _enforce(report, result.manifest)
    logger.info(
        "Verified {} row(s) of snapshot {!r} from {}.",
        report.rows_checked,
        result.manifest.tag,
        asset.path,
    )
    return read.frame


def _enforce(report: ContinuityReport, manifest: SnapshotManifest) -> None:
    """Raise when a report holds anything at all, logging first."""
    if report.passed:
        return
    message = (
        f"Snapshot {manifest.tag!r} failed continuity with "
        f"{len(report.seam_mismatches)} seam mismatch(es) and "
        f"{len(report.validation.findings)} grid finding(s)."
    )
    logger.error(
        "Snapshot {!r} failed continuity: {} seam mismatch(es) {}, {} grid finding(s).",
        manifest.tag,
        len(report.seam_mismatches),
        [mismatch.kind.value for mismatch in report.seam_mismatches],
        len(report.validation.findings),
    )
    raise SnapshotContinuityError(message, report)


def _build_report(
    frame: pl.DataFrame,
    validation: ValidationReport,
    manifest: SnapshotManifest,
    profile: SourceProfile,
) -> ContinuityReport:
    """Assemble the seam mismatches alongside an already-computed grid report."""
    return ContinuityReport(
        rows_checked=frame.height,
        seam_mismatches=tuple(_check_seams(frame, manifest, profile)),
        validation=validation,
    )


def _check_seams(
    frame: pl.DataFrame, manifest: SnapshotManifest, profile: SourceProfile
) -> list[SeamMismatch]:
    """Compare the manifest's statements against itself and against the frame."""
    mismatches = _check_manifest_span(manifest, profile)

    if frame.height != manifest.row_count:
        mismatches.append(
            SeamMismatch(
                kind=SeamKind.ROW_COUNT,
                expected=manifest.row_count,
                observed=frame.height,
            )
        )

    if not _bounds_are_readable(frame, profile):
        logger.warning(
            "Skipping the first/last timestamp checks for snapshot {!r}: the "
            "timestamp column is empty, missing, or holds a null.",
            manifest.tag,
        )
        return mismatches

    timestamps = frame.get_column(profile.timestamp_column)
    first = int(timestamps[0])
    last = int(timestamps[-1])
    if first != manifest.first_timestamp:
        mismatches.append(
            SeamMismatch(
                kind=SeamKind.FIRST_TIMESTAMP,
                expected=manifest.first_timestamp,
                observed=first,
            )
        )
    if last != manifest.last_timestamp:
        mismatches.append(
            SeamMismatch(
                kind=SeamKind.LAST_TIMESTAMP,
                expected=manifest.last_timestamp,
                observed=last,
            )
        )
    return mismatches


def _check_manifest_span(
    manifest: SnapshotManifest, profile: SourceProfile
) -> list[SeamMismatch]:
    """Check the manifest's row count against the span it declares.

    A complete grid holds exactly one row per cadence step from the first
    open through the last, inclusive. When the manifest's own row count
    disagrees with that, no frame can satisfy every statement at once,
    and saying so names the broken manifest rather than blaming the data.

    A span that is not a whole number of cadence steps is a mismatch on
    its own, whatever the row count: the two declared bounds are not on
    the same grid. The reported ``observed`` count is then the floor,
    which is the closest honest thing to say about a span that does not
    divide.
    """
    cadence_seconds = profile.cadence.total_seconds
    span = manifest.last_timestamp - manifest.first_timestamp
    implied = span // cadence_seconds + 1
    if span % cadence_seconds == 0 and implied == manifest.row_count:
        return []
    return [
        SeamMismatch(
            kind=SeamKind.MANIFEST_SPAN,
            expected=manifest.row_count,
            observed=implied,
        )
    ]


def _bounds_are_readable(frame: pl.DataFrame, profile: SourceProfile) -> bool:
    """Report whether the frame's first and last timestamps can be read.

    This mirrors the guard
    :func:`ohlc_toolkit.source.validation.validate_source_frame` applies
    before its own row-level checks, and for the same reason: a null
    timestamp can hide what is really a gap, so reading one as a boundary
    would launder the corruption instead of reporting it. The grid report
    has already recorded the schema or null-value finding by the time
    this returns False.
    """
    if frame.height == 0:
        return False
    if profile.timestamp_column not in frame.columns:
        return False
    return frame.get_column(profile.timestamp_column).null_count() == 0
