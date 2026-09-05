"""A polars-native CSV reader for raw source frames.

This reader never sorts, fills, drops, or de-duplicates rows: what is on
disk is what comes back, so validation (see
:mod:`ohlc_toolkit.source.validation`) sees the data exactly as the
provider published it. A reader that tidies its input first cannot report
that the input needed tidying, which is the whole point of reading it
here rather than anywhere else.
"""

import gzip
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, overload

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.source.profile import ColumnKind, SourceProfile
from ohlc_toolkit.source.validation import (
    ValidationMode,
    ValidationReport,
    validate_source_frame,
)
from ohlc_toolkit.temporal import bounded_echo

logger = get_logger(__name__)

# The only two numeric kinds a profile can declare; each maps to one
# concrete polars dtype so the CSV parser never falls back to inference
# for a declared column.
_COLUMN_KIND_DTYPES: dict[ColumnKind, type[pl.DataType]] = {
    ColumnKind.INTEGER: pl.Int64,
    ColumnKind.FLOATING: pl.Float64,
}


@dataclass(frozen=True)
class SourceReadResult:
    """A source frame paired with its report-mode validation report.

    Returned by :func:`read_source_csv` when reading in
    :attr:`~ohlc_toolkit.source.validation.ValidationMode.REPORT` mode, so
    the frame and its (possibly failing) report travel together instead
    of as a bare, order-ambiguous tuple.

    Attributes:
        frame: The raw frame exactly as read from disk.
        report: The validation report for ``frame``.

    """

    frame: pl.DataFrame
    report: ValidationReport


@overload
def read_source_csv(
    path: str | os.PathLike[str],
    profile: SourceProfile,
    *,
    mode: Literal[ValidationMode.STRICT],
    max_rows: int | None = None,
) -> pl.DataFrame: ...


@overload
def read_source_csv(
    path: str | os.PathLike[str],
    profile: SourceProfile,
    *,
    mode: Literal[ValidationMode.REPORT],
    max_rows: int | None = None,
) -> SourceReadResult: ...


def read_source_csv(
    path: str | os.PathLike[str],
    profile: SourceProfile,
    *,
    mode: ValidationMode,
    max_rows: int | None = None,
) -> pl.DataFrame | SourceReadResult:
    """Read a raw source CSV (plain or gzipped) and validate it.

    The frame is read with an explicit schema derived from
    ``profile.raw_schema``, so a provider's dtype is pinned by
    declaration rather than inferred from a sample of rows. The file is
    never sorted, filled, or de-duplicated.

    Args:
        path: Path to a plain or gzip-compressed CSV file. Polars detects
            gzip compression from the file itself; no flag is needed.
        profile: The profile describing the expected columns, cadence,
            and timestamp column.
        mode: ``ValidationMode.STRICT`` returns the frame on success or
            raises on any finding; ``ValidationMode.REPORT`` always
            returns both the frame and its report.
        max_rows: Upper bound on rows read from the file, or ``None``
            for no bound. For a caller who already knows how many rows
            the file MUST contain: reading one row past that number is
            enough to prove the file too long, without a longer -- or
            adversarially compressed -- file ever being fully resident.

    Returns:
        The raw frame in strict mode; a :class:`SourceReadResult` in
        report mode.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        NoDataError: If the file reads as empty, capped or not.
        SourceValidationError: In strict mode, when validation produces
            one or more findings.

    """
    frame = _read_raw_frame(path, profile, max_rows)

    if mode is ValidationMode.STRICT:
        validate_source_frame(frame, profile, mode=ValidationMode.STRICT)
        return frame

    report = validate_source_frame(frame, profile, mode=ValidationMode.REPORT)
    return SourceReadResult(frame=frame, report=report)


# The first two bytes of a gzip member, which is how the file itself says
# it is compressed -- the same way the reader lets polars decide, rather
# than by trusting a suffix.
_GZIP_MAGIC = b"\x1f\x8b"


def _holds_no_data(path: str | os.PathLike[str]) -> bool:
    """Report whether a REGULAR file decompresses to nothing at all.

    Only regular files are looked at, and the reason is not tidiness. This
    opens the path, reads from it and closes it, and the read that follows
    opens it again -- which is free on a file and destructive on anything
    else. Asking for two bytes does not read two bytes: the buffered
    reader asks the operating system for 128 KiB, so on a FIFO, a stream
    or a character device the probe consumes the data and the real read
    finds an empty input, or blocks waiting for a writer that has already
    gone. Those inputs go straight to the read, exactly as they did before
    this guard existed.

    On a regular file the cost is one buffer fill however large the file
    is, so it does not scale with a nine-hundred-megabyte history.

    Says NO when it cannot tell. A missing path, a permission error, a
    truncated archive: every one of those is the read's to report, in the
    read's own class and words, at the point where this package already
    logs and re-raises. A probe deciding the class of a failure it merely
    stumbled into is how a cap would start changing what is raised, which
    is the very thing this guard exists to stop.
    """
    resolved = Path(path)
    try:
        if not resolved.is_file():
            return False
        with resolved.open("rb") as handle:
            head = handle.read(len(_GZIP_MAGIC))
            if head != _GZIP_MAGIC:
                return not head
        with gzip.open(resolved, "rb") as archive:
            return not archive.read(1)
    except (OSError, EOFError):
        return False


def _require_data(path: str | os.PathLike[str]) -> None:
    """Refuse a file with no data before a capped read can panic on it.

    polars raises ``NoDataError`` for an empty file, which a caller can
    catch -- but not when a row cap is in play. Measured at polars 1.44.1:
    an empty gzip archive read with any ``n_rows``, zero included, aborts
    with a ``pyo3_runtime.PanicException``. That is a ``BaseException``
    and not an ``Exception``, so no caller's ``except PolarsError``, and
    not even ``except Exception``, can catch it: it escapes every refusal
    this package documents. This reader is reached with a cap from the
    snapshot path, where the file arrives off the internet, so the case is
    not hypothetical.

    The guard runs on EVERY read, not only capped ones. Only the capped
    shape panics, so a capped-only guard would fix the abort -- and would
    leave an empty file raising one class with a cap and another without
    one, which is the same defect in a quieter form. One buffer fill per
    read of a regular file is a small price for a promise with no
    exceptions.

    Raises:
        NoDataError: If the file holds no data. The class is polars' own
            and is exactly what the uncapped read raises for the same
            file, so a cap changes what is read and never what is raised.

    """
    if not _holds_no_data(path):
        return
    logger.error("Source file holds no data: {}", bounded_echo(path))
    raise pl.exceptions.NoDataError(f"Source file {bounded_echo(path)} holds no data.")


def _read_raw_frame(
    path: str | os.PathLike[str], profile: SourceProfile, max_rows: int | None
) -> pl.DataFrame:
    """Read a raw CSV with dtypes pinned by the profile's raw schema."""
    schema_overrides = {
        name: _COLUMN_KIND_DTYPES[kind] for name, kind in profile.raw_schema.items()
    }
    _require_data(path)
    logger.debug(
        "Reading source frame {} for profile {}.",
        bounded_echo(path),
        bounded_echo(profile.name),
    )
    try:
        # os.fspath() normalizes any PathLike (not just pathlib.Path) to
        # the str/bytes union polars' read_csv is typed to accept.
        return pl.read_csv(
            os.fspath(path), schema_overrides=schema_overrides, n_rows=max_rows
        )
    except FileNotFoundError:
        logger.error("Source file not found: {}", bounded_echo(path))
        raise
