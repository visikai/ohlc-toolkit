"""A polars-native CSV reader for raw source frames.

This reader never sorts, fills, drops, or de-duplicates rows: what is on
disk is what comes back, so validation (see
:mod:`ohlc_toolkit.source.validation`) sees the data exactly as the
provider published it. A reader that tidies its input first cannot report
that the input needed tidying, which is the whole point of reading it
here rather than anywhere else.
"""

import os
from dataclasses import dataclass
from typing import Literal, overload

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.source.profile import ColumnKind, SourceProfile
from ohlc_toolkit.source.validation import (
    ValidationMode,
    ValidationReport,
    validate_source_frame,
)

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
        SourceValidationError: In strict mode, when validation produces
            one or more findings.

    """
    frame = _read_raw_frame(path, profile, max_rows)

    if mode is ValidationMode.STRICT:
        validate_source_frame(frame, profile, mode=ValidationMode.STRICT)
        return frame

    report = validate_source_frame(frame, profile, mode=ValidationMode.REPORT)
    return SourceReadResult(frame=frame, report=report)


def _read_raw_frame(
    path: str | os.PathLike[str], profile: SourceProfile, max_rows: int | None
) -> pl.DataFrame:
    """Read a raw CSV with dtypes pinned by the profile's raw schema."""
    schema_overrides = {
        name: _COLUMN_KIND_DTYPES[kind] for name, kind in profile.raw_schema.items()
    }
    logger.debug("Reading source frame {!r} for profile {!r}.", path, profile.name)
    try:
        # os.fspath() normalizes any PathLike (not just pathlib.Path) to
        # the str/bytes union polars' read_csv is typed to accept.
        return pl.read_csv(
            os.fspath(path), schema_overrides=schema_overrides, n_rows=max_rows
        )
    except FileNotFoundError:
        logger.error("Source file not found: {}", path)
        raise
