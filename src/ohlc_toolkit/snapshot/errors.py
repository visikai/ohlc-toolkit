"""Exceptions raised when a published snapshot cannot be trusted.

Both exceptions here subclass
:class:`ohlc_toolkit.temporal.DataValidationError`: each describes
published data that failed a check, not a caller mistake. Caller-shaped
problems -- an empty tag, a destination that is not a directory, a
non-positive timeout -- raise
:class:`ohlc_toolkit.temporal.ConfigError` instead, at the same boundary.

The third snapshot exception, ``SnapshotContinuityError``, lives in
:mod:`ohlc_toolkit.snapshot.continuity` next to the report type it
carries, following the same placement as
:class:`ohlc_toolkit.source.validation.SourceValidationError`. Defining it
here would mean this module importing the report and the report module
importing this one.

:func:`bounded_echo` lives here because every one of its uses is in an
error message or in the log line that precedes the raise: keeping it with
the exceptions is what stops a pathological manifest from turning one
refusal into an unbounded log line.
"""

from ohlc_toolkit.temporal import DataValidationError


class SnapshotManifestError(DataValidationError):
    """A release manifest could not be parsed, or contradicts itself.

    Raised before any asset is fetched. A manifest this package cannot
    read in full is not partially believed: there is no way to verify an
    asset whose declared digest was never understood.
    """


class SnapshotIntegrityError(DataValidationError):
    """A release asset could not be fetched, or failed its declared digest.

    Covers every byte-level refusal: an asset the release does not serve,
    a body larger or smaller than the manifest declared, and a body whose
    SHA-256 does not match. In every case the asset never reaches its
    final path.
    """
