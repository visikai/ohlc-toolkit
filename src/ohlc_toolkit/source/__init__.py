"""Source profiles, raw-frame validation, and a polars-native CSV reader.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.source`` directly.
"""

from ohlc_toolkit.source.profile import (
    BITSTAMP_BTCUSD_1M,
    Availability,
    ColumnKind,
    SourceProfile,
)
from ohlc_toolkit.source.reader import SourceReadResult, read_source_csv
from ohlc_toolkit.source.validation import (
    Finding,
    FindingKind,
    SourceValidationError,
    ValidationMode,
    ValidationReport,
    validate_source_frame,
)

__all__ = [
    "BITSTAMP_BTCUSD_1M",
    "Availability",
    "ColumnKind",
    "Finding",
    "FindingKind",
    "SourceProfile",
    "SourceReadResult",
    "SourceValidationError",
    "ValidationMode",
    "ValidationReport",
    "read_source_csv",
    "validate_source_frame",
]
