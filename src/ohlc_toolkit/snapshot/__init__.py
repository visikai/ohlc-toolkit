"""Fetching and verifying a published monthly full-history snapshot.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.snapshot`` directly.
"""

from ohlc_toolkit.snapshot.errors import (
    SnapshotIntegrityError,
    SnapshotManifestError,
)
from ohlc_toolkit.snapshot.manifest import (
    MANIFEST_ASSET_NAME,
    MAX_MANIFEST_BYTES,
    SUPPORTED_SCHEMA_VERSION,
    AssetRecord,
    SnapshotManifest,
    parse_manifest,
)
from ohlc_toolkit.snapshot.release import (
    BITSTAMP_BTCUSD_1M_REPOSITORY,
    BITSTAMP_HISTORY_CSV_ASSET,
    DEFAULT_RELEASE_HOST,
    SnapshotRelease,
    is_plain_asset_name,
)

__all__ = [
    "BITSTAMP_BTCUSD_1M_REPOSITORY",
    "BITSTAMP_HISTORY_CSV_ASSET",
    "DEFAULT_RELEASE_HOST",
    "MANIFEST_ASSET_NAME",
    "MAX_MANIFEST_BYTES",
    "SUPPORTED_SCHEMA_VERSION",
    "AssetRecord",
    "SnapshotIntegrityError",
    "SnapshotManifest",
    "SnapshotManifestError",
    "SnapshotRelease",
    "is_plain_asset_name",
    "parse_manifest",
]
