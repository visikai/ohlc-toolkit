"""Fetching and verifying a published monthly full-history snapshot.

This namespace is not re-exported from the top-level ``ohlc_toolkit``
package; import from ``ohlc_toolkit.snapshot`` directly.
"""

from ohlc_toolkit.snapshot.continuity import (
    ContinuityReport,
    SeamKind,
    SeamMismatch,
    SnapshotContinuityError,
    read_snapshot_frame,
    verify_snapshot_continuity,
)
from ohlc_toolkit.snapshot.errors import (
    SnapshotIntegrityError,
    SnapshotManifestError,
)
from ohlc_toolkit.snapshot.fetcher import (
    ExistingAssetPolicy,
    FetchedAsset,
    SnapshotFetchResult,
    fetch_snapshot,
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
from ohlc_toolkit.snapshot.transport import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    AssetTransport,
    HttpAssetTransport,
)

__all__ = [
    "BITSTAMP_BTCUSD_1M_REPOSITORY",
    "BITSTAMP_HISTORY_CSV_ASSET",
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_RELEASE_HOST",
    "DEFAULT_TIMEOUT_SECONDS",
    "MANIFEST_ASSET_NAME",
    "MAX_MANIFEST_BYTES",
    "SUPPORTED_SCHEMA_VERSION",
    "AssetRecord",
    "AssetTransport",
    "ContinuityReport",
    "ExistingAssetPolicy",
    "FetchedAsset",
    "HttpAssetTransport",
    "SeamKind",
    "SeamMismatch",
    "SnapshotContinuityError",
    "SnapshotFetchResult",
    "SnapshotIntegrityError",
    "SnapshotManifest",
    "SnapshotManifestError",
    "SnapshotRelease",
    "fetch_snapshot",
    "is_plain_asset_name",
    "parse_manifest",
    "read_snapshot_frame",
    "verify_snapshot_continuity",
]
