"""Strict parsing of a published release's JSON manifest.

The manifest is the only statement of what a release contains and what
each asset should hash to. Every later step trusts it, so nothing here is
lenient: a manifest that is short one key, one schema version ahead, or
one character wrong in a digest is refused outright rather than partially
believed. There is no repair path and no default value.

The manifest carries no digest of itself -- nothing could compute one
over bytes that would then have to contain it -- so a successful strict
parse is the whole of its verification, and a manifest that declares
itself as an asset is refused for the same reason.
"""

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Any

import orjson

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.snapshot.errors import SnapshotManifestError, bounded_echo
from ohlc_toolkit.snapshot.release import is_plain_asset_name

logger = get_logger(__name__)

MANIFEST_ASSET_NAME = "manifest.json"

# The one manifest layout this package knows how to read. A manifest
# declaring any other version is refused, never read with today's
# assumptions: a future version may reuse a key with a new meaning.
SUPPORTED_SCHEMA_VERSION = 1

# An explicit cap on a payload fetched before anything about it is known.
# The published manifest is 739 bytes; a mebibyte leaves several orders of
# magnitude of headroom while still bounding the download.
MAX_MANIFEST_BYTES = 1 << 20

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")

_REQUIRED_KEYS = (
    "schema_version",
    "tag",
    "as_of",
    "first_timestamp",
    "last_timestamp",
    "row_count",
    "generation_revision",
    "assets",
)


@dataclass(frozen=True)
class AssetRecord:
    """One asset a release manifest declares, with its size and digest.

    Attributes:
        name: The asset's plain filename within the release.
        size_bytes: The exact published length, used as the download cap
            and checked against what actually landed.
        sha256: The lowercase hex SHA-256 the downloaded bytes must have.

    """

    name: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        """Validate the record against the grammar every check downstream assumes.

        Raises:
            SnapshotManifestError: If the name is not a plain filename,
                the size is not a positive integer, or the digest is not
                64 lowercase hex characters.

        """
        if not is_plain_asset_name(self.name):
            logger.error(
                "Rejecting manifest asset name {}: not a plain filename.",
                bounded_echo(self.name),
            )
            raise SnapshotManifestError(
                f"Manifest declares an invalid asset name: "
                f"{bounded_echo(self.name)}. An asset name must be a plain "
                "filename, so a manifest cannot steer a write out of the "
                "caller's directory."
            )
        if isinstance(self.size_bytes, bool) or not isinstance(self.size_bytes, int):
            logger.error(
                "Rejecting asset {!r}: declared size {} is not an integer.",
                self.name,
                bounded_echo(self.size_bytes),
            )
            raise SnapshotManifestError(
                f"Asset {self.name!r} declares a non-integer size: "
                f"{bounded_echo(self.size_bytes)}."
            )
        if self.size_bytes <= 0:
            logger.error(
                "Rejecting asset {!r}: declared size {} is not positive.",
                self.name,
                self.size_bytes,
            )
            raise SnapshotManifestError(
                f"Asset {self.name!r} declares a non-positive size: {self.size_bytes}."
            )
        if not isinstance(self.sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.sha256
        ):
            logger.error(
                "Rejecting asset {!r}: declared sha256 {} is not 64 lowercase "
                "hex characters.",
                self.name,
                bounded_echo(self.sha256),
            )
            raise SnapshotManifestError(
                f"Asset {self.name!r} declares an invalid sha256 digest: "
                f"{bounded_echo(self.sha256)}."
            )


@dataclass(frozen=True)
class SnapshotManifest:
    """Everything a release states about itself, parsed and typed.

    Attributes:
        schema_version: The manifest layout version; always
            :data:`SUPPORTED_SCHEMA_VERSION` on a parsed instance.
        tag: The release tag the manifest claims to describe.
        as_of: The offset-aware instant the snapshot was generated at.
        first_timestamp: The Unix-second open of the history's first row.
        last_timestamp: The Unix-second open of the history's last row.
        row_count: How many rows the history holds.
        generation_revision: The dataset repository revision that
            produced the snapshot.
        assets: Asset name to its declared record. Stored as a read-only
            mapping, so mutating whatever was passed in cannot later
            change what this manifest says. Because this field is a
            mapping, instances of this class are not hashable.

    """

    schema_version: int
    tag: str
    as_of: datetime
    first_timestamp: int
    last_timestamp: int
    row_count: int
    generation_revision: str
    assets: Mapping[str, AssetRecord]

    def __post_init__(self) -> None:
        """Take a defensive, read-only copy of the asset mapping."""
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))


def parse_manifest(raw: bytes) -> SnapshotManifest:
    """Parse and strictly validate a release manifest's bytes.

    Args:
        raw: The manifest payload exactly as published.

    Returns:
        The parsed manifest, with every declared field present and typed.

    Raises:
        SnapshotManifestError: If the payload exceeds
            :data:`MAX_MANIFEST_BYTES`, is not a JSON object, is missing
            any required key, declares an unsupported schema version, or
            holds a field or asset record that fails its own check.

    """
    _require_bounded_size(raw)
    payload = _decode(raw)

    schema_version = _require_int(payload, "schema_version")
    if schema_version != SUPPORTED_SCHEMA_VERSION:
        logger.error(
            "Refusing manifest schema_version {}: this package reads version {}.",
            schema_version,
            SUPPORTED_SCHEMA_VERSION,
        )
        raise SnapshotManifestError(
            f"Manifest declares schema_version {schema_version}; this package "
            f"reads version {SUPPORTED_SCHEMA_VERSION} and will not guess at "
            "another."
        )

    first_timestamp = _require_int(payload, "first_timestamp")
    last_timestamp = _require_int(payload, "last_timestamp")
    if last_timestamp < first_timestamp:
        logger.error(
            "Refusing manifest: last_timestamp {} precedes first_timestamp {}.",
            last_timestamp,
            first_timestamp,
        )
        raise SnapshotManifestError(
            f"Manifest last_timestamp {last_timestamp} precedes first_timestamp "
            f"{first_timestamp}: a history cannot end before it starts."
        )

    row_count = _require_int(payload, "row_count")
    if row_count <= 0:
        logger.error("Refusing manifest: row_count {} is not positive.", row_count)
        raise SnapshotManifestError(
            f"Manifest row_count must be positive, got {row_count}."
        )

    return SnapshotManifest(
        schema_version=schema_version,
        tag=_require_text(payload, "tag"),
        as_of=_require_instant(payload, "as_of"),
        first_timestamp=first_timestamp,
        last_timestamp=last_timestamp,
        row_count=row_count,
        generation_revision=_require_text(payload, "generation_revision"),
        assets=_parse_assets(payload),
    )


def _require_bounded_size(raw: bytes) -> None:
    """Refuse a payload larger than the cap before spending a parse on it."""
    if len(raw) > MAX_MANIFEST_BYTES:
        logger.error(
            "Refusing a {}-byte manifest payload: it exceeds the {}-byte cap.",
            len(raw),
            MAX_MANIFEST_BYTES,
        )
        raise SnapshotManifestError(
            f"Manifest payload of {len(raw)} bytes exceeds the "
            f"{MAX_MANIFEST_BYTES}-byte cap."
        )


def _decode(raw: bytes) -> Mapping[str, Any]:
    """Decode the payload into a JSON object, refusing anything else."""
    try:
        payload = orjson.loads(raw)
    except orjson.JSONDecodeError as error:
        logger.error("Manifest payload is not valid JSON: {}", error)
        raise SnapshotManifestError("Manifest payload is not valid JSON.") from error
    if not isinstance(payload, dict):
        logger.error(
            "Manifest payload decoded to a {}, not a JSON object.",
            type(payload).__name__,
        )
        raise SnapshotManifestError(
            f"Manifest payload must be a JSON object, got a {type(payload).__name__}."
        )
    return payload


def _require_key(payload: Mapping[str, Any], key: str) -> Any:
    """Read a required key, refusing a manifest that omits it."""
    if key not in payload:
        logger.error("Manifest is missing the required key {!r}.", key)
        raise SnapshotManifestError(
            f"Manifest is missing the required key {key!r}. Required keys are "
            f"{list(_REQUIRED_KEYS)}."
        )
    return payload[key]


def _require_int(payload: Mapping[str, Any], key: str) -> int:
    """Read a required integer key.

    ``bool`` is rejected explicitly: it is a subclass of ``int``, so a
    manifest saying ``"row_count": true`` would otherwise sail through as
    a row count of one.
    """
    value = _require_key(payload, key)
    if isinstance(value, bool) or not isinstance(value, int):
        logger.error(
            "Manifest key {!r} must be an integer, got {}.", key, bounded_echo(value)
        )
        raise SnapshotManifestError(
            f"Manifest key {key!r} must be an integer, got {bounded_echo(value)}."
        )
    return value


def _require_text(payload: Mapping[str, Any], key: str) -> str:
    """Read a required non-empty string key."""
    value = _require_key(payload, key)
    if not isinstance(value, str) or not value:
        logger.error(
            "Manifest key {!r} must be a non-empty string, got {}.",
            key,
            bounded_echo(value),
        )
        raise SnapshotManifestError(
            f"Manifest key {key!r} must be a non-empty string, got "
            f"{bounded_echo(value)}."
        )
    return value


def _require_instant(payload: Mapping[str, Any], key: str) -> datetime:
    """Read a required ISO-8601 instant that carries a UTC offset."""
    text = _require_text(payload, key)
    try:
        instant = datetime.fromisoformat(text)
    except ValueError as error:
        logger.error(
            "Manifest key {!r} is not an ISO-8601 instant: {}",
            key,
            bounded_echo(text),
        )
        raise SnapshotManifestError(
            f"Manifest key {key!r} must be an ISO-8601 instant, got "
            f"{bounded_echo(text)}."
        ) from error
    if instant.tzinfo is None:
        logger.error(
            "Manifest key {!r} carries no UTC offset: {}", key, bounded_echo(text)
        )
        raise SnapshotManifestError(
            f"Manifest key {key!r} must carry a UTC offset, got "
            f"{bounded_echo(text)}: a local time does not name an instant."
        )
    return instant


def _parse_assets(payload: Mapping[str, Any]) -> Mapping[str, AssetRecord]:
    """Parse the asset table, refusing anything this package could not verify."""
    raw_assets = _require_key(payload, "assets")
    if not isinstance(raw_assets, dict):
        logger.error(
            "Manifest key 'assets' must be a JSON object, got {}.",
            bounded_echo(raw_assets),
        )
        raise SnapshotManifestError(
            f"Manifest key 'assets' must be a JSON object, got "
            f"{bounded_echo(raw_assets)}."
        )
    if not raw_assets:
        logger.error("Manifest key 'assets' declares nothing.")
        raise SnapshotManifestError(
            "Manifest key 'assets' declares no assets: a release with nothing "
            "in it is a broken publish, not an empty success."
        )
    return {name: _parse_asset(name, entry) for name, entry in raw_assets.items()}


def _parse_asset(name: Any, entry: Any) -> AssetRecord:
    """Parse one asset record, refusing a name or shape that cannot be trusted.

    The name grammar is not re-checked here: :class:`AssetRecord` enforces
    it in its own constructor, which is the single place a record can come
    into existence. Every echo below therefore still goes through
    ``bounded_echo``, because at this point the name is untrusted input.
    """
    if name == MANIFEST_ASSET_NAME:
        logger.error("Rejecting a manifest that declares {!r} as an asset.", name)
        raise SnapshotManifestError(
            f"A manifest must not declare {MANIFEST_ASSET_NAME!r} as an asset: "
            "nothing could verify that digest against the bytes carrying it."
        )
    if not isinstance(entry, dict):
        logger.error(
            "Asset {} must be described by a JSON object, got {}.",
            bounded_echo(name),
            bounded_echo(entry),
        )
        raise SnapshotManifestError(
            f"Asset {bounded_echo(name)} must be described by a JSON object, "
            f"got {bounded_echo(entry)}."
        )
    for required in ("bytes", "sha256"):
        if required not in entry:
            logger.error(
                "Asset {} record is missing {!r}.", bounded_echo(name), required
            )
            raise SnapshotManifestError(
                f"Asset {bounded_echo(name)} record is missing the required key "
                f"{required!r}."
            )
    return AssetRecord(name=name, size_bytes=entry["bytes"], sha256=entry["sha256"])
