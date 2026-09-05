"""Fetch a release's declared assets, verifying every byte before use.

The rule this module exists to enforce: bytes reach their final path only
after their SHA-256 matches what the manifest declared. Downloads land on
a temporary name in the destination directory and are renamed into place
with :func:`os.replace` -- an atomic rename within one filesystem -- only
once the size and digest both check out. On any failure the temporary
file is removed, so a reader that finds a file under an asset's name has
found verified bytes, always.

A fetch is deliberately **not** transactional. An asset that already
matched its declared digest is finished work, not partial state; deleting
it because a later asset failed would throw away a verified download and
force it back over the wire. Re-fetching then asks only for what is
missing.

The manifest itself is always re-fetched. It carries no digest of its own
(nothing could compute one over bytes that would then have to contain
it), so a cached copy proves nothing about the release being fetched, and
at well under a kibibyte it is not worth caching. Its verification is a
successful strict parse plus a check that it names the release actually
asked for.
"""

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, unique
from pathlib import Path
from types import MappingProxyType

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.snapshot.errors import (
    SnapshotIntegrityError,
    SnapshotManifestError,
)
from ohlc_toolkit.snapshot.manifest import (
    MANIFEST_ASSET_NAME,
    MAX_MANIFEST_BYTES,
    AssetRecord,
    SnapshotManifest,
    parse_manifest,
)
from ohlc_toolkit.snapshot.release import SnapshotRelease
from ohlc_toolkit.snapshot.transport import AssetTransport, HttpAssetTransport
from ohlc_toolkit.temporal import ConfigError, bounded_echo

logger = get_logger(__name__)

# Temporary downloads are hidden (leading dot) and suffixed, so a reader
# globbing for the published names cannot pick one up, and a leftover
# from a killed process is obvious for what it is.
TEMP_SUFFIX = ".part"


@unique
class ExistingAssetPolicy(Enum):
    """What to do with an already-present file whose digest does not match.

    An already-present file whose digest *does* match is never
    re-downloaded under either policy: it is exactly the asset asked for.

    REFUSE is the default because silently overwriting a caller's file is
    the one outcome that cannot be undone. REPLACE overwrites it, but
    only through the same verify-then-rename path, so a replacement that
    fails its own digest leaves the original exactly where it was.
    """

    REFUSE = "refuse"
    REPLACE = "replace"


@dataclass(frozen=True)
class FetchedAsset:
    """One verified asset sitting at its final path.

    Attributes:
        name: The asset's plain filename within the release.
        path: Where the verified bytes now are.
        sha256: The digest they were verified against.
        size_bytes: The size they were verified against.
        was_downloaded: True when this fetch transferred the bytes; False
            when an already-present copy matched the declared digest and
            was kept.

    """

    name: str
    path: Path
    sha256: str
    size_bytes: int
    was_downloaded: bool


@dataclass(frozen=True)
class SnapshotFetchResult:
    """What one fetch of one release produced.

    Attributes:
        release: The release that was fetched.
        directory: The directory the caller named, now holding the assets.
        manifest: The parsed manifest every asset was verified against.
        manifest_path: Where that manifest's bytes were written.
        manifest_sha256: The snapshot identity -- the SHA-256 over the
            manifest bytes exactly as fetched. Per-asset digests cannot
            reveal a wholesale swap of the manifest and its assets
            together, because a swapped manifest describes its swapped
            assets correctly; this one value changes.
        assets: Asset name to its verified record. Stored as a read-only
            mapping; because this field is a mapping, instances of this
            class are not hashable.

    """

    release: SnapshotRelease
    directory: Path
    manifest: SnapshotManifest
    manifest_path: Path
    manifest_sha256: str
    assets: Mapping[str, FetchedAsset]

    def __post_init__(self) -> None:
        """Take a defensive, read-only copy of the asset mapping."""
        object.__setattr__(self, "assets", MappingProxyType(dict(self.assets)))


def fetch_snapshot(
    release: SnapshotRelease,
    directory: str | os.PathLike[str],
    *,
    transport: AssetTransport | None = None,
    existing: ExistingAssetPolicy = ExistingAssetPolicy.REFUSE,
    expected_manifest_sha256: str | None = None,
) -> SnapshotFetchResult:
    """Fetch every asset a release declares, verifying each before it lands.

    Args:
        release: The release to fetch.
        directory: Where to put the assets. Created if missing; there is
            no default, because a library should not pick a directory on
            a caller's disk.
        transport: How bytes are moved. Defaults to
            :class:`~ohlc_toolkit.snapshot.transport.HttpAssetTransport`;
            substitute one to fetch from somewhere else.
        existing: What to do with an already-present file whose digest
            does not match. Defaults to refusing.
        expected_manifest_sha256: If given, the snapshot identity the
            fetched manifest must have -- the SHA-256 over its bytes. A
            caller holding the identity of the snapshot it means can
            demand exactly that one, which refuses both a wholesale
            manifest swap and a release re-cut under the same tag.

    Returns:
        The fetch result, including the parsed manifest and one record
        per verified asset.

    Raises:
        ConfigError: If ``directory`` exists and is not a directory, or
            an asset's final path is occupied by something that is not a
            regular file.
        SnapshotManifestError: If the manifest cannot be parsed, or names
            a different release than the one asked for.
        SnapshotIntegrityError: If any asset cannot be fetched, is not
            the size its manifest declared, or fails its declared
            SHA-256 -- or if ``expected_manifest_sha256`` is given and
            the fetched manifest's identity is not it.

    """
    resolved = _prepare_directory(directory)
    active = HttpAssetTransport() if transport is None else transport
    manifest_path = resolved / MANIFEST_ASSET_NAME
    manifest, manifest_sha256 = _fetch_manifest(release, active, manifest_path)
    if (
        expected_manifest_sha256 is not None
        and manifest_sha256 != expected_manifest_sha256
    ):
        logger.error(
            "Refusing snapshot {}: manifest identity is {}, expected {}.",
            bounded_echo(release.tag),
            manifest_sha256,
            bounded_echo(expected_manifest_sha256),
        )
        raise SnapshotIntegrityError(
            f"The fetched manifest for {bounded_echo(release.tag)} has identity "
            f"{manifest_sha256}, not the expected "
            f"{bounded_echo(expected_manifest_sha256)}. Either the caller's "
            "record is stale or the published release changed under its tag."
        )

    assets = {
        record.name: _resolve_asset(release, active, resolved, record, existing)
        for record in manifest.assets.values()
    }
    logger.info(
        "Fetched snapshot {} into {}: {} asset(s), {} newly downloaded.",
        bounded_echo(release.tag),
        bounded_echo(str(resolved)),
        len(assets),
        sum(asset.was_downloaded for asset in assets.values()),
    )
    return SnapshotFetchResult(
        release=release,
        directory=resolved,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        manifest_path=manifest_path,
        assets=assets,
    )


def _prepare_directory(directory: str | os.PathLike[str]) -> Path:
    """Resolve and create the destination, refusing a non-directory path."""
    resolved = Path(directory)
    if resolved.exists() and not resolved.is_dir():
        logger.error(
            "Refusing to fetch into {}: it exists and is not a directory.",
            bounded_echo(str(resolved)),
        )
        raise ConfigError(
            f"Snapshot destination {bounded_echo(str(resolved))} exists and is not "
            "a directory."
        )
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _fetch_manifest(
    release: SnapshotRelease, transport: AssetTransport, manifest_path: Path
) -> tuple[SnapshotManifest, str]:
    """Fetch, parse, and place the manifest, returning it with its identity.

    The bytes are only moved to ``manifest_path`` after they parse and
    after they turn out to describe this release, so a manifest on disk
    is always one this package could read. The returned identity is the
    SHA-256 over exactly the bytes that were parsed and placed.
    """
    url = release.asset_url(MANIFEST_ASSET_NAME)
    temp_path = _make_temp_path(manifest_path)
    try:
        transport.download(url, temp_path, max_bytes=MAX_MANIFEST_BYTES)
        raw = temp_path.read_bytes()
        manifest = parse_manifest(raw)
        if manifest.tag != release.tag:
            logger.error(
                "Refusing the manifest at {}: it declares tag {}, not {}.",
                bounded_echo(url),
                bounded_echo(manifest.tag),
                bounded_echo(release.tag),
            )
            raise SnapshotManifestError(
                f"Manifest at {bounded_echo(url)} declares tag "
                f"{bounded_echo(manifest.tag)}, but the release being fetched is "
                f"{bounded_echo(release.tag)}."
            )
        os.replace(temp_path, manifest_path)
    finally:
        temp_path.unlink(missing_ok=True)
    return manifest, hashlib.sha256(raw).hexdigest()


def _resolve_asset(
    release: SnapshotRelease,
    transport: AssetTransport,
    directory: Path,
    record: AssetRecord,
    existing: ExistingAssetPolicy,
) -> FetchedAsset:
    """Return a verified asset, downloading it only if it is not already there."""
    destination = directory / record.name
    if destination.exists():
        if not destination.is_file():
            logger.error(
                "Refusing asset {!r}: {} exists and is not a regular file.",
                record.name,
                bounded_echo(str(destination)),
            )
            raise ConfigError(
                f"Snapshot asset path for {record.name!r} exists and is not a "
                f"regular file: {bounded_echo(str(destination))}."
            )
        present = _digest_of(destination)
        if present == record.sha256:
            logger.info(
                "Asset {!r} is already present with the declared digest; "
                "not re-downloading it.",
                record.name,
            )
            return FetchedAsset(
                name=record.name,
                path=destination,
                sha256=record.sha256,
                size_bytes=record.size_bytes,
                was_downloaded=False,
            )
        if existing is ExistingAssetPolicy.REFUSE:
            logger.error(
                "Asset {!r} is already present with sha256 {}, not the declared {}.",
                record.name,
                present,
                record.sha256,
            )
            raise SnapshotIntegrityError(
                f"Asset {record.name!r} is already present at "
                f"{bounded_echo(str(destination))} "
                f"with sha256 {present}, not the declared {record.sha256}. It "
                "is left untouched; pass ExistingAssetPolicy.REPLACE to "
                "overwrite it deliberately."
            )
        logger.warning(
            "Replacing asset {!r}: its sha256 {} does not match the declared {}.",
            record.name,
            present,
            record.sha256,
        )

    _download_verified(release, transport, destination, record)
    return FetchedAsset(
        name=record.name,
        path=destination,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        was_downloaded=True,
    )


def _download_verified(
    release: SnapshotRelease,
    transport: AssetTransport,
    destination: Path,
    record: AssetRecord,
) -> None:
    """Download one asset to a temporary name and rename it only once verified."""
    url = release.asset_url(record.name)
    temp_path = _make_temp_path(destination)
    try:
        transport.download(url, temp_path, max_bytes=record.size_bytes)
        _verify_size(temp_path, record, url)
        _verify_digest(temp_path, record, url)
        os.replace(temp_path, destination)
        logger.info(
            "Verified and placed asset {!r} at {}.",
            record.name,
            bounded_echo(str(destination)),
        )
    finally:
        # After a successful rename this is already gone; after any
        # failure it is the partly-written body nobody may read.
        temp_path.unlink(missing_ok=True)


def _make_temp_path(final_path: Path) -> Path:
    """Create an empty, uniquely-named temporary file beside ``final_path``.

    It shares a directory with its destination so the later rename is a
    same-filesystem, atomic ``os.replace`` rather than a copy.
    """
    descriptor, name = tempfile.mkstemp(
        dir=final_path.parent, prefix=f".{final_path.name}.", suffix=TEMP_SUFFIX
    )
    os.close(descriptor)
    return Path(name)


def _verify_size(temp_path: Path, record: AssetRecord, url: str) -> None:
    """Check the landed size against the manifest.

    A short body cannot match the declared digest either, but size is the
    cheaper check and says plainly that the transfer was truncated rather
    than that the data was tampered with.
    """
    landed = temp_path.stat().st_size
    if landed != record.size_bytes:
        logger.error(
            "Asset {!r} from {} landed at {} bytes, not the declared {}.",
            record.name,
            bounded_echo(url),
            landed,
            record.size_bytes,
        )
        raise SnapshotIntegrityError(
            f"Asset {record.name!r} from {bounded_echo(url)} landed at {landed} "
            "bytes, not "
            f"the {record.size_bytes} bytes its manifest declared."
        )


def _verify_digest(temp_path: Path, record: AssetRecord, url: str) -> None:
    """Check the landed bytes against the manifest's declared SHA-256."""
    digest = _digest_of(temp_path)
    if digest != record.sha256:
        logger.error(
            "Asset {!r} from {} hashes to {}, not the declared {}.",
            record.name,
            bounded_echo(url),
            digest,
            record.sha256,
        )
        raise SnapshotIntegrityError(
            f"Asset {record.name!r} from {bounded_echo(url)} has sha256 {digest}, "
            "not the "
            f"declared {record.sha256}."
        )


def _digest_of(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file, read in bounded blocks."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
