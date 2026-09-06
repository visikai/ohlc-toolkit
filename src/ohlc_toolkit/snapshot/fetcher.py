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
from ohlc_toolkit.snapshot.release import (
    DEFAULT_RELEASE_HOST,
    SnapshotRelease,
)
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


def verify_snapshot_on_disk(
    directory: str | os.PathLike[str],
    *,
    repository: str,
    host: str = DEFAULT_RELEASE_HOST,
) -> SnapshotFetchResult:
    """Verify a snapshot already on disk against the manifest beside it.

    :func:`fetch_snapshot` answers "did these bytes arrive intact". This
    answers the other question a consumer has: "are these still the
    bytes". They are different questions and neither implies the other --
    a directory verified in August is not thereby a directory verified
    now, and nothing about a directory stops something else writing to
    it. Nothing is fetched here and no transport is used.

    What it proves is that the directory is INTERNALLY CONSISTENT: these
    bytes are the bytes this manifest describes. It does not prove the
    manifest is the one you meant. A different, self-consistent release
    verifies clean and is reported as itself, which is the right
    behaviour -- the identity it returns is what a caller records, so a
    substitution is visible in the record rather than hidden by it.
    Pinning an expected identity is a separate feature and is not this.

    Nor is containment. An asset that is a SYMLINK to a file outside the
    directory verifies clean, exactly as it does on the fetch path: what
    is checked is that the bytes reachable at each declared path are the
    bytes the manifest declares, not where those bytes live. A directory
    assembled by something hostile can therefore point outward and still
    be internally consistent, which is worth knowing before treating a
    clean verification as a statement about the filesystem.

    The result is the same type :func:`fetch_snapshot` returns, so
    :func:`~ohlc_toolkit.snapshot.continuity.read_snapshot_frame` consumes
    it unchanged. Every asset reports ``was_downloaded=False``, because
    nothing was.

    The returned release is assembled from the two parties that each know
    part of it: ``repository`` and ``host`` come from the caller, because
    a manifest does not record where it was published, and the tag comes
    from the manifest, because that is the one field it does record.
    Neither is defaulted to this project's own release, which would be
    the same mistake :func:`fetch_snapshot` avoids by refusing to pick a
    directory on a caller's disk -- and this function knows LESS about
    provenance than that one does, not more, since it fetched nothing.

    Args:
        directory: The directory holding the manifest and its assets.
        repository: The repository the snapshot came from, ``owner/name``.
            Required and not defaulted: a repository this function
            invented would be recorded as provenance by a caller who
            trusted it, and its ``asset_url`` would resolve somewhere
            with no relationship to the bytes just verified.
        host: The release host. Defaults to
            :data:`~ohlc_toolkit.snapshot.release.DEFAULT_RELEASE_HOST`,
            which is the same default :class:`SnapshotRelease` itself
            applies, so this adds no assumption of its own.

    Returns:
        The verified assets, the parsed manifest, and the snapshot
        identity.

    Raises:
        ConfigError: If no manifest is there to check against, or if an
            asset path exists and is not a regular file. Both are a
            caller pointing this at the wrong directory, which is a
            different kind of wrong from a file that does not match.
        SnapshotIntegrityError: If the manifest declares no assets, or an
            asset it declares is absent, the wrong size, or hashes to
            something else. The same class :func:`fetch_snapshot` raises
            for the same findings, so one ``except`` covers both
            questions.
        SnapshotManifestError: If the manifest's own bytes do not parse
            or fail its schema, or if the tag it declares is not a usable
            release tag.
        OSError: If the manifest or an asset is present but cannot be
            read.

    """
    base = Path(directory)
    manifest_path = base / MANIFEST_ASSET_NAME
    if not manifest_path.is_file():
        logger.warning("No snapshot manifest at {}.", bounded_echo(str(manifest_path)))
        raise ConfigError(
            f"No snapshot manifest at {bounded_echo(str(manifest_path))}; there "
            "is nothing to verify these files against."
        )

    raw = manifest_path.read_bytes()
    manifest = parse_manifest(raw)
    # The identity is the digest of the manifest bytes exactly as
    # published. Per-asset digests cannot reveal a wholesale swap of a
    # manifest and its assets together, because a swapped manifest
    # describes its swapped assets correctly; this one value changes.
    manifest_sha256 = hashlib.sha256(raw).hexdigest()

    # Refused HERE and not only by `parse_manifest`, which refuses it
    # today. If that guard is ever relaxed, the loop below verifies
    # nothing and reports success -- a verification that passes over zero
    # assets is worse than none, because it looks like one.
    if not manifest.assets:
        logger.error("Manifest {} declares no assets.", bounded_echo(manifest_sha256))
        raise SnapshotIntegrityError(
            f"Manifest {bounded_echo(manifest_sha256)} declares no assets, so "
            f"verifying it proves nothing about {bounded_echo(str(base))}."
        )

    assets = {
        record.name: _verified_on_disk(base / record.name, record)
        for record in manifest.assets.values()
    }
    logger.info(
        "Verified {} on-disk asset(s) against manifest {} (tag {!r}).",
        len(assets),
        manifest_sha256,
        manifest.tag,
    )
    # `parse_manifest` reads the tag as any non-empty string, and
    # `SnapshotRelease` applies the stricter release grammar. Checked here
    # so a hostile tag is refused as what it is -- a manifest this cannot
    # use -- rather than escaping as the ConfigError that means a caller
    # pointed at the wrong directory. The caller pointed at the right one.
    try:
        release = SnapshotRelease(repository=repository, tag=manifest.tag, host=host)
    except ConfigError as error:
        logger.error(
            "Manifest {} declares an unusable release tag: {}",
            bounded_echo(manifest_sha256),
            bounded_echo(manifest.tag),
        )
        raise SnapshotManifestError(
            f"Manifest {bounded_echo(manifest_sha256)} declares the tag "
            f"{bounded_echo(manifest.tag)}, which is not a usable release tag."
        ) from error

    return SnapshotFetchResult(
        release=release,
        directory=base,
        manifest=manifest,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        assets=assets,
    )


def _verified_on_disk(path: Path, record: AssetRecord) -> FetchedAsset:
    """Check one on-disk asset against the record the manifest declares.

    Size before digest, through the same two helpers the fetch path uses
    -- a truncated file and a substituted one are different findings and
    the messages say which.
    """
    if not path.is_file():
        if path.exists():
            logger.warning(
                "Refusing asset {!r}: {} exists and is not a regular file.",
                record.name,
                bounded_echo(str(path)),
            )
            raise ConfigError(
                f"Snapshot asset path for {record.name!r} exists and is not a "
                f"regular file: {bounded_echo(str(path))}."
            )
        logger.error(
            "Manifest declares asset {!r}, absent at {}.",
            record.name,
            bounded_echo(str(path)),
        )
        raise SnapshotIntegrityError(
            f"The manifest declares asset {record.name!r}, which is not at "
            f"{bounded_echo(str(path))}."
        )

    source = str(path)
    _verify_size(path, record, source)
    _verify_digest(path, record, source)
    return FetchedAsset(
        name=record.name,
        path=path,
        sha256=record.sha256,
        size_bytes=record.size_bytes,
        was_downloaded=False,
    )


def _prepare_directory(directory: str | os.PathLike[str]) -> Path:
    """Resolve and create the destination, refusing a non-directory path."""
    resolved = Path(directory)
    if resolved.exists() and not resolved.is_dir():
        logger.warning(
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
            logger.warning(
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


def _verify_size(path: Path, record: AssetRecord, source: str) -> None:
    """Check a file's size against the manifest.

    A short body cannot match the declared digest either, but size is the
    cheaper check and says plainly that the file is truncated rather than
    that the data was tampered with.

    ``source`` is where the bytes came from, for the message alone: a URL
    when they were just downloaded, a path when they were already on
    disk. The check itself does not care which.
    """
    landed = path.stat().st_size
    if landed != record.size_bytes:
        logger.error(
            "Asset {!r} from {} is {} bytes, not the declared {}.",
            record.name,
            bounded_echo(source),
            landed,
            record.size_bytes,
        )
        raise SnapshotIntegrityError(
            f"Asset {record.name!r} from {bounded_echo(source)} is {landed} "
            f"bytes, not the {record.size_bytes} bytes its manifest declared."
        )


def _verify_digest(path: Path, record: AssetRecord, source: str) -> None:
    """Check a file's bytes against the manifest's declared SHA-256."""
    digest = _digest_of(path)
    if digest != record.sha256:
        logger.error(
            "Asset {!r} from {} hashes to {}, not the declared {}.",
            record.name,
            bounded_echo(source),
            digest,
            record.sha256,
        )
        raise SnapshotIntegrityError(
            f"Asset {record.name!r} from {bounded_echo(source)} has sha256 "
            f"{digest}, not the declared {record.sha256}."
        )


def _digest_of(path: Path) -> str:
    """Return the lowercase hex SHA-256 of a file, read in bounded blocks."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()
