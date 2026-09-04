"""Locally-built release fixtures shared across the ``test_snapshot`` modules.

Every byte these tests verify is produced here, in-process, and every
SHA-256 they assert against is computed from those same bytes at test
time. No digest is ever transcribed by hand, so a fixture and the
manifest describing it cannot drift apart, and no default-run test needs
the network.

The names deliberately do not match the real published release: a fixture
that looks like the real thing invites someone to "fix" it against the
real thing.
"""

import gzip
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import orjson

from ohlc_toolkit.snapshot.errors import SnapshotIntegrityError
from ohlc_toolkit.snapshot.manifest import MANIFEST_ASSET_NAME
from ohlc_toolkit.snapshot.release import SnapshotRelease

CADENCE_SECONDS = 60

FIXTURE_TAG = "example-1m-2026-08"
FIXTURE_REPOSITORY = "example/dataset"
FIXTURE_HOST = "https://example.invalid"
FIXTURE_AS_OF = "2026-09-01T02:07:12Z"
FIXTURE_REVISION = "0ec7256120c82643547d1a96a4dae6a0d953970a"
FIXTURE_SCHEMA_VERSION = 1

# A round-minute start, so the fixture sits on the same zero-phase grid the
# published data does.
FIXTURE_START = 1_325_376_060
FIXTURE_ROWS = 10

HISTORY_ASSET = "example_1min.csv.gz"
PARQUET_ASSET = "example_1min.parquet"
PROVENANCE_ASSET = "example_1min_provenance.csv"

_CSV_HEADER = "timestamp,open,high,low,close,volume"


def sha256_hex(payload: bytes) -> str:
    """Return the lowercase hex SHA-256 of ``payload``."""
    return hashlib.sha256(payload).hexdigest()


def history_timestamps(
    *,
    start: int = FIXTURE_START,
    rows: int = FIXTURE_ROWS,
    cadence_seconds: int = CADENCE_SECONDS,
    omit: Sequence[int] = (),
) -> list[int]:
    """Build the timestamps of a synthetic minute history.

    Args:
        start: The first row's Unix-second interval open.
        rows: How many grid positions to generate before omissions.
        cadence_seconds: The spacing between successive opens.
        omit: Grid positions to leave out, which is how a test makes a gap.

    Returns:
        The remaining timestamps, in ascending order.

    """
    omitted = set(omit)
    return [
        start + index * cadence_seconds for index in range(rows) if index not in omitted
    ]


def build_history_csv(timestamps: Sequence[int]) -> bytes:
    """Render a six-column history CSV over ``timestamps``.

    Args:
        timestamps: One Unix-second interval open per row, written in the
            order given so a test can hand over unsorted or duplicated
            rows on purpose.

    Returns:
        The CSV as UTF-8 bytes, header included.

    """
    lines = [_CSV_HEADER]
    for index, timestamp in enumerate(timestamps):
        price = 100.0 + index
        lines.append(f"{timestamp},{price},{price},{price},{price},1.0")
    return ("\n".join(lines) + "\n").encode("utf-8")


def gzip_bytes(payload: bytes) -> bytes:
    """Gzip ``payload`` with a fixed mtime so the result is byte-stable."""
    return gzip.compress(payload, mtime=0)


def build_asset_entry(payload: bytes) -> dict[str, Any]:
    """Describe one asset the way a manifest declares it."""
    return {"bytes": len(payload), "sha256": sha256_hex(payload)}


def build_manifest_payload(
    *,
    assets: Mapping[str, bytes],
    timestamps: Sequence[int],
    tag: str = FIXTURE_TAG,
) -> dict[str, Any]:
    """Build a manifest payload that truthfully describes its own fixtures.

    Args:
        assets: Asset name to the exact bytes published under it. Sizes
            and digests are derived from these bytes, never asserted.
        timestamps: The history the manifest claims to summarize; the
            first, last, and row-count statements are read off it.
        tag: The release tag the manifest declares.

    Returns:
        A plain dictionary, ready to mutate into a malformed variant.

    """
    return {
        "schema_version": FIXTURE_SCHEMA_VERSION,
        "tag": tag,
        "as_of": FIXTURE_AS_OF,
        "first_timestamp": timestamps[0],
        "last_timestamp": timestamps[-1],
        "row_count": len(timestamps),
        "generation_revision": FIXTURE_REVISION,
        "assets": {
            name: build_asset_entry(payload) for name, payload in assets.items()
        },
    }


def encode_manifest(payload: Mapping[str, Any]) -> bytes:
    """Serialize a manifest payload the way a release publishes it."""
    return orjson.dumps(payload)


def build_default_assets() -> dict[str, bytes]:
    """Build the three published assets a default fixture release carries."""
    timestamps = history_timestamps()
    return {
        HISTORY_ASSET: gzip_bytes(build_history_csv(timestamps)),
        PARQUET_ASSET: b"not really parquet, and no check here reads it as such",
        PROVENANCE_ASSET: (
            b"start_timestamp,end_timestamp,duration_minutes,flag\n"
            b"1362229320,1362233460,69,suspected_outage\n"
        ),
    }


class RecordingTransport:
    """An in-process stand-in for the HTTP transport, backed by a URL map.

    It honours the same contract the shipped transport does: it writes the
    published bytes to the destination it is handed, refuses a body larger
    than the caller's cap without writing it, and raises
    ``SnapshotIntegrityError`` for a URL the release does not serve. Every
    requested URL is recorded, so a test can show that a re-fetch asked
    for nothing it already had.
    """

    def __init__(self, payloads: Mapping[str, bytes]) -> None:
        """Serve exactly the URL-to-bytes map given."""
        self.payloads = dict(payloads)
        self.requested: list[str] = []

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None:
        """Write the bytes published at ``url`` to ``destination``."""
        self.requested.append(url)
        payload = self.payloads.get(url)
        if payload is None:
            raise SnapshotIntegrityError(f"No asset is published at {url}.")
        if len(payload) > max_bytes:
            raise SnapshotIntegrityError(
                f"Body at {url} exceeds the {max_bytes}-byte cap declared for it."
            )
        destination.write_bytes(payload)


@dataclass(frozen=True)
class ReleaseFixture:
    """A fixture release whose manifest truthfully describes its own bytes.

    Attributes:
        release: The release identity the URLs are composed from.
        manifest_bytes: The manifest payload served under the release.
        assets: Asset name to the exact bytes published under it.
        payloads: URL to bytes, covering the manifest and every asset.

    """

    release: SnapshotRelease
    manifest_bytes: bytes
    assets: Mapping[str, bytes]
    payloads: Mapping[str, bytes]

    def transport(
        self, payloads: Mapping[str, bytes] | None = None
    ) -> RecordingTransport:
        """Build a transport serving this release, or a perturbed variant."""
        return RecordingTransport(self.payloads if payloads is None else payloads)

    def serving(self, asset_name: str, payload: bytes | None) -> dict[str, bytes]:
        """Return this release's payload map with one asset replaced or removed.

        Args:
            asset_name: The asset to perturb.
            payload: The bytes to serve instead, or None to serve nothing
                at all, which is how a test models a missing asset.

        Returns:
            A new payload map; this fixture's own map is left alone.

        """
        served = dict(self.payloads)
        url = self.release.asset_url(asset_name)
        if payload is None:
            del served[url]
        else:
            served[url] = payload
        return served

    def url_for(self, asset_name: str) -> str:
        """Return the URL this release serves one asset under."""
        return self.release.asset_url(asset_name)


def build_release_fixture(
    *,
    assets: Mapping[str, bytes] | None = None,
    timestamps: Sequence[int] | None = None,
    tag: str = FIXTURE_TAG,
    manifest_tag: str | None = None,
    manifest_bytes: bytes | None = None,
) -> ReleaseFixture:
    """Build a self-consistent fixture release.

    Args:
        assets: Asset name to bytes; defaults to the three-asset layout
            the real releases publish.
        timestamps: The history the manifest summarizes; defaults to the
            complete grid the default history asset holds.
        tag: The tag the release is fetched by.
        manifest_tag: The tag the manifest itself declares, when a test
            needs it to disagree with ``tag``.
        manifest_bytes: A replacement manifest payload, for tests that
            need it malformed rather than merely wrong.

    Returns:
        The fixture, including the URL-to-bytes map a transport serves.

    """
    resolved_assets = dict(build_default_assets() if assets is None else assets)
    resolved_timestamps = list(
        history_timestamps() if timestamps is None else timestamps
    )
    payload = build_manifest_payload(
        assets=resolved_assets,
        timestamps=resolved_timestamps,
        tag=tag if manifest_tag is None else manifest_tag,
    )
    encoded = encode_manifest(payload) if manifest_bytes is None else manifest_bytes
    release = SnapshotRelease(repository=FIXTURE_REPOSITORY, tag=tag, host=FIXTURE_HOST)
    payloads = {release.asset_url(MANIFEST_ASSET_NAME): encoded}
    for name, asset_payload in resolved_assets.items():
        payloads[release.asset_url(name)] = asset_payload
    return ReleaseFixture(
        release=release,
        manifest_bytes=encoded,
        assets=resolved_assets,
        payloads=payloads,
    )
