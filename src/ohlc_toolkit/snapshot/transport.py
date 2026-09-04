"""How release-asset bytes reach the local filesystem.

This is the only module in the subpackage that touches a network, and it
is deliberately the whole of that surface: :class:`AssetTransport` is the
seam everything else is written against, so the fetcher's verification,
temp-file, and idempotence behaviour is testable without a socket.

The transport does not decide where an asset ends up or whether it is
trustworthy. It writes the bytes it is given to the path it is handed,
under an explicit byte cap, and raises this package's own error on any
failure. Deciding the path, checking the digest, and renaming into place
belong to :mod:`ohlc_toolkit.snapshot.fetcher`.

``requests`` is this module's dependency and nothing else's: it is the
only import of it left in the package, so the cost of the dependency is
paid for exactly this transport.

This module reads no credential and echoes none into any log or message
-- but ``requests`` at module level means ambient environment
configuration applies: an ``HTTP(S)_PROXY``, a ``REQUESTS_CA_BUNDLE``,
and a ``~/.netrc`` entry for the release host WILL be honoured, netrc
turning into an ``Authorization`` header on the wire. That is left on
deliberately: the assets are public and need no auth, proxy support is
what keeps this usable behind one, and requests drops authorization on a
cross-host redirect. A deployment that must not send ambient credentials
anywhere should provide its own :class:`AssetTransport`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import requests

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.snapshot.errors import SnapshotIntegrityError
from ohlc_toolkit.temporal import ConfigError

logger = get_logger(__name__)

# Generous enough for a slow link on a hundred-megabyte asset, since it
# bounds each socket read rather than the whole transfer, and finite so a
# stalled connection cannot hang a fetch forever.
DEFAULT_TIMEOUT_SECONDS = 60.0

# One mebibyte per read: large enough that a hundred-megabyte asset costs
# a hundred-ish iterations, small enough that the byte cap is enforced
# long before a hostile body could exhaust memory.
DEFAULT_CHUNK_BYTES = 1 << 20

_HTTP_OK = 200


class AssetTransport(Protocol):
    """Writes the bytes published at a URL to a local path.

    Implementations must treat ``max_bytes`` as a hard cap enforced while
    reading, not a size checked afterwards, and must raise
    :class:`~ohlc_toolkit.snapshot.errors.SnapshotIntegrityError` -- not a
    library-specific error -- when the resource cannot be fetched.
    """

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None:
        """Write the bytes published at ``url`` to ``destination``.

        Args:
            url: The resource to read.
            destination: The path to write, which the caller owns and
                will move or delete afterwards.
            max_bytes: The hard cap on how many bytes to accept.

        """
        ...  # pragma: no cover - protocol declaration


@dataclass(frozen=True)
class HttpAssetTransport:
    """Streams one release asset over HTTP into a local file.

    Attributes:
        timeout_seconds: The per-read socket timeout. Must be positive: a
            request with no timeout has no bound at all.
        chunk_bytes: How many bytes to read per iteration. Must be
            positive.

    """

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    chunk_bytes: int = DEFAULT_CHUNK_BYTES

    def __post_init__(self) -> None:
        """Reject a configuration that would leave a download unbounded.

        Raises:
            ConfigError: If either setting is not positive.

        """
        if self.timeout_seconds <= 0:
            logger.warning(
                "Rejecting a transport with timeout_seconds={}.", self.timeout_seconds
            )
            raise ConfigError(
                f"HttpAssetTransport timeout_seconds must be positive, got "
                f"{self.timeout_seconds}."
            )
        if self.chunk_bytes <= 0:
            logger.warning(
                "Rejecting a transport with chunk_bytes={}.", self.chunk_bytes
            )
            raise ConfigError(
                f"HttpAssetTransport chunk_bytes must be positive, got "
                f"{self.chunk_bytes}."
            )

    def download(self, url: str, destination: Path, *, max_bytes: int) -> None:
        """Stream ``url`` into ``destination``, stopping at ``max_bytes``.

        Args:
            url: The release asset's download URL.
            destination: The temporary path the caller wants written.
            max_bytes: The size the manifest declared for this asset,
                enforced as a hard cap while reading.

        Raises:
            SnapshotIntegrityError: If the request fails, the response
                status is not 200, or the body outgrows ``max_bytes``.

        """
        logger.debug("Fetching release asset from {} (cap {} bytes).", url, max_bytes)
        try:
            response = requests.get(
                url, stream=True, timeout=self.timeout_seconds, allow_redirects=True
            )
        except requests.RequestException as error:
            logger.error("Release asset at {} could not be fetched: {}", url, error)
            raise SnapshotIntegrityError(
                f"Release asset at {url} could not be fetched: {error}."
            ) from error

        with response:
            if response.status_code != _HTTP_OK:
                logger.error(
                    "Release asset at {} returned HTTP {}.", url, response.status_code
                )
                raise SnapshotIntegrityError(
                    f"Release asset at {url} returned HTTP "
                    f"{response.status_code}; the release does not serve it."
                )
            self._stream_to(response, url, destination, max_bytes)

    def _stream_to(
        self,
        response: requests.Response,
        url: str,
        destination: Path,
        max_bytes: int,
    ) -> None:
        """Write the response body, refusing it the moment it outgrows the cap."""
        written = 0
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=self.chunk_bytes):
                written += len(chunk)
                if written > max_bytes:
                    logger.error(
                        "Release asset at {} exceeds its declared size of {} "
                        "bytes; refusing it mid-stream after {} bytes.",
                        url,
                        max_bytes,
                        written,
                    )
                    raise SnapshotIntegrityError(
                        f"Release asset at {url} exceeds the {max_bytes}-byte "
                        "size its manifest declared."
                    )
                handle.write(chunk)
        logger.debug("Wrote {} bytes from {} to {}.", written, url, destination)
