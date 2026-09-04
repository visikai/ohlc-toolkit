"""The identity of one published release, and where its assets live.

This module owns the grammar of a release's coordinates: the repository,
the tag, and the name of an asset within it. Each grammar is deliberately
narrow, and a value outside it is refused rather than escaped -- escaping
would silently change which release, or which file, was asked for.

An asset name in particular must be a plain filename. The same predicate
guards the names a manifest declares, so a manifest can never steer a
write outside the directory the caller named, nor a fetch outside the
release path.
"""

import re
from dataclasses import dataclass

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.snapshot.errors import bounded_echo
from ohlc_toolkit.temporal import ConfigError

logger = get_logger(__name__)

DEFAULT_RELEASE_HOST = "https://github.com"

# The public dataset this package's Bitstamp profile describes, and the
# six-column full-history asset within its monthly releases.
BITSTAMP_BTCUSD_1M_REPOSITORY = "ff137/bitstamp-btcusd-minute-data"
BITSTAMP_HISTORY_CSV_ASSET = "btcusd_bitstamp_1min.csv.gz"

# A filename cap, so a hostile manifest cannot make a log line or a path
# arbitrarily long.
MAX_ASSET_NAME_CHARS = 128

_REPOSITORY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*"
)
_TAG_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]*")
_HOST_PATTERN = re.compile(r"https?://[A-Za-z0-9][A-Za-z0-9.:-]*")
_ASSET_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def is_plain_asset_name(name: object) -> bool:
    """Report whether a name is a plain filename, safe to append to a path.

    A plain filename starts with an alphanumeric character and contains
    only alphanumerics, dots, underscores, and hyphens. That excludes the
    empty string, anything containing a path separator, anything starting
    with a dot, and anything long enough to be a denial-of-service in a
    log line.

    Args:
        name: The candidate name, from a manifest or from a caller.

    Returns:
        True if the name may be joined onto a directory or a URL path.

    """
    if not isinstance(name, str) or len(name) > MAX_ASSET_NAME_CHARS:
        return False
    return _ASSET_NAME_PATTERN.fullmatch(name) is not None


@dataclass(frozen=True)
class SnapshotRelease:
    """One named release of a published dataset repository.

    A release is identified by a repository and a tag, and nothing else.
    Asset URLs are composed from that identity rather than discovered, so
    fetching needs no API call, no token, and no listing step that could
    disagree with the manifest.

    Attributes:
        repository: The ``owner/name`` of the dataset repository.
        tag: The release tag, e.g. ``bitstamp-btcusd-1m-2026-08``.
        host: The origin serving the release, with a scheme and no
            trailing slash. Defaults to :data:`DEFAULT_RELEASE_HOST`;
            override it to read from a mirror.

    """

    repository: str
    tag: str
    host: str = DEFAULT_RELEASE_HOST

    def __post_init__(self) -> None:
        """Validate every coordinate against its grammar.

        Raises:
            ConfigError: If the repository is not ``owner/name``, the tag
                is outside the supported grammar, or the host is not a
                bare HTTP origin.

        """
        if not isinstance(self.repository, str) or not _REPOSITORY_PATTERN.fullmatch(
            self.repository
        ):
            logger.warning(
                "Rejecting release repository {}: not an 'owner/name' pair.",
                bounded_echo(self.repository),
            )
            raise ConfigError(
                f"Release repository must be an 'owner/name' pair, got "
                f"{bounded_echo(self.repository)}."
            )
        if not isinstance(self.tag, str) or not _TAG_PATTERN.fullmatch(self.tag):
            logger.warning(
                "Rejecting release tag {}: outside the supported grammar.",
                bounded_echo(self.tag),
            )
            raise ConfigError(
                f"Release tag must start with an alphanumeric and contain only "
                f"alphanumerics, '.', '_', '+', or '-', got "
                f"{bounded_echo(self.tag)}. A tag that would need escaping is "
                "refused rather than escaped, because escaping changes which "
                "release is fetched."
            )
        if not isinstance(self.host, str) or not _HOST_PATTERN.fullmatch(self.host):
            logger.warning(
                "Rejecting release host {}: not a bare http(s) origin.",
                bounded_echo(self.host),
            )
            raise ConfigError(
                f"Release host must be a bare http(s) origin with no trailing "
                f"slash, got {bounded_echo(self.host)}."
            )

    def asset_url(self, asset_name: str) -> str:
        """Compose the canonical download URL for one asset of this release.

        Args:
            asset_name: A plain filename published under this release,
                including ``manifest.json`` itself.

        Returns:
            The full download URL.

        Raises:
            ConfigError: If ``asset_name`` is not a plain filename.

        """
        if not is_plain_asset_name(asset_name):
            logger.warning(
                "Rejecting asset name {} for release {!r}.",
                bounded_echo(asset_name),
                self.tag,
            )
            raise ConfigError(
                f"Release asset name must be a plain filename, got "
                f"{bounded_echo(asset_name)}."
            )
        return (
            f"{self.host}/{self.repository}/releases/download/{self.tag}/{asset_name}"
        )
