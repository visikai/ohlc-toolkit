"""Tests for the identity of a published release and its asset URLs.

A release is named by a repository and a tag, and nothing else. The
grammar for both is deliberately narrow: a value outside it is refused
rather than URL-escaped, because escaping silently changes which release
you asked for.
"""

import pytest
from ohlc_toolkit.snapshot.manifest import MANIFEST_ASSET_NAME
from ohlc_toolkit.snapshot.release import (
    BITSTAMP_BTCUSD_1M_REPOSITORY,
    BITSTAMP_HISTORY_CSV_ASSET,
    DEFAULT_RELEASE_HOST,
    SnapshotRelease,
)

from ohlc_toolkit.temporal import ConfigError
from tests.test_snapshot.factories import (
    FIXTURE_HOST,
    FIXTURE_REPOSITORY,
    FIXTURE_TAG,
    HISTORY_ASSET,
)

_PUBLISHED_TAG = "bitstamp-btcusd-1m-2026-08"


def test_an_asset_url_is_the_canonical_release_download_url() -> None:
    """The URL is composed from the release identity, never discovered."""
    release = SnapshotRelease(
        repository=BITSTAMP_BTCUSD_1M_REPOSITORY, tag=_PUBLISHED_TAG
    )

    assert release.asset_url(BITSTAMP_HISTORY_CSV_ASSET) == (
        f"{DEFAULT_RELEASE_HOST}/{BITSTAMP_BTCUSD_1M_REPOSITORY}/releases/download/"
        f"{_PUBLISHED_TAG}/{BITSTAMP_HISTORY_CSV_ASSET}"
    )


def test_the_manifest_has_a_url_like_any_other_asset() -> None:
    """The manifest is fetched from the same release, by the same rule."""
    release = SnapshotRelease(repository=FIXTURE_REPOSITORY, tag=FIXTURE_TAG)

    assert release.asset_url(MANIFEST_ASSET_NAME).endswith(
        f"/{FIXTURE_TAG}/{MANIFEST_ASSET_NAME}"
    )


def test_a_custom_host_replaces_only_the_host() -> None:
    """Pointing at a mirror does not change the path layout."""
    release = SnapshotRelease(
        repository=FIXTURE_REPOSITORY, tag=FIXTURE_TAG, host=FIXTURE_HOST
    )

    assert release.asset_url(HISTORY_ASSET) == (
        f"{FIXTURE_HOST}/{FIXTURE_REPOSITORY}/releases/download/"
        f"{FIXTURE_TAG}/{HISTORY_ASSET}"
    )


def test_a_release_is_frozen() -> None:
    """A release identity cannot be edited after it is stated."""
    release = SnapshotRelease(repository=FIXTURE_REPOSITORY, tag=FIXTURE_TAG)

    with pytest.raises(AttributeError):
        release.tag = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "repository",
    ["", "noslash", "too/many/slashes", "/leading", "trailing/", "own er/repo"],
    ids=["empty", "no_slash", "extra_slash", "empty_owner", "empty_name", "space"],
)
def test_a_repository_outside_the_owner_name_grammar_is_refused(
    repository: str,
) -> None:
    """A repository is exactly one owner and one name."""
    with pytest.raises(ConfigError, match="repository"):
        SnapshotRelease(repository=repository, tag=FIXTURE_TAG)


@pytest.mark.parametrize(
    "tag",
    ["", "with/slash", "with space", "-leading-dash", "with\nnewline"],
    ids=["empty", "slash", "space", "leading_dash", "newline"],
)
def test_a_tag_outside_the_supported_grammar_is_refused(tag: str) -> None:
    """A tag that would need escaping is refused instead of escaped."""
    with pytest.raises(ConfigError, match="tag"):
        SnapshotRelease(repository=FIXTURE_REPOSITORY, tag=tag)


@pytest.mark.parametrize(
    "host",
    ["", "example.invalid", "ftp://example.invalid", "https://example.invalid/"],
    ids=["empty", "no_scheme", "wrong_scheme", "trailing_slash"],
)
def test_a_host_that_is_not_a_bare_http_origin_is_refused(host: str) -> None:
    """The host is an origin; path pieces belong to the URL builder."""
    with pytest.raises(ConfigError, match="host"):
        SnapshotRelease(repository=FIXTURE_REPOSITORY, tag=FIXTURE_TAG, host=host)


@pytest.mark.parametrize(
    "asset_name",
    ["", "../escape", "nested/asset", "back\\slash", ".hidden"],
    ids=["empty", "traversal", "subdirectory", "backslash", "leading_dot"],
)
def test_an_asset_name_that_is_not_a_plain_filename_has_no_url(
    asset_name: str,
) -> None:
    """A name that could reshape the URL path is refused at the boundary."""
    release = SnapshotRelease(repository=FIXTURE_REPOSITORY, tag=FIXTURE_TAG)

    with pytest.raises(ConfigError, match="asset name"):
        release.asset_url(asset_name)


def test_the_published_repository_constant_is_a_valid_release_identity() -> None:
    """The shipped constant is usable as-is, not a string needing repair."""
    release = SnapshotRelease(
        repository=BITSTAMP_BTCUSD_1M_REPOSITORY, tag=_PUBLISHED_TAG
    )

    assert release.repository == BITSTAMP_BTCUSD_1M_REPOSITORY
    assert release.host == DEFAULT_RELEASE_HOST


if __name__ == "__main__":
    pytest.main([__file__])
