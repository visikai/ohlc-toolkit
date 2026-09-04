"""Tests for the HTTP transport that streams one release asset to disk.

The transport is the only part of this subpackage that touches the
network, so it is also the only part these tests stub out. Nothing here
opens a socket: ``requests.get`` is replaced with a fake that replays a
fixed list of chunks, which is enough to pin the two properties that
matter -- the body reaches the destination unchanged, and a body that
outgrows its declared size is cut off rather than written out.
"""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests

from ohlc_toolkit.snapshot.errors import SnapshotIntegrityError
from ohlc_toolkit.snapshot.transport import (
    DEFAULT_CHUNK_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    HttpAssetTransport,
)
from ohlc_toolkit.temporal import ConfigError

_URL = "https://example.invalid/owner/name/releases/download/tag/asset.csv.gz"
_BODY_CHUNKS = [b"first-", b"second-", b"third"]
_BODY = b"".join(_BODY_CHUNKS)
_GENEROUS_CAP = 1024
_NOT_FOUND = 404
_CONSUMED_CHUNKS_BEFORE_REFUSAL = 2
_CONFIGURED_TIMEOUT_SECONDS = 5.0
_SUCCESS = 200


class _FakeResponse:
    """A stand-in for a streamed ``requests`` response."""

    def __init__(self, chunks: list[bytes], status_code: int = _SUCCESS) -> None:
        self._chunks = chunks
        self.status_code = status_code
        self.consumed: list[bytes] = []

    def __enter__(self) -> Any:
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def iter_content(self, chunk_size: int) -> Any:
        """Replay the fixed chunks, recording what the caller actually took."""
        assert chunk_size > 0
        for chunk in self._chunks:
            self.consumed.append(chunk)
            yield chunk


def _patch_get(response: object) -> Any:
    """Patch the transport's ``requests.get`` to return a prepared response."""
    return patch("ohlc_toolkit.snapshot.transport.requests.get", return_value=response)


def test_the_streamed_body_lands_at_the_destination(tmp_path: Path) -> None:
    """Every chunk is written, in order, with nothing added or dropped."""
    destination = tmp_path / "asset.csv.gz"
    response = _FakeResponse(list(_BODY_CHUNKS))

    with _patch_get(response):
        HttpAssetTransport().download(_URL, destination, max_bytes=_GENEROUS_CAP)

    assert destination.read_bytes() == _BODY


def test_the_request_is_streamed_and_bounded_by_the_configured_timeout(
    tmp_path: Path,
) -> None:
    """An unbounded request is the failure mode this transport must not have."""
    destination = tmp_path / "asset.csv.gz"
    response = _FakeResponse(list(_BODY_CHUNKS))

    with _patch_get(response) as fake_get:
        HttpAssetTransport(timeout_seconds=_CONFIGURED_TIMEOUT_SECONDS).download(
            _URL, destination, max_bytes=_GENEROUS_CAP
        )

    assert fake_get.call_args.kwargs["stream"] is True
    assert fake_get.call_args.kwargs["timeout"] == _CONFIGURED_TIMEOUT_SECONDS


def test_a_non_success_status_is_refused(tmp_path: Path) -> None:
    """An asset the release does not serve fails as an integrity problem."""
    destination = tmp_path / "asset.csv.gz"
    response = _FakeResponse([], status_code=_NOT_FOUND)

    with _patch_get(response), pytest.raises(SnapshotIntegrityError, match="404"):
        HttpAssetTransport().download(_URL, destination, max_bytes=_GENEROUS_CAP)

    assert not destination.exists()


def test_a_body_larger_than_its_declared_size_is_refused(tmp_path: Path) -> None:
    """The manifest's declared size is an enforced cap, not a hint."""
    destination = tmp_path / "asset.csv.gz"
    response = _FakeResponse(list(_BODY_CHUNKS))

    with (
        _patch_get(response),
        pytest.raises(SnapshotIntegrityError, match="exceeds"),
    ):
        HttpAssetTransport().download(_URL, destination, max_bytes=len(_BODY) - 1)


def test_the_cap_stops_the_stream_rather_than_checking_it_afterwards(
    tmp_path: Path,
) -> None:
    """A hostile body is cut off mid-stream, not downloaded then measured."""
    destination = tmp_path / "asset.csv.gz"
    response = _FakeResponse(list(_BODY_CHUNKS))
    # A cap that the first two chunks together overshoot, so a transport
    # that only checked at the end would have taken all three.
    cap = len(_BODY_CHUNKS[0]) + 1

    with _patch_get(response), pytest.raises(SnapshotIntegrityError):
        HttpAssetTransport().download(_URL, destination, max_bytes=cap)

    assert len(response.consumed) == _CONSUMED_CHUNKS_BEFORE_REFUSAL


def test_a_request_failure_is_refused_as_an_integrity_failure(
    tmp_path: Path,
) -> None:
    """A network error is this package's error, not a bare requests error."""
    destination = tmp_path / "asset.csv.gz"

    with (
        patch(
            "ohlc_toolkit.snapshot.transport.requests.get",
            side_effect=requests.ConnectionError("no route"),
        ),
        pytest.raises(SnapshotIntegrityError, match="could not be fetched"),
    ):
        HttpAssetTransport().download(_URL, destination, max_bytes=_GENEROUS_CAP)


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0], ids=["zero", "negative"])
def test_a_non_positive_timeout_is_refused(timeout_seconds: float) -> None:
    """A transport without a positive timeout has no bound at all."""
    with pytest.raises(ConfigError, match="timeout_seconds"):
        HttpAssetTransport(timeout_seconds=timeout_seconds)


@pytest.mark.parametrize("chunk_bytes", [0, -1], ids=["zero", "negative"])
def test_a_non_positive_chunk_size_is_refused(chunk_bytes: int) -> None:
    """A non-positive chunk size cannot make progress through a stream."""
    with pytest.raises(ConfigError, match="chunk_bytes"):
        HttpAssetTransport(chunk_bytes=chunk_bytes)


def test_the_defaults_are_positive() -> None:
    """The shipped defaults satisfy the transport's own invariants."""
    transport = HttpAssetTransport()

    assert transport.timeout_seconds == DEFAULT_TIMEOUT_SECONDS
    assert transport.chunk_bytes == DEFAULT_CHUNK_BYTES
    assert DEFAULT_TIMEOUT_SECONDS > 0
    assert DEFAULT_CHUNK_BYTES > 0


if __name__ == "__main__":
    pytest.main([__file__])
