"""Tests for strict parsing of a release's JSON manifest.

The manifest is the only statement of what a release contains and what it
should hash to. Everything downstream trusts it, so nothing here is
lenient: a manifest that is short one key, one version ahead, or one
character wrong in a digest is refused outright rather than partially
believed.
"""

from copy import deepcopy
from datetime import UTC, datetime
from typing import Any

import pytest
from ohlc_toolkit.snapshot.errors import SnapshotManifestError
from ohlc_toolkit.snapshot.manifest import (
    MANIFEST_ASSET_NAME,
    MAX_MANIFEST_BYTES,
    SUPPORTED_SCHEMA_VERSION,
    AssetRecord,
    SnapshotManifest,
    parse_manifest,
)

from ohlc_toolkit.temporal import DataValidationError
from tests.test_snapshot.factories import (
    FIXTURE_REVISION,
    FIXTURE_TAG,
    HISTORY_ASSET,
    PARQUET_ASSET,
    PROVENANCE_ASSET,
    build_default_assets,
    build_manifest_payload,
    encode_manifest,
    history_timestamps,
    sha256_hex,
)

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

_EXPECTED_ASSET_COUNT = 3
_EXPECTED_ROW_COUNT = 10
_EXPECTED_AS_OF = datetime(2026, 9, 1, 2, 7, 12, tzinfo=UTC)


def _payload() -> dict[str, Any]:
    """Build a truthful manifest payload over the default fixture assets."""
    return build_manifest_payload(
        assets=build_default_assets(), timestamps=history_timestamps()
    )


def _parse(payload: dict[str, Any]) -> SnapshotManifest:
    """Encode and parse a payload in one step."""
    return parse_manifest(encode_manifest(payload))


def _mutated(**changes: Any) -> dict[str, Any]:
    """Return the default payload with top-level keys replaced."""
    payload = _payload()
    payload.update(changes)
    return payload


def test_a_truthful_manifest_parses_into_every_declared_field() -> None:
    """Each statement the release makes survives parsing with its own type."""
    manifest = _parse(_payload())

    assert manifest.schema_version == SUPPORTED_SCHEMA_VERSION
    assert manifest.tag == FIXTURE_TAG
    assert manifest.as_of == _EXPECTED_AS_OF
    assert manifest.first_timestamp == history_timestamps()[0]
    assert manifest.last_timestamp == history_timestamps()[-1]
    assert manifest.row_count == _EXPECTED_ROW_COUNT
    assert manifest.generation_revision == FIXTURE_REVISION


def test_every_declared_asset_is_carried_with_its_size_and_digest() -> None:
    """Asset records repeat exactly what the manifest declared."""
    assets = build_default_assets()

    manifest = _parse(_payload())

    assert sorted(manifest.assets) == sorted(assets)
    assert len(manifest.assets) == _EXPECTED_ASSET_COUNT
    for name, payload in assets.items():
        record = manifest.assets[name]
        assert record.name == name
        assert record.size_bytes == len(payload)
        assert record.sha256 == sha256_hex(payload)


def test_the_asset_mapping_is_read_only() -> None:
    """A caller cannot add an asset to a parsed manifest after the fact."""
    manifest = _parse(_payload())

    with pytest.raises(TypeError):
        manifest.assets["injected"] = manifest.assets[HISTORY_ASSET]  # type: ignore[index]


def test_the_manifest_is_frozen() -> None:
    """A parsed manifest cannot be edited into disagreeing with the release."""
    manifest = _parse(_payload())

    with pytest.raises(AttributeError):
        manifest.row_count = 1  # type: ignore[misc]


@pytest.mark.parametrize("key", _REQUIRED_KEYS)
def test_a_manifest_missing_any_required_key_is_refused(key: str) -> None:
    """Every declared key is required; none of them defaults."""
    payload = _payload()
    del payload[key]

    with pytest.raises(SnapshotManifestError, match=key):
        _parse(payload)


def test_a_future_schema_version_is_refused_rather_than_guessed_at() -> None:
    """A manifest one version ahead is not read with today's assumptions."""
    payload = _mutated(schema_version=SUPPORTED_SCHEMA_VERSION + 1)

    with pytest.raises(SnapshotManifestError, match="schema_version"):
        _parse(payload)


def test_undecodable_bytes_are_refused() -> None:
    """Bytes that are not JSON at all fail as a manifest problem, not a ValueError."""
    with pytest.raises(SnapshotManifestError, match="not valid JSON"):
        parse_manifest(b"{ this is not json")


def test_a_json_document_that_is_not_an_object_is_refused() -> None:
    """A JSON array parses as JSON but is not a manifest."""
    with pytest.raises(SnapshotManifestError, match="JSON object"):
        parse_manifest(b"[1, 2, 3]")


def test_an_oversized_payload_is_refused_before_it_is_parsed() -> None:
    """The manifest has an explicit size cap; the published one is under 1 KiB."""
    oversized = b" " * (MAX_MANIFEST_BYTES + 1)

    with pytest.raises(SnapshotManifestError, match="exceeds"):
        parse_manifest(oversized)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("tag", ""),
        ("tag", 7),
        ("generation_revision", ""),
        ("generation_revision", None),
        ("schema_version", "1"),
        ("row_count", "10"),
        ("row_count", 0),
        ("row_count", -1),
        ("first_timestamp", 1.5),
        ("as_of", ""),
        ("as_of", "not a timestamp"),
        ("as_of", 1_756_692_432),
    ],
    ids=[
        "empty_tag",
        "numeric_tag",
        "empty_revision",
        "null_revision",
        "string_schema_version",
        "string_row_count",
        "zero_row_count",
        "negative_row_count",
        "float_first_timestamp",
        "empty_as_of",
        "unparseable_as_of",
        "numeric_as_of",
    ],
)
def test_a_malformed_scalar_field_is_refused(key: str, value: Any) -> None:
    """Each scalar statement is checked for type and for sense, not just presence."""
    with pytest.raises(SnapshotManifestError, match=key):
        _parse(_mutated(**{key: value}))


def test_a_boolean_is_not_accepted_where_an_integer_is_declared() -> None:
    """``True`` is an int in Python; a manifest saying ``true`` is still wrong."""
    with pytest.raises(SnapshotManifestError, match="row_count"):
        _parse(_mutated(row_count=True))


def test_a_naive_as_of_is_refused() -> None:
    """An as-of time without an offset does not identify an instant."""
    with pytest.raises(SnapshotManifestError, match="as_of"):
        _parse(_mutated(as_of="2026-09-01T02:07:12"))


def test_a_last_timestamp_before_the_first_is_refused() -> None:
    """A manifest cannot describe a history that ends before it starts."""
    timestamps = history_timestamps()
    payload = _mutated(last_timestamp=timestamps[0] - 1)

    with pytest.raises(SnapshotManifestError, match="last_timestamp"):
        _parse(payload)


def test_a_single_row_history_is_accepted() -> None:
    """First and last may coincide: a one-row history is legal, not a mistake."""
    timestamps = history_timestamps(rows=1)
    payload = build_manifest_payload(
        assets=build_default_assets(), timestamps=timestamps
    )

    manifest = _parse(payload)

    assert manifest.first_timestamp == manifest.last_timestamp
    assert manifest.row_count == 1


def test_a_manifest_declaring_no_assets_is_refused() -> None:
    """A release with nothing in it is a broken publish, not an empty success."""
    with pytest.raises(SnapshotManifestError, match="assets"):
        _parse(_mutated(assets={}))


def test_an_assets_value_that_is_not_an_object_is_refused() -> None:
    """The assets field is a mapping of name to record, not a list."""
    with pytest.raises(SnapshotManifestError, match="assets"):
        _parse(_mutated(assets=[HISTORY_ASSET]))


@pytest.mark.parametrize(
    "entry",
    [
        {"sha256": "0" * 64},
        {"bytes": 10},
        {"bytes": 10, "sha256": "0" * 63},
        {"bytes": 10, "sha256": "0" * 65},
        {"bytes": 10, "sha256": "Z" * 64},
        {"bytes": 10, "sha256": "A" * 64},
        {"bytes": 10, "sha256": 12345},
        {"bytes": 0, "sha256": "0" * 64},
        {"bytes": -1, "sha256": "0" * 64},
        {"bytes": "10", "sha256": "0" * 64},
        {"bytes": True, "sha256": "0" * 64},
    ],
    ids=[
        "missing_bytes",
        "missing_sha256",
        "short_digest",
        "long_digest",
        "non_hex_digest",
        "uppercase_digest",
        "numeric_digest",
        "zero_bytes",
        "negative_bytes",
        "string_bytes",
        "boolean_bytes",
    ],
)
def test_a_malformed_asset_record_is_refused(entry: dict[str, Any]) -> None:
    """An asset this package cannot verify byte-for-byte is not accepted."""
    payload = _payload()
    payload["assets"][HISTORY_ASSET] = entry

    with pytest.raises(SnapshotManifestError, match=HISTORY_ASSET):
        _parse(payload)


def test_an_asset_entry_that_is_not_an_object_is_refused() -> None:
    """Each asset entry is a record; a bare digest string is not one."""
    payload = _payload()
    payload["assets"][HISTORY_ASSET] = "0" * 64

    with pytest.raises(SnapshotManifestError, match=HISTORY_ASSET):
        _parse(payload)


@pytest.mark.parametrize(
    "name",
    [
        "../escaped.csv.gz",
        "nested/asset.csv.gz",
        "back\\slash.csv.gz",
        ".hidden.csv.gz",
        "/absolute.csv.gz",
        "",
        "spaced name.csv.gz",
        "x" * 200,
    ],
    ids=[
        "parent_traversal",
        "subdirectory",
        "backslash",
        "leading_dot",
        "absolute",
        "empty",
        "space",
        "oversized",
    ],
)
def test_an_asset_name_that_is_not_a_plain_filename_is_refused(name: str) -> None:
    """A manifest cannot steer a write outside the caller's directory."""
    payload = _payload()
    entry = payload["assets"].pop(HISTORY_ASSET)
    payload["assets"][name] = entry

    with pytest.raises(SnapshotManifestError, match="asset name"):
        _parse(payload)


def test_the_manifest_asset_is_not_itself_declarable() -> None:
    """The manifest cannot list itself: nothing could verify that digest."""
    payload = _payload()
    payload["assets"][MANIFEST_ASSET_NAME] = payload["assets"][HISTORY_ASSET]

    with pytest.raises(SnapshotManifestError, match=MANIFEST_ASSET_NAME):
        _parse(payload)


def test_parsing_does_not_mutate_the_caller_visible_payload() -> None:
    """Parsing reads; it never edits the bytes or the structure it was handed."""
    payload = _payload()
    before = deepcopy(payload)

    _parse(payload)

    assert payload == before


def test_an_asset_record_is_frozen() -> None:
    """A verified digest cannot be edited after the record is built."""
    record = AssetRecord(name=PARQUET_ASSET, size_bytes=1, sha256="0" * 64)

    with pytest.raises(AttributeError):
        record.sha256 = "1" * 64  # type: ignore[misc]


def test_an_asset_record_validates_its_own_digest() -> None:
    """Constructing a record directly is held to the same digest grammar."""
    with pytest.raises(SnapshotManifestError, match="sha256"):
        AssetRecord(name=PROVENANCE_ASSET, size_bytes=1, sha256="nope")


def test_manifest_failures_are_data_validation_errors() -> None:
    """A bad manifest is bad published data, catchable as such."""
    with pytest.raises(DataValidationError):
        parse_manifest(b"{}")


if __name__ == "__main__":
    pytest.main([__file__])
