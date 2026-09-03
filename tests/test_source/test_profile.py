"""Tests for source profiles and the Bitstamp minute-data profile."""

import unittest

import polars as pl

from ohlc_toolkit.source.profile import (
    BITSTAMP_BTCUSD_1M,
    Availability,
    ColumnKind,
    SourceProfile,
)
from ohlc_toolkit.temporal import ConfigError, Duration


def _make_profile(**overrides: object) -> SourceProfile:
    """Build a minimal valid SourceProfile, overriding selected fields."""
    fields: dict[str, object] = {
        "name": "test-source-1m",
        "cadence": "1m",
        "timestamp_column": "timestamp",
        "availability": Availability.CLOSE_TIME,
        "raw_schema": {
            "timestamp": ColumnKind.INTEGER,
            "open": ColumnKind.FLOATING,
        },
    }
    fields.update(overrides)
    return SourceProfile(**fields)  # type: ignore[arg-type]


class TestSourceProfileConstruction(unittest.TestCase):
    """Test cases for constructing a SourceProfile."""

    def test_accepts_a_duration_instance_for_cadence(self):
        """A pre-built Duration is accepted and stored as-is."""
        profile = _make_profile(cadence=Duration.parse("1m"))
        self.assertEqual(profile.cadence, Duration.parse("1m"))

    def test_accepts_a_duration_string_for_cadence(self):
        """A compact duration string is coerced into a Duration."""
        profile = _make_profile(cadence="1m")
        self.assertEqual(profile.cadence, Duration(60))

    def test_rejects_a_zero_cadence(self):
        """A zero-length cadence never advances and is rejected."""
        with self.assertRaises(ConfigError):
            _make_profile(cadence="0s")

    def test_rejects_an_invalid_cadence_string(self):
        """A malformed duration string fails at construction."""
        with self.assertRaises(ConfigError):
            _make_profile(cadence="not-a-duration")

    def test_rejects_an_empty_name(self):
        """An empty source name is never valid."""
        with self.assertRaises(ConfigError):
            _make_profile(name="")

    def test_rejects_an_empty_timestamp_column(self):
        """An empty timestamp column name is never valid."""
        with self.assertRaises(ConfigError):
            _make_profile(timestamp_column="")

    def test_rejects_an_empty_raw_schema(self):
        """A profile must declare at least one raw column."""
        with self.assertRaises(ConfigError):
            _make_profile(raw_schema={})

    def test_rejects_a_timestamp_column_absent_from_raw_schema(self):
        """The timestamp column must itself be a declared raw column."""
        with self.assertRaises(ConfigError):
            _make_profile(
                timestamp_column="open_time",
                raw_schema={"timestamp": ColumnKind.INTEGER},
            )

    def test_profile_is_frozen(self):
        """A constructed profile cannot be mutated."""
        profile = _make_profile()
        with self.assertRaises(Exception):  # noqa: B017 - dataclasses.FrozenInstanceError
            profile.name = "changed"  # type: ignore[misc]


class TestSourceProfileDerivation(unittest.TestCase):
    """Test cases for deriving half-open interval bounds from a raw frame."""

    def test_derives_open_and_close_time_for_a_single_row(self):
        """A single row derives its open/close time from timestamp and cadence."""
        profile = _make_profile(cadence="1m")
        raw = pl.DataFrame({"timestamp": [1420070400], "open": [1.0]})

        bounds = profile.derive_interval_bounds(raw)

        self.assertEqual(bounds.get_column("open_time").to_list(), [1420070400])
        self.assertEqual(bounds.get_column("close_time").to_list(), [1420070460])

    def test_derived_columns_are_int64(self):
        """Derived open_time/close_time are always int64, regardless of input."""
        profile = _make_profile(cadence="1m")
        raw = pl.DataFrame(
            {"timestamp": pl.Series([1420070400], dtype=pl.Int32), "open": [1.0]}
        )

        bounds = profile.derive_interval_bounds(raw)

        self.assertEqual(bounds.schema["open_time"], pl.Int64)
        self.assertEqual(bounds.schema["close_time"], pl.Int64)

    def test_derives_bounds_for_multiple_rows_vectorized(self):
        """Multiple rows each derive their own half-open interval."""
        profile = _make_profile(cadence="1m")
        raw = pl.DataFrame({"timestamp": [0, 60, 120], "open": [1.0, 2.0, 3.0]})

        bounds = profile.derive_interval_bounds(raw)

        self.assertEqual(bounds.get_column("open_time").to_list(), [0, 60, 120])
        self.assertEqual(bounds.get_column("close_time").to_list(), [60, 120, 180])


class TestBitstampProfile(unittest.TestCase):
    """Test cases for the shipped Bitstamp BTC/USD 1-minute profile."""

    def test_name_and_cadence(self):
        """The profile identifies itself and carries a 1-minute cadence."""
        self.assertEqual(BITSTAMP_BTCUSD_1M.name, "bitstamp-btcusd-1m")
        self.assertEqual(BITSTAMP_BTCUSD_1M.cadence, Duration.parse("1m"))

    def test_timestamp_column_and_availability(self):
        """The raw timestamp column is the interval open; availability is close-time."""
        self.assertEqual(BITSTAMP_BTCUSD_1M.timestamp_column, "timestamp")
        self.assertEqual(BITSTAMP_BTCUSD_1M.availability, Availability.CLOSE_TIME)

    def test_raw_schema_matches_the_published_six_columns(self):
        """The declared raw schema matches Bitstamp's public six-column layout."""
        self.assertEqual(
            BITSTAMP_BTCUSD_1M.raw_schema,
            {
                "timestamp": ColumnKind.INTEGER,
                "open": ColumnKind.FLOATING,
                "high": ColumnKind.FLOATING,
                "low": ColumnKind.FLOATING,
                "close": ColumnKind.FLOATING,
                "volume": ColumnKind.FLOATING,
            },
        )

    def test_derivation_example(self):
        """A known timestamp derives the documented open/close example."""
        raw = pl.DataFrame({"timestamp": [1420070400]})

        bounds = BITSTAMP_BTCUSD_1M.derive_interval_bounds(raw)

        self.assertEqual(bounds.get_column("open_time").to_list(), [1420070400])
        self.assertEqual(bounds.get_column("close_time").to_list(), [1420070460])


class TestSourceProfilePhase(unittest.TestCase):
    """Test cases for the profile's declared cadence-grid phase."""

    def test_phase_defaults_to_zero(self):
        """A profile that declares no phase sits on the plain epoch grid."""
        self.assertEqual(_make_profile().phase, Duration(0))

    def test_phase_accepts_a_string_and_normalizes_to_duration(self):
        """The constructor boundary accepts a compact duration string."""
        self.assertEqual(_make_profile(phase="30s").phase, Duration(30))

    def test_phase_must_be_smaller_than_the_cadence(self):
        """A phase at or beyond the cadence is not a grid offset."""
        for phase in ("1m", "2m"):
            with self.subTest(phase=phase):
                with self.assertRaises(ConfigError):
                    _make_profile(phase=phase)

    def test_bitstamp_profile_declares_the_zero_phase(self):
        """The published minute grid sits on round minute boundaries."""
        self.assertEqual(BITSTAMP_BTCUSD_1M.phase, Duration(0))


if __name__ == "__main__":
    unittest.main()
