"""Tests for the Duration value type and the compact duration grammar."""

import re
import unittest

from ohlc_toolkit.temporal.duration import (
    Duration,
    coerce_duration,
    validate_cadence,
    validate_window_duration,
)
from ohlc_toolkit.temporal.errors import ConfigError

# [0-9] rather than \d: the helper's output contract is ASCII digits only,
# and \d would let a non-ASCII Unicode digit slip through the assertion.
INDEX_COUNT_PATTERN = re.compile(r"[0-9]+i")


class TestDurationConstruction(unittest.TestCase):
    """Test cases for constructing Duration values."""

    def test_constructs_from_non_negative_integer_seconds(self):
        """A Duration stores exact integer seconds."""
        self.assertEqual(Duration(5400).total_seconds, 5400)

    def test_zero_is_a_valid_duration(self):
        """The zero duration is representable and constructs cleanly."""
        duration = Duration(0)
        self.assertEqual(duration.total_seconds, 0)
        self.assertEqual(str(duration), "0s")

    def test_negative_seconds_raise_config_error(self):
        """A negative second count is never a valid duration."""
        with self.assertRaises(ConfigError):
            Duration(-1)

    def test_non_integer_seconds_raise_config_error(self):
        """A fractional second count is rejected: seconds are the atomic unit."""
        with self.assertRaises(ConfigError):
            Duration(1.5)  # type: ignore[arg-type]

    def test_bool_seconds_raise_config_error(self):
        """A bool is an int subtype in Python but is not a valid duration value."""
        with self.assertRaises(ConfigError):
            Duration(True)  # type: ignore[arg-type]
        with self.assertRaises(ConfigError):
            Duration(False)  # type: ignore[arg-type]


class TestDurationEqualityOrderingAndHashing(unittest.TestCase):
    """Test cases for comparing and hashing Duration values."""

    def test_equal_second_counts_are_equal(self):
        """Two Durations with the same seconds compare equal."""
        self.assertEqual(Duration(60), Duration(60))

    def test_different_second_counts_are_not_equal(self):
        """Two Durations with different seconds compare unequal."""
        self.assertNotEqual(Duration(60), Duration(61))

    def test_total_ordering_matches_total_seconds(self):
        """Durations sort the same way their total_seconds would."""
        self.assertLess(Duration(60), Duration(120))
        self.assertLessEqual(Duration(60), Duration(60))
        self.assertGreater(Duration(120), Duration(60))
        self.assertGreaterEqual(Duration(60), Duration(60))

    def test_durations_sort_as_expected(self):
        """Sorting a list of Durations orders by total_seconds."""
        durations = [Duration(90), Duration(0), Duration(3600)]
        self.assertEqual(sorted(durations), [Duration(0), Duration(90), Duration(3600)])

    def test_durations_are_hashable_and_usable_as_dict_keys(self):
        """Duration is hashable, so it can key a dict or live in a set."""
        mapping = {Duration(60): "one minute"}
        self.assertEqual(mapping[Duration(60)], "one minute")
        self.assertIn(Duration(60), {Duration(60), Duration(120)})


class TestDurationRepr(unittest.TestCase):
    """Test cases for Duration's debug representation."""

    def test_repr_is_useful_and_identifies_the_type(self):
        """repr() names the type and shows the underlying seconds."""
        text = repr(Duration(60))
        self.assertIn("Duration", text)
        self.assertIn("60", text)


class TestDurationFormatting(unittest.TestCase):
    """Test cases for str(Duration(...)) canonical formatting."""

    def test_str_examples(self):
        """str() emits the greedy canonical decomposition, largest unit first."""
        cases = {
            5400: "1h30m",
            90000: "1d1h",
            60: "1m",
            0: "0s",
        }
        for seconds, expected in cases.items():
            with self.subTest(seconds=seconds):
                self.assertEqual(str(Duration(seconds)), expected)

    def test_zero_components_are_omitted(self):
        """A duration with a zero-valued unit does not print that unit."""
        one_day_and_one_second = 86400 + 1
        self.assertEqual(str(Duration(one_day_and_one_second)), "1d1s")


class TestDurationParsing(unittest.TestCase):
    """Test cases for Duration.parse and the compact duration grammar."""

    def test_parses_a_multi_component_duration(self):
        """'1h30m' parses to 5400 seconds."""
        self.assertEqual(Duration.parse("1h30m"), Duration(5400))

    def test_parses_every_supported_unit(self):
        """Each single-unit component parses to the documented seconds value."""
        cases = {
            "1w": 604800,
            "1d": 86400,
            "1h": 3600,
            "1m": 60,
            "1s": 1,
        }
        for text, expected_seconds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(Duration.parse(text), Duration(expected_seconds))

    def test_grammar_valid_non_canonical_magnitudes_normalize(self):
        """Magnitudes that exceed the next unit's threshold still normalize."""
        cases = {
            "90m": 90 * 60,
            "60s": 60,
            "0m": 0,
        }
        for text, expected_seconds in cases.items():
            with self.subTest(text=text):
                self.assertEqual(Duration.parse(text), Duration(expected_seconds))

    def test_non_string_input_raises_config_error(self):
        """Duration.parse only ever accepts str."""
        for value in (123, 1.5, None, Duration(60)):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    Duration.parse(value)  # type: ignore[arg-type]

    def test_rejects_out_of_order_units(self):
        """Units must appear in strictly descending order: w>d>h>m>s."""
        with self.assertRaises(ConfigError):
            Duration.parse("30m1h")

    def test_rejects_duplicate_units(self):
        """Each unit may appear at most once."""
        with self.assertRaises(ConfigError):
            Duration.parse("1m1m")

    def test_rejects_bare_integers(self):
        """A number with no unit suffix is not a duration."""
        with self.assertRaises(ConfigError):
            Duration.parse("90")

    def test_rejects_calendar_and_unknown_units(self):
        """Calendar months and any unit outside w/d/h/m/s are rejected."""
        for text in ("1mo", "3M", "1y"):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    Duration.parse(text)

    def test_rejects_empty_and_whitespace_only_strings(self):
        """An empty or blank string has no components to parse."""
        for text in ("", "   "):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    Duration.parse(text)

    def test_rejects_signed_values(self):
        """Signs are not part of the grammar; durations are unsigned."""
        for text in ("+1h", "-1h"):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    Duration.parse(text)

    def test_rejects_decimal_magnitudes(self):
        """Fractional magnitudes are not part of the grammar."""
        with self.assertRaises(ConfigError):
            Duration.parse("1.5h")

    def test_rejects_internal_leading_and_trailing_whitespace(self):
        """Whitespace anywhere in the string is rejected."""
        for text in (" 1h", "1h ", "1h 30m"):
            with self.subTest(text=text):
                with self.assertRaises(ConfigError):
                    Duration.parse(text)

    def test_round_trip_examples(self):
        """Duration.parse(str(d)) == d for representative durations."""
        for seconds in (0, 1, 60, 5400, 90000, 604800):
            with self.subTest(seconds=seconds):
                duration = Duration(seconds)
                self.assertEqual(Duration.parse(str(duration)), duration)


class TestCoerceDuration(unittest.TestCase):
    """Test cases for the Duration | str boundary coercion helper."""

    def test_returns_a_duration_unchanged(self):
        """An existing Duration passes through untouched."""
        duration = Duration(60)
        self.assertIs(coerce_duration(duration), duration)

    def test_parses_a_string(self):
        """A string is strictly parsed into a Duration."""
        self.assertEqual(coerce_duration("1m"), Duration(60))

    def test_rejects_an_invalid_string(self):
        """An invalid string still raises ConfigError through coercion."""
        with self.assertRaises(ConfigError):
            coerce_duration("90")

    def test_rejects_unsupported_types(self):
        """Only Duration and str are accepted at this boundary."""
        for value in (60, 1.5, None, [60]):
            with self.subTest(value=value):
                with self.assertRaises(ConfigError):
                    coerce_duration(value)  # type: ignore[arg-type]


class TestValidateWindowDurationAndCadence(unittest.TestCase):
    """Test cases for the strictly-positive validators."""

    def test_validate_window_duration_accepts_duration_and_string(self):
        """Both a Duration and an equivalent string are accepted."""
        self.assertEqual(validate_window_duration(Duration(60)), Duration(60))
        self.assertEqual(validate_window_duration("1m"), Duration(60))

    def test_validate_window_duration_rejects_zero(self):
        """Zero window durations are never valid, in either input form."""
        with self.assertRaises(ConfigError):
            validate_window_duration(Duration(0))
        with self.assertRaises(ConfigError):
            validate_window_duration("0s")

    def test_validate_cadence_accepts_duration_and_string(self):
        """Both a Duration and an equivalent string are accepted."""
        self.assertEqual(validate_cadence(Duration(60)), Duration(60))
        self.assertEqual(validate_cadence("1m"), Duration(60))

    def test_validate_cadence_rejects_zero(self):
        """Zero cadences are never valid, in either input form."""
        with self.assertRaises(ConfigError):
            validate_cadence(Duration(0))
        with self.assertRaises(ConfigError):
            validate_cadence("0s")


class TestPolarsIndexCountHelper(unittest.TestCase):
    """Test cases for the polars index-count interop helper."""

    def test_examples(self):
        """Known durations format as the documented index-count strings."""
        self.assertEqual(Duration.parse("1h").to_polars_index_count(), "3600i")
        self.assertEqual(Duration.parse("1w").to_polars_index_count(), "604800i")

    def test_output_always_fullmatches_digits_then_i(self):
        """The helper never emits a calendar-aware unit suffix."""
        for seconds in (0, 1, 60, 5400, 90000, 604800):
            with self.subTest(seconds=seconds):
                text = Duration(seconds).to_polars_index_count()
                self.assertTrue(INDEX_COUNT_PATTERN.fullmatch(text))


class TestNonAsciiDigitRejection(unittest.TestCase):
    r"""Test cases rejecting non-ASCII Unicode digits in duration strings.

    Python's ``\d`` matches every Unicode decimal digit and ``int()``
    happily converts them, so a grammar built on ``\d`` would accept
    inputs like Arabic-Indic or fullwidth digits that the canonical
    formatter can never emit — and therefore can never round-trip.
    """

    def test_arabic_indic_digit_is_rejected(self):
        """An Arabic-Indic digit (U+0661) is not part of the grammar."""
        with self.assertRaises(ConfigError):
            Duration.parse("\u0661h")

    def test_thai_digit_is_rejected(self):
        """A Thai digit (U+0E55) is not part of the grammar."""
        with self.assertRaises(ConfigError):
            Duration.parse("\u0e55m")

    def test_fullwidth_digit_is_rejected(self):
        """A fullwidth digit (U+FF11) is not part of the grammar."""
        with self.assertRaises(ConfigError):
            Duration.parse("\uff11h")

    def test_mixed_ascii_and_extended_arabic_digit_is_rejected(self):
        """A non-ASCII digit is rejected even mixed in with ASCII ones."""
        with self.assertRaises(ConfigError):
            Duration.parse("1\u06f0m")


class TestRejectedInputQuotingIsBounded(unittest.TestCase):
    """Test cases bounding how much rejected input is echoed back."""

    def test_error_message_for_huge_garbage_input_is_bounded(self):
        """A rejected megabyte string must not produce a megabyte message."""
        garbage = "x" * 10_000
        with self.assertRaises(ConfigError) as ctx:
            Duration.parse(garbage)
        self.assertLess(len(str(ctx.exception)), 200)


if __name__ == "__main__":
    unittest.main()
