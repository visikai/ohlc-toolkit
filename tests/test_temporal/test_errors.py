"""Tests for the temporal package's exception taxonomy."""

import unittest

from ohlc_toolkit.temporal.errors import (
    ConfigError,
    CoverageError,
    DataValidationError,
)


class TestErrorTaxonomy(unittest.TestCase):
    """Test cases for the temporal exception taxonomy."""

    def test_config_error_inherits_directly_from_exception(self):
        """ConfigError is a flat subclass of Exception."""
        self.assertEqual(ConfigError.__bases__, (Exception,))

    def test_data_validation_error_inherits_directly_from_exception(self):
        """DataValidationError is a flat subclass of Exception."""
        self.assertEqual(DataValidationError.__bases__, (Exception,))

    def test_coverage_error_inherits_directly_from_exception(self):
        """CoverageError is a flat subclass of Exception."""
        self.assertEqual(CoverageError.__bases__, (Exception,))

    def test_error_types_are_siblings_not_related_to_each_other(self):
        """The three error types share no subclass relationship."""
        error_types = (ConfigError, DataValidationError, CoverageError)
        for left in error_types:
            for right in error_types:
                if left is right:
                    continue
                with self.subTest(left=left, right=right):
                    self.assertFalse(issubclass(left, right))

    def test_each_error_can_be_raised_and_caught_with_a_message(self):
        """Each error type behaves like a normal Exception when raised."""
        for error_type in (ConfigError, DataValidationError, CoverageError):
            with self.subTest(error_type=error_type):
                with self.assertRaises(error_type) as raised:
                    raise error_type("boom")
                self.assertEqual(str(raised.exception), "boom")


if __name__ == "__main__":
    unittest.main()
