"""Verify Polars deprecation warnings fail the configured test policy."""

import warnings

import polars as pl
import pytest


def test_polars_deprecation_errors_even_when_attributed_to_caller() -> None:
    """Catch the caller-attributed warning and prove the filter catches it."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        pl.enable_string_cache()

    assert len(caught) == 1
    assert caught[0].category is DeprecationWarning
    assert caught[0].filename == __file__

    with pytest.raises(DeprecationWarning, match="string cache has been replaced"):
        pl.enable_string_cache()
