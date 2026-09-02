"""Property-based tests for the Duration value type and duration grammar."""

from itertools import pairwise

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from ohlc_toolkit.temporal.duration import Duration
from ohlc_toolkit.temporal.errors import ConfigError

_UNITS_DESCENDING = ("w", "d", "h", "m", "s")
_UNIT_SECONDS = {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1}

_unit_subsets = st.lists(
    st.sampled_from(_UNITS_DESCENDING), min_size=1, max_size=5, unique=True
)

# The greedy canonical decomposition never lets a unit's magnitude reach the
# size of the next-larger unit (e.g. 7 "d" would actually be 1 "w"). "w" has
# no larger unit, so it is effectively unbounded.
_CANONICAL_MAGNITUDE_MAX = {"w": 999, "d": 6, "h": 23, "m": 59, "s": 59}


@st.composite
def _canonical_duration_strings(draw: st.DrawFn) -> str:
    """Build a canonical duration string: unique units, descending, no zeros."""
    ordered_units = sorted(draw(_unit_subsets), key=_UNITS_DESCENDING.index)
    magnitudes = [
        draw(st.integers(min_value=1, max_value=_CANONICAL_MAGNITUDE_MAX[unit]))
        for unit in ordered_units
    ]
    return "".join(
        f"{magnitude}{unit}"
        for magnitude, unit in zip(magnitudes, ordered_units, strict=True)
    )


@st.composite
def _grammar_valid_duration_strings(draw: st.DrawFn) -> tuple[str, int]:
    """Build any grammar-valid duration string, possibly non-canonical."""
    ordered_units = sorted(draw(_unit_subsets), key=_UNITS_DESCENDING.index)
    magnitudes = draw(
        st.lists(
            st.integers(min_value=0, max_value=999),
            min_size=len(ordered_units),
            max_size=len(ordered_units),
        )
    )
    text = "".join(
        f"{magnitude}{unit}"
        for magnitude, unit in zip(magnitudes, ordered_units, strict=True)
    )
    expected_total = sum(
        magnitude * _UNIT_SECONDS[unit]
        for magnitude, unit in zip(magnitudes, ordered_units, strict=True)
    )
    return text, expected_total


@given(st.integers(min_value=0, max_value=10_000_000))
def test_parse_of_str_round_trips_from_seconds(seconds: int) -> None:
    """Duration.parse(str(Duration(n))) == Duration(n) for any valid n."""
    duration = Duration(seconds)
    assert Duration.parse(str(duration)) == duration


@given(_canonical_duration_strings())
def test_canonical_string_round_trips_through_parse(canonical: str) -> None:
    """str(Duration.parse(s)) == s for every canonical string s."""
    assert str(Duration.parse(canonical)) == canonical


@given(_grammar_valid_duration_strings())
def test_grammar_valid_strings_normalize_to_arithmetic_sum(
    case: tuple[str, int],
) -> None:
    """Any grammar-valid string parses to the sum of its unit components."""
    text, expected_total_seconds = case
    assert Duration.parse(text).total_seconds == expected_total_seconds


# Zero digits of non-ASCII decimal-digit scripts (all Unicode category Nd,
# so all matched by \d and converted by int()): Arabic-Indic, Extended
# Arabic-Indic, Devanagari, Thai, and fullwidth forms. Escapes, not glyphs,
# so the intent is visible and no confusable literals live in the source.
_NON_ASCII_DIGIT_ZEROS = ("\u0660", "\u06f0", "\u0966", "\u0e50", "\uff10")


def _is_strictly_descending_unique(units: list[str]) -> bool:
    """Report whether a unit sequence is valid grammar ordering."""
    ranks = [_UNITS_DESCENDING.index(unit) for unit in units]
    return all(a < b for a, b in pairwise(ranks))


@st.composite
def _misordered_or_duplicated_unit_strings(draw: st.DrawFn) -> str:
    """Build a grammar-shaped string whose unit sequence violates ordering."""
    units = draw(st.lists(st.sampled_from(_UNITS_DESCENDING), min_size=2, max_size=6))
    assume(not _is_strictly_descending_unique(units))
    magnitudes = [draw(st.integers(min_value=0, max_value=999)) for _ in units]
    return "".join(
        f"{magnitude}{unit}" for magnitude, unit in zip(magnitudes, units, strict=True)
    )


@st.composite
def _non_ascii_digit_duration_strings(draw: st.DrawFn) -> str:
    """Build a canonical string with at least one digit swapped to another script."""
    canonical = draw(_canonical_duration_strings())
    digit_positions = [i for i, char in enumerate(canonical) if char.isdigit()]
    position = draw(st.sampled_from(digit_positions))
    zero = draw(st.sampled_from(_NON_ASCII_DIGIT_ZEROS))
    replacement = chr(ord(zero) + int(canonical[position]))
    return canonical[:position] + replacement + canonical[position + 1 :]


@given(_misordered_or_duplicated_unit_strings())
def test_misordered_or_duplicate_units_are_rejected(text: str) -> None:
    """Any unit sequence that repeats or misorders units fails to parse."""
    with pytest.raises(ConfigError):
        Duration.parse(text)


@given(_non_ascii_digit_duration_strings())
def test_non_ascii_digits_are_rejected(text: str) -> None:
    """A duration string containing any non-ASCII digit fails to parse."""
    with pytest.raises(ConfigError):
        Duration.parse(text)


@given(
    st.integers(min_value=0, max_value=10_000_000),
    st.integers(min_value=0, max_value=10_000_000),
)
def test_ordering_is_consistent_with_total_seconds(a: int, b: int) -> None:
    """Duration comparisons agree with comparing total_seconds directly."""
    assert (Duration(a) < Duration(b)) == (a < b)
    assert (Duration(a) <= Duration(b)) == (a <= b)
    assert (Duration(a) == Duration(b)) == (a == b)
    assert (Duration(a) > Duration(b)) == (a > b)
    assert (Duration(a) >= Duration(b)) == (a >= b)
