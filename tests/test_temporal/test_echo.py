"""The one bounded echo every module quotes untrusted input through."""

from enum import Enum

import polars as pl
import pytest

from ohlc_toolkit.temporal import MAX_ECHO_CHARS, ConfigError, bounded_echo
from ohlc_toolkit.temporal.echo import enum_from_payload


def test_a_short_value_is_echoed_as_its_repr() -> None:
    """Quotes and escapes are shown, so a string cannot masquerade."""
    assert bounded_echo("1m") == "'1m'"
    assert bounded_echo("a\nb") == "'a\\nb'"


def test_a_non_string_is_echoed_as_its_repr() -> None:
    """The helper takes any value; a dtype names itself."""
    assert bounded_echo(pl.Int64) == "Int64"
    assert bounded_echo(42) == "42"


def test_an_oversized_value_is_truncated_with_its_full_length() -> None:
    """The echo is bounded and the note states what was truncated.

    The note counts the REPRESENTATION -- for a plain string, the raw
    length plus its two repr quotes -- so it describes exactly the text
    that was cut, not a different measure.
    """
    raw = "s" * 500
    echoed = bounded_echo(raw)
    assert raw not in echoed
    assert echoed.startswith("'" + "s" * (MAX_ECHO_CHARS - 1))
    assert echoed.endswith("... (502 chars total)")
    assert len(echoed) <= MAX_ECHO_CHARS + len("... (502 chars total)")


def test_the_boundary_length_is_not_truncated() -> None:
    """A representation of exactly the cap passes through whole."""
    raw = "s" * (MAX_ECHO_CHARS - 2)  # repr adds two quotes
    assert bounded_echo(raw) == repr(raw)


def test_the_cap_measures_the_representation_not_the_raw_length() -> None:
    """A raw 79-character string truncates; its 78-character neighbour does not.

    The cap bounds what is WRITTEN -- the repr, quotes included -- not
    the input's own length. Under the raw-length rule this helper
    replaced, plain-ASCII inputs of raw length 79 and 80 were echoed
    whole; measuring the representation moves that boundary down by the
    two quote characters, and measuring anything else would re-open the
    escape-expansion hole where an 80-character input rendered as
    hundreds of characters of log line.
    """
    assert "total" not in bounded_echo("x" * 78)
    assert "total" in bounded_echo("x" * 79)
    assert "total" in bounded_echo("x" * 80)


class _Colour(Enum):
    """A throwaway enum, so this test needs nothing from a higher layer."""

    RED = "red"


def test_a_huge_non_string_enum_payload_is_echoed_bounded() -> None:
    """The non-string refusal echoes through the bounded helper.

    This branch previously interpolated the raw value unbounded, so a
    500-element list would have produced a multi-kilobyte refusal; the
    consolidation routed it through the shared bounded echo, and this
    pins that improvement as deliberate.
    """
    # Generous ceiling: the bounded echo caps the interpolated value at 80
    # visible characters, so the whole refusal stays well under this while
    # the unbounded form measured multiple kilobytes.
    bounded_message_ceiling = 300

    with pytest.raises(ConfigError) as caught:
        enum_from_payload(_Colour, ["garbage"] * 500, label="colour")
    assert len(str(caught.value)) < bounded_message_ceiling
