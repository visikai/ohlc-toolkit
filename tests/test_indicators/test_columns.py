"""The one collision guard, and the deletion mutant each caller keeps.

A guard wired into one caller does not protect another, so each call site
has its own test here. That is the point of the class: unifying three
copies into one implementation is worth nothing if the wiring is what
breaks.
"""

import polars as pl
import pytest

from ohlc_toolkit.returns import (
    ReturnMethod,
    add_backward_returns,
    add_forward_returns,
)
from ohlc_toolkit.temporal import ConfigError, require_absent_columns
from ohlc_toolkit.windows import annotate_windows

_CADENCE = 60
_BASE = 1_700_000_000


def _window_frame(rows: int = 6) -> pl.DataFrame:
    """Build a minimal frame the return and annotation steps accept."""
    closes = [_BASE + index * _CADENCE for index in range(rows)]
    return pl.DataFrame(
        {
            "open_time": [close - _CADENCE for close in closes],
            "close_time": closes,
            "open": [100.0 + index for index in range(rows)],
            "high": [101.0 + index for index in range(rows)],
            "low": [99.0 + index for index in range(rows)],
            "close": [100.5 + index for index in range(rows)],
            "volume": [5.0] * rows,
            "src_count": [1] * rows,
            "coverage_seconds": [_CADENCE] * rows,
            "traded_seconds": [_CADENCE] * rows,
        },
        schema_overrides={
            "open_time": pl.Int64,
            "close_time": pl.Int64,
            "src_count": pl.UInt32,
            "coverage_seconds": pl.Int64,
            "traded_seconds": pl.Int64,
        },
    )


def test_the_guard_names_every_colliding_column_and_the_caller_s_remedy() -> None:
    """The finding is shared; the advice belongs to the call site."""
    frame = _window_frame().with_columns(pl.lit(1.0).alias("mine"))

    with pytest.raises(ConfigError, match="mine") as caught:
        require_absent_columns(frame, ("mine", "yours"), remedy="pick another name.")

    message = str(caught.value)
    assert "pick another name." in message
    # Only what actually collided is named: "yours" is not there.
    assert "yours" not in message


def test_the_guard_passes_when_nothing_collides() -> None:
    """The ordinary path, so the refusal is not the only exercised one."""
    require_absent_columns(_window_frame(), ("brand", "new"), remedy="unused.")


def test_a_column_name_is_bounded_before_it_is_echoed() -> None:
    """A name comes from a caller and can be as long as a caller likes.

    One of the two implementations this replaced echoed them raw, which
    is how a single bad argument turns one refusal into an unbounded
    message.
    """
    enormous = "x" * 10_000
    frame = _window_frame().with_columns(pl.lit(1.0).alias(enormous))

    with pytest.raises(ConfigError) as caught:
        require_absent_columns(frame, (enormous,), remedy="unused.")

    assert len(str(caught.value)) < len(enormous)


def test_the_backward_return_caller_is_wired_to_it() -> None:
    """Its own test, because wiring is per caller and not per guard."""
    frame = _window_frame()
    once = add_backward_returns(
        frame, horizon="2m", cadence="1m", method=ReturnMethod.LOG
    )

    with pytest.raises(ConfigError, match="already carries"):
        add_backward_returns(once, horizon="2m", cadence="1m", method=ReturnMethod.LOG)


def test_the_forward_return_caller_is_wired_to_it() -> None:
    """Two columns rather than one, and both are found."""
    frame = _window_frame()
    once = add_forward_returns(
        frame, horizon="2m", cadence="1m", method=ReturnMethod.LOG
    )

    with pytest.raises(ConfigError, match="already carries"):
        add_forward_returns(once, horizon="2m", cadence="1m", method=ReturnMethod.LOG)


def test_the_annotation_caller_is_wired_to_it() -> None:
    """And keeps its own remedy, which names the prefix it takes."""
    frame = _window_frame()
    intervals = pl.DataFrame(
        {
            "start_timestamp": [_BASE],
            "end_timestamp": [_BASE + _CADENCE],
            "flag": ["outage"],
        },
        schema={
            "start_timestamp": pl.Int64,
            "end_timestamp": pl.Int64,
            "flag": pl.String,
        },
    )
    once = annotate_windows(frame, intervals)

    with pytest.raises(ConfigError, match="another prefix"):
        annotate_windows(once, intervals)


if __name__ == "__main__":
    pytest.main([__file__])
