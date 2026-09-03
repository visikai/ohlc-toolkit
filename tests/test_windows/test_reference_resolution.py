"""Strict resolution-time rules and range semantics of the window oracle.

Every rule here is an error, never a warning: an unresolvable schedule
cannot be silently rounded, widened, or shifted onto a grid the caller did
not ask for.
"""

import polars as pl
import pytest

from ohlc_toolkit.source.profile import Availability, ColumnKind, SourceProfile
from ohlc_toolkit.temporal import ConfigError
from ohlc_toolkit.windows import (
    ExplicitRange,
    MaterializationRule,
    compute_reference_windows,
)
from tests.test_windows.factories import (
    SourceRow,
    frame_from_rows,
    profile_for,
)

_MINUTE_CANDLES: tuple[SourceRow, ...] = (
    (0, 100.0, 110.0, 90.0, 105.0, 1.0),
    (60, 101.0, 111.0, 91.0, 106.0, 2.0),
    (120, 102.0, 112.0, 92.0, 107.0, 4.0),
    (180, 103.0, 113.0, 93.0, 108.0, 8.0),
)

_ANY_RANGE = ExplicitRange(start=120, end=241)


@pytest.mark.parametrize(
    ("window", "emit_every", "anchor", "phase_seconds", "expected_message"),
    [
        pytest.param(
            "1m30s",
            "1m",
            "0s",
            0,
            "Window duration must be a whole multiple",
            id="window_is_not_a_whole_number_of_source_candles",
        ),
        pytest.param(
            "2m",
            "1m30s",
            "0s",
            0,
            "Emit cadence must be a whole multiple",
            id="emit_cadence_is_not_a_whole_number_of_source_candles",
        ),
        pytest.param(
            "2m",
            "30s",
            "0s",
            0,
            "Emit cadence must not be shorter than",
            id="emit_cadence_is_shorter_than_the_source_cadence",
        ),
        pytest.param(
            "30s",
            "1m",
            "0s",
            0,
            "Window duration must not be shorter than",
            id="window_is_shorter_than_the_source_cadence",
        ),
        pytest.param(
            "2m",
            "2m",
            "30s",
            0,
            "does not land on the source close-time grid",
            id="anchor_puts_the_emit_grid_between_source_close_times",
        ),
        pytest.param(
            "2m",
            "2m",
            "0s",
            30,
            "does not land on the source close-time grid",
            id="source_phase_puts_every_close_time_off_the_emit_grid",
        ),
    ],
)
def test_unresolvable_schedules_raise_config_error(
    window: str,
    emit_every: str,
    anchor: str,
    phase_seconds: int,
    expected_message: str,
) -> None:
    """Each strict resolution rule rejects its own schedule, with its own reason."""
    with pytest.raises(ConfigError, match=expected_message):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60, phase_seconds=phase_seconds),
            window=window,
            emit_every=emit_every,
            anchor=anchor,
            materialization=_ANY_RANGE,
        )


def test_a_matching_source_phase_resolves_cleanly() -> None:
    """A phased source is fine as long as the anchor shares that phase.

    This is the positive control for the close-time-grid rule: the same
    30s-phased profile that fails with a 0s anchor resolves with a 30s one.
    """
    rows: tuple[SourceRow, ...] = (
        (30, 100.0, 110.0, 90.0, 105.0, 1.0),
        (90, 101.0, 111.0, 91.0, 106.0, 2.0),
    )
    result = compute_reference_windows(
        frame_from_rows(rows),
        profile_for(60, phase_seconds=30),
        window="2m",
        emit_every="2m",
        anchor="30s",
        materialization=ExplicitRange(start=150, end=151),
    )

    assert result.get_column("close_time").to_list() == [150]
    assert result.get_column("src_count").to_list() == [2]


def test_zero_window_is_rejected() -> None:
    """A zero-length window carries no data and cannot be resolved."""
    with pytest.raises(ConfigError):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="0s",
            emit_every="1m",
            materialization=_ANY_RANGE,
        )


def test_zero_emit_cadence_is_rejected() -> None:
    """A zero emit cadence never advances and cannot be resolved."""
    with pytest.raises(ConfigError):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="1m",
            emit_every="0s",
            materialization=_ANY_RANGE,
        )


def test_skip_warmup_without_a_fully_covered_tick_raises() -> None:
    """When no window is ever fully covered there is no honest start tick."""
    with pytest.raises(ConfigError, match="fully covered"):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="30m",
            emit_every="1m",
            materialization=MaterializationRule.SKIP_WARMUP,
        )


def test_skip_warmup_over_a_gap_that_no_window_bridges_raises() -> None:
    """Enough rows but never enough consecutive rows is still no start tick."""
    rows: tuple[SourceRow, ...] = (
        (0, 100.0, 110.0, 90.0, 105.0, 1.0),
        (120, 101.0, 111.0, 91.0, 106.0, 2.0),
        (240, 102.0, 112.0, 92.0, 107.0, 4.0),
        (360, 103.0, 113.0, 93.0, 108.0, 8.0),
    )
    with pytest.raises(ConfigError, match="fully covered"):
        compute_reference_windows(
            frame_from_rows(rows),
            profile_for(60),
            window="2m",
            emit_every="1m",
            materialization=MaterializationRule.SKIP_WARMUP,
        )


def test_skip_warmup_on_an_empty_frame_raises() -> None:
    """An empty frame has no coverage to measure, so warmup cannot be skipped."""
    with pytest.raises(ConfigError, match="empty source frame"):
        compute_reference_windows(
            frame_from_rows(()),
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization=MaterializationRule.SKIP_WARMUP,
        )


def test_an_unknown_materialization_rule_name_is_rejected() -> None:
    """Only the documented rule names resolve; a typo must not be guessed at."""
    with pytest.raises(ConfigError, match="Unknown materialization rule"):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization="skip-warmup",
        )


def test_a_rejected_rule_name_is_echoed_back_bounded() -> None:
    """A huge bad value must not become a huge log line or error message."""
    oversized = "s" * 500
    with pytest.raises(ConfigError, match="Unknown materialization rule") as caught:
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization=oversized,
        )

    message = str(caught.value)
    assert oversized not in message
    assert "500 chars total" in message


def test_a_materialization_of_an_unsupported_type_is_rejected() -> None:
    """A bare tuple is not a materialization: the pair must be explicit."""
    with pytest.raises(ConfigError, match="Expected an ExplicitRange"):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization=(0, 120),  # type: ignore[arg-type]
        )


def test_an_inverted_explicit_range_is_rejected() -> None:
    """An end before its start is a mistake, not an empty range."""
    with pytest.raises(ConfigError, match="must not precede"):
        ExplicitRange(start=240, end=120)


def test_a_non_integer_explicit_range_bound_is_rejected() -> None:
    """Range bounds are exact Unix seconds; a float bound is not exact."""
    with pytest.raises(ConfigError, match="must be an int"):
        ExplicitRange(start=0.0, end=120)  # type: ignore[arg-type]


def test_an_explicit_range_is_half_open_in_ticks() -> None:
    """A tick at ``start`` emits; a tick at ``end`` does not."""
    included = compute_reference_windows(
        frame_from_rows(_MINUTE_CANDLES),
        profile_for(60),
        window="1m",
        emit_every="1m",
        materialization=ExplicitRange(start=60, end=120),
    )

    assert included.get_column("close_time").to_list() == [60]


def test_a_profile_without_the_ohlcv_columns_is_rejected() -> None:
    """The oracle needs all five OHLCV roles declared to aggregate anything."""
    profile = SourceProfile.create(
        name="timestamp-only",
        timestamp_column="timestamp",
        availability=Availability.CLOSE_TIME,
        raw_schema={"timestamp": ColumnKind.INTEGER},
        cadence="1m",
    )
    with pytest.raises(ConfigError, match="must declare"):
        compute_reference_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile,
            window="1m",
            emit_every="1m",
            materialization=_ANY_RANGE,
        )


def test_a_frame_missing_a_declared_column_is_rejected() -> None:
    """A declared column that is absent from the frame is a hard error."""
    frame = frame_from_rows(_MINUTE_CANDLES).drop("volume")
    with pytest.raises(ConfigError, match="does not contain"):
        compute_reference_windows(
            frame,
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization=_ANY_RANGE,
        )


def test_resolution_runs_before_the_frame_is_read() -> None:
    """A bad schedule is rejected even when the frame could never be read.

    Resolution errors describe the caller's configuration, so they must not
    depend on -- or be masked by -- the shape of the data.
    """
    unusable = pl.DataFrame({"nothing": [1, 2, 3]})
    with pytest.raises(ConfigError, match="Window duration must be a whole multiple"):
        compute_reference_windows(
            unusable,
            profile_for(60),
            window="1m30s",
            emit_every="1m",
            materialization=_ANY_RANGE,
        )


if __name__ == "__main__":
    pytest.main([__file__])
