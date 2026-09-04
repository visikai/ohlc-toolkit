"""The window engine refuses exactly what the oracle refuses, and says so.

Resolution is a statement about the caller's configuration, not about the
data, so a fast engine that quietly accepted a schedule the oracle
rejects would be a different library with the same name. Almost every
case here runs both implementations over the same input and asserts that
they raise the same :class:`~ohlc_toolkit.temporal.ConfigError` with the
same message.

The exception is the last test, which covers the one resolution-time
decision that is not a refusal: an enormous emit grid is allowed, and is
allowed loudly.
"""

from collections.abc import Callable

import polars as pl
import pytest

from ohlc_toolkit.source.profile import (
    Availability,
    ColumnKind,
    SourceProfile,
)
from ohlc_toolkit.temporal import ConfigError
from ohlc_toolkit.windows import (
    ExplicitRange,
    Materialization,
    MaterializationRule,
    compute_reference_windows,
    compute_windows,
    engine,
)
from tests.test_windows.factories import SourceRow, frame_from_rows, profile_for

_MINUTE_CANDLES: tuple[SourceRow, ...] = (
    (0, 100.0, 110.0, 90.0, 105.0, 1.0),
    (60, 101.0, 111.0, 91.0, 106.0, 2.0),
    (120, 102.0, 112.0, 92.0, 107.0, 4.0),
    (180, 103.0, 113.0, 93.0, 108.0, 8.0),
)

_ANY_RANGE = ExplicitRange(start=120, end=241)

_Compute = Callable[..., pl.DataFrame]
_IMPLEMENTATIONS: tuple[_Compute, ...] = (compute_reference_windows, compute_windows)


def _both_refuse(  # noqa: PLR0913 - one keyword per schedule knob under test
    frame: pl.DataFrame,
    profile: SourceProfile,
    *,
    window: str,
    emit_every: str,
    anchor: str = "0s",
    materialization: Materialization,
    expected_message: str,
) -> None:
    """Assert both implementations raise the same message for one schedule."""
    messages = []
    for compute in _IMPLEMENTATIONS:
        with pytest.raises(ConfigError, match=expected_message) as caught:
            compute(
                frame,
                profile,
                window=window,
                emit_every=emit_every,
                anchor=anchor,
                materialization=materialization,
            )
        messages.append(str(caught.value))

    assert messages[0] == messages[1]


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
        pytest.param(
            "0s",
            "1m",
            "0s",
            0,
            "Window duration must be strictly positive",
            id="zero_window",
        ),
        pytest.param(
            "1m",
            "0s",
            "0s",
            0,
            "Cadence must be strictly positive",
            id="zero_emit_cadence",
        ),
    ],
)
def test_unresolvable_schedules_are_refused_identically(
    window: str,
    emit_every: str,
    anchor: str,
    phase_seconds: int,
    expected_message: str,
) -> None:
    """Each strict resolution rule fires in the engine exactly as in the oracle."""
    _both_refuse(
        frame_from_rows(_MINUTE_CANDLES),
        profile_for(60, phase_seconds=phase_seconds),
        window=window,
        emit_every=emit_every,
        anchor=anchor,
        materialization=_ANY_RANGE,
        expected_message=expected_message,
    )


def test_skip_warmup_without_a_fully_covered_tick_is_refused_identically() -> None:
    """No honest start tick is a configuration error in both implementations."""
    _both_refuse(
        frame_from_rows(_MINUTE_CANDLES),
        profile_for(60),
        window="30m",
        emit_every="1m",
        materialization=MaterializationRule.SKIP_WARMUP,
        expected_message="fully covered",
    )


def test_skip_warmup_over_an_unbridgeable_gap_is_refused_identically() -> None:
    """Enough rows but never enough consecutive rows is still no start tick."""
    rows: tuple[SourceRow, ...] = (
        (0, 100.0, 110.0, 90.0, 105.0, 1.0),
        (120, 101.0, 111.0, 91.0, 106.0, 2.0),
        (240, 102.0, 112.0, 92.0, 107.0, 4.0),
        (360, 103.0, 113.0, 93.0, 108.0, 8.0),
    )
    _both_refuse(
        frame_from_rows(rows),
        profile_for(60),
        window="2m",
        emit_every="1m",
        materialization=MaterializationRule.SKIP_WARMUP,
        expected_message="fully covered",
    )


def test_skip_warmup_on_an_empty_frame_is_refused_identically() -> None:
    """An empty frame has no coverage to measure, so warmup cannot be skipped."""
    _both_refuse(
        frame_from_rows(()),
        profile_for(60),
        window="1m",
        emit_every="1m",
        materialization=MaterializationRule.SKIP_WARMUP,
        expected_message="empty source frame",
    )


def test_an_unknown_materialization_rule_name_is_refused_identically() -> None:
    """Only the documented rule names resolve; a typo must not be guessed at."""
    _both_refuse(
        frame_from_rows(_MINUTE_CANDLES),
        profile_for(60),
        window="1m",
        emit_every="1m",
        materialization="skip-warmup",
        expected_message="Unknown materialization rule",
    )


def test_a_rejected_rule_name_is_echoed_back_bounded() -> None:
    """A huge bad value must not become a huge log line or error message."""
    oversized = "s" * 500
    with pytest.raises(ConfigError, match="Unknown materialization rule") as caught:
        compute_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization=oversized,
        )

    message = str(caught.value)
    assert oversized not in message
    # The length note counts the echoed REPRESENTATION -- the 500-char
    # input plus its two repr quotes -- so the reader of a truncated
    # echo knows the size of exactly what was truncated.
    assert "502 chars total" in message


def test_a_materialization_of_an_unsupported_type_is_refused_identically() -> None:
    """A bare tuple is not a materialization: the pair must be explicit."""
    _both_refuse(
        frame_from_rows(_MINUTE_CANDLES),
        profile_for(60),
        window="1m",
        emit_every="1m",
        materialization=(0, 120),  # type: ignore[arg-type]
        expected_message="Expected an ExplicitRange",
    )


def test_a_profile_without_the_ohlcv_columns_is_refused_identically() -> None:
    """The engine needs all five OHLCV roles declared to aggregate anything."""
    profile = SourceProfile.create(
        name="timestamp-only",
        timestamp_column="timestamp",
        availability=Availability.CLOSE_TIME,
        raw_schema={"timestamp": ColumnKind.INTEGER},
        cadence="1m",
    )
    _both_refuse(
        frame_from_rows(_MINUTE_CANDLES),
        profile,
        window="1m",
        emit_every="1m",
        materialization=_ANY_RANGE,
        expected_message="must declare",
    )


def test_a_frame_missing_a_declared_column_is_refused_identically() -> None:
    """A declared column that is absent from the frame is a hard error."""
    _both_refuse(
        frame_from_rows(_MINUTE_CANDLES).drop("volume"),
        profile_for(60),
        window="1m",
        emit_every="1m",
        materialization=_ANY_RANGE,
        expected_message="does not contain",
    )


def test_resolution_runs_before_the_frame_is_read() -> None:
    """A bad schedule is rejected even when the frame could never be read.

    Resolution errors describe the caller's configuration, so they must not
    depend on -- or be masked by -- the shape of the data.
    """
    _both_refuse(
        pl.DataFrame({"nothing": [1, 2, 3]}),
        profile_for(60),
        window="1m30s",
        emit_every="1m",
        materialization=_ANY_RANGE,
        expected_message="Window duration must be a whole multiple",
    )


def test_a_matching_source_phase_resolves_cleanly() -> None:
    """A phased source is fine as long as the anchor shares that phase.

    The positive control for the close-time-grid rule: the same 30s-phased
    profile that fails with a 0s anchor resolves with a 30s one.
    """
    rows: tuple[SourceRow, ...] = (
        (30, 100.0, 110.0, 90.0, 105.0, 1.0),
        (90, 101.0, 111.0, 91.0, 106.0, 2.0),
    )
    result = compute_windows(
        frame_from_rows(rows),
        profile_for(60, phase_seconds=30),
        window="2m",
        emit_every="2m",
        anchor="30s",
        materialization=ExplicitRange(start=150, end=151),
    )

    assert result.get_column("close_time").to_list() == [150]
    assert result.get_column("src_count").to_list() == [2]


def test_an_enormous_emit_grid_is_materialized_but_never_quietly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A huge grid is a huge allocation, and the caller gets told so.

    The engine holds one row per emit tick and one per source candle at
    once. It does not refuse a large grid -- a long history at a fine emit
    cadence is a legitimate thing to ask for -- but the cost is logged
    instead of being paid in silence.

    The cap is lowered for this test rather than the grid being inflated
    to reach it: allocating twenty million ticks to observe one log line
    would make the suite pay the very cost the warning is about.
    """
    monkeypatch.setattr(engine, "_MAX_UNWARNED_TICKS", 2)
    logged: list[str] = []
    sink_id = engine.logger.add(logged.append, level="WARNING", format="{message}")
    try:
        result = compute_windows(
            frame_from_rows(_MINUTE_CANDLES),
            profile_for(60),
            window="1m",
            emit_every="1m",
            materialization=ExplicitRange(start=60, end=241),
        )
    finally:
        engine.logger.remove(sink_id)

    # Warned about, and then computed anyway: the cap is a warning, not a
    # refusal.
    assert result.get_column("close_time").to_list() == [60, 120, 180, 240]
    assert [message for message in logged if "emit ticks over" in message]


if __name__ == "__main__":
    pytest.main([__file__])
