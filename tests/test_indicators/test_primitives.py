"""The primitive contract: what a frame must be before a primitive reads it.

Cutler's RSI stands in for "a primitive" throughout. These tests are
about the contract rather than the arithmetic -- the identity a column is
named from, the three ways harness output can be wrong for the primitive
holding it, and the one path that writes a column onto a frame.
"""

import pytest

from ohlc_toolkit.indicators import (
    CutlersRSI,
    FeatureFamily,
    NormalizationClass,
    PhasedLookback,
    add_indicator,
    indicator_identity,
    phased_lookback,
    require_phased_inputs,
)
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from tests.test_indicators.factories import BASE, phased_from_closes, rising
from tests.test_windows.factories import frame_from_rows, profile_for

_MINUTE = 60
_PERIOD = 3
_LOOKBACK = _PERIOD + 1
_RSI = CutlersRSI()


def test_the_identity_is_derived_from_the_primitive_and_the_grid() -> None:
    """Nothing about the name is passed in; every field is read off something."""
    phased = phased_from_closes(
        [rising(_LOOKBACK)], lookback=_LOOKBACK, window_seconds=8760
    )

    identity = indicator_identity(_RSI, phased, period=_PERIOD)

    assert identity.indicator == "rsi"
    assert identity.family is FeatureFamily.PHASED
    assert identity.period == _PERIOD
    assert identity.window == Duration(8760)
    assert identity.normalization is NormalizationClass.BOUNDED_BY_CONSTRUCTION
    assert identity.column_name == "rsi_p3_w2h26m"


def test_a_frame_resolved_for_another_lookback_is_refused() -> None:
    """`L = 14` handed to an indicator that needs 15 is not a rounding error."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)

    with pytest.raises(ConfigError, match="reads 15 phased window"):
        require_phased_inputs(_RSI, phased, period=14, fields=("close",))


def test_a_missing_field_is_refused_by_name() -> None:
    """A primitive says which columns it reads, and they are checked."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)

    with pytest.raises(ConfigError, match="'nowhere'"):
        require_phased_inputs(_RSI, phased, period=_PERIOD, fields=("close", "nowhere"))


def test_a_list_that_is_not_the_lookback_s_length_is_refused() -> None:
    """The harness never emits one; a hand-built record can.

    A short list would be averaged over the length the primitive expected
    rather than the length it got, which is a wrong number rather than a
    missing one.
    """
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK + 1)

    with pytest.raises(ConfigError, match="not 5 long"):
        require_phased_inputs(_RSI, phased, period=_LOOKBACK, fields=("close",))


def test_a_null_list_is_not_a_ragged_one() -> None:
    """A tick with no inputs is legal; the check must not read it as short."""
    phased = phased_from_closes([rising(_LOOKBACK), None], lookback=_LOOKBACK)

    checked = require_phased_inputs(_RSI, phased, period=_PERIOD, fields=("close",))

    assert checked == _LOOKBACK


def test_the_writer_appends_the_column_and_keeps_the_frame() -> None:
    """The lookback columns stay, so a second primitive can read them."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)

    written = add_indicator(phased, _RSI, period=_PERIOD)

    assert written.columns == [*phased.frame.columns, "rsi_p3_w3m"]
    assert written.height == phased.frame.height


def test_writing_the_same_indicator_twice_is_refused() -> None:
    """The collision guard, on the path that derives the name."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)
    once = add_indicator(phased, _RSI, period=_PERIOD)

    with pytest.raises(ConfigError, match="already carries"):
        add_indicator(
            PhasedLookback(frame=once, grid=phased.grid), _RSI, period=_PERIOD
        )


def test_a_primitive_runs_over_real_harness_output() -> None:
    """The wiring: engine to harness to primitive, with nothing hand-built.

    Every other test here builds the record directly. This one is the
    check that what the harness really emits is what the contract
    describes -- the list order, the lengths, and the null rule.
    """
    rows = [
        (
            BASE + index * _MINUTE,
            100.0 + index,
            101.0 + index,
            99.0 + index,
            100.5 + index,
            5.0,
        )
        for index in range(60)
    ]
    frame = compute_windows(
        frame_from_rows(rows),
        profile_for(_MINUTE),
        window="3m",
        emit_every="1m",
        materialization=ExplicitRange(start=BASE + 180, end=BASE + 60 * _MINUTE + 1),
    )
    phased = phased_lookback(frame, window="3m", emit_every="3m", lookback=_LOOKBACK)

    values = _RSI.values(phased, period=_PERIOD)

    # Prices rise by one per minute throughout, so every complete tick is
    # the only-rises convention and nothing in between.
    assert set(values.drop_nulls().to_list()) == {100.0}
    assert values.len() == phased.frame.height


if __name__ == "__main__":
    pytest.main([__file__])
