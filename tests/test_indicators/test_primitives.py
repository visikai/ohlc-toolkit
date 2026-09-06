"""The primitive contract: what a frame must be before a primitive reads it.

Cutler's RSI stands in for "a primitive" throughout. These tests are
about the contract rather than the arithmetic -- the identity a column is
named from, the three ways harness output can be wrong for the primitive
holding it, and the one path that writes a column onto a frame.
"""

import polars as pl
import pytest

from ohlc_toolkit.indicators import (
    CutlersRSI,
    FeatureFamily,
    IndicatorPrimitive,
    LogVolumeRatio,
    NormalizationClass,
    PhasedLookback,
    PriceToMovingAverage,
    RelativeRange,
    add_indicator,
    indicator_identity,
    phased_lookback,
    require_phased_inputs,
)
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.windows import ExplicitRange, compute_windows
from tests.test_indicators.factories import (
    BASE,
    phased_from_closes,
    phased_from_fields,
    rising,
)
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

    with pytest.raises(ConfigError, match="does not carry"):
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


def test_the_writer_returns_a_record_the_next_call_can_read() -> None:
    """The lookback columns and the grid both survive, so calls chain."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)

    written = add_indicator(phased, _RSI, period=_PERIOD)

    assert written.frame.columns == [*phased.frame.columns, "rsi_p3_w3m"]
    assert written.grid == phased.grid
    assert written.frame["rsi_p3_w3m"].to_list() == [100.0]


def test_writing_the_same_indicator_twice_is_refused() -> None:
    """The collision guard, on the path that derives the name."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)
    once = add_indicator(phased, _RSI, period=_PERIOD)

    with pytest.raises(ConfigError, match="already carries"):
        add_indicator(once, _RSI, period=_PERIOD)


def test_a_field_of_the_wrong_element_dtype_is_refused() -> None:
    """A wrong dtype is a different NUMBER, not a failure.

    Float32 closes near 1.678e7 put the differences below the type's
    resolution: the reading comes back 75.0 where Float64 reads 80.0,
    with nothing logged and no guard fired.
    """
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)
    narrowed = PhasedLookback(
        frame=phased.frame.with_columns(pl.col("close").cast(pl.List(pl.Float32))),
        grid=phased.grid,
    )

    with pytest.raises(ConfigError, match="element type the harness declares"):
        require_phased_inputs(_RSI, narrowed, period=_PERIOD, fields=("close",))


def test_a_column_that_is_not_a_phased_field_is_refused() -> None:
    """`close_time` is present and is not one of the phased fields."""
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)

    with pytest.raises(ConfigError, match="does not carry"):
        require_phased_inputs(
            _RSI, phased, period=_PERIOD, fields=("close", "close_time")
        )


def test_a_bare_string_of_fields_is_refused() -> None:
    """The mistake no annotation can make unrepresentable.

    A `str` IS a `Collection[str]`, so passing one type-checks and would
    be read one character at a time.
    """
    phased = phased_from_closes([rising(_LOOKBACK)], lookback=_LOOKBACK)

    with pytest.raises(ConfigError, match="not a single string"):
        require_phased_inputs(
            _RSI,
            phased,
            period=_PERIOD,
            fields="close",  # type: ignore[arg-type]
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
    # the only-rises convention and nothing in between. The null count is
    # the harness's own -- the first ticks of the range have no `L`th
    # window behind them -- and reading it off `close` rather than off
    # the result is what makes this an agreement rather than a tautology.
    assert set(values.drop_nulls().to_list()) == {100.0}
    assert values.null_count() == phased.frame["close"].null_count()
    assert values.null_count() > 0


@pytest.mark.parametrize(
    ("primitive", "expected"),
    [
        (CutlersRSI(), "bounded_by_construction"),
        (RelativeRange(), "stationarized"),
        (LogVolumeRatio(), "stationarized"),
        (PriceToMovingAverage(), "stationarized"),
    ],
    ids=lambda value: getattr(value, "name", value),
)
def test_every_primitive_s_identity_carries_the_class_it_declares(
    primitive: IndicatorPrimitive, expected: str
) -> None:
    """The class is on the record, not on the name -- so it has to be read."""
    phased = phased_from_closes(
        [rising(_LOOKBACK)], lookback=primitive.lookback(_PERIOD)
    )

    identity = indicator_identity(primitive, phased, period=_PERIOD)

    assert identity.normalization.value == expected
    assert identity.normalization is primitive.normalization


def test_three_primitives_chain_over_one_lookback() -> None:
    """The reason the writer returns the record: a recipe computes several.

    The three that read `L = P + 1` share one harness call, and each
    appends beside the last. Before the writer returned a record this
    read `PhasedLookback(frame=..., grid=...)` between every call, and a
    caller who got the grid wrong would have silently renamed a column.
    """
    phased = phased_from_fields(
        {
            "high": [[10.0] * _LOOKBACK],
            "low": [[6.0] * _LOOKBACK],
            "close": [[8.0] * _LOOKBACK],
            "volume": [[5.0] * _LOOKBACK],
        },
        lookback=_LOOKBACK,
    )

    written = phased
    for primitive in (CutlersRSI(), RelativeRange(), LogVolumeRatio()):
        written = add_indicator(written, primitive, period=_PERIOD)

    assert written.frame.columns[-3:] == [
        "rsi_p3_w3m",
        "relrange_p3_w3m",
        "logvolratio_p3_w3m",
    ]
    assert written.grid == phased.grid
    # Every reading is present: a chain that had lost the grid would have
    # named a column for another window and left this one all null.
    assert written.frame.select(pl.all().null_count()).row(0).count(0) == len(
        written.frame.columns
    )


if __name__ == "__main__":
    pytest.main([__file__])
