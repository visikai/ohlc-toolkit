"""Feature identity: the derived name, the round trip, and the two counts.

The four published names are written out as literals. A test that built
its expectation from the same formula the code uses would agree with the
code whatever the formula was.
"""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ohlc_toolkit.indicators import (
    BANNED_NAME_PART,
    FeatureFamily,
    FeatureIdentity,
    NormalizationClass,
    effective_history,
    effective_n,
)
from ohlc_toolkit.temporal import ConfigError, Duration

_PUBLISHED = [
    ("rsi", FeatureFamily.PHASED, 14, "21m", "rsi_p14_w21m"),
    ("relrange", FeatureFamily.PHASED, 7, "56m", "relrange_p7_w56m"),
    ("logvolratio", FeatureFamily.PHASED, 14, "2h26m", "logvolratio_p14_w2h26m"),
    ("mapos", FeatureFamily.PHASED, 21, "6h20m", "mapos_p21_w6h20m"),
]


@pytest.mark.parametrize(
    ("indicator", "family", "period", "window", "name"), _PUBLISHED
)
def test_the_published_names_are_exactly_these(
    indicator: str, family: FeatureFamily, period: int, window: str, name: str
) -> None:
    """Written out, not recomputed from the formula under test."""
    identity = FeatureIdentity(
        indicator=indicator,
        family=family,
        period=period,
        window=Duration.parse(window),
    )

    assert identity.column_name == name


@pytest.mark.parametrize(
    ("indicator", "family", "period", "window", "name"), _PUBLISHED
)
def test_a_published_name_parses_back_to_what_produced_it(
    indicator: str, family: FeatureFamily, period: int, window: str, name: str
) -> None:
    """Round trip, from the name rather than from the record.

    Parsing the string is the direction that matters: a stored artifact
    hands a reader column names and nothing else.
    """
    parsed = FeatureIdentity.parse(name)

    assert parsed.indicator == indicator
    assert parsed.family is family
    assert parsed.period == period
    assert parsed.window == Duration.parse(window)
    assert parsed.column_name == name


def test_the_dense_family_is_representable_before_it_exists() -> None:
    """So that shipping it renames no phased column.

    The family character is in the name even though one artifact carries
    one family, which breaks the rule that a constant field belongs in
    the manifest. It is the one exception, and this is what it buys: a
    rename is a break for every consumer holding a stored frame, and one
    character now is cheaper than that later.
    """
    dense = FeatureIdentity(
        indicator="rsi",
        family=FeatureFamily.DENSE,
        period=14,
        window=Duration.parse("21m"),
    )

    assert dense.column_name == "rsi_d14_w21m"
    assert FeatureIdentity.parse("rsi_d14_w21m") == dense
    # And it does not collide with the phased column of the same shape.
    assert (
        dense.column_name
        != FeatureIdentity(
            indicator="rsi",
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("21m"),
        ).column_name
    )


@given(
    indicator=st.sampled_from(["rsi", "relrange", "mapos", "adx"]),
    family=st.sampled_from(list(FeatureFamily)),
    period=st.integers(min_value=1, max_value=200),
    window_seconds=st.integers(min_value=1, max_value=1_209_600),
)
def test_two_identities_differing_anywhere_get_different_names(
    indicator: str, family: FeatureFamily, period: int, window_seconds: int
) -> None:
    """Every identity field reaches the name, so no two can share one.

    Checked by round trip rather than by comparing pairs: if the name
    recovers the identity exactly, two different identities cannot have
    produced the same name.
    """
    identity = FeatureIdentity(
        indicator=indicator,
        family=family,
        period=period,
        window=Duration(window_seconds),
    )

    assert FeatureIdentity.parse(identity.column_name) == identity


@pytest.mark.parametrize(
    ("column", "match"),
    [
        ("rsi_x14_w21m", r"names no feature family"),
        ("rsi_pfourteen_w21m", r"not a feature column name"),
        ("rsi_p14_wnotaduration", r"duration"),
        ("rsi_p14", r"not a feature column name"),
        ("rsi_p14_w21m_extra", r"duration"),
        ("timestamp_p14_w21m", BANNED_NAME_PART),
        ("rsi_p14_w21m_timestamp", BANNED_NAME_PART),
    ],
)
def test_a_name_that_could_not_have_been_derived_is_refused(
    column: str, match: str
) -> None:
    """Parsing refuses rather than guessing at a shape it does not know."""
    with pytest.raises(ConfigError, match=match):
        FeatureIdentity.parse(column)


@pytest.mark.parametrize(
    ("indicator", "match"),
    [
        ("RSI", r"lowercase"),
        ("log_vol", r"underscore"),
        ("14rsi", r"lowercase"),
        ("", r"lowercase"),
        ("close_timestamp", BANNED_NAME_PART),
    ],
)
def test_an_indicator_name_that_would_not_parse_back_is_refused(
    indicator: str, match: str
) -> None:
    """An underscore in particular, since it separates the three parts."""
    with pytest.raises(ConfigError, match=match):
        FeatureIdentity(
            indicator=indicator,
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("21m"),
        )


@pytest.mark.parametrize(
    ("period", "match"),
    [(0, r"strictly positive"), (-1, r"strictly positive"), (True, r"must be an int")],
)
def test_a_period_that_is_not_a_positive_int_is_refused(
    period: object, match: str
) -> None:
    """A period of ``True`` is nobody's intention and would mean one."""
    with pytest.raises(ConfigError, match=match):
        FeatureIdentity(
            indicator="rsi",
            family=FeatureFamily.PHASED,
            period=period,  # type: ignore[arg-type]
            window=Duration.parse("21m"),
        )


def test_a_family_that_is_not_a_member_is_refused() -> None:
    """The string "p" is not the family; the member is."""
    with pytest.raises(ConfigError, match="must be a FeatureFamily"):
        FeatureIdentity(
            indicator="rsi",
            family="p",  # type: ignore[arg-type]
            period=14,
            window=Duration.parse("21m"),
        )


def test_a_zero_window_is_refused() -> None:
    """A zero window names no span, and every count below divides by it."""
    with pytest.raises(ConfigError, match="strictly positive"):
        FeatureIdentity(
            indicator="rsi",
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("0s"),
        )


class TestTheTwoCounts:
    """Effective history and effective-N, on hand-computed literals."""

    @pytest.mark.parametrize(
        ("period", "window", "expected"),
        [(3, "3m", "9m"), (14, "21m", "4h54m"), (1, "1w", "1w"), (7, "2h26m", "17h2m")],
    )
    def test_effective_history_is_the_product(
        self, period: int, window: str, expected: str
    ) -> None:
        """`L * W`, because the phased windows do not overlap."""
        assert effective_history(period, window) == Duration.parse(expected)

    @pytest.mark.parametrize(
        ("history", "window", "period", "expected"),
        [
            # A day holds 480 three-minute windows; in blocks of 3 that is 160.
            ("1d", "3m", 3, 160),
            # Exactly one block, and not one more.
            ("9m", "3m", 3, 1),
            # A second short of a block is no blocks at all.
            ("8m59s", "3m", 3, 0),
            ("1w", "1d", 1, 7),
        ],
    )
    def test_effective_n_counts_whole_blocks(
        self, history: str, window: str, period: int, expected: int
    ) -> None:
        """Whole blocks only: a partial span is not evidence of anything."""
        assert effective_n(history, window, period) == expected

    def test_effective_n_is_not_a_statistical_effective_sample_size(self) -> None:
        """The docstring says so, and the docstring is the contract here.

        The two are recorded under different names because reading one as
        the other overstates how much independent evidence a feature has.
        A statistical effective N accounts for autocorrelation between
        blocks and is smaller and model-dependent; this counts blocks.
        """
        assert "not an effective sample size" in " ".join(
            (effective_n.__doc__ or "").lower().split()
        )


def test_the_normalization_classes_are_exactly_the_three() -> None:
    """Three, spelled out. A fourth is a decision, not an addition."""
    assert [member.value for member in NormalizationClass] == [
        "bounded_by_construction",
        "stationarized",
        "empirically_normalized",
    ]


def test_an_indicator_that_is_not_a_string_is_refused() -> None:
    """A name is text, and a number here would be a column called 14."""
    with pytest.raises(ConfigError, match="must be a str"):
        FeatureIdentity(
            indicator=14,  # type: ignore[arg-type]
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("21m"),
        )


if __name__ == "__main__":
    pytest.main([__file__])
