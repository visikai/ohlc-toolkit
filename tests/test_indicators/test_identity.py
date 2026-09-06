"""Feature identity: the derived name, the round trip, and the two counts.

The four published names are written out as literals. A test that built
its expectation from the same formula the code uses would agree with the
code whatever the formula was.
"""

from collections.abc import Callable

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ohlc_toolkit.indicators import (
    FeatureFamily,
    FeatureIdentity,
    NormalizationClass,
    effective_history,
    effective_n,
)
from ohlc_toolkit.indicators import identity as identity_module
from ohlc_toolkit.indicators.identity import BANNED_NAME_PART
from ohlc_toolkit.temporal import ConfigError, Duration
from ohlc_toolkit.temporal.echo import MAX_ECHO_CHARS

# One class for every fixture below: this file is about the NAME, and the
# normalization class is the one identity field the name deliberately does
# not carry.
_CLASS = NormalizationClass.BOUNDED_BY_CONSTRUCTION
# A phrase only the banned-part refusal produces, and one no input under
# test contains. Matching on `BANNED_NAME_PART` instead would match the
# echoed input whatever guard had fired.
_BANNED_PART_PHRASE = "close_time is the time key"

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
        normalization=_CLASS,
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
    parsed = FeatureIdentity.parse(name, normalization=_CLASS)

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
        normalization=_CLASS,
    )

    assert dense.column_name == "rsi_d14_w21m"
    assert FeatureIdentity.parse("rsi_d14_w21m", normalization=_CLASS) == dense
    # And it does not collide with the phased column of the same shape.
    assert (
        dense.column_name
        != FeatureIdentity(
            indicator="rsi",
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("21m"),
            normalization=_CLASS,
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
        normalization=_CLASS,
    )

    assert FeatureIdentity.parse(identity.column_name, normalization=_CLASS) == identity


@pytest.mark.parametrize(
    ("column", "match"),
    [
        ("rsi_x14_w21m", r"names no feature family"),
        ("rsi_pfourteen_w21m", r"not a feature column name"),
        ("rsi_p14_wnotaduration", r"duration"),
        ("rsi_p14", r"not a feature column name"),
        ("rsi_p14_w21m_extra", r"duration"),
        ("timestamp_p14_w21m", _BANNED_PART_PHRASE),
        ("rsi_p14_w21m_timestamp", _BANNED_PART_PHRASE),
    ],
)
def test_a_name_that_could_not_have_been_derived_is_refused(
    column: str, match: str
) -> None:
    """Parsing refuses rather than guessing at a shape it does not know."""
    with pytest.raises(ConfigError, match=match):
        FeatureIdentity.parse(column, normalization=_CLASS)


@pytest.mark.parametrize(
    ("indicator", "match"),
    [
        ("RSI", r"lowercase"),
        ("log_vol", r"underscore"),
        ("14rsi", r"lowercase"),
        ("", r"lowercase"),
        ("close_timestamp", _BANNED_PART_PHRASE),
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
            normalization=_CLASS,
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
            normalization=_CLASS,
        )


def test_a_family_that_is_not_a_member_is_refused() -> None:
    """The string "p" is not the family; the member is."""
    with pytest.raises(ConfigError, match="must be a FeatureFamily"):
        FeatureIdentity(
            indicator="rsi",
            family="p",  # type: ignore[arg-type]
            period=14,
            window=Duration.parse("21m"),
            normalization=_CLASS,
        )


def test_a_zero_window_is_refused() -> None:
    """A zero window names no span, and every count below divides by it."""
    with pytest.raises(ConfigError, match="strictly positive"):
        FeatureIdentity(
            indicator="rsi",
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("0s"),
            normalization=_CLASS,
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
            normalization=_CLASS,
        )


class TestTheBannedNamePart:
    """The refusal that three earlier tests appeared to cover and did not.

    Each case below is one where the banned-part guard is the ONLY thing
    that can refuse, and each matches on a phrase the input does not
    contain. The earlier tests matched on `BANNED_NAME_PART` itself,
    which is the literal "timestamp" and appears in every input -- so the
    match succeeded against the echoed input whatever had refused, and
    weakening the guard from `in` to `==` left the suite green while
    `timestampfoo_p14_w21m` became constructible.
    """

    @pytest.mark.parametrize(
        "indicator", ["timestampfoo", "mytimestamp", "atimestampb"]
    )
    def test_an_indicator_carrying_the_part_anywhere_is_refused(
        self, indicator: str
    ) -> None:
        """Nothing else about these names is wrong.

        Each is lowercase, alphanumeric, starts with a letter and has no
        underscore, so the indicator pattern accepts every one. Only the
        banned-part guard can refuse them.
        """
        with pytest.raises(ConfigError, match="close_time is the time key"):
            FeatureIdentity(
                indicator=indicator,
                family=FeatureFamily.PHASED,
                period=14,
                window=Duration.parse("21m"),
                normalization=_CLASS,
            )

    @pytest.mark.parametrize(
        "column", ["mytimestamp_p14_w21m", "timestampfoo_p14_w21m"]
    )
    def test_a_column_carrying_the_part_anywhere_is_refused(self, column: str) -> None:
        """These parse cleanly as names; only the part makes them illegal."""
        with pytest.raises(ConfigError, match="close_time is the time key"):
            FeatureIdentity.parse(column, normalization=_CLASS)

    def test_a_name_that_merely_looks_similar_is_not_caught(self) -> None:
        """The other direction, so the guard is not refusing everything.

        "stamps" contains no banned part. Without this, a guard that
        refused every indicator would pass every test above.
        """
        identity = FeatureIdentity(
            indicator="stamps",
            family=FeatureFamily.PHASED,
            period=1,
            window=Duration.parse("1m"),
            normalization=_CLASS,
        )

        assert identity.column_name == "stamps_p1_w1m"


class TestTheCountsRefuseTheirOwnArguments:
    """The `Raises:` both count functions promise, which nothing enforced.

    Their validators were reachable only through `FeatureIdentity`, so
    stripping every check from the two functions left the suite green --
    and `effective_n("1d", "0s", 3)` became a `ZeroDivisionError` out of a
    function documented to raise `ConfigError`.
    """

    @pytest.mark.parametrize(
        ("period", "window", "match"),
        [
            (0, "3m", r"strictly positive"),
            (3, "0s", r"window must be strictly positive"),
            (True, "3m", r"must be an int"),
        ],
    )
    def test_effective_history_refuses_its_arguments(
        self, period: object, window: str, match: str
    ) -> None:
        """Each argument, and the message names which one."""
        with pytest.raises(ConfigError, match=match):
            effective_history(period, window)  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        ("history", "window", "period", "match"),
        [
            ("0s", "3m", 3, r"history_range must be strictly positive"),
            ("1d", "0s", 3, r"window must be strictly positive"),
            ("1d", "3m", 0, r"strictly positive"),
        ],
    )
    def test_effective_n_refuses_and_says_which_argument(
        self, history: str, window: str, period: object, match: str
    ) -> None:
        """Two durations, so the message has to name the one it refused.

        Without the label both zero cases produced the same sentence,
        naming neither -- and a `ZeroDivisionError` was one edit away.
        """
        with pytest.raises(ConfigError, match=match):
            effective_n(history, window, period)  # type: ignore[arg-type]


def test_a_window_too_long_to_parse_is_still_a_config_error() -> None:
    """`parse` takes a name off a stored artifact, so it cannot leak.

    A numeric component of more than 4300 digits trips CPython's own
    integer-parsing limit inside `Duration.parse`, which raises a bare
    `ValueError`. This entry point documents `ConfigError`, so it
    converts rather than letting a caller discover the difference.
    """
    absurd = "9" * 5000 + "m"

    with pytest.raises(ConfigError):
        FeatureIdentity.parse(f"rsi_p14_w{absurd}", normalization=_CLASS)


def test_the_record_carries_a_normalization_class_the_name_does_not() -> None:
    """§8's rule holds: what varies between columns is in the name.

    The class does not vary between the columns of one feature, so it
    lives in the record the manifest gets and not in the name. That is
    also why `parse` takes it as an argument: the name cannot vouch for
    it, and defaulting it would invent the one field it does not carry.
    """
    identity = FeatureIdentity(
        indicator="rsi",
        family=FeatureFamily.PHASED,
        period=14,
        window=Duration.parse("21m"),
        normalization=NormalizationClass.STATIONARIZED,
    )

    assert identity.normalization is NormalizationClass.STATIONARIZED
    assert "stationarized" not in identity.column_name
    assert (
        FeatureIdentity.parse(
            identity.column_name, normalization=NormalizationClass.STATIONARIZED
        )
        == identity
    )


def test_a_normalization_that_is_not_a_member_is_refused() -> None:
    """The string is not the class; the member is."""
    with pytest.raises(ConfigError, match="must be a NormalizationClass"):
        FeatureIdentity(
            indicator="rsi",
            family=FeatureFamily.PHASED,
            period=14,
            window=Duration.parse("21m"),
            normalization="stationarized",  # type: ignore[arg-type]
        )


_ENORMOUS_CHARS = 10_000
# The fixed prose of any refusal here plus its echoes, with room to
# spare, and orders of magnitude below the argument that provokes it.
_MAX_REFUSAL_CHARS = 6 * MAX_ECHO_CHARS


def _both_exits_bounded(trip: Callable[[], object]) -> None:
    """Run ``trip`` expecting a refusal; hold message AND log under the ceiling."""
    logged: list[str] = []
    sink_id = identity_module.logger.add(
        logged.append, level="WARNING", format="{message}"
    )
    try:
        with pytest.raises(ConfigError) as raised:
            trip()
    finally:
        identity_module.logger.remove(sink_id)
    assert len(str(raised.value)) < _MAX_REFUSAL_CHARS
    assert logged, "the refusal logs before it raises; nothing was captured"
    assert len(logged[-1]) < _MAX_REFUSAL_CHARS


class TestEveryEchoIsBoundedAtItsOwnSite:
    """One test per refusal that quotes something a caller supplied.

    The rule these hold in place is stated in ``temporal/echo.py``:
    enforced at each site by that site's own test, never by a truncating
    sink. Without them, replacing every ``bounded_echo`` in this module
    with ``repr`` leaves the suite green.
    """

    def test_an_unparsable_column_is_bounded(self) -> None:
        """The whole column name is the caller's and has no length it must have."""
        _both_exits_bounded(
            lambda: FeatureIdentity.parse("!" * _ENORMOUS_CHARS, normalization=_CLASS)
        )

    def test_an_unknown_family_is_bounded(self) -> None:
        """The family is one character, but the column carrying it is not.

        The pattern caps what this site echoes at a single letter, so the
        test cannot fail on today's code. It fails on the edit that would
        matter: echoing the matched COLUMN here instead of the family.
        """
        column = f"{'a' * _ENORMOUS_CHARS}_z14_w21m"
        _both_exits_bounded(lambda: FeatureIdentity.parse(column, normalization=_CLASS))

    def test_an_unparsable_window_is_bounded(self) -> None:
        """The window is the rest of the name after ``_w``, of any length."""
        column = f"rsi_p14_w{'9' * _ENORMOUS_CHARS}"
        _both_exits_bounded(lambda: FeatureIdentity.parse(column, normalization=_CLASS))

    def test_a_malformed_indicator_is_bounded(self) -> None:
        """A constructed identity takes its indicator straight from a caller."""
        _both_exits_bounded(
            lambda: FeatureIdentity(
                indicator="X" * _ENORMOUS_CHARS,
                family=FeatureFamily.PHASED,
                period=14,
                window=Duration.parse("21m"),
                normalization=_CLASS,
            )
        )

    def test_a_banned_indicator_is_bounded(self) -> None:
        """The banned-part refusal quotes the name it banned."""
        _both_exits_bounded(
            lambda: FeatureIdentity(
                indicator=BANNED_NAME_PART + "x" * _ENORMOUS_CHARS,
                family=FeatureFamily.PHASED,
                period=14,
                window=Duration.parse("21m"),
                normalization=_CLASS,
            )
        )


if __name__ == "__main__":
    pytest.main([__file__])
