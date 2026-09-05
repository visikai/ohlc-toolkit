"""Type refusals log the offending TYPE, never the offending value.

A guard that rejects a value for being the wrong kind already tells the
exception ``type(x).__name__``; for a while the warning beside it echoed
``repr(x)`` instead. That is the same information as the exception's, at
whatever size the caller's object renders to -- so the right fix is not
to bound the echo but to log what the exception logs. These tests hold
every such site to that rule with a value whose ``repr`` is enormous and
distinctive: the type name must appear in the warning and the ``repr``
must not.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest

from ohlc_toolkit.returns import primitives
from ohlc_toolkit.schedules import cadence, generators, identity, registry
from ohlc_toolkit.schedules.generators import DedupRule, RoundingRule
from ohlc_toolkit.temporal import ConfigError, Duration, duration
from ohlc_toolkit.windows import quality, resolution
from ohlc_toolkit.windows.quality import QualityMode, WindowQualityPolicy

if TYPE_CHECKING:
    from loguru._logger import Logger

# A substring no legitimate log line contains, repeated until the repr is
# far larger than any bounded rendering would allow.
_MARKER = "LOUDREPR"
_REPR_REPEATS = 20_000


class _Loud:
    """A wrong-typed value whose repr would swamp a log line if echoed."""

    def __repr__(self) -> str:
        """Render something large and unmistakable."""
        return _MARKER * _REPR_REPEATS


# Typed Any on purpose: every guard below declares a narrower parameter, and
# handing it the wrong type is the whole point of the test.
_LOUD: Any = _Loud()


@dataclass(frozen=True)
class Site:
    """One type-refusing guard: the logger it warns through, and how to trip it."""

    logger: "Logger"
    trip: Callable[[], object]
    name: str


_SITES = (
    Site(
        resolution.logger,
        lambda: resolution.coerce_materialization(_LOUD),
        "materialization",
    ),
    Site(registry.logger, lambda: registry.named_schedule(_LOUD), "schedule name"),
    Site(duration.logger, lambda: Duration.parse(_LOUD), "duration text"),
    Site(duration.logger, lambda: duration.coerce_duration(_LOUD), "duration value"),
    Site(cadence.logger, lambda: cadence._validated_divisor(_LOUD), "divisor"),
    Site(cadence.logger, lambda: cadence._normalized_allowed(_LOUD), "allowed set"),
    Site(
        cadence.logger, lambda: cadence._require_sequence(_LOUD, label="x"), "sequence"
    ),
    Site(cadence.logger, lambda: cadence._coerced_pair(_LOUD), "cadence pair"),
    Site(quality.logger, lambda: WindowQualityPolicy(mode=_LOUD), "quality mode"),
    Site(
        quality.logger,
        lambda: WindowQualityPolicy(mode=QualityMode.FILTER, gate_mode=_LOUD),
        "gate mode",
    ),
    Site(
        quality.logger, lambda: quality._validated_min_coverage(_LOUD), "min_coverage"
    ),
    Site(primitives.logger, lambda: primitives._require_method(_LOUD), "return method"),
    Site(
        generators.logger,
        lambda: generators._validated_coefficient(_LOUD),
        "coefficient",
    ),
    Site(generators.logger, lambda: generators._validated_count(_LOUD), "count"),
    Site(
        generators.logger,
        lambda: generators._require_rules(_LOUD, DedupRule.DROP_LATER_REPEATS),
        "rounding rule",
    ),
    Site(
        generators.logger,
        lambda: generators._require_rules(RoundingRule.NEAREST_TIES_AWAY, _LOUD),
        "dedup rule",
    ),
    Site(
        generators.logger,
        lambda: generators.require_resolved_windows((_LOUD,)),
        "schedule window",
    ),
    Site(
        identity.logger,
        lambda: identity.mapping_from_payload(_LOUD, label="x"),
        "mapping",
    ),
    Site(
        identity.logger,
        lambda: identity.duration_from_payload(_LOUD, label="x"),
        "payload duration",
    ),
    Site(
        identity.logger,
        lambda: identity.durations_from_payload(_LOUD, label="x"),
        "payload durations",
    ),
    Site(
        identity.logger,
        lambda: identity.optional_text_from_payload(_LOUD, label="x"),
        "payload text",
    ),
    Site(
        generators.logger,
        lambda: generators.explicit(_LOUD),
        "explicit schedule",
    ),
    Site(
        duration.logger,
        lambda: Duration(_LOUD),
        "duration seconds",
    ),
    Site(
        resolution.logger,
        lambda: resolution.ExplicitRange(_LOUD, 0),
        "materialization range bound",
    ),
)


@pytest.mark.parametrize("site", _SITES, ids=[site.name for site in _SITES])
def test_a_type_refusal_logs_the_type_and_not_the_value(site: Site) -> None:
    """Both exits name the offending type; the offending repr appears in neither."""
    logged: list[str] = []
    sink_id = site.logger.add(logged.append, level="WARNING", format="{message}")
    try:
        with pytest.raises(ConfigError) as raised:
            site.trip()
    finally:
        site.logger.remove(sink_id)

    assert logged, "the refusal warns before it raises; nothing was captured"
    line = logged[-1]
    assert _Loud.__name__ in line
    assert _MARKER not in line
    message = str(raised.value)
    assert _Loud.__name__ in message
    assert _MARKER not in message
