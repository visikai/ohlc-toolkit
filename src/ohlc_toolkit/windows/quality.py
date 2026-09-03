"""A window quality-policy step, composed after the aggregator.

This module is not part of the engine or the oracle: it consumes their
nine-column output (:mod:`ohlc_toolkit.windows.engine`,
:mod:`ohlc_toolkit.windows.reference`) as an independent, later step, and
never feeds back into either. It reads exactly two of those nine columns
-- ``src_count`` and ``coverage_seconds`` -- plus ``close_time`` for
naming an offending row in a message. It never reads or alters ``open``,
``high``, ``low``, ``close``, or ``volume``.

A :class:`WindowQualityPolicy` is a frozen, JSON-round-trippable identity
-- a recipe can record it the same way it records a schedule -- and
:func:`apply_quality_policy` is the single entry point that interprets
one against a frame:

- ``PASS_THROUGH`` returns the frame unchanged: still an explicit,
  recorded step, useful so a recipe can name "no quality policy" the same
  way it names any other choice.
- ``FILTER`` drops rows whose ``coverage_seconds`` falls below the
  policy's threshold, returning a new frame. The input is never mutated,
  row order is otherwise preserved, and no OHLCV value is touched.
- ``GATE`` checks the same threshold without dropping anything. In
  ``GateMode.STRICT`` a violation logs and raises
  :class:`~ohlc_toolkit.temporal.errors.CoverageError`; in
  ``GateMode.REPORT`` it always returns a :class:`QualityReport` and never
  raises. This mirrors the strict/report split in
  :mod:`ohlc_toolkit.source.validation`, applied to windows instead of a
  raw source frame.

Threshold rounding
-------------------

``min_coverage`` is a fraction in ``[0, 1]``. The threshold compared
against ``coverage_seconds`` is the ordinary IEEE-754 product
``min_coverage * window_seconds``, computed once, as a single
double-precision multiplication -- not by first converting
``min_coverage`` to an exact rational via :class:`fractions.Fraction`.
That distinction matters at the boundary: a decimal literal such as
``0.9`` is not exactly representable in binary floating point, so its
true stored value is a hair above ``0.9``. Multiplying that exact stored
value by an integer window (via ``Fraction``) preserves that hair and can
push a mathematically-round product like ``0.9 * 300 == 270`` a fraction
above ``270``, which would then round UP to a threshold of ``271`` under
any ceiling rule -- silently tightening the policy by a whole second and
failing a row that is genuinely at exactly 90% coverage. Plain
double-precision multiplication does not have this problem: IEEE-754
multiplication is correctly rounded to the nearest representable double,
and for every ``(min_coverage, window_seconds)`` pair used in practice
here that nearest double is the mathematically intended product exactly.
A row is kept (``FILTER``) or considered covered (``GATE``) exactly when
``coverage_seconds >= min_coverage * window_seconds``, that product taken
literally. ``min_coverage=1.0`` therefore reduces to the full-coverage
requirement ``coverage_seconds == W`` for any real engine output, because
``coverage_seconds`` a window can report never exceeds ``W``.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, unique
from typing import Self

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import ConfigError, CoverageError, Duration, coerce_duration

logger = get_logger(__name__)

# The only columns this step is permitted to read. Requiring them up
# front, rather than letting a missing column surface later as a bare
# polars ColumnNotFoundError, keeps the failure at this module's own
# boundary and in this module's own words.
_REQUIRED_COLUMNS = ("close_time", "src_count", "coverage_seconds")


@unique
class QualityMode(Enum):
    """Which of the three window quality-policy behaviours applies.

    Attributes:
        PASS_THROUGH: Return the frame unchanged.
        FILTER: Drop rows below the coverage threshold, returning a new
            frame.
        GATE: Check the coverage threshold without dropping rows; react
            per :class:`GateMode`.

    """

    PASS_THROUGH = "pass_through"
    FILTER = "filter"
    GATE = "gate"


@unique
class GateMode(Enum):
    """How a ``GATE`` policy reacts to a coverage violation.

    Mirrors :class:`ohlc_toolkit.source.validation.ValidationMode`'s
    strict/report split, applied to windows instead of a raw source
    frame.

    Attributes:
        STRICT: Raise :class:`~ohlc_toolkit.temporal.errors.CoverageError`
            when any row falls below the threshold.
        REPORT: Never raise; always return the findings as a
            :class:`QualityReport`.

    """

    STRICT = "strict"
    REPORT = "report"


@dataclass(frozen=True)
class WindowQualityPolicy:
    """A frozen, serializable window quality-policy identity.

    Two policies with equal fields are equal, so a recipe can compare a
    stored identity against a freshly constructed one. ``min_coverage``
    and ``gate_mode`` are always recorded, even for a ``PASS_THROUGH``
    policy that ignores both: a recipe that later swaps
    ``PASS_THROUGH`` for ``FILTER`` then changes exactly one field rather
    than growing a new shape.

    Attributes:
        mode: Which behaviour :func:`apply_quality_policy` applies.
        min_coverage: The minimum fraction of the window duration a row's
            ``coverage_seconds`` must reach, in ``[0, 1]``. ``1.0``
            requires full coverage. Defaults to ``1.0``.
        gate_mode: For ``GATE``, whether a violation raises or is
            reported. Ignored by ``PASS_THROUGH`` and ``FILTER``.
            Defaults to :attr:`GateMode.STRICT`.

    """

    mode: QualityMode
    min_coverage: float = 1.0
    gate_mode: GateMode = GateMode.STRICT

    def __post_init__(self) -> None:
        """Reject a malformed mode, gate_mode, or min_coverage.

        Raises:
            ConfigError: If ``mode`` is not a :class:`QualityMode`,
                ``gate_mode`` is not a :class:`GateMode`, or
                ``min_coverage`` is not a non-NaN ``int``/``float`` in
                ``[0, 1]``.

        """
        if not isinstance(self.mode, QualityMode):
            logger.warning("Rejecting non-QualityMode mode: {!r}", self.mode)
            raise ConfigError(
                f"mode must be a QualityMode, got {type(self.mode).__name__}"
            )
        if not isinstance(self.gate_mode, GateMode):
            logger.warning("Rejecting non-GateMode gate_mode: {!r}", self.gate_mode)
            raise ConfigError(
                f"gate_mode must be a GateMode, got {type(self.gate_mode).__name__}"
            )
        _validate_min_coverage(self.min_coverage)

    def to_dict(self) -> dict[str, str | float]:
        """Serialize this identity to a deterministic, JSON-compatible dict.

        Returns:
            A dict with exactly the keys ``"mode"``, ``"min_coverage"``,
            and ``"gate_mode"``, using only ``str`` and ``float`` values,
            in that fixed key order.

        """
        return {
            "mode": self.mode.value,
            "min_coverage": self.min_coverage,
            "gate_mode": self.gate_mode.value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        """Reconstruct a policy identity from its :meth:`to_dict` form.

        Args:
            data: A mapping holding ``"mode"``, ``"min_coverage"``, and
                ``"gate_mode"``, as produced by :meth:`to_dict`.

        Returns:
            The reconstructed policy.

        Raises:
            ConfigError: If a required key is missing, ``"mode"`` or
                ``"gate_mode"`` does not name a known member, or
                ``min_coverage`` fails its own validation.

        """
        missing = [
            key for key in ("mode", "min_coverage", "gate_mode") if key not in data
        ]
        if missing:
            logger.warning("Rejecting policy dict missing key(s): {}", missing)
            raise ConfigError(f"Policy dict is missing key(s): {missing}")

        try:
            mode = QualityMode(data["mode"])
        except ValueError as error:
            logger.warning("Rejecting unknown quality mode: {!r}", data["mode"])
            raise ConfigError(f"Unknown quality mode: {data['mode']!r}") from error

        try:
            gate_mode = GateMode(data["gate_mode"])
        except ValueError as error:
            logger.warning("Rejecting unknown gate_mode: {!r}", data["gate_mode"])
            raise ConfigError(f"Unknown gate_mode: {data['gate_mode']!r}") from error

        min_coverage = data["min_coverage"]
        _validate_min_coverage(min_coverage)
        return cls(mode=mode, min_coverage=float(min_coverage), gate_mode=gate_mode)  # type: ignore[arg-type]


def _validate_min_coverage(value: object) -> None:
    """Reject a ``min_coverage`` that is not a non-NaN number in ``[0, 1]``.

    Raises:
        ConfigError: If ``value`` is not an ``int``/``float`` (``bool`` is
            rejected too, even though it is an ``int`` subtype), is NaN,
            or lies outside ``[0, 1]``.

    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        logger.warning("Rejecting non-numeric min_coverage: {!r}", value)
        raise ConfigError(
            f"min_coverage must be an int or float, got {type(value).__name__}"
        )
    if math.isnan(value):
        logger.warning("Rejecting NaN min_coverage.")
        raise ConfigError("min_coverage must not be NaN.")
    if value < 0 or value > 1:
        logger.warning("Rejecting out-of-range min_coverage: {}", value)
        raise ConfigError(f"min_coverage must be in [0, 1], got {value}.")


@dataclass(frozen=True)
class QualityReport:
    """The outcome of evaluating a ``GATE`` policy's coverage check.

    Attributes:
        rows_checked: How many rows were present in the evaluated frame.
        threshold_seconds: The minimum ``coverage_seconds`` a row had to
            reach to pass, derived from ``min_coverage`` and the window
            duration.
        offending_count: How many rows fell below ``threshold_seconds``.
        first_offending_close_time: The ``close_time`` of the first
            offending row, in frame order. ``None`` when nothing
            offended, including over an empty frame.

    """

    rows_checked: int
    threshold_seconds: float
    offending_count: int
    first_offending_close_time: int | None

    @property
    def passed(self) -> bool:
        """Report whether every row met the coverage threshold."""
        return self.offending_count == 0


def _require_quality_columns(frame: pl.DataFrame) -> None:
    """Check that the frame carries every column this step reads.

    Raises:
        ConfigError: If ``close_time``, ``src_count``, or
            ``coverage_seconds`` is absent from ``frame``.

    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        logger.warning("Rejecting frame missing quality column(s): {}", missing)
        raise ConfigError(
            f"A window quality policy requires column(s) {missing}; apply it "
            "to an engine-produced window frame."
        )


def _threshold_seconds(min_coverage: float, window_seconds: int) -> float:
    """Compute the minimum ``coverage_seconds`` a row must reach to pass.

    See the module docstring's "Threshold rounding" section for why this
    is a single ordinary float multiplication rather than exact rational
    arithmetic on ``min_coverage``'s raw binary value.

    Args:
        min_coverage: The minimum coverage fraction, in ``[0, 1]``.
        window_seconds: The window duration ``W``, in seconds.

    Returns:
        The threshold, in seconds. A row passes when its
        ``coverage_seconds >= `` this value.

    """
    return min_coverage * window_seconds


def _filter_frame(frame: pl.DataFrame, threshold_seconds: float) -> pl.DataFrame:
    """Return a new frame holding only rows at or above the threshold."""
    return frame.filter(pl.col("coverage_seconds") >= threshold_seconds)


def _evaluate_gate(frame: pl.DataFrame, threshold_seconds: float) -> QualityReport:
    """Check every row's coverage against the threshold, without mutating."""
    coverage = frame.get_column("coverage_seconds")
    mask = coverage < threshold_seconds
    offending_count = int(mask.sum())

    first_offending_close_time: int | None = None
    if offending_count > 0:
        first_offending_index = int(mask.arg_true()[0])
        first_offending_close_time = int(
            frame.get_column("close_time")[first_offending_index]
        )

    return QualityReport(
        rows_checked=frame.height,
        threshold_seconds=threshold_seconds,
        offending_count=offending_count,
        first_offending_close_time=first_offending_close_time,
    )


def apply_quality_policy(
    frame: pl.DataFrame,
    policy: WindowQualityPolicy,
    *,
    window: Duration | str,
) -> pl.DataFrame | QualityReport:
    """Apply a window quality policy to an engine-produced window frame.

    Never mutates ``frame``. Never reads or alters ``open``, ``high``,
    ``low``, ``close``, or ``volume``.

    Args:
        frame: A window frame such as
            :func:`~ohlc_toolkit.windows.engine.compute_windows` produces,
            carrying at least ``close_time``, ``src_count``, and
            ``coverage_seconds``.
        policy: The policy identity to apply.
        window: The window duration ``W`` the frame was aggregated over,
            as a :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string.

    Returns:
        For :attr:`QualityMode.PASS_THROUGH`: ``frame``, unchanged.
        For :attr:`QualityMode.FILTER`: a new frame with every row whose
        ``coverage_seconds`` falls below the threshold dropped, other
        rows and their order preserved.
        For :attr:`QualityMode.GATE` in :attr:`GateMode.STRICT`: ``frame``,
        unchanged, when every row meets the threshold.
        For :attr:`QualityMode.GATE` in :attr:`GateMode.REPORT`: the
        :class:`QualityReport`, always.

    Raises:
        ConfigError: If ``frame`` is missing a required column, or if
            ``window`` cannot be coerced to a
            :class:`~ohlc_toolkit.temporal.Duration`.
        CoverageError: For :attr:`QualityMode.GATE` in
            :attr:`GateMode.STRICT`, when any row falls below the
            threshold. The message names the first offending row's
            ``close_time`` and a bounded summary: the offending-row count
            and the threshold.

    """
    _require_quality_columns(frame)
    window_duration = coerce_duration(window)

    if policy.mode is QualityMode.PASS_THROUGH:
        logger.debug("Quality policy pass-through: {} row(s) unchanged.", frame.height)
        return frame

    threshold_seconds = _threshold_seconds(
        policy.min_coverage, window_duration.total_seconds
    )

    if policy.mode is QualityMode.FILTER:
        filtered = _filter_frame(frame, threshold_seconds)
        logger.debug(
            "Quality policy filter: kept {}/{} row(s) at >= {}s coverage.",
            filtered.height,
            frame.height,
            threshold_seconds,
        )
        return filtered

    report = _evaluate_gate(frame, threshold_seconds)

    if policy.gate_mode is GateMode.REPORT:
        if not report.passed:
            logger.warning(
                "Quality gate (report): {}/{} row(s) below the {}s coverage threshold.",
                report.offending_count,
                report.rows_checked,
                threshold_seconds,
            )
        return report

    if not report.passed:
        logger.error(
            "Quality gate (strict): {}/{} row(s) below the {}s coverage "
            "threshold; first offending close_time={}.",
            report.offending_count,
            report.rows_checked,
            threshold_seconds,
            report.first_offending_close_time,
        )
        raise CoverageError(
            f"Window quality gate failed: {report.offending_count}/"
            f"{report.rows_checked} row(s) have coverage_seconds below the "
            f"required {threshold_seconds}s threshold; first offending "
            f"close_time={report.first_offending_close_time}."
        )

    logger.debug(
        "Quality gate (strict): all {} row(s) meet the {}s coverage threshold.",
        report.rows_checked,
        threshold_seconds,
    )
    return frame
