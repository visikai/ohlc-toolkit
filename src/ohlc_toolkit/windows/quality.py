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

``min_coverage`` is a fraction in ``[0, 1]``, which a caller writes as a
decimal literal. The threshold is that literal's DECIMAL INTENT times
the window, held exactly: ``Fraction(str(min_coverage))`` recovers the
decimal as an exact rational, and multiplying by the integer window
gives an exact rational threshold ``p / q`` in lowest terms. A row is
kept (``FILTER``) or counted as covered (``GATE``) exactly when

    ``coverage_seconds * q >= p``

Both sides are ordinary integers -- ``coverage_seconds`` is an exact
whole-second count, which this module requires of the column it reads --
so no float ever touches the decision, and there is no rounding rule
left to get wrong in either direction.

That integer comparison is applied in the algebraically identical form
``coverage_seconds >= least_integer_at_or_above(p / q)``, with the bound
derived once in Python. The two are the same predicate for every integer
``coverage_seconds`` (that is the only reason the second form is used),
and the second never has to materialize ``coverage_seconds * q``:
``min_coverage`` may legally be a subnormal such as ``5e-324``, whose
decimal intent carries a 322-digit denominator that no dataframe literal
can hold. Exactness is kept and the frame-side arithmetic stays small.
:attr:`QualityReport.threshold_seconds` still reports the unrounded
``p / q``.

Two simpler formulations are wrong at the boundary, in opposite
directions, and both were tried here first:

- The ordinary IEEE-754 product ``min_coverage * window_seconds`` is too
  STRICT. Correct rounding of the multiplication does not rescue it,
  because the multiplicand is already not the decimal the caller wrote:
  ``0.55 * 180`` evaluates to ``99.00000000000001``, ``0.56 * 100`` to
  ``56.00000000000001``, and ``0.17 * 300`` to ``51.00000000000001``. A
  window at exactly 55% of a three-minute window is then dropped for
  missing a threshold nobody asked for. Sweeping every two-decimal
  fraction in ``[0, 1]`` against every whole-second window from ``1s``
  to ``1h`` -- 363,600 pairs -- turns up 571 pairs where the float
  product and the exact threshold disagree about some integer coverage.
  All 571 are too strict; none are too loose.
- Taking ``Fraction(min_coverage)`` -- the exact value of the STORED
  DOUBLE rather than of the decimal -- and rounding the product up is
  too strict on a different set of pairs. ``0.9`` is stored as a hair
  above nine tenths, so ``Fraction(0.9) * 300`` is a hair above ``270``
  and ceils to ``271``: a row at exactly 90% coverage of a five-minute
  window fails a 90% policy.

Reading the literal's decimal intent settles both cases:
``Fraction("0.55") * 180`` is exactly ``99``, and ``Fraction("0.9") *
300`` is exactly ``270``. ``str(float)`` is the shortest string that
round-trips back to the same double, so it recovers the decimal the
caller wrote whenever they wrote one -- which is why
:meth:`WindowQualityPolicy.to_dict` can go on storing a plain float and
still name the same threshold after a JSON round trip.

``min_coverage=1.0`` reduces to the full-coverage requirement
``coverage_seconds == W`` for any real engine output, because the
``coverage_seconds`` a window can report never exceeds ``W``.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, unique
from fractions import Fraction
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

    The window duration ``W`` is deliberately NOT part of this identity,
    even though the threshold is meaningless without it. ``W`` belongs to
    the schedule the frame was aggregated over, and a recipe already
    records that schedule; duplicating it here would create a second
    copy to keep in step, and a policy whose recorded ``W`` disagreed
    with the schedule's would be a contradiction with no obvious winner.
    The consequence is explicit and intended: the same policy identity
    applied to two frames aggregated over different windows filters them
    differently, because "90% of a window" is a different number of
    seconds for each. Reproducibility comes from the pair -- this
    identity together with the recorded schedule -- not from this
    identity alone.

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

        ``min_coverage`` is stored as the plain float, not as a decimal
        string: JSON has no rational type, and the threshold is derived
        from ``str(min_coverage)``, which is the shortest string that
        round-trips back to the same double. A stored float therefore
        names the same threshold after a round trip as it did before one.

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
        threshold_seconds: The exact minimum ``coverage_seconds`` a row
            had to reach to pass, derived from ``min_coverage`` and the
            window duration. A :class:`~fractions.Fraction` rather than a
            float, so it IS the decision boundary the check applied
            rather than the nearest double to it: a caller re-deriving a
            verdict from this value gets the same answer the policy gave.
            It compares directly against ``int`` and ``float``.
        offending_count: How many rows fell below ``threshold_seconds``.
        first_offending_close_time: The ``close_time`` of the first
            offending row, in frame order. ``None`` when nothing
            offended, including over an empty frame.

    """

    rows_checked: int
    threshold_seconds: Fraction
    offending_count: int
    first_offending_close_time: int | None

    @property
    def passed(self) -> bool:
        """Report whether every row met the coverage threshold."""
        return self.offending_count == 0


def _require_quality_columns(frame: pl.DataFrame) -> None:
    """Check that the frame carries every column this step reads, in the right kind.

    ``coverage_seconds`` must be an integer column. That is what the
    engine emits -- a whole number of seconds, never accumulated -- and
    the threshold comparison relies on it: an exact rational threshold
    can be compared against a whole second exactly, whereas a fractional
    coverage would have no exact answer to give. Refusing such a column
    here says so, rather than quietly answering a slightly different
    question.

    Raises:
        ConfigError: If ``close_time``, ``src_count``, or
            ``coverage_seconds`` is absent from ``frame``, or if
            ``coverage_seconds`` is not an integer column.

    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        logger.warning("Rejecting frame missing quality column(s): {}", missing)
        raise ConfigError(
            f"A window quality policy requires column(s) {missing}; apply it "
            "to an engine-produced window frame."
        )

    coverage_dtype = frame.schema["coverage_seconds"]
    if not coverage_dtype.is_integer():
        logger.warning("Rejecting non-integer coverage_seconds: {}", coverage_dtype)
        raise ConfigError(
            "coverage_seconds must be an integer count of whole seconds, got "
            f"{coverage_dtype}; apply this policy to an engine-produced window "
            "frame."
        )


def _threshold_seconds(min_coverage: float, window_seconds: int) -> Fraction:
    """Compute the exact minimum ``coverage_seconds`` a row must reach to pass.

    See the module docstring's "Threshold rounding" section for why this
    reads ``min_coverage``'s decimal intent via ``str`` rather than
    multiplying in floating point or converting the stored double.

    Args:
        min_coverage: The minimum coverage fraction, in ``[0, 1]``.
        window_seconds: The window duration ``W``, in seconds.

    Returns:
        The exact threshold, in seconds, in lowest terms. A row passes
        when its ``coverage_seconds >= `` this value.

    """
    return Fraction(str(min_coverage)) * window_seconds


def _least_passing_seconds(threshold_seconds: Fraction) -> int:
    """Return the smallest whole second that meets the exact threshold.

    For an integer ``c``, ``c * q >= p`` holds exactly when ``c`` is at
    least the least integer at or above ``p / q``, so this bound decides
    every row the same way the exact integer comparison would. See the
    module docstring's "Threshold rounding" section for why the bound is
    derived here rather than cross-multiplied on the frame.

    Args:
        threshold_seconds: The exact threshold, in lowest terms.

    Returns:
        The least whole second at or above ``threshold_seconds``.

    """
    return -(-threshold_seconds.numerator // threshold_seconds.denominator)


def _filter_frame(frame: pl.DataFrame, minimum_seconds: int) -> pl.DataFrame:
    """Return a new frame holding only rows at or above the threshold."""
    return frame.filter(pl.col("coverage_seconds") >= minimum_seconds)


def _evaluate_gate(
    frame: pl.DataFrame, threshold_seconds: Fraction, minimum_seconds: int
) -> QualityReport:
    """Check every row's coverage against the threshold, without mutating."""
    coverage = frame.get_column("coverage_seconds")
    mask = coverage < minimum_seconds
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
    minimum_seconds = _least_passing_seconds(threshold_seconds)

    if policy.mode is QualityMode.FILTER:
        filtered = _filter_frame(frame, minimum_seconds)
        logger.debug(
            "Quality policy filter: kept {}/{} row(s) at >= {}s coverage.",
            filtered.height,
            frame.height,
            minimum_seconds,
        )
        return filtered

    report = _evaluate_gate(frame, threshold_seconds, minimum_seconds)

    if policy.gate_mode is GateMode.REPORT:
        if not report.passed:
            logger.warning(
                "Quality gate (report): {}/{} row(s) below the {}s coverage minimum.",
                report.offending_count,
                report.rows_checked,
                minimum_seconds,
            )
        return report

    if not report.passed:
        logger.error(
            "Quality gate (strict): {}/{} row(s) below the {}s coverage "
            "minimum; first offending close_time={}.",
            report.offending_count,
            report.rows_checked,
            minimum_seconds,
            report.first_offending_close_time,
        )
        raise CoverageError(
            f"Window quality gate failed: {report.offending_count}/"
            f"{report.rows_checked} row(s) have coverage_seconds below the "
            f"required minimum of {minimum_seconds}s; first offending "
            f"close_time={report.first_offending_close_time}."
        )

    logger.debug(
        "Quality gate (strict): all {} row(s) meet the {}s coverage minimum.",
        report.rows_checked,
        minimum_seconds,
    )
    return frame
