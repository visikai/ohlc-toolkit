"""A window quality-policy step, composed after the aggregator.

This module is not part of the engine or the oracle: it consumes their
nine-column output (:mod:`ohlc_toolkit.windows.engine`,
:mod:`ohlc_toolkit.windows.reference`) as an independent, later step, and
never feeds back into either. It reads exactly two of those nine columns
-- ``coverage_seconds``, and ``close_time`` for naming an offending row.
It never reads or alters ``open``, ``high``, ``low``, ``close``,
``volume``, ``open_time``, or ``src_count``.

A :class:`WindowQualityPolicy` is a frozen, JSON-round-trippable identity
-- a recipe can record it the same way it records a schedule -- and
:func:`apply_quality_policy` is the single entry point that interprets
one against a frame. Every mode that returns at all returns the same
thing: a :class:`QualityPolicyResult` pairing the resulting frame with
the :class:`QualityReport` measured over the input. One shape, in every
mode, so no caller has to discriminate a return value by type, and no
mode's findings are thrown away:

- ``PASS_THROUGH`` returns the frame unchanged: still an explicit,
  recorded step, useful so a recipe can name "no quality policy" the same
  way it names any other choice. Its report still measures the frame, so
  a recorded no-op records what it declined to act on.
- ``FILTER`` drops rows whose ``coverage_seconds`` falls below the
  policy's threshold -- and rows that state no coverage at all -- and
  returns a new frame. The input is never mutated, row order is
  otherwise preserved, and no OHLCV value is touched. The report
  accounts for exactly the rows dropped: both come from one mask, so
  they cannot disagree.
- ``GATE`` checks the same threshold without dropping anything. In
  ``GateMode.STRICT`` a violation logs and raises
  :class:`WindowCoverageError`, which carries the whole
  :class:`QualityReport`; in ``GateMode.REPORT`` it never raises. This
  mirrors the strict/report split in
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
left to get wrong in either direction. A row whose ``coverage_seconds``
is null meets nothing: it has not been shown to reach the threshold, so
every mode treats it as offending and ``FILTER`` drops it.

That integer comparison is applied in the algebraically identical form
``coverage_seconds >= least_integer_at_or_above(p / q)``, with the bound
derived once in Python. The two are the same predicate for every integer
``coverage_seconds`` (that is the only reason the second form is used),
and the second never has to materialize ``coverage_seconds * q``:
``min_coverage`` may legally be a subnormal such as ``5e-324``, whose
decimal intent carries a 324-digit denominator that no dataframe literal
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
from ohlc_toolkit.temporal import (
    ConfigError,
    CoverageError,
    Duration,
    validate_window_duration,
)

logger = get_logger(__name__)

# The only columns this step reads, and therefore the only ones it
# requires. Requiring them up front -- present, and both as Int64
# specifically -- rather than letting a missing column surface
# later as a bare polars ColumnNotFoundError or a narrow integer width
# as a bare OverflowError, keeps the failure at this module's own
# boundary and in this module's own words.
#
# `src_count` is deliberately not among them. It was required and never
# read: a requirement that refuses a frame for lacking something no
# check consults buys no safety, and would refuse a projection carrying
# exactly the two columns this step does consult. It is also redundant
# with what is here -- an engine window's coverage is its source count
# times the cadence -- and this step has no cadence with which to read
# it back the other way, so there is nothing it could add to a report.
_REQUIRED_COLUMNS = ("close_time", "coverage_seconds")


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
        STRICT: Raise :class:`WindowCoverageError` -- a
            :class:`~ohlc_toolkit.temporal.errors.CoverageError` carrying
            the report -- when any row falls below the threshold.
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
        _validated_min_coverage(self.min_coverage)

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

        min_coverage = _validated_min_coverage(data["min_coverage"])
        return cls(mode=mode, min_coverage=min_coverage, gate_mode=gate_mode)


def _validated_min_coverage(value: object) -> float:
    """Return a ``min_coverage`` as a float, rejecting anything unusable.

    Returning the validated value, rather than only raising, is what lets
    :meth:`WindowQualityPolicy.from_dict` pass a value read out of an
    untyped mapping straight to the constructor: the check has already
    established it is a number, and saying so in the return type spares
    the caller a cast the checker would otherwise have to be told to
    ignore.

    Args:
        value: The candidate ``min_coverage``, of any type.

    Returns:
        ``value`` as a ``float``.

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
    return float(value)


@dataclass(frozen=True)
class QualityReport:
    """The outcome of measuring a frame against a policy's coverage threshold.

    Produced in every mode, including the ones that do not act on it: a
    ``PASS_THROUGH`` policy records what it declined to act on, and a
    ``FILTER`` policy's report accounts for exactly the rows it dropped.

    Attributes:
        rows_checked: How many rows were present in the evaluated frame.
        threshold_seconds: The exact minimum ``coverage_seconds`` a row
            had to reach to pass, derived from ``min_coverage`` and the
            window duration. A :class:`~fractions.Fraction` rather than a
            float, so it IS the decision boundary the check applied
            rather than the nearest double to it: a caller re-deriving a
            verdict from this value gets the same answer the policy gave.
            It compares directly against ``int`` and ``float``.
        offending_count: How many rows failed to meet
            ``threshold_seconds`` -- rows below it, plus rows that stated
            no coverage at all.
        null_coverage_count: How many of those offending rows had a null
            ``coverage_seconds``. Always ``<= offending_count``. Reported
            separately because "below the bar" and "no measurement" are
            different problems with different fixes, even though the gate
            refuses both.
        first_offending_close_time: The ``close_time`` of the first
            offending row, in ROW order -- the frame is never sorted and
            no sortedness is assumed, so on an out-of-order frame this
            need not be the offender with the smallest ``close_time``.
            ``None`` when nothing offended, including over an empty
            frame.

    """

    rows_checked: int
    threshold_seconds: Fraction
    offending_count: int
    null_coverage_count: int
    first_offending_close_time: int | None

    @property
    def passed(self) -> bool:
        """Report whether every row met the coverage threshold."""
        return self.offending_count == 0


class WindowCoverageError(CoverageError):
    """A window frame failed a strict :attr:`QualityMode.GATE` policy.

    Carries the full :class:`QualityReport` as a typed attribute rather
    than as a dynamically-set one, so callers can read the offending
    count, the exact threshold, and the first offending ``close_time``
    directly, with no message parsing, no runtime attribute check, and no
    ``type: ignore`` required. Mirrors
    :class:`~ohlc_toolkit.source.validation.SourceValidationError`, which
    does the same for a raw source frame.

    Subclasses :class:`~ohlc_toolkit.temporal.errors.CoverageError`, so
    code that catches the taxonomy's base class keeps working.

    Attributes:
        report: The quality report that triggered this error.

    """

    def __init__(self, message: str, report: QualityReport) -> None:
        """Store the message and the report that produced it.

        Args:
            message: A short, human-readable summary of the failure.
            report: The full quality report for the evaluated frame.

        """
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class QualityPolicyResult:
    """A window frame paired with the quality report measured over its input.

    Returned by :func:`apply_quality_policy` from every non-raising path,
    so the frame and its (possibly failing) report travel together
    instead of as a bare, order-ambiguous tuple -- and so no mode has to
    be told apart from another by the type of what it returned. Mirrors
    :class:`~ohlc_toolkit.source.reader.SourceReadResult`, which does the
    same for a raw source frame and its validation report.

    Attributes:
        frame: The resulting frame: the input unchanged for
            ``PASS_THROUGH`` and for a ``GATE`` that did not raise, or a
            new row-subset for ``FILTER``.
        report: The report measured over the INPUT frame, whatever mode
            produced it. Its ``rows_checked`` is therefore the input's
            height, not ``frame``'s, and for ``FILTER``
            ``frame.height == report.rows_checked -
            report.offending_count``.

    """

    frame: pl.DataFrame
    report: QualityReport


def _require_quality_columns(frame: pl.DataFrame) -> None:
    """Check that the frame carries every column this step reads, in the right kind.

    ``coverage_seconds`` must be an ``Int64`` column -- exactly the type
    the engine emits and the schema declares, not merely any integer
    width. The threshold comparison relies on the integer part: an exact
    rational threshold can be compared against a whole second exactly,
    whereas a fractional coverage would have no exact answer to give.
    The width matters separately: a narrower column overflows inside
    polars when compared against a large whole-second minimum (``Int8``
    against a one-hour window), and ``UInt64`` cannot be safely widened
    -- a strict cast raises near the top of its range and a lenient one
    yields a null, which this gate exists to refuse. Refusing every
    other width here keeps both failures at this module's boundary and
    in this module's words.

    ``close_time`` is held to the same word. The report names the first
    offending row by reading this column with ``int(...)``: a Float64
    close time would be silently truncated into a name that is not a row
    in the frame, and a String or Datetime one would surface as a
    foreign TypeError only when an offender exists -- so a clean frame
    with the wrong kind would pass while a dirty one crashed elsewhere.

    Raises:
        ConfigError: If ``close_time`` or ``coverage_seconds`` is absent
            from ``frame``, or if either is not an ``Int64`` column.

    """
    missing = [name for name in _REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        logger.warning("Rejecting frame missing quality column(s): {}", missing)
        raise ConfigError(
            f"A window quality policy requires column(s) {missing}; apply it "
            "to an engine-produced window frame."
        )

    coverage_dtype = frame.schema["coverage_seconds"]
    if coverage_dtype != pl.Int64:
        logger.warning("Rejecting non-Int64 coverage_seconds: {}", coverage_dtype)
        raise ConfigError(
            "coverage_seconds must be an Int64 count of whole seconds, got "
            f"{coverage_dtype}; apply this policy to an engine-produced window "
            "frame."
        )

    close_time_dtype = frame.schema["close_time"]
    if close_time_dtype != pl.Int64:
        logger.warning("Rejecting non-Int64 close_time: {}", close_time_dtype)
        raise ConfigError(
            "close_time must be an Int64 Unix second, got "
            f"{close_time_dtype}; apply this policy to an engine-produced "
            "window frame."
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


def _offending_mask(frame: pl.DataFrame, minimum_seconds: int) -> pl.Series:
    """Return the row mask of coverages that do not meet the threshold.

    Never mutates ``frame``. Every mode decides from this one mask --
    what ``FILTER`` drops is what the report counts -- so the returned
    frame and the report it travels with cannot disagree about which rows
    offended.

    A null ``coverage_seconds`` compares to neither side, and filling
    that null verdict with ``True`` is what makes this gate fail closed:
    a row that states no coverage has not been shown to meet the
    threshold, and left as a null it would vanish from the count
    entirely -- ``Series.sum`` skips nulls -- so a strict gate would
    report a clean frame while having checked one row fewer than it
    said. :mod:`ohlc_toolkit.source.validation` treats a null in a column
    it reads the same way, for the same reason.
    """
    return (frame.get_column("coverage_seconds") < minimum_seconds).fill_null(
        value=True
    )


def _build_report(
    frame: pl.DataFrame, mask: pl.Series, threshold_seconds: Fraction
) -> QualityReport:
    """Summarize an offending-row mask into a bounded report."""
    offending_count = int(mask.sum())
    null_coverage_count = frame.get_column("coverage_seconds").null_count()

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
        null_coverage_count=null_coverage_count,
        first_offending_close_time=first_offending_close_time,
    )


def _gate_failure_message(report: QualityReport, minimum_seconds: int) -> str:
    """Summarize a failed strict gate in one bounded line.

    Every number here comes from the report, which is attached to the
    raised error too: this is a convenience for a human reading a
    traceback, never the only way to reach a finding.
    """
    unstated = (
        f" {report.null_coverage_count} of those state no coverage at all."
        if report.null_coverage_count
        else ""
    )
    return (
        f"Window quality gate failed: {report.offending_count}/"
        f"{report.rows_checked} row(s) do not meet the required "
        f"coverage_seconds minimum of {minimum_seconds}s; first offending "
        f"close_time={report.first_offending_close_time}.{unstated}"
    )


def apply_quality_policy(
    frame: pl.DataFrame,
    policy: WindowQualityPolicy,
    *,
    window: Duration | str,
) -> QualityPolicyResult:
    """Apply a window quality policy to an engine-produced window frame.

    Never mutates ``frame``. Never reads or alters ``open``, ``high``,
    ``low``, ``close``, or ``volume``.

    Args:
        frame: A window frame such as
            :func:`~ohlc_toolkit.windows.engine.compute_windows` produces,
            carrying at least ``close_time`` and an integer
            ``coverage_seconds``.
        policy: The policy identity to apply.
        window: The window duration ``W`` the frame was aggregated over,
            as a :class:`~ohlc_toolkit.temporal.Duration` or a compact
            duration string.

    Returns:
        A :class:`QualityPolicyResult` in every mode that returns at all,
        pairing the resulting frame with the report measured over
        ``frame``. Its ``.frame`` is ``frame`` itself for
        :attr:`QualityMode.PASS_THROUGH` and for a :attr:`QualityMode.GATE`
        that did not raise, and a new row-subset -- rows below the
        threshold or stating no coverage dropped, the rest in their
        original order -- for :attr:`QualityMode.FILTER`.

    Raises:
        ConfigError: If ``frame`` is missing a required column, if
            either required column is not ``Int64``, or if
            ``window`` cannot be coerced to a
            :class:`~ohlc_toolkit.temporal.Duration` or coerces to the
            zero duration. A zero window is refused for the same reason
            :func:`~ohlc_toolkit.windows.resolution.resolve_schedule`
            refuses one: here, every row meets a 0s threshold, so
            accepting it would quietly disarm the gate.
        WindowCoverageError: For :attr:`QualityMode.GATE` in
            :attr:`GateMode.STRICT`, when any row falls below the
            threshold. A :class:`~ohlc_toolkit.temporal.errors.CoverageError`
            carrying the whole :class:`QualityReport` as ``.report``. Its
            message repeats a bounded summary of that report: the
            offending-row count, the whole-second minimum, and the FIRST
            OFFENDING row's ``close_time``. "First" means first in ROW
            order. This function states no sortedness precondition and
            never sorts, so on a frame whose rows are not in time order
            the row it names need not be the offender with the smallest
            ``close_time``.

    """
    _require_quality_columns(frame)
    window_duration = validate_window_duration(window)

    threshold_seconds = _threshold_seconds(
        policy.min_coverage, window_duration.total_seconds
    )
    minimum_seconds = _least_passing_seconds(threshold_seconds)
    mask = _offending_mask(frame, minimum_seconds)
    report = _build_report(frame, mask, threshold_seconds)

    if policy.mode is QualityMode.PASS_THROUGH:
        logger.debug("Quality policy pass-through: {} row(s) unchanged.", frame.height)
        return QualityPolicyResult(frame=frame, report=report)

    if policy.mode is QualityMode.FILTER:
        filtered = frame.filter(~mask)
        logger.debug(
            "Quality policy filter: kept {}/{} row(s) at >= {}s coverage.",
            filtered.height,
            frame.height,
            minimum_seconds,
        )
        return QualityPolicyResult(frame=filtered, report=report)

    if policy.gate_mode is GateMode.REPORT:
        if not report.passed:
            logger.warning(
                "Quality gate (report): {}/{} row(s) miss the {}s coverage "
                "minimum ({} state none at all).",
                report.offending_count,
                report.rows_checked,
                minimum_seconds,
                report.null_coverage_count,
            )
        return QualityPolicyResult(frame=frame, report=report)

    if not report.passed:
        logger.error(
            "Quality gate (strict): {}/{} row(s) miss the {}s coverage minimum "
            "({} state none at all); first offending close_time={}.",
            report.offending_count,
            report.rows_checked,
            minimum_seconds,
            report.null_coverage_count,
            report.first_offending_close_time,
        )
        raise WindowCoverageError(
            _gate_failure_message(report, minimum_seconds), report
        )

    logger.debug(
        "Quality gate (strict): all {} row(s) meet the {}s coverage minimum.",
        report.rows_checked,
        minimum_seconds,
    )
    return QualityPolicyResult(frame=frame, report=report)
