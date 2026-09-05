"""Join a sparse interval sidecar onto window frames as opaque flags.

This is a later step over the aggregator's output, in the same sense
that :mod:`ohlc_toolkit.windows.quality` is: it consumes a window frame
such as :func:`~ohlc_toolkit.windows.engine.compute_windows` produces,
reads exactly two of its columns -- ``open_time`` and ``close_time`` --
and appends exactly two of its own. It never reads or alters ``open``,
``high``, ``low``, ``close``, ``volume``, ``src_count`` or
``coverage_seconds``, and never feeds back into the engine or the oracle.

An annotation is a half-open interval ``[start, end)`` carrying a flag.
A window is the half-open interval ``[open_time, close_time)``. The two
overlap by ``max(0, min(close_time, end) - max(open_time, start))``
seconds, so an interval that ends exactly at a window's open, or starts
exactly at its close, touches nothing -- half-open on both sides, as the
window contract and the sidecar contract each state of themselves.

For every window the transform records two things:

- ``<prefix>_flags``: the DISTINCT flags of every interval overlapping
  the window, sorted, as a list of strings; an empty list when none does.
- ``<prefix>_overlap_seconds``: how many seconds of the window fall
  inside the UNION of all annotated intervals. Two intervals covering the
  same seconds count those seconds once, so the value never exceeds the
  window's own length, whatever the sidecar repeats.

Flags are opaque. Nothing here checks them against any allowed set: the
sidecar is owned by whoever publishes it, and this transform reports what
it says rather than judging it. Interpreting a flag -- masking, excluding,
weighting -- is a downstream decision this module does not make.

Cost is one vectorised expression per interval over the whole frame, so
the work is proportional to windows times intervals. That is the right
shape for what a sidecar is -- a sparse table of dozens to thousands of
operational events beside millions of candles -- and the wrong shape for
a dense per-row annotation, which is not what this step is for.
"""

import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from ohlc_toolkit.config.logging import get_logger
from ohlc_toolkit.temporal import ConfigError, bounded_echo

logger = get_logger(__name__)

# The two window columns this step reads, and what each must be.
_WINDOW_BOUND_COLUMNS = ("open_time", "close_time")
_INT64_UNIX_SECOND = "an Int64 Unix second"

# The column names appended, before the caller's prefix.
_FLAGS_SUFFIX = "flags"
_OVERLAP_SUFFIX = "overlap_seconds"

# The prefix the two appended columns carry unless a caller names another.
DEFAULT_ANNOTATION_PREFIX = "annotation"


@dataclass(frozen=True)
class AnnotationColumns:
    """Which columns of an annotation frame hold the interval and its flag.

    The defaults are the sidecar contract's own names, so a published
    provenance file is read without configuration; any other layout names
    its three columns here. Columns beyond these three are ignored by the
    transform and kept by the reader.

    Attributes:
        start: The column holding each interval's inclusive start, as an
            Int64 Unix second.
        end: The column holding each interval's exclusive end, as an
            Int64 Unix second.
        flag: The column holding each interval's opaque String flag.

    """

    start: str = "start_timestamp"
    end: str = "end_timestamp"
    flag: str = "flag"

    def __post_init__(self) -> None:
        """Refuse a name that is not a non-empty string, or one used twice.

        Raises:
            ConfigError: If any of the three names is not a ``str``, is
                empty, or repeats another.

        """
        for label, name in (
            ("start", self.start),
            ("end", self.end),
            ("flag", self.flag),
        ):
            if not isinstance(name, str):
                logger.warning(
                    "Rejecting a non-string annotation {} column: {}",
                    label,
                    type(name).__name__,
                )
                raise ConfigError(
                    f"Annotation {label} column must be a str, got "
                    f"{type(name).__name__}."
                )
            if not name:
                logger.warning("Rejecting an empty annotation {} column name.", label)
                raise ConfigError(f"Annotation {label} column name must not be empty.")
        names = (self.start, self.end, self.flag)
        if len(set(names)) < len(names):
            logger.warning(
                "Rejecting annotation columns that repeat a name: {}, {}, {}.",
                bounded_echo(self.start),
                bounded_echo(self.end),
                bounded_echo(self.flag),
            )
            raise ConfigError(
                "Annotation start, end and flag columns must be three distinct "
                f"names, got {bounded_echo(self.start)}, {bounded_echo(self.end)} "
                f"and {bounded_echo(self.flag)}."
            )


# One shared default rather than a fresh instance per call, so a caller who
# passes nothing and a caller who passes ``AnnotationColumns()`` mean the
# same thing and the signature carries no call in a default.
DEFAULT_ANNOTATION_COLUMNS = AnnotationColumns()


def annotate_windows(
    frame: pl.DataFrame,
    annotations: pl.DataFrame,
    *,
    columns: AnnotationColumns = DEFAULT_ANNOTATION_COLUMNS,
    prefix: str = DEFAULT_ANNOTATION_PREFIX,
) -> pl.DataFrame:
    """Append each window's overlapping flags and overlap seconds.

    Never mutates ``frame`` or ``annotations``. Never reads or alters
    ``open``, ``high``, ``low``, ``close``, ``volume``, ``src_count`` or
    ``coverage_seconds``; the returned frame is ``frame``'s columns, in
    their order, followed by exactly two new ones.

    Args:
        frame: A window frame such as
            :func:`~ohlc_toolkit.windows.engine.compute_windows` produces,
            carrying at least ``open_time`` and ``close_time`` as Int64
            Unix seconds.
        annotations: One row per half-open interval ``[start, end)`` with
            its flag, in the columns ``columns`` names. Other columns are
            ignored. Zero rows is a legitimate sidecar and annotates
            nothing.
        columns: Which columns of ``annotations`` hold the start, the end
            and the flag.
        prefix: The stem of the two appended columns,
            ``<prefix>_flags`` and ``<prefix>_overlap_seconds``.

    Returns:
        ``frame`` with ``<prefix>_flags`` (List of String: the distinct
        overlapping flags, sorted, empty when none) and
        ``<prefix>_overlap_seconds`` (Int64: seconds of the window inside
        the union of all intervals) appended.

    Raises:
        ConfigError: If ``frame`` is not a DataFrame or lacks an Int64
            ``open_time`` or ``close_time``; if ``annotations`` is not a
            DataFrame, lacks a named column, holds a start or end that is
            not Int64 or a flag that is not String, holds a null in any
            of the three, or holds an interval whose end does not exceed
            its start; if ``prefix`` is not a non-empty ``str``; or if
            ``frame`` already carries either column this call would add.

    """
    _require_window_bounds(frame)
    intervals = _require_annotations(annotations, columns)
    flags_column, overlap_column = _output_columns(prefix)
    _require_absent_columns(frame, (flags_column, overlap_column))

    open_time = pl.col("open_time")
    close_time = pl.col("close_time")
    rows = [
        (int(start), int(end), str(flag)) for start, end, flag in intervals.iter_rows()
    ]

    if rows:
        hits = [
            pl.when((open_time < end) & (close_time > start))
            .then(pl.lit(flag))
            .otherwise(None)
            for start, end, flag in rows
        ]
        flags = pl.concat_list(hits).list.drop_nulls().list.unique().list.sort()
        overlaps = [
            (
                pl.min_horizontal(close_time, pl.lit(end))
                - pl.max_horizontal(open_time, pl.lit(start))
            ).clip(lower_bound=0)
            for start, end in _merged(rows)
        ]
        overlap = pl.sum_horizontal(overlaps)
    else:
        flags = pl.lit([], dtype=pl.List(pl.String))
        overlap = pl.lit(0)

    annotated = frame.with_columns(
        flags.cast(pl.List(pl.String)).alias(flags_column),
        overlap.cast(pl.Int64).alias(overlap_column),
    )
    logger.debug(
        "Annotated {} window(s) against {} interval(s).", frame.height, len(rows)
    )
    return annotated


def read_annotations(
    path: str | os.PathLike[str],
    *,
    columns: AnnotationColumns = DEFAULT_ANNOTATION_COLUMNS,
) -> pl.DataFrame:
    """Read an interval sidecar CSV, typing and checking its three columns.

    Every column the file carries is kept; the three ``columns`` names are
    read as Int64, Int64 and String and held to the same rules
    :func:`annotate_windows` applies, so a file this returns is one that
    function accepts. Rows come back sorted by start.

    Args:
        path: The CSV file, with a header row.
        columns: Which columns hold the start, the end and the flag.

    Returns:
        The sidecar as a frame, sorted by its start column.

    Raises:
        FileNotFoundError: If ``path`` is not an existing file.
        ConfigError: If the file cannot be parsed into the declared
            kinds, lacks a named column, or holds a null or an interval
            whose end does not exceed its start.

    """
    resolved = Path(path)
    if not resolved.is_file():
        logger.error("Annotation file {} does not exist.", bounded_echo(str(resolved)))
        raise FileNotFoundError(
            f"Annotation file {bounded_echo(str(resolved))} does not exist."
        )
    logger.debug("Reading annotations from {}.", bounded_echo(str(resolved)))
    declared = {
        columns.start: pl.Int64,
        columns.end: pl.Int64,
        columns.flag: pl.String,
    }
    try:
        header = pl.read_csv(resolved, n_rows=0).columns
        table = pl.read_csv(
            resolved,
            schema_overrides={
                name: kind for name, kind in declared.items() if name in header
            },
        )
    except pl.exceptions.PolarsError as error:
        logger.error(
            "Annotation file {} could not be read: {}",
            bounded_echo(str(resolved)),
            bounded_echo(error),
        )
        raise ConfigError(
            f"Annotation file {bounded_echo(str(resolved))} could not be read "
            f"as the declared kinds: {bounded_echo(error)}"
        ) from error
    _require_annotations(table, columns)
    logger.info(
        "Read {} annotation interval(s) from {}.",
        table.height,
        bounded_echo(str(resolved)),
    )
    return table.sort(columns.start)


def _require_window_bounds(frame: object) -> None:
    """Check that ``frame`` is a window frame carrying Int64 bounds.

    Raises:
        ConfigError: If ``frame`` is not a DataFrame, lacks ``open_time``
            or ``close_time``, or carries either as anything but Int64.

    """
    if not isinstance(frame, pl.DataFrame):
        logger.warning(
            "Rejecting a non-DataFrame window frame: {}", type(frame).__name__
        )
        raise ConfigError(
            f"Windows to annotate must be a polars DataFrame, got "
            f"{type(frame).__name__}."
        )
    missing = [name for name in _WINDOW_BOUND_COLUMNS if name not in frame.columns]
    if missing:
        logger.warning("Rejecting a window frame missing column(s): {}", missing)
        raise ConfigError(
            f"Annotating windows requires column(s) {missing}; apply this step "
            "to an engine-produced window frame."
        )
    for name in _WINDOW_BOUND_COLUMNS:
        dtype = frame.schema[name]
        if dtype != pl.Int64:
            logger.warning("Rejecting non-Int64 {}: {}", name, bounded_echo(dtype))
            raise ConfigError(
                f"{name} must be {_INT64_UNIX_SECOND}, got {bounded_echo(dtype)}; "
                "apply this step to an engine-produced window frame."
            )


def _require_annotations(
    annotations: object, columns: AnnotationColumns
) -> pl.DataFrame:
    """Check an annotation frame and return its three columns, start-sorted.

    Raises:
        ConfigError: If ``annotations`` is not a DataFrame, lacks a named
            column, holds a start or end that is not Int64 or a flag that
            is not String, holds a null in any of the three, or holds an
            interval whose end does not exceed its start.

    """
    if not isinstance(annotations, pl.DataFrame):
        logger.warning(
            "Rejecting non-DataFrame annotations: {}", type(annotations).__name__
        )
        raise ConfigError(
            f"Annotations must be a polars DataFrame, got {type(annotations).__name__}."
        )
    declared = (
        (columns.start, pl.Int64, _INT64_UNIX_SECOND),
        (columns.end, pl.Int64, _INT64_UNIX_SECOND),
        (columns.flag, pl.String, "a String flag"),
    )
    missing = [name for name, _, _ in declared if name not in annotations.columns]
    if missing:
        echoed = ", ".join(bounded_echo(name) for name in missing)
        logger.warning("Rejecting annotations missing column(s): {}", echoed)
        raise ConfigError(f"Annotations must carry column(s) {echoed}.")
    for name, kind, meaning in declared:
        dtype = annotations.schema[name]
        if dtype != kind:
            logger.warning(
                "Rejecting annotation column {} of dtype {}.",
                bounded_echo(name),
                bounded_echo(dtype),
            )
            raise ConfigError(
                f"Annotation column {bounded_echo(name)} must be {meaning}, got "
                f"{bounded_echo(dtype)}."
            )
    selected = annotations.select(columns.start, columns.end, columns.flag)
    nulls = selected.null_count().row(0)
    if any(nulls):
        logger.warning(
            "Rejecting annotations with null(s): {} start, {} end, {} flag.", *nulls
        )
        raise ConfigError(
            f"Annotations must not hold nulls; found {nulls[0]} in the start "
            f"column, {nulls[1]} in the end column and {nulls[2]} in the flag "
            "column."
        )
    inverted = selected.filter(pl.col(columns.end) <= pl.col(columns.start))
    if inverted.height:
        first_start = int(inverted.get_column(columns.start)[0])
        first_end = int(inverted.get_column(columns.end)[0])
        logger.warning(
            "Rejecting {} annotation interval(s) whose end does not exceed its "
            "start; the first is [{}, {}).",
            inverted.height,
            first_start,
            first_end,
        )
        raise ConfigError(
            f"Every annotation interval must satisfy start < end; {inverted.height} "
            f"do not, the first being [{first_start}, {first_end})."
        )
    return selected.sort(columns.start)


def _output_columns(prefix: object) -> tuple[str, str]:
    """Name the two appended columns from the caller's prefix.

    Raises:
        ConfigError: If ``prefix`` is not a non-empty ``str``.

    """
    if not isinstance(prefix, str):
        logger.warning(
            "Rejecting a non-string annotation prefix: {}", type(prefix).__name__
        )
        raise ConfigError(
            f"Annotation column prefix must be a str, got {type(prefix).__name__}."
        )
    if not prefix:
        logger.warning("Rejecting an empty annotation prefix.")
        raise ConfigError("Annotation column prefix must not be empty.")
    return f"{prefix}_{_FLAGS_SUFFIX}", f"{prefix}_{_OVERLAP_SUFFIX}"


def _require_absent_columns(frame: pl.DataFrame, names: tuple[str, str]) -> None:
    """Refuse to write over a column the frame already carries.

    Raises:
        ConfigError: If ``frame`` already has either of ``names``.

    """
    present = [name for name in names if name in frame.columns]
    if present:
        echoed = ", ".join(bounded_echo(name) for name in present)
        logger.warning("Refusing to overwrite existing column(s): {}", echoed)
        raise ConfigError(
            f"The frame already carries column(s) {echoed}; choose another "
            "prefix rather than overwriting them."
        )


def _merged(rows: list[tuple[int, int, str]]) -> list[tuple[int, int]]:
    """Merge ``[start, end)`` intervals into disjoint ones, in start order.

    The flag is dropped: the union is what the overlap column measures,
    and a second flag over the same seconds must not count them twice.
    Two intervals that merely touch (one's end equal to the next's
    start) are merged too; they cover a contiguous run of seconds, and
    summing them separately would give the same answer.
    """
    merged: list[tuple[int, int]] = []
    for start, end, _ in sorted(rows):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
