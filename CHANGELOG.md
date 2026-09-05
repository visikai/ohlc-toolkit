# Changelog

This file starts at 1.0.0. Earlier versions (0.1.0 through 0.4.0) are
recorded as [GitHub releases](https://github.com/visikai/ohlc-toolkit/releases)
against their tags, and are not restated here.

## Unreleased

### Added

- `windows.annotate_windows`, `windows.read_annotations`,
  `windows.AnnotationColumns` and `windows.AnnotationValidationError`: join
  a sparse half-open interval sidecar onto a window frame as opaque flags
  with union overlap accounting, reading only `open_time` and `close_time`
  and appending exactly two columns. The reader keeps file order, takes an
  optional row cap that refuses rather than truncates, and raises
  `AnnotationValidationError` (a `DataValidationError`) for a sidecar it
  cannot make intervals from, or one holding more rows than the cap.
- `scripts/echo_sweep.py`: lists every `logger.*` argument and `raise`
  interpolation in `src/` that the echo rule does not visibly bound, with a
  count, so a sweep's figures can be reproduced by anyone. It reports; it
  does not gate.

- `source.FindingKind.NON_FINITE_VALUES`: a new finding kind, for a NaN or
  either infinity in a declared price or volume column. It is kept distinct
  from `NULL_VALUES` because a null is an absent cell and a NaN is a present
  cell that is not a number, and nothing is coerced: making a NaN into a null
  is a repair this validator does not perform.

### Fixed

- A capped read of a file holding no data aborted the interpreter instead of
  refusing. `source.read_source_csv` passes a row cap through to polars, and
  at polars 1.44.1 an empty gzip archive read with any `n_rows` -- zero
  included -- raises `pyo3_runtime.PanicException`, a `BaseException` that no
  caller's `except PolarsError`, or even `except Exception`, can catch. It was
  reachable from `snapshot.read_snapshot_frame`, which always caps its read at
  the manifest's row count plus one, so a published release carrying an empty
  history asset with a correct digest went straight past every documented
  refusal. A file that reads as empty is now refused with polars' own
  `NoDataError`, on capped and uncapped reads alike, so a cap changes what is
  read and never what is raised. One case changes class on the uncapped path
  as a result: a character device that reads as empty, such as `/dev/null`,
  previously raised `OSError` and now says it holds no data.

### Changed

- **Breaking.** `snapshot.read_snapshot_frame` raises `SnapshotIntegrityError`,
  not `ConfigError`, when the named asset is absent from a fetched release. That
  class already covers "an asset the release does not serve" and is what the
  fetcher raises for the same condition; the asset name also has a default, so
  the refusal fires on calls that configured nothing, where a release failing
  to carry the asset this package asks for is an integrity failure and not the
  caller's mistake.
- **Breaking.** A NaN or either infinity in a declared price or volume column
  is now INVALID source data and is refused. It used to validate completely
  clean -- the null check counts nulls, and a NaN is not a null -- while
  propagating through every window that averaged it. Any frame or file that
  relied on the old silence now fails, with no call-site change and by three
  routes: `source.validate_source_frame` in `STRICT` mode raises
  `SourceValidationError`; `source.read_source_csv` in `STRICT` mode does the
  same, and note that a CSV holding the literal text `nan` or `inf` in a price
  column parses to a non-finite float under a profile's pinned schema and is
  therefore now refused; and `snapshot.verify_snapshot_continuity`, and so
  `snapshot.read_snapshot_frame`, refuses a published history carrying one.
  The two routes that take a mode return the finding instead in `REPORT`
  mode; continuity verification has no mode and always refuses. Under the
  style guide's rule that a breaking behavioural change belongs in a
  documented major version, the release carrying this is a major one.
- Every echo of a value the package did not choose -- a name, a tag, a
  path, a URL, third-party error text -- now reaches a log line or an
  error message only through `temporal.bounded_echo`; type refusals log
  the offending type instead of the offending value.
- The log line before a raise follows the exception's branch: `warning`
  before a `ConfigError` (the caller's own argument is refused), `error`
  before a data, integrity, coverage or file error (input from outside the
  call failed; the file branch is `OSError` in full) and before a bare
  re-raise, and a `polars.exceptions.PolarsError` this package raises or
  re-raises is on the error side too. Five sites moved to match;
  `tests/test_refusal_levels.py` holds every raise in the package to the
  pairing, and a new exception class must be classified into a branch before
  it is raised -- the test fails a class on neither branch.

## 1.0.0

A breaking rewrite. Everything 0.4.0 exported is removed, and the
replacement surface is Polars-native.

### Removed

The whole 0.4 public API, with no aliases, no deprecation warnings, and
no compatibility shims:

- `read_ohlc_csv`, and the `ohlc_toolkit.csv_reader` module
- `transform_ohlc` and `rolling_ohlc`, and the `ohlc_toolkit.transform`
  module
- `DatasetDownloader`, and the `ohlc_toolkit.bitstamp_dataset_downloader`
  module
- `parse_timeframe`, `format_timeframe`, `validate_timeframe`,
  `validate_timeframe_format`, and the `ohlc_toolkit.timeframes` module
- `calculate_percentage_return`, and the `ohlc_toolkit.future_returns`
  package
- The vendored `ohlc_toolkit.pandas_ta` port
- `ohlc_toolkit.utils` (`infer_time_step`, `check_data_integrity`),
  `ohlc_toolkit.exceptions` (`DatasetEmptyError`), and the
  `DEFAULT_COLUMNS` / `DEFAULT_DTYPE` constants in `ohlc_toolkit.config`
- The `examples/` scripts, which exercised only the above

The replacements are not renames. They take different arguments, return
different types, and mean different things — which is why no shim is
provided. 0.4.x remains installable from PyPI; pin `ohlc-toolkit<1` to
stay on it.

### Added

`ohlc_toolkit.__all__` is now six subpackages, imported by the top-level
package but not flattened into it:

- **`temporal`** — `Duration`, an exact whole-second value type with one
  compact grammar, plus duration/cadence validators and the
  `ConfigError` / `DataValidationError` / `CoverageError` taxonomy.
- **`source`** — `SourceProfile` declares a source's cadence, phase,
  timestamp column and raw schema; `read_source_csv` reads against one
  without sorting, filling, dropping or de-duplicating; and
  `validate_source_frame` returns findings as data, with a strict mode
  that raises.
- **`windows`** — `compute_windows`, a Polars-native aggregator whose
  window membership is decided by close time rather than by row count,
  checked against `compute_reference_windows`, a deliberately quadratic
  brute-force oracle. `apply_quality_policy` is a separate later step
  over their output.
- **`schedules`** — window-scale generators (`log_spaced`,
  `metallic_recurrence`, `explicit`) and emit-cadence rules
  (`w_over_k`, `explicit_pairs`), each recording the request that
  produced it and named by a content hash that is verified on read-back.
- **`returns`** — `add_backward_returns` (causal) and
  `add_forward_returns` (not), locating counterparts by exact close-time
  equality rather than by shifting rows. Forward values carry an
  `available_at` column stating when they may first be read.
- **`snapshot`** — `fetch_snapshot` downloads a published dataset release
  and lets no byte reach its final path until size and SHA-256 match the
  release manifest, with the manifest digest usable as a pinnable
  snapshot identity.

### Changed

- `pandas` and `tqdm` are no longer dependencies. The runtime
  dependencies are `polars`, `requests`, `loguru` and `orjson`.
- The Python floor rises: 0.4.0 allowed Python 3.10 (`^3.10`); 1.0
  requires `>=3.11`, tested on 3.11 through 3.14.
- The project description no longer advertises timeframe transformation.

### Unchanged

- Apache-2.0 licensed.
