# Changelog

This file starts at 1.0.0. Earlier versions (0.1.0 through 0.4.0) are
recorded as [GitHub releases](https://github.com/visikai/ohlc-toolkit/releases)
against their tags, and are not restated here.

## Unreleased

**This release is a major version.** The windowed-candle frame gains a
tenth column, and "exactly N columns" is a shape every consumer sees and
this repository's own tests pin. The alternative considered was an
additive release with the column opt-in, keeping the default at nine.
That was rejected on three counts: the default would then contradict the
schema this release exists to ship; the quality-policy step now REQUIRES
`traded_seconds`, so an opt-in would have to be threaded through two
modules and the policy would need a mode meaning "no traded column",
which is the v1-reinterpretation the schema explicitly rules out; and the
only known consumer is already planning to move its own bound. A 2.0.0
with one consumer is cheap now and expensive later. The version string is
not bumped in this entry -- the release does that -- but the decision is
recorded here because it is the reason the entries below are breaking.

### Added

- **`indicators.FeatureIdentity`**, the record a feature column's name is
  DERIVED from: `{indicator}_{family}{period}_w{window}`, as in
  `rsi_p14_w21m`. A field that varies within a frame goes in the name; a
  field constant across the artifact goes in the manifest. Deriving the
  name rather than passing it beside the record means the two cannot
  disagree -- there is nowhere for a second spelling to live. The name
  parses back to the identity it came from, and refuses one it could not
  have produced.
  One exception to the varies/constant rule, made deliberately: the
  family character is in the name even though an artifact carries one
  family, so that shipping the dense family renames no phased column. A
  rename breaks every consumer holding a stored frame.
- **`indicators.NormalizationClass`**, three members: bounded by
  construction, stationarized, empirically normalized. Carried by
  `FeatureIdentity` as a required field and deliberately absent from the
  column name: it does not vary between the columns of one feature, and
  the rule above puts what varies in the name. A primitive declares its
  class rather than having it inferred, because the answer follows from
  how the number is built rather than from how one sample looks -- which
  also means a reader parsing a stored column has to supply it from the
  manifest that recorded it, and `FeatureIdentity.parse` takes it as a
  keyword argument rather than inventing a default for the one field the
  name cannot vouch for.
- **`indicators.IndicatorPrimitive`**, the protocol every input
  indicator satisfies: its short name, the lookback count `L` a period
  `P` implies, the normalization class its values already carry, and one
  column of values over harness output. A protocol rather than a base
  class -- a primitive inherits no state and is only ever read
  structurally. Two things are deliberately not the primitive's to state:
  effective history, which is `L * W` and belongs to the grid the harness
  resolved, and its own column name, which is derived from
  `FeatureIdentity` so that the name and the manifest cannot disagree.
  `require_phased_inputs` refuses harness output four ways: assembled for
  a different lookback, missing a field the primitive reads, carrying a
  field whose element dtype is not the one the harness declares, or
  carrying lists that are not `L` long. The dtype way is not a failure but
  a different NUMBER -- `Float32` closes near `1.678e7` put the
  differences below the type's resolution and read 75.0 where `Float64`
  reads 80.0, with nothing logged. A bare `str` passed as `fields` is
  refused rather than read one character at a time.
  `add_indicator` computes one and appends it, refusing a frame that
  already carries the column, and returns the record rather than a bare
  frame so that a recipe computing four primitives over one lookback
  chains them instead of unwrapping and rewrapping three times.
- **`indicators.CutlersRSI`**, the first primitive:
  `100 - 100 / (1 + G / D)`, where `G` and `D` are the SIMPLE means of
  the positive and negative close-to-close changes across the `P` changes
  that `P + 1` phased windows hold, so `L = P + 1`. Bounded by
  construction.
  Cutler's and not Wilder's: Wilder's smoothing is a recursion with
  infinite memory, so its value depends on where the series was seeded
  and two artifacts built from different history starts disagree about
  the same window. Cutler's reads exactly `P` changes, which is what
  makes a window's value a function of that window's inputs and nothing
  else. Wilder's may be added later as a separately named indicator
  carrying its own warm-up and restart rule; it is never substituted.
  Three boundary conventions, so that a reading is never undefined and
  never non-finite: only rises reads `100`, only falls reads `0`, and
  every one of the `P` changes exactly zero reads `50` -- the symmetric
  limit, which is not the same as a window with too little trading in it,
  and the harness has already turned that one into a null. Any null among
  the `P + 1` inputs nulls that tick's reading and no other.
  The implementation evaluates the closed form `100 * (G / (G + D))`,
  which is the same number by algebra but divides by a sum of
  non-negative totals rather than by `D`, so the two one-sided
  conventions are arithmetic rather than branches and no path can divide
  by zero. The parenthesis is load-bearing: scaling before dividing
  rounds the product first and returns `100.00000000000001` at `D = 0`
  and `G = 256842.5`, outside the range the indicator claims.
  The means' common divisor cancels in that form, so the implementation
  sums and never divides by `P`; what the period really fixes is the
  NUMBER of changes, which the lookback guard checks. The two one-sided
  conventions are exact at their boundaries and are not symmetric just
  beyond them: `G = 1.0` with `D = 1.11e-16` reads exactly `100.0`, while
  `G = 5e-324` with `D = 1.0` reads `4.94e-322` rather than `0.0`.
  An infinite change total is refused before it can be read through: a
  finite up total over an infinite down total is `0.0`, which is in
  range, finite, and indistinguishable from the only-falls convention.
- **`indicators.effective_history`** and **`indicators.effective_n`**.
  The first is `L * W`. The second counts INDEPENDENT BLOCKS -- whole
  non-overlapping windows in a range, divided by the lookback -- and is
  explicitly not a statistical effective sample size, which accounts for
  autocorrelation and is smaller. They carry different names because
  reading one as the other overstates how much independent evidence a
  feature has.
- **`temporal.require_absent_columns`**, one guard against writing over a
  column a frame already carries. It existed twice before, in
  `returns.primitives` and `windows.annotations`, and the two copies had
  already diverged: the annotations copy bounded what it echoed, the
  returns copy did not. Only one of them needed to. The annotations copy
  echoes a CALLER's prefix, of any length; the returns copy echoes names
  this package composes from an enum member and a duration label, and the
  longest such name is 60 characters even at the largest horizon Int64
  seconds can express -- under the 80-character echo bound, and one of the
  cases `temporal/echo.py` names as legitimately needing none. So the
  drift was real and the exposure was not. What makes the bound necessary
  is publishing the guard: its `names` argument is now a caller's, and
  bounded at the one site. Three call sites in two modules, each keeping
  its own remedy sentence -- the finding is shared, the advice about it is
  not.
  The guard refuses a bare `str` out loud rather than trusting the
  annotation: a `str` IS a `Collection[str]`, so `require_absent_columns(
  frame, "close", ...)` type-checks, iterates the characters, finds no
  column named `c`, and lets the overwrite proceed.
  The returns path's refusal text and log prefix changed to the shared
  wording; the columns it names and the exception it raises did not.
- **`indicators.RelativeRange`**, **`indicators.LogVolumeRatio`** and
  **`indicators.PriceToMovingAverage`**, the other three primitives. All
  three are STATIONARIZED: ratios or log ratios that remove the price or
  volume scale without estimating anything from a sample. One unit
  convention across the slice, because a series running from $5 to
  $100k+ makes any price-unit quantity incomparable with itself a year
  later.
  - `relrange`, `L = P + 1`: the mean true range over the `P` windows
    that have a predecessor, divided by the current close. A window's
    true range uses the PREVIOUS PHASED window's close -- the window one
    `W` earlier, not the previous row of any frame. Computed as
    `max(high, prev) - min(low, prev)`, which is the three-candidate
    definition exactly rather than approximately: subtraction is
    monotone, so no candidate can win by rounding. The tempting
    arithmetic-only form `clip(high - prev, 0) + clip(prev - low, 0)` is
    not exact where `prev` lies inside the bar.
  - `logvolratio`, `L = P + 1`: the natural log of the current volume
    over the MEDIAN volume of the `P` windows before it. A median because
    volume is heavy-tailed and one outlying window would drag a mean
    baseline toward itself, flattening the events the indicator exists to
    show; a log because the plain ratio lives on the same tail, and
    doubling and halving should be the same distance from zero. The
    current window is excluded from its own baseline, so a spike cannot
    shrink the ratio it is supposed to produce.
    A zero or negative volume among present inputs is REFUSED rather than
    read through: a present window has traded seconds at or above the
    recipe's threshold and therefore volume above zero, so a zero is a
    violation upstream, not a null or a negative infinity to interpret.
  - `mapos`, `L = P`: the natural log of the current close over the
    simple mean close of the `P` windows ending at the tick, the current
    one INCLUDED -- which is what makes `P` identical closes read exactly
    `0.0`. At `P = 1` the mean is the current close itself and every
    reading is identically zero.
    This replaces a moving-average log SLOPE, which is nearly
    proportional to the `P x W` backward log return this package already
    computes. Over a seeded random walk of 600 minutes at `W = 3m` and
    `P = 3`, the slope correlates with that return at **+0.999999** while
    price-to-mean correlates at **+0.764172**: where the mean MOVED and
    where the price SITS relative to it are different facts.
  A non-positive close is refused for both price ratios, on the same
  reasoning: a windowed candle's close is a traded price, so a zero one
  is a violation upstream rather than a division to perform. A non-finite
  intermediate is refused in every primitive, checked on the values a
  reading is assembled from rather than on the reading: an infinite
  denominator gives a finite quotient that looks like an ordinary
  reading.
  The RSI's non-finite refusal MESSAGE changed with that unification --
  it now names the intermediate rather than the change total, and the
  exception type and the condition are unchanged.
- **`ohlc_toolkit.indicators`**, a seventh subpackage, holding the phased
  lookback every indicator reads through: at each tick of the emit grid,
  the `L` non-overlapping windows of duration `W` ending at `t`, `t - W`,
  `t - 2W`, and so on. `phased_lookback` is the fast path,
  `phased_lookback_reference` the brute-force oracle it is checked
  against.
  The inputs are read from the window's SOURCE-cadence materialization,
  not from its own `E`-cadence frame, because `{t - kW}` lies on the emit
  grid only when `E` divides `W` -- which under the cadence rules this
  package resolves, it frequently does not. At `W = 2h26m` and `E = 3m`
  the emit frame holds none of the phases at all.
  Lookups are exact equality on `close_time`, never a shift or an as-of
  match. A window whose `traded_seconds` falls below the threshold is a
  null input, applied by the harness itself whatever quality mode the
  frame was written under -- report mode removes nothing. Any null among
  the `L` inputs nulls the whole output row rather than leaving a list
  with a hole in it, so no indicator downstream has to remember the rule.
  The effective history `L * W` is reported by the harness rather than
  left to callers to multiply.
  A schema v1 frame is refused and named as one; so is a frame spanning a
  different window, an emit cadence off the source grid, and an anchor
  whose grid never lands on a row of the frame -- that last one would
  otherwise answer "no data" to a question that was really "this anchor
  does not belong to this frame".
- **`snapshot.verify_snapshot_on_disk`**, which verifies a snapshot
  already on disk against the manifest beside it: presence, size and
  SHA-256 for every declared asset, no network, and the same
  `SnapshotFetchResult` that `read_snapshot_frame` already consumes.
  `fetch_snapshot` answers "did these bytes arrive intact"; this answers
  "are these still the bytes", and neither implies the other. Byte-level
  refusals raise `SnapshotIntegrityError`, the same class the fetch path
  uses, so one `except` covers both questions. A manifest declaring no
  assets is refused by this function itself rather than only by the
  parser it calls: a verification that passes over zero assets is worse
  than none, because it looks like one. `repository` is required and has
  no default -- a repository this function invented would be recorded as
  provenance by a caller who trusted it, and this function knows less
  about where the bytes came from than `fetch_snapshot` does, having
  fetched nothing.
- **A count-valued `LookbackSchedule`**, with `metallic_lookback`,
  `log_spaced_lookback` and `explicit_lookback`. Its members are period
  counts rather than durations: the same `21` is twenty-one minutes on a
  1m frame and twenty-one weeks on a 1w one. It is a separate type rather
  than a mode on `WindowSchedule`, because a lookback borrowed from a
  window schedule would be silently coupled to it, and because
  `require_resolved_windows` refuses anything but a `Duration` — a guard
  worth keeping rather than loosening. The arithmetic is shared: the
  recurrence, the log-spaced placement and the quantize/bound/dedup rule
  are one implementation called with counts where the window schedule
  calls it with seconds.
  Identities cannot collide. A lookback records its members under
  `"periods"` and a window schedule under `"windows"`, and `3` is not
  `"3m"`, so each reader refuses the other's payload for a missing key
  rather than coercing it.
- `windows` output gains **`traded_seconds`** (Int64), the tenth column:
  the summed duration of the included source intervals whose `volume` is
  greater than zero. `coverage_seconds` says the source had rows;
  `traded_seconds` says those rows held trades. On a
  complete-by-construction grid coverage is full everywhere, so nothing
  in the first nine columns could tell a dead window from a quiet one.
  The predicate is volume, never `high != low` -- on the public
  one-minute grid 13.71% of traded minutes trade at a single price, and a
  flatness test would report every one of them as untraded. A null volume
  is not a trade, and neither is a NaN -- polars answers `NaN > 0` with
  True where Python answers False, so the fast path and the reference
  oracle are made to agree explicitly rather than by luck. An INFINITE
  volume is counted as a trade by both, because `inf > 0` is true in
  either language; it is invalid source data and validation is where it
  is refused.
- `WindowQualityPolicy` gains **`min_traded_seconds`** (int, default
  `0`), a second threshold consuming `traded_seconds` with the same
  modes, the same fail-closed null handling and the same single mask. It
  is a DURATION and not a fraction of the window: the lowest useful
  setting must admit a window exactly when at least one included interval
  traded, which as a fraction is `d / W` for the source cadence `d` -- a
  number the policy cannot compute, since it does not record `W`, and one
  that is not exactly representable anyway. As a duration that setting is
  `1`. The default of `0` admits everything, and `from_dict` treats the
  key as optional, so a policy recorded before this threshold existed
  reads back exactly as it was written rather than acquiring a bar nobody
  chose.
- `QualityReport` gains `traded_threshold_seconds`,
  `coverage_offending_count`, `traded_offending_count` and
  `null_traded_count`. The two offending counts overlap -- a row can miss
  both bars -- so they can sum past `offending_count`, which is the union
  the gate refuses. The strict-gate message names the threshold that bit
  and omits the one that did not.

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

- The size-mismatch message on the snapshot fetch path reads "is N bytes"
  where it read "landed at N bytes". Visible to a 1.x caller reading
  messages, which is not API and not recommended; recorded because the
  wording is now shared with a path where nothing landed.

- `windows.read_annotations` said a path "does not exist" when it was a
  directory. It now says which it is, so a caller is not sent looking for a
  missing file that is present.
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
  read and never what is raised. Only regular files are checked this way: a
  FIFO, a stream or a character device is read exactly as before, because the
  check opens the path and the read opens it again, which is free on a file
  and destructive on anything that cannot be reopened.

### Changed

- **BREAKING: `compute_windows` and `compute_reference_windows` return
  ten columns, not nine.** Anything asserting the exact column list, or
  reading columns positionally, changes. A v1 nine-column artifact is
  superseded rather than reinterpreted: no reader invents a
  `traded_seconds` for it.
- **BREAKING: `apply_quality_policy` requires `traded_seconds`.** The
  step reads three of the ten columns now, and a frame without the new
  one is refused with a `ConfigError` rather than silently checked
  against one threshold. A caller that projects a frame down to what the
  step consults must keep all three.
- A generated schedule's maximum is now compared against the QUANTIZED
  term rather than the raw one. Generation produces the first term past
  the bound and lets resolution decide, so a term of 21.434 against a
  maximum of 21 is kept — it quantizes to exactly 21, which is inside the
  bound — where it was previously dropped for a value the schedule never
  uses. Every other bound in the resolver is already applied after
  quantization; the generation bound was not.
  **This changes some existing schedule ids.** The rule, which is what
  you need to tell whether you are affected: a generated schedule gains a
  window when its `maximum` lands within half a grain of a generated
  term, and the window it gains is always the `maximum` itself. Gaining a
  window changes the content hash that names the schedule. A schedule
  that previously resolved to nothing at all can now resolve to one
  window, so a call that raised `ConfigError` can return.

  Reproducible: `metallic_recurrence(coefficient=1.618, seed="1m",
  grain="1m", maximum="5m")` resolved to `['1m', '3m']` before and
  `['1m', '3m', '5m']` now.

  Measured NOT to change: every schedule in this repository's suite, and
  the eleven-window schedule of the one known consumer that records a
  `schedule_id`. The named legacy schedule is unaffected structurally
  rather than by measurement -- it is an `explicit` list and never passes
  through this code at all.
- **BREAKING: positional construction of `WindowQualityPolicy` and
  `QualityReport` changes.** `min_traded_seconds` is inserted between
  `min_coverage` and `gate_mode` rather than appended, so
  `WindowQualityPolicy(QualityMode.GATE, 0.9, GateMode.REPORT)` -- legal
  at 1.0.0 -- now raises `ConfigError: min_traded_seconds must be an int,
  got GateMode`. `QualityReport` likewise gains four fields at interior
  positions. Both fail loudly rather than silently mis-assigning, and
  keyword construction is unaffected; the fields are ordered by what they
  mean rather than by when they were added, because the order is part of
  a public dataclass for as long as the major version lasts.
- **Dependency floors are raised so the published wheel cannot carry a
  version with a published advisory against it.** `orjson` to `>=3.11.6` and `requests`
  to `>=2.33.0`, the first versions clearing the advisories against them;
  and `urllib3 >=2.7.0` and `idna >=3.15` are now declared, though
  nothing here imports either. `requests` carries them into every install
  and its own ranges (`urllib3<3,>=1.26`, `idna<4,>=2.5`) admit versions
  with published advisories, three of them high. None of this was visible in
  the lockfile, which pins clear versions and binds only this repository.
  Measured: against 1.0.0 a consumer pinning `orjson==3.10.18`, or
  `urllib3==1.26.20` and `idna==2.5`, resolved cleanly with no conflict
  and no warning; against these floors the same pins are refused.
- **`certifi` is now declared, at `>=2024.7.4`, and `polars`'s floor is
  raised from `>=1.35.0` to `>=1.38.1`.** `certifi` is the fourth
  dependency `requests` carries into every install, and its own
  declaration is `certifi>=2023.5.7` -- the exact version PYSEC-2023-135
  and PYSEC-2024-230 are against. It is declared without an upper bound,
  unlike its neighbours, because it is a dated snapshot of a root store on
  a CalVer scheme: there is no major version to cap, and capping one would
  strand a consumer on an expired bundle. `polars>=1.35.0` was never a
  usable bound -- 1.35.0 pins `polars-runtime-32==1.35.0`, which is
  yanked -- and 1.35.1 through 1.37.1 raise `InvalidOperationError` out of
  `windows.compute_windows` where later versions skip a null, so a
  consumer resolving to any of them gets behaviour this library's own
  tests contradict. Measured on 3.11 and 3.14, which agree.
- **A `Minimum versions` workflow installs at `--resolution lowest` and
  fails if the floors are not what a consumer would actually get.** It
  checks that every declared floor is the version that installs, audits
  that closure against the advisory database, and runs the suite there. No
  other job had ever installed at the floors, which is why the exposure
  above could sit in the metadata unnoticed.
- `windows.compute_windows`'s documented equivalence with the reference
  oracle is stated as conditional on VALID input, in both places it was
  claimed. Neither function validates, so either can be handed a frame the
  source contract rejects, and on two kinds of such input the equivalence
  fails: on a NaN price the two return different answers and neither is
  correct, and on a null price the oracle raises a bare `TypeError` while
  the engine proceeds -- so "same refusals" was false as well. Infinities
  are NOT affected; there the two agree exactly. On a gap, a duplicate, an
  off-phase timestamp or rows out of order they also agree exactly.
- Both functions' precondition lists now name every shape they do not
  detect -- a gap, a duplicate, an off-phase timestamp, rows out of order,
  a non-finite price -- and state the null exception rather than implying
  the two behave alike on it. `compute_reference_windows` carries the
  normative copy and said it "will not detect ... a null price", which its
  own behaviour contradicts. On a null the engine skips it in `high` and
  `low` and propagates it into `open` and `close`, so it is not simply
  dropped either. Documentation and tests only; no behaviour changes, and
  no guard is added to either function.
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
