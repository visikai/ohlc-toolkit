# Style Guide

How we write Python in **ohlc-toolkit**. Inspired in part by
[TigerBeetle's TIGER_STYLE.md](https://github.com/tigerbeetle/tigerbeetle/blob/main/docs/TIGER_STYLE.md).

Design goals, in order:

1. **Safety** — correctness, predictability, no silent failures, no hidden
   gap-fills on ingest paths.
2. **Clarity** — durations, window kinds, and quality flags are explicit.
3. **Compatibility** — this is a published PyPI library. Exported behaviour
   is a contract.

---

## Core philosophy

- Prefer the boring, obvious solution. Cleverness is expensive.
- Always say why. Rationale belongs in comments, commit messages, and pull
  requests — not lost to time.
- Do not leave half-broken public behaviour behind a TODO.

## Safety

### Handle every error explicitly

Never swallow an exception. Log the failure path, then raise (or return an
explicit error) deliberately. Specific exceptions first; broad `Exception` last.

**Every `raise` is preceded by a log.** Warning for noteworthy-but-non-critical;
`error` or `exception` for unexpected or bad.

### Validate at boundaries, trust internal types

Validate at CSV, download, and public-function edges. Once past the boundary,
trust your types. Do not re-validate deep in pure helpers.

### Make invalid states unrepresentable

- Prefer enums and union types over flag soup.
- New public APIs should use unit-bearing durations such as `"1s"` or `"1m"`,
  not a bare integer whose unit is implied.
- Existing `transform_ohlc(..., timeframe=5)` minute integers are legacy.
  Do not extend that pattern; do not silently change it either.

### No unbounded growth

Every read, download, retry, and aggregation has an explicit cap with a
reason.

## Python

The package currently supports Python 3.11+. Write code that stays inside
that range:

- No `from __future__ import annotations` unless it actually fixes something.
- Prefer `T | None` over `Optional[T]`, `list[int]` over `List[int]`.
- Import generics from `collections.abc`.

Do not bump `requires-python` as a drive-by change.

## Naming

- `snake_case` for functions, methods, variables, and modules.
- `PascalCase` for classes and enums.
- `SCREAMING_SNAKE_CASE` for module-level constants.
- Full words over abbreviations. Exceptions: `id`, `url`, `ohlc`, `csv`,
  `http`.
- Append units: `timeout_seconds`, `step_size_minutes`.
- Booleans read as assertions: `is_valid`, `has_gap`, `should_retry`.
- Functions are verb phrases: `read_ohlc_csv`, `transform_ohlc`.

## Type hints

Always on public functions and methods. Use `# type: ignore` sparingly, and
only for tooling gaps.

Trust the typing. Do not re-cast values that are already the right type.

## Docstrings

Google style, enforced by Ruff `D`. One-line summary in active voice, blank
line, then `Args:` / `Returns:` / `Raises:` when the signature warrants it.

```python
def parse_timeframe(timeframe: str, *, to_minutes: bool = False) -> int:
    """Parse a unit-bearing timeframe string into seconds or minutes.

    Args:
        timeframe: A string such as ``1h15m``.
        to_minutes: When True, return whole minutes instead of seconds.

    Returns:
        The duration as an integer in the requested unit.
    """
```

## Logging

- Obtain loggers via `ohlc_toolkit.config.logging.get_logger(__name__)`.
- Use `{}` placeholders. **Never** f-strings or `%`-formatting in log calls.

```python
logger.info("Read {} rows from {}", len(frame), path)  # Good
logger.info(f"Read {len(frame)} rows from {path}")  # Bad
```

- `debug` for important I/O and branch decisions.
- `info` for significant state changes.
- `warning` for noteworthy-but-recoverable (including quality warnings that
  do not raise).
- `error` / `exception` for unexpected failure (`exception` inside `except:`).
- Every `raise` preceded by a log.

## DataFrames

Use **pandas**. Do not introduce a second DataFrame library unless a release
explicitly migrates the public API.

## Public API

Exported names live in `ohlc_toolkit.__all__`:

- `DatasetDownloader`
- `read_ohlc_csv`
- `transform_ohlc`
- `parse_timeframe`, `format_timeframe`
- `validate_timeframe`, `validate_timeframe_format`

Update in-repo call sites rather than adding `_old_name = new_name` aliases
for code only this repository consumes. Breaking changes to exported names,
defaults, or on-disk CSV behaviour belong in a documented major version.

## Tooling

- **Dependency management:** uv (`pyproject.toml` + `uv.lock`).
- **Lint / format:** Ruff. Line length 88 for format, 120 for pycodestyle.
  Rule selection is in `pyproject.toml`.
- **Type check:** mypy.
- **Tests:** pytest.

```bash
mise exec -- uv sync --all-groups
mise exec -- uv run pytest
mise exec -- uv run ruff check .
mise exec -- uv run ruff format .
mise exec -- uv run mypy .
```

## Testing

- Tests live under `tests/`.
- Functions `test_*`, classes `Test*`.
- Golden fixtures and invariant tests are preferred for resampling and
  returns.
- Test behaviour, not internal call sequences.
- Do not skip or xfail without a recorded reason.

Write a failing test before the code that makes it pass. For bug fixes,
reproduce the bug with a test first.

## Complexity

Keep functions small. `# noqa` is an explicit escape hatch; if several
complexity ignores pile up on one function, split it.

When in doubt, choose the smaller, more obvious option that does not break
the public contract.
