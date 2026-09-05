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

**Every `raise` is preceded by a log**, at the level the exception's branch
fixes: `warning` before a `ConfigError` (the caller's own argument is refused
and the caller can fix it); `error` before a data, integrity, coverage or
file error (input from outside the call failed). `tests/test_refusal_levels.py`
checks the pairing.

### Validate at boundaries, trust internal types

Validate at CSV, download, and public-function edges. Once past the boundary,
trust your types. Do not re-validate deep in pure helpers.

### Make invalid states unrepresentable

- Prefer enums and union types over flag soup.
- Durations are a `Duration` or a compact string such as `"1s"` or `"1m"`,
  never a bare integer whose unit is implied. `coerce_duration` accepts
  either spelling at a boundary; past it, the value is a `Duration`.

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
- Append units: `timeout_seconds`, `total_seconds`, `max_bytes`.
- Booleans read as assertions: `is_valid`, `has_gap`, `should_retry`.
- Functions are verb phrases: `read_source_csv`, `compute_windows`.

## Type hints

Always on public functions and methods. Use `# type: ignore` sparingly, and
only for tooling gaps.

Trust the typing. Do not re-cast values that are already the right type.

## Docstrings

Google style, enforced by Ruff `D`. One-line summary in active voice, blank
line, then `Args:` / `Returns:` / `Raises:` when the signature warrants it.

```python
@classmethod
def parse(cls, text: str) -> Self:
    """Parse a compact duration string into a Duration.

    Args:
        text: One or more ``<digits><unit>`` components with units
            strictly descending in the order w > d > h > m > s, each
            unit used at most once, and no separators, signs,
            decimals, or whitespace.

    Returns:
        The parsed Duration.

    Raises:
        ConfigError: If ``text`` is not a ``str``, does not match the
            grammar, or repeats/misorders a unit.
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
  do not raise), and before every `ConfigError`.
- `error` / `exception` for unexpected failure (`exception` inside `except:`),
  and before every data, integrity, coverage or file error.
- Every `raise` preceded by a log, at the level its exception fixes (see
  "Handle every error explicitly").

## DataFrames

**polars**, and only polars. pandas left with the 0.4 surface at 1.0, and
nothing depends on it any more. Do not add a second DataFrame library.

## Public API

`ohlc_toolkit.__all__` is six subpackages, and nothing else:

- `temporal` — `Duration`, its grammar, and the error taxonomy
- `source` — profiles, validation, and the raw-frame reader
- `windows` — the aggregation engine, its oracle, and the quality policy
- `schedules` — scale schedules and emit-cadence rules
- `returns` — backward and forward returns
- `snapshot` — fetching and verifying a published release

The top-level package imports all six, so `import ohlc_toolkit` reaches
every one of them. Names are **not** flattened into the top level:
`ohlc_toolkit.Duration` does not exist, and adding it would give up the
subpackage qualifier that says which stage a name belongs to.

A new public name goes in the subpackage that owns it and in that
subpackage's `__all__`. Update call sites rather than adding
`_old_name = new_name` aliases. Breaking changes to exported names,
defaults, or on-disk behaviour belong in a documented major version —
which is what 1.0 is, and why it carries no shims for anything 0.4
exported.

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
