"""Every raise is preceded by a log line at the level its exception fixes.

``ohlc_toolkit.temporal.errors`` states the pairing: ``warning`` before a
``ConfigError`` -- the caller's own argument is refused, and the caller can
fix the call -- and ``error`` before a data, integrity, coverage or file
error -- input from outside the call failed. These tests walk every module
under ``src/ohlc_toolkit`` by AST and hold each ``raise`` to it, so the rule
is checked on every run instead of remembered at review time.

The log a raise pairs with is the last ``logger.<level>(...)`` statement
before it in the same block. A raise with no such log in its block fails
too: the style guide says every raise is preceded by one.
"""

import ast
import builtins
from collections.abc import Iterator
from pathlib import Path

import pytest

from ohlc_toolkit import snapshot, source, temporal, windows
from ohlc_toolkit.temporal import ConfigError, CoverageError, DataValidationError

_PACKAGE = Path(__file__).parents[1] / "src" / "ohlc_toolkit"
# Where a raised exception's name is looked up, in order.
_NAMESPACES = (temporal, source, snapshot, windows, builtins)
# Levels that count as "error" for the pairing.
_ERROR_LEVELS = frozenset({"error", "exception", "critical"})
# Fewer raises than this means the walk is broken, not that the package is
# quiet: the survey that wrote the rule counted 145.
_MIN_EXPECTED_RAISES = 100


def _resolve(name: str) -> type[BaseException]:
    """Turn the name a ``raise`` uses into the class it names."""
    for namespace in _NAMESPACES:
        found = getattr(namespace, name, None)
        if isinstance(found, type) and issubclass(found, BaseException):
            return found
    pytest.fail(f"{name} is raised but is not a known exception class; classify it")


def _expected_level(exception: type[BaseException] | None) -> str:
    """Say which log level the pairing requires before raising ``exception``."""
    if exception is None:
        # A bare re-raise hands an external failure on unchanged.
        return "error"
    if issubclass(exception, ConfigError):
        return "warning"
    if issubclass(exception, DataValidationError | CoverageError | OSError):
        return "error"
    pytest.fail(f"{exception.__name__} is on neither branch of the pairing")


def _logger_level(statement: ast.stmt) -> str | None:
    """Return the level of a ``logger.<level>(...)`` statement, else ``None``."""
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    func = statement.value.func
    if (
        isinstance(func, ast.Attribute)
        and isinstance(func.value, ast.Name)
        and func.value.id == "logger"
    ):
        return func.attr
    return None


def _raised_name(node: ast.Raise) -> str | None:
    """Name the exception a ``raise`` constructs, or ``None`` for a bare re-raise."""
    if node.exc is None:
        return None
    if isinstance(node.exc, ast.Call):
        return ast.unparse(node.exc.func)
    return ast.unparse(node.exc)


def _child_blocks(statement: ast.stmt) -> Iterator[list[ast.stmt]]:
    """Yield every statement list nested directly inside ``statement``."""
    for field in ("body", "orelse", "finalbody"):
        block = getattr(statement, field, None)
        if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
            yield block
    for handler in getattr(statement, "handlers", ()):
        yield handler.body
    for case in getattr(statement, "cases", ()):
        yield case.body


def _pairs(body: list[ast.stmt]) -> Iterator[tuple[ast.Raise, str | None]]:
    """Yield each raise in ``body`` with the level of the last log before it."""
    last: str | None = None
    for statement in body:
        level = _logger_level(statement)
        if level is not None:
            last = level
        if isinstance(statement, ast.Raise):
            yield statement, last
        for block in _child_blocks(statement):
            yield from _pairs(block)


def _modules() -> list[Path]:
    """Every module of the package, in a stable order."""
    return sorted(_PACKAGE.rglob("*.py"))


@pytest.mark.parametrize(
    "module", _modules(), ids=lambda path: str(path.relative_to(_PACKAGE))
)
def test_every_raise_logs_at_the_level_its_exception_fixes(module: Path) -> None:
    """Each raise follows the pairing the taxonomy module states."""
    wrong: list[str] = []
    for raise_node, level in _pairs(ast.parse(module.read_text()).body):
        name = _raised_name(raise_node)
        expected = _expected_level(None if name is None else _resolve(name))
        actual = "error" if level in _ERROR_LEVELS else level
        if actual != expected:
            wrong.append(
                f"{module.relative_to(_PACKAGE)}:{raise_node.lineno} raises "
                f"{name or 'again'} after logger.{level}; the pairing wants {expected}"
            )
    assert not wrong, "\n".join(wrong)


def test_the_walk_sees_the_package_raises() -> None:
    """A walker that found nothing would pass vacuously; hold it to a floor."""
    seen = sum(
        1 for module in _modules() for _ in _pairs(ast.parse(module.read_text()).body)
    )
    assert seen >= _MIN_EXPECTED_RAISES, seen
