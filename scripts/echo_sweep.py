"""List every echo in the package that the echo rule does not visibly bound.

The rule in ``ohlc_toolkit.temporal.echo`` says a value the package did not
choose reaches a log line or an error message only through ``bounded_echo``.
This script walks ``src/ohlc_toolkit`` by AST and prints every ``logger.*``
argument and every ``raise`` f-string interpolation that is not one of the
shapes the rule accepts on sight: a constant, an UPPER_CASE module name,
``bounded_echo(...)``, ``len(...)``, ``type(...).__name__``, an enum
``.value``, a helper that bounds internally, or arithmetic and joins over
those. What it prints is the list a reader still has to classify by hand --
first-party literals, numbers a guard already validated, values bounded
where they were made -- and the count at the end is the figure to quote when
saying a sweep was done. It reports; it does not gate.

Usage::

    uv run python scripts/echo_sweep.py [PACKAGE_DIR]

``PACKAGE_DIR`` defaults to ``src/ohlc_toolkit`` under the repository root.
"""

import ast
import sys
from pathlib import Path

# Helpers whose result is bounded whatever they are given.
_BOUNDING_CALLS = frozenset({"bounded_echo", "len", "_echo_asset_names", "sha256_hex"})
# Calls that are bounded exactly when every argument is.
_TRANSPARENT_CALLS = frozenset(
    {"join", "round", "sorted", "int", "float", "repr", "str", "sum", "min", "max", "abs"}
)
# Attributes that render small whatever object they hang off.
_SMALL_ATTRIBUTES = frozenset({"__name__", "value"})
# How much of an offending expression to print.
_SHOWN_CHARS = 70


def _call_is_bounded(node: ast.Call) -> bool:
    """Decide a call: bounding helpers always, transparent ones by their arguments."""
    func = node.func
    if isinstance(func, ast.Name):
        name = func.id
    elif isinstance(func, ast.Attribute):
        name = func.attr
    else:
        return False
    if name in _BOUNDING_CALLS:
        return True
    if name in _TRANSPARENT_CALLS:
        return all(_is_bounded(arg) for arg in node.args) and all(
            _is_bounded(keyword.value) for keyword in node.keywords
        )
    return False


def _is_bounded(node: ast.AST) -> bool:
    """Decide whether an expression is one the echo rule accepts on sight."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.Name):
        return node.id.isupper()
    if isinstance(node, ast.Call):
        return _call_is_bounded(node)
    if isinstance(node, ast.Attribute):
        return node.attr in _SMALL_ATTRIBUTES
    if isinstance(node, ast.BinOp):
        return _is_bounded(node.left) and _is_bounded(node.right)
    if isinstance(node, ast.ListComp | ast.GeneratorExp):
        return _is_bounded(node.elt)
    if isinstance(node, ast.JoinedStr):
        return all(
            _is_bounded(value.value)
            for value in node.values
            if isinstance(value, ast.FormattedValue)
        )
    return False


def _is_logger_call(node: ast.AST) -> bool:
    """Match ``logger.<anything>(...)``."""
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "logger"
    )


def _sweep_file(path: Path, root: Path) -> list[str]:
    """Return one line per unbounded echo in ``path``."""
    shown = path.relative_to(root)
    findings: list[str] = []
    for node in ast.walk(ast.parse(path.read_text())):
        if _is_logger_call(node) and isinstance(node, ast.Call):
            for arg in node.args[1:]:
                if not _is_bounded(arg):
                    text = ast.unparse(arg)[:_SHOWN_CHARS]
                    findings.append(f"LOG   {shown}:{arg.lineno}: {text}")
            if node.args and isinstance(node.args[0], ast.JoinedStr):
                text = ast.unparse(node.args[0])[:_SHOWN_CHARS]
                findings.append(f"LOGF  {shown}:{node.args[0].lineno}: {text}")
        if isinstance(node, ast.Raise) and node.exc is not None:
            for sub in ast.walk(node.exc):
                if isinstance(sub, ast.FormattedValue) and not _is_bounded(sub.value):
                    text = ast.unparse(sub.value)[:_SHOWN_CHARS]
                    findings.append(f"RAISE {shown}:{sub.lineno}: {text}")
    return findings


def main(argv: list[str]) -> int:
    """Print every finding, then the count; exit 0 regardless."""
    default = Path(__file__).resolve().parents[1] / "src" / "ohlc_toolkit"
    root = Path(argv[1]) if len(argv) > 1 else default
    findings = [
        line for path in sorted(root.rglob("*.py")) for line in _sweep_file(path, root)
    ]
    for line in findings:
        print(line)
    print(f"{len(findings)} echo(es) under {root} not visibly bounded; classify by hand.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
