"""The schedules package sits below the windows package, and stays there.

``windows.resolution`` is the natural CONSUMER of a resolved schedule,
so an import in the other direction is a cycle waiting for the first
card that teaches resolution to accept a schedule. The one such edge
that ever existed carried only a text helper, now shared from
``ohlc_toolkit.temporal``; this test keeps the direction honest.
"""

import ast
from pathlib import Path

import ohlc_toolkit.schedules


def test_no_schedules_module_imports_the_windows_package() -> None:
    """Every import in every schedules module stays below windows."""
    package_dir = Path(ohlc_toolkit.schedules.__file__).parent
    offenders: list[str] = []
    for module_path in sorted(package_dir.glob("*.py")):
        tree = ast.parse(module_path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("ohlc_toolkit.windows"):
                    offenders.append(f"{module_path.name}: from {node.module}")
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith("ohlc_toolkit.windows"):
                        offenders.append(f"{module_path.name}: import {alias.name}")
    assert offenders == []
