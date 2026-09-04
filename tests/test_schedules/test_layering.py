"""The schedules package sits below the windows package, and stays there.

``windows.resolution`` is the natural CONSUMER of a resolved schedule,
so an import in the other direction is a cycle waiting for the first
change that teaches resolution to accept a schedule. The one such edge
that ever existed carried only a text helper, now shared from
``ohlc_toolkit.temporal``; this test keeps the direction honest.
"""

import ast
from pathlib import Path

import ohlc_toolkit.schedules

# Package depth of ``ohlc_toolkit.schedules.<module>``: a relative import
# with one leading dot resolves inside schedules, two dots reach
# ohlc_toolkit itself -- where ``from .. import windows`` becomes the
# edge this test forbids.
_PARENT_LEVEL = 2


def _windows_imports(source: str, filename: str) -> list[str]:
    """Return every statement in ``source`` that imports the windows package.

    Catches the absolute spellings (``import ohlc_toolkit.windows``,
    ``from ohlc_toolkit.windows import ...``), the bare-name spelling an
    IDE auto-import produces (``from ohlc_toolkit import windows``), and
    the relative spelling (``from .. import windows``). Deferred,
    class-body, TYPE_CHECKING and try/except forms are all reached
    because the whole tree is walked.
    """
    offenders: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.startswith("ohlc_toolkit.windows"):
                offenders.append(f"{filename}: from {module}")
            elif module == "ohlc_toolkit" or (
                node.level >= _PARENT_LEVEL and module == ""
            ):
                for alias in node.names:
                    if alias.name == "windows" or alias.name.startswith("windows."):
                        offenders.append(
                            f"{filename}: from {module or '..'} import windows"
                        )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("ohlc_toolkit.windows"):
                    offenders.append(f"{filename}: import {alias.name}")
    return offenders


def test_the_detector_fires_on_every_forbidden_spelling() -> None:
    """Positive control: a detector that detects nothing proves nothing.

    Each spelling below is one a real module could carry -- including
    the bare-name form an IDE auto-import produces and the relative
    form. If the detector goes quiet on any of them, the layering test
    is decoration.
    """
    spellings = [
        "import ohlc_toolkit.windows",
        "from ohlc_toolkit.windows import resolution",
        "from ohlc_toolkit.windows.resolution import resolve_schedule",
        "from ohlc_toolkit import windows",
        "from .. import windows",
    ]
    for spelling in spellings:
        assert _windows_imports(spelling, "control.py") != [], spelling


def test_the_detector_stays_quiet_on_allowed_imports() -> None:
    """The control's mirror: ordinary imports must not trip it."""
    allowed = "\n".join(
        [
            "import math",
            "from ohlc_toolkit.temporal import bounded_echo",
            "from ohlc_toolkit import temporal",
            "from . import identity",
        ]
    )
    assert _windows_imports(allowed, "control.py") == []


def test_no_schedules_module_imports_the_windows_package() -> None:
    """Every import in every schedules module stays below windows."""
    package_dir = Path(ohlc_toolkit.schedules.__file__).parent
    module_paths = sorted(package_dir.rglob("*.py"))
    assert module_paths != []

    offenders: list[str] = []
    for module_path in module_paths:
        offenders.extend(_windows_imports(module_path.read_text(), module_path.name))
    assert offenders == []
