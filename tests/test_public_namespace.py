"""The 1.0 public namespace, pinned.

These are pins, not a red/green pair, and the distinction is worth being
honest about. A deletion has no failing test to catch a skip: written
before the deletion, "read_ohlc_csv is not importable" is simply a false
statement about code that is still there, and the only thing that turns
it green is the deletion itself. So these were written after, and what
they buy is the other direction in time -- a later commit that adds a
compatibility alias, or flattens a name into the top level, or drops a
subpackage out of the passthrough, fails here instead of shipping.

Two of them need a fresh interpreter to mean anything. Once any test in
this session has run ``import ohlc_toolkit.windows``, the submodule is
bound as an attribute of the parent package by the import system --
whatever ``ohlc_toolkit/__init__.py`` does or does not do. Asserting
``hasattr(ohlc_toolkit, "windows")`` in this process would therefore pass
even with an empty ``__init__``, so the passthrough is checked in a
subprocess that has imported nothing else.
"""

import subprocess
import sys
from importlib import import_module

import pytest

import ohlc_toolkit

# The whole of the 1.0 public surface, as subpackages. Spelled out rather
# than derived from the package directory: a test that discovers the
# answer from the thing under test cannot notice the thing changing.
EXPECTED_SUBPACKAGES = (
    "returns",
    "schedules",
    "snapshot",
    "source",
    "temporal",
    "windows",
)

# Every name 0.4 exported from the top level, plus the one public helper
# that lived a level down. None may come back, under any spelling.
RETIRED_NAMES = (
    "DatasetDownloader",
    "calculate_percentage_return",
    "format_timeframe",
    "parse_timeframe",
    "read_ohlc_csv",
    "transform_ohlc",
    "validate_timeframe",
    "validate_timeframe_format",
)

# Every module the 1.0 break removed.
RETIRED_MODULES = (
    "ohlc_toolkit.bitstamp_dataset_downloader",
    "ohlc_toolkit.csv_reader",
    "ohlc_toolkit.exceptions",
    "ohlc_toolkit.future_returns",
    "ohlc_toolkit.pandas_ta",
    "ohlc_toolkit.timeframes",
    "ohlc_toolkit.transform",
    "ohlc_toolkit.utils",
)


def _run(source: str) -> subprocess.CompletedProcess[str]:
    """Execute ``source`` in a fresh interpreter and return the result.

    Args:
        source: The program to run.

    Returns:
        The completed process, with output captured as text.

    """
    return subprocess.run(
        [sys.executable, "-c", source], capture_output=True, text=True, check=False
    )


def test_all_is_exactly_the_six_subpackages() -> None:
    """``__all__`` names the contract surface and nothing else."""
    assert tuple(ohlc_toolkit.__all__) == EXPECTED_SUBPACKAGES


def test_all_is_sorted_and_free_of_duplicates() -> None:
    """One entry per subpackage, in a stable order a diff can read."""
    assert ohlc_toolkit.__all__ == sorted(set(ohlc_toolkit.__all__))


@pytest.mark.parametrize("name", EXPECTED_SUBPACKAGES)
def test_each_exported_name_is_the_subpackage_it_names(name: str) -> None:
    """An exported name resolves to the module of that name, not a copy."""
    attribute = getattr(ohlc_toolkit, name)
    assert attribute is import_module(f"ohlc_toolkit.{name}")


def test_a_bare_import_reaches_every_subpackage() -> None:
    """``import ohlc_toolkit`` alone is enough to use the whole surface.

    Run cold, in a process that has imported nothing else, because in
    this one the submodules are already bound by other tests' imports.
    """
    attributes = " and ".join(f"ohlc_toolkit.{name}" for name in EXPECTED_SUBPACKAGES)
    result = _run(f"import ohlc_toolkit; assert {attributes}; print('ok')")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"


def test_a_star_import_binds_exactly_the_six() -> None:
    """``from ohlc_toolkit import *`` brings in the subpackages, no names.

    Cold again: a star import honours ``__all__``, so what it binds is
    the assertion, but a stale submodule already in ``sys.modules`` would
    not change the answer while an unrelated import in this session
    could.
    """
    source = (
        "from ohlc_toolkit import *\n"
        "bound = sorted(n for n in dir() if not n.startswith('_'))\n"
        "print(','.join(bound))\n"
    )
    result = _run(source)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == ",".join(EXPECTED_SUBPACKAGES)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_no_retired_name_is_a_top_level_attribute(name: str) -> None:
    """No 0.4 export survives at the top level, aliased or otherwise."""
    assert not hasattr(ohlc_toolkit, name)
    with pytest.raises(AttributeError):
        getattr(ohlc_toolkit, name)


@pytest.mark.parametrize("name", RETIRED_NAMES)
def test_no_retired_name_is_exported(name: str) -> None:
    """Nor does one linger in ``__all__`` without an attribute behind it."""
    assert name not in ohlc_toolkit.__all__


@pytest.mark.parametrize("module", RETIRED_MODULES)
def test_no_retired_module_is_importable(module: str) -> None:
    """Every deleted module is gone from the package, not merely unexported."""
    with pytest.raises(ModuleNotFoundError):
        import_module(module)


def test_pandas_is_not_a_dependency_of_the_package() -> None:
    """Importing the whole surface pulls in no pandas.

    The 1.0 break is a change of DataFrame library, and an import left
    behind somewhere would keep the dependency alive in every consumer's
    environment while the pyproject claimed otherwise. Checked in a cold
    interpreter, since pytest itself imports a great deal.
    """
    source = (
        "import sys\n"
        "import ohlc_toolkit\n"
        "for name in ohlc_toolkit.__all__:\n"
        "    getattr(ohlc_toolkit, name)\n"
        "print(sorted(m for m in sys.modules if m in {'pandas', 'tqdm'}))\n"
    )
    result = _run(source)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
