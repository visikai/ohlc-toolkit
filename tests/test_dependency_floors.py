"""The floor check itself, checked instead of trusted.

``scripts/check_dependency_floors.py`` is a gate: under a lowest
resolution it decides whether the lower bounds in ``pyproject.toml`` are
the versions a resolver actually uses. A gate that silently passes on a
declaration it did not understand is worse than no gate, so its refusals
are what most of this file pins -- an unparseable requirement, a
dependency declared with no lower bound, a version it cannot compare, and
a declared distribution that is not installed all have to be errors.

The one place the two spellings of the same version must NOT be treated
as a miss is trailing zeros: ``idna>=3.15`` is satisfied exactly by
``3.15``, and a naive string comparison would report a false miss on
``3.15.0``.
"""

import importlib.metadata
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "check_dependency_floors.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_dependency_floors", _SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


floors = _load()


def _pyproject(tmp_path: Path, *dependencies: str) -> Path:
    # Single-quoted TOML literals, so a requirement carrying an
    # environment marker with its own double quotes still writes cleanly.
    body = ",\n".join(f"    '{item}'" for item in dependencies)
    path = tmp_path / "pyproject.toml"
    path.write_text(f"[project]\ndependencies = [\n{body}\n]\n", encoding="utf-8")
    return path


def test_every_runtime_dependency_of_this_package_declares_one_numeric_floor() -> None:
    """The real file parses, which is what the CI lane depends on."""
    declared = floors.declared_floors(_REPO_ROOT / "pyproject.toml")

    assert "polars" in declared
    assert "certifi" in declared
    for name, floor in declared.items():
        assert floors._release(floor), f"{name} declares an empty floor"


def test_a_dependency_with_no_lower_bound_is_refused(tmp_path: Path) -> None:
    """An uncapped-below dependency is exactly the gap this script closes."""
    path = _pyproject(tmp_path, "loguru<0.8.0")

    with pytest.raises(floors.FloorCheckError, match=r"0 lower bounds"):
        floors.declared_floors(path)


def test_two_lower_bounds_are_refused(tmp_path: Path) -> None:
    """Which of the two binds is not this script's guess to make."""
    path = _pyproject(tmp_path, "loguru>=0.7.3,>=0.7.9")

    with pytest.raises(floors.FloorCheckError, match=r"2 lower bounds"):
        floors.declared_floors(path)


@pytest.mark.parametrize(
    "requirement",
    [
        "loguru @ https://example.invalid/loguru.whl",
        'requests>=2.33.0; python_version < "3.12"',
        "requests[socks]>=2.33.0",
        "loguru",
    ],
)
def test_a_requirement_shape_it_does_not_recognise_is_refused(
    tmp_path: Path, requirement: str
) -> None:
    """Extras, markers, URLs and bare names are refused, not skipped."""
    path = _pyproject(tmp_path, requirement)

    with pytest.raises(floors.FloorCheckError, match=r"cannot parse"):
        floors.declared_floors(path)


def test_a_lower_bound_that_is_not_numeric_is_refused(tmp_path: Path) -> None:
    """A release-candidate floor has no digit-by-digit meaning here."""
    path = _pyproject(tmp_path, "polars>=1.38.1rc2,<2.0.0")

    with pytest.raises(floors.FloorCheckError, match=r"0 lower bounds"):
        floors.declared_floors(path)


def test_a_non_numeric_version_cannot_be_compared() -> None:
    """The comparison refuses rather than falling back on string equality."""
    with pytest.raises(floors.FloorCheckError, match=r"non-numeric version"):
        floors._release("2024.7.4.post1")


def test_trailing_zeros_do_not_make_a_floor_look_missed() -> None:
    """``3.15`` and ``3.15.0`` are the same release, so neither is a miss."""
    assert floors._release("3.15") == floors._release("3.15.0")
    assert floors._release("1.38.1") != floors._release("1.38.10")
    assert floors._release("0") == floors._release("0.0")


def test_a_declared_distribution_that_is_not_installed_is_refused() -> None:
    """Nothing to compare against is an error, not a satisfied floor."""
    with pytest.raises(floors.FloorCheckError, match=r"declared but not installed"):
        floors.installed_version("ohlc-toolkit-no-such-distribution")


def test_the_installed_version_of_a_real_distribution_is_reported() -> None:
    """The happy path of the lookup, against a package that is certainly here."""
    assert floors.installed_version("polars") == importlib.metadata.version("polars")


@pytest.fixture
def _pinned_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, str]]:
    versions: dict[str, str] = {}
    monkeypatch.setattr(
        floors.importlib.metadata, "version", lambda name: versions[name]
    )
    yield versions


@pytest.mark.usefixtures("_pinned_versions")
def test_the_check_passes_when_every_floor_is_the_installed_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    _pinned_versions: dict[str, str],
) -> None:
    """The green case names every dependency it compared."""
    _pinned_versions.update({"loguru": "0.7.3", "idna": "3.15.0"})
    path = _pyproject(tmp_path, "loguru>=0.7.3,<0.8.0", "idna>=3.15,<4.0.0")

    assert floors.main(["check", str(path)]) == floors.EXIT_OK

    out = capsys.readouterr().out
    assert "All 2 declared floors are the versions installed." in out
    assert "loguru" in out
    assert "idna" in out


@pytest.mark.usefixtures("_pinned_versions")
def test_the_check_fails_when_a_floor_is_not_the_version_installed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    _pinned_versions: dict[str, str],
) -> None:
    """A yanked or unbuildable floor resolves upwards, and that is the miss."""
    _pinned_versions.update({"polars": "1.35.1", "idna": "3.15"})
    path = _pyproject(tmp_path, "polars>=1.35.0,<2.0.0", "idna>=3.15,<4.0.0")

    assert floors.main(["check", str(path)]) == floors.EXIT_MISSED

    captured = capsys.readouterr()
    assert "polars" in captured.out
    assert "!= 1.35.1" in captured.out
    assert "idna" in captured.out
    assert "1 declared floor(s) are not the version" in captured.err


def test_the_check_refuses_with_its_own_exit_code(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A refusal is distinguishable from a miss, so CI can tell them apart."""
    path = _pyproject(tmp_path, "loguru")

    assert floors.main(["check", str(path)]) == floors.EXIT_REFUSED

    assert "REFUSED: cannot parse" in capsys.readouterr().err


def test_the_pyproject_argument_defaults_to_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Run with no argument, it reads ``pyproject.toml`` where it was invoked."""
    _pyproject(tmp_path, "loguru")
    monkeypatch.chdir(tmp_path)

    assert floors.main(["check"]) == floors.EXIT_REFUSED

    assert "REFUSED" in capsys.readouterr().err
