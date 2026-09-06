"""Check that a lowest resolution installs exactly the declared floors.

Every runtime dependency in ``pyproject.toml`` carries a lower bound, and
those bounds are the only thing standing between a consumer of the published
wheel and a version with an advisory against it. Nothing else in this
repository exercises them: the lockfile pins versions far above every floor
and CI resolves at the newest compatible version, so the suite passes
whether a floor is right or wrong.

Run against an environment built with ``uv pip install --resolution lowest``,
this script reports every declared floor that is NOT the version which
actually installed. A mismatch means the bound in the metadata is not the
bound in effect -- the floor names a yanked release, or one with no wheel
for the running interpreter, and a resolver quietly used something else.
That is how ``polars>=1.35.0`` went unnoticed: 1.35.0 could never be
installed, because its pinned runtime companion was yanked.

It refuses rather than guesses. A requirement it cannot parse, a floor
without a lower bound, a version it cannot compare digit by digit, or a
declared distribution that is not installed is an error, not a pass.

Usage::

    uv run python scripts/check_dependency_floors.py [PYPROJECT]

``PYPROJECT`` defaults to ``pyproject.toml`` in the repository root. Exits
non-zero if any floor is not the version installed.
"""

from __future__ import annotations

import importlib.metadata
import re
import sys
import tomllib
from pathlib import Path

#: ``name>=1.2.3`` or ``name>=1.2.3,<2.0.0``. Deliberately narrow: extras,
#: environment markers and URL requirements are not what this repository
#: declares, and a shape this does not recognise is refused rather than
#: skipped.
_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"(?P<specifiers>(?:\s*[<>=!~]=?\s*[^,\s]+)(?:\s*,\s*[<>=!~]=?\s*[^,\s]+)*)$"
)

_LOWER_BOUND = re.compile(r"^>=\s*(?P<version>[0-9]+(?:\.[0-9]+)*)$")

#: Exit codes. A REFUSAL is kept distinct from a MISS so that a broken
#: declaration cannot be read as a floor that merely drifted.
EXIT_OK = 0
EXIT_MISSED = 1
EXIT_REFUSED = 2


class FloorCheckError(Exception):
    """Raised when a declaration cannot be checked at all."""


def declared_floors(pyproject: Path) -> dict[str, str]:
    """Read the lower bound of every runtime dependency.

    Raises ``FloorCheckError`` for a requirement this cannot parse or one
    that declares no lower bound, because an unchecked dependency is the
    exact gap this script exists to close.
    """
    project = tomllib.loads(pyproject.read_text(encoding="utf-8"))["project"]
    floors: dict[str, str] = {}
    for requirement in project["dependencies"]:
        match = _REQUIREMENT.match(requirement.strip())
        if match is None:
            msg = f"cannot parse the requirement {requirement!r}"
            raise FloorCheckError(msg)
        bounds = [
            bound.group("version")
            for specifier in match.group("specifiers").split(",")
            if (bound := _LOWER_BOUND.match(specifier.strip())) is not None
        ]
        if len(bounds) != 1:
            msg = (
                f"{requirement!r} declares {len(bounds)} lower bounds of the "
                f"form >=N.N; exactly one is required"
            )
            raise FloorCheckError(msg)
        floors[match.group("name")] = bounds[0]
    return floors


def _release(version: str) -> tuple[int, ...]:
    """Turn a dotted numeric version into a comparable tuple.

    Trailing zeros are dropped so that ``3.15`` and ``3.15.0`` compare
    equal. Anything that is not purely numeric -- a release candidate, a
    post-release, a local version -- is refused, since this script cannot
    say what such a floor means for a resolver.
    """
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", version) is None:
        msg = f"cannot compare the non-numeric version {version!r}"
        raise FloorCheckError(msg)
    parts = [int(part) for part in version.split(".")]
    while len(parts) > 1 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def installed_version(name: str) -> str:
    """Report the version of a declared distribution in this environment."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError as error:
        msg = f"{name} is declared but not installed"
        raise FloorCheckError(msg) from error


def main(argv: list[str]) -> int:
    """Compare every declared floor with the installed version."""
    pyproject = Path(argv[1] if len(argv) > 1 else "pyproject.toml")
    try:
        floors = declared_floors(pyproject)
        rows = [
            (name, floor, installed_version(name)) for name, floor in floors.items()
        ]
        mismatched = [
            (name, floor, found)
            for name, floor, found in rows
            if _release(floor) != _release(found)
        ]
    except FloorCheckError as error:
        print(f"REFUSED: {error}", file=sys.stderr)
        return EXIT_REFUSED

    width = max(len(name) for name, _, _ in rows)
    for name, floor, found in rows:
        mark = "!=" if (name, floor, found) in mismatched else "=="
        print(f"{name:<{width}}  declared >={floor:<12} installed {mark} {found}")

    if mismatched:
        print(
            f"\n{len(mismatched)} declared floor(s) are not the version a lowest "
            f"resolution installs, so the bound in the metadata is not the bound "
            f"in effect.",
            file=sys.stderr,
        )
        return EXIT_MISSED
    print(f"\nAll {len(rows)} declared floors are the versions installed.")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main(sys.argv))  # pragma: no cover - command line entry
