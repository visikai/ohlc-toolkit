"""The workflows' action pins, checked instead of claimed.

Every ``uses:`` under .github/workflows must name an immutable commit
SHA and carry the version it points at beside it. A branch or tag ref
moves under the repository between runs, which matters most for the one
action here that holds publishing rights.

This file exists because a commit message once stated that every action
in this repository was SHA-pinned while the publishing action floated on
``release/v1``. Prose could say that; nothing checked it. Now something
does.
"""

import re
from dataclasses import dataclass
from pathlib import Path

_WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"

# A step's action reference, with the version comment written beside
# every pin. Both `- uses:` and `uses:` spellings appear in these files.
_USES = re.compile(
    r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+)(?:\s+#\s*(?P<comment>.*\S))?\s*$"
)
# owner/repo@<40 hex>. Anything else -- a tag, a branch, a short SHA --
# is mutable or ambiguous and fails.
_PINNED = re.compile(r"^[^@\s]+/[^@\s]+@[0-9a-f]{40}$")
_VERSION = re.compile(r"^v\d+\.\d+\.\d+$")

# Census pins. Without them a glob that stops matching, or a regex that
# stops parsing, would leave every assertion below iterating an empty
# list and passing vacuously. Update these deliberately when a workflow
# or a step is added.
_EXPECTED_WORKFLOWS = 5
_EXPECTED_REFERENCES = 20


@dataclass(frozen=True)
class Reference:
    """One ``uses:`` reference, located by file and line."""

    workflow: str
    line: int
    ref: str
    comment: str | None

    def __str__(self) -> str:
        """Render as the file:line the reader needs to open."""
        return f"{self.workflow}:{self.line} -> {self.ref}"


def _references() -> list[Reference]:
    """Collect every action reference across the workflow files."""
    found: list[Reference] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            match = _USES.match(line)
            if match is None:
                continue
            found.append(
                Reference(
                    workflow=path.name,
                    line=number,
                    ref=match["ref"],
                    comment=match["comment"],
                )
            )
    return found


def test_the_workflow_census_is_pinned() -> None:
    """Refuse to let a broken parser turn the checks below into no-ops."""
    workflows = sorted(path.name for path in _WORKFLOWS.glob("*.yml"))
    assert len(workflows) == _EXPECTED_WORKFLOWS, workflows
    references = _references()
    assert len(references) == _EXPECTED_REFERENCES, [str(r) for r in references]


def test_every_action_is_pinned_to_a_commit_sha() -> None:
    """Fail on any action referenced by a mutable ref."""
    floating = [str(r) for r in _references() if not _PINNED.match(r.ref)]
    assert not floating, f"actions not pinned to a commit SHA: {floating}"


def test_every_pin_names_the_version_it_points_at() -> None:
    """Fail on a pin whose SHA is not paired with a readable version."""
    unlabelled = [
        str(r)
        for r in _references()
        if r.comment is None or not _VERSION.match(r.comment)
    ]
    assert not unlabelled, f"pins without a version comment: {unlabelled}"
