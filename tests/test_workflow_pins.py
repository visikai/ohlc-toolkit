"""The workflows' action pins, checked instead of claimed.

Every ``uses:`` in a workflow under ``.github/workflows`` -- ``.yml`` or
``.yaml``, both of which GitHub honours -- must name an immutable commit
SHA and carry the version it points at beside it. A branch or tag ref
moves under the repository between runs, which matters most for the one
action here that holds publishing rights. Composite actions under
``.github/actions/`` are outside this claim; none exist today.

Two limits worth stating rather than discovering. These tests check a
pin's FORM, not its truth: an invented SHA with a lying version comment
satisfies every offline assertion here. The network-marked test closes
the half that matters, by asking the registry whether the pinned image
is still published at all.

This file exists because a commit message once stated that every action
in this repository was SHA-pinned while the publishing action floated on
``release/v1``. Prose could say that; nothing checked it. Now something
does.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
import requests

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

# The action holding publishing rights, and the registry it resolves
# through. It builds its own container reference from the ref it is
# called with, so the pinned SHA doubles as an image tag.
_PUBLISH_ACTION = "pypa/gh-action-pypi-publish"
_EXPECTED_PUBLISH_PINS = 1
_GHCR_TOKEN_URL = "https://ghcr.io/token?scope=repository:{repo}:pull&service=ghcr.io"
_GHCR_MANIFEST_URL = "https://ghcr.io/v2/{repo}/manifests/{ref}"
_MANIFEST_ACCEPT = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)
_HTTP_OK = 200
_TIMEOUT_SECONDS = 30


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
    """Collect every action reference across the workflow files.

    Both suffixes are globbed because GitHub runs both, so checking only
    one leaves a whole workflow -- publishing rights and all -- invisible
    to every assertion in this file.
    """
    found: list[Reference] = []
    for path in sorted(
        list(_WORKFLOWS.glob("*.yml")) + list(_WORKFLOWS.glob("*.yaml"))
    ):
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
    # Globbed here independently of `_references()` on purpose: sharing a
    # helper would let one broken glob hide from the census meant to
    # catch it.
    workflows = sorted(
        path.name
        for path in list(_WORKFLOWS.glob("*.yml")) + list(_WORKFLOWS.glob("*.yaml"))
    )
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


@pytest.mark.network
def test_the_pinned_publish_image_is_still_published() -> None:
    """Fail here rather than at release time if the image was pruned.

    Pinning this action buys immutability at an unusual cost. Because it
    derives its container tag from the ref it is called with, the pin's
    runnability depends on a registry tag the upstream project prunes:
    measured against ghcr, v1.12.4 and newer resolve while v1.11.0 and
    older return 404. A pin left un-bumped past that window does not go
    stale-but-working -- it stops running, on the release path, after
    the tag and GitHub release already exist. This asks the registry
    before that can happen.
    """
    pins = [r for r in _references() if r.ref.startswith(f"{_PUBLISH_ACTION}@")]
    assert len(pins) == _EXPECTED_PUBLISH_PINS, [str(r) for r in pins]
    sha = pins[0].ref.partition("@")[2]

    token = requests.get(
        _GHCR_TOKEN_URL.format(repo=_PUBLISH_ACTION), timeout=_TIMEOUT_SECONDS
    ).json()["token"]
    response = requests.head(
        _GHCR_MANIFEST_URL.format(repo=_PUBLISH_ACTION, ref=sha),
        headers={"Authorization": f"Bearer {token}", "Accept": _MANIFEST_ACCEPT},
        timeout=_TIMEOUT_SECONDS,
    )
    assert response.status_code == _HTTP_OK, (
        f"ghcr no longer serves {_PUBLISH_ACTION}:{sha} "
        f"(HTTP {response.status_code}). The pin has aged out of the "
        "registry's retention window and must be bumped before the next "
        "release, which would otherwise fail after publishing the tag."
    )
