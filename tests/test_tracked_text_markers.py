"""Fail when tracked text contains a workspace marker.

Every path from `git ls-files` is read. A file is binary, and skipped, only
when its bytes contain a NUL in the first 8192 bytes. Filenames, extensions,
and size are not consulted. Lockfiles in `EXCLUSIONS` are the only named
skips. In this file, only the line that states the regex literal is skipped.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
THIS_FILE = Path(__file__).resolve().relative_to(REPO).as_posix()
SNIFF_BYTES = 8192

# fmt: off
PATTERN = re.compile(r"\b(?:root|strategy|models|live-data|execution|datasets|inputs|evaluation|targets):\d+\b|mdboard|foresight-lab", re.IGNORECASE)  # noqa: E501
# fmt: on

# One-line justification per filename. This is not an extension list.
EXCLUSIONS = {
    "uv.lock": "Generated dependency lock; tool output, not authored prose.",
    "poetry.lock": "Generated dependency lock; tool output, not authored prose.",
    "package-lock.json": "Generated dependency lock; tool output, not authored prose.",
}


def tracked_paths() -> list[str]:
    """List paths tracked by Git."""
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return [item for item in completed.stdout.decode().split("\0") if item]


def is_binary(path: Path) -> bool:
    """Return whether the file contains a NUL in its first 8 KiB."""
    with path.open("rb") as handle:
        sample = handle.read(SNIFF_BYTES)
    return b"\0" in sample


def is_pattern_definition(rel: str, line: str) -> bool:
    """Identify the sole line that defines the marker regex."""
    return rel == THIS_FILE and line.strip().startswith("PATTERN = re.compile(")


def marker_hits() -> list[str]:
    """Return marker locations in tracked text files."""
    hits: list[str] = []
    for rel in tracked_paths():
        if Path(rel).name in EXCLUSIONS:
            continue
        path = REPO / rel
        if not path.is_file() or is_binary(path):
            continue
        with path.open(
            encoding="utf-8", errors="surrogateescape", newline=""
        ) as handle:
            lines = handle.readlines()
        for number, line in enumerate(lines, start=1):
            if is_pattern_definition(rel, line):
                continue
            if PATTERN.search(line):
                hits.append(f"{rel}:{number}")
    return hits


def test_tracked_text_has_no_workspace_markers() -> None:
    """Reject markers in tracked text outside the exclusions."""
    for name, reason in EXCLUSIONS.items():
        assert "\n" not in reason and reason.strip(), name
    hits = marker_hits()
    assert not hits, "\n".join(hits)


def test_non_utf8_tracked_text_is_scanned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch ASCII markers inside non-UTF-8 tracked files."""
    subprocess.run(["git", "init", "--quiet"], cwd=tmp_path, check=True)
    path = tmp_path / "tracked.txt"
    marker = b"root" + bytes((58,)) + b"123"
    path.write_bytes(b"\xff " + marker + b"\n")
    subprocess.run(["git", "add", path.name], cwd=tmp_path, check=True)
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    assert marker_hits() == ["tracked.txt:1"]
