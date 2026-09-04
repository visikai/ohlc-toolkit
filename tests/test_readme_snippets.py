"""The README's code blocks, executed — the pasted outputs are assertions.

Every ``python`` fence in README.md is parsed out of the file itself (no
copies that can drift) and executed. Where a ``text`` fence follows
immediately, captured stdout must match it byte for byte, after dropping
the progress-log lines the README's prose says are omitted. A block with
no pasted output is a claim: every bare expression in it must hold.
Editing a snippet or its pasted output alone fails here.
"""

import ast
import contextlib
import io
import re
from dataclasses import dataclass
from pathlib import Path

import pytest

_README = Path(__file__).parents[1] / "README.md"

# The toolkit logs progress to stdout (colorized by default); the README
# states those lines are omitted from its pasted outputs, so the
# comparison drops them too: strip ANSI, then match the loguru prefix.
_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_LOG_LINE = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+ \| ")
# A pasted-output fence counts as paired only when nothing but one blank
# line separates it from its snippet.
_MAX_PAIR_GAP_LINES = 2
# The README currently carries four python fences: quickstart (network),
# validation and schedules (offline, with pasted output), and the
# bare-expression Duration block. Parser rot fails this count.
_EXPECTED_TOTAL, _EXPECTED_OFFLINE = 4, 2
# The Duration block carries exactly two bare comparisons; a reformat
# that removes or demotes one must fail loud, never pass vacuously.
_EXPECTED_BARE_CLAIMS = 2
# The one snippet that fetches the released dataset lives under this
# heading. Routing is by SECTION — a structural signal a code edit
# cannot move — never by matching the snippet's text.
_NETWORK_SECTIONS = frozenset({"Quickstart"})


@dataclass(frozen=True)
class Fence:
    """One fenced code block, located by its opening line in README.md."""

    lang: str
    section: str
    start_line: int
    end_line: int
    code: str


def _fences(text: str) -> list[Fence]:
    """Return every fenced block in document order, tagged by its section."""
    fences: list[Fence] = []
    lang: str | None = None
    section = ""
    start = 0
    body: list[str] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if lang is None and line.startswith("## "):
            section = line.removeprefix("## ").strip()
        if line.startswith("```"):
            if lang is None:
                lang = line[3:].strip()
                start = number
                body = []
            else:
                fences.append(
                    Fence(lang, section, start, number, "\n".join(body) + "\n")
                )
                lang = None
        elif lang is not None:
            body.append(line)
    return fences


def _snippets() -> list[tuple[Fence, str | None]]:
    """Pair each python fence with the text fence that immediately follows."""
    fences = _fences(_README.read_text(encoding="utf-8"))
    pairs: list[tuple[Fence, str | None]] = []
    for index, fence in enumerate(fences):
        if fence.lang != "python":
            continue
        expected = None
        if index + 1 < len(fences):
            follower = fences[index + 1]
            if (
                follower.lang == "text"
                and follower.start_line - fence.end_line <= _MAX_PAIR_GAP_LINES
            ):
                expected = follower.code
        pairs.append((fence, expected))
    return pairs


def _printed_output(buffer: str) -> str:
    """Drop progress-log lines; return what a reader compares to the README."""
    lines = [
        line for line in buffer.splitlines() if not _LOG_LINE.match(_ANSI.sub("", line))
    ]
    return "\n".join(lines) + "\n" if lines else ""


def _run(fence: Fence) -> str:
    """Execute a snippet in a fresh namespace and return its printed output."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(fence.code, f"README.md:{fence.start_line}", "exec"), {})
    return _printed_output(buffer.getvalue())


_PAIRS = _snippets()
_OFFLINE = [
    (fence, expected)
    for fence, expected in _PAIRS
    if expected is not None and fence.section not in _NETWORK_SECTIONS
]
_BARE = [fence for fence, expected in _PAIRS if expected is None]
_NETWORK = [
    (fence, expected)
    for fence, expected in _PAIRS
    if expected is not None and fence.section in _NETWORK_SECTIONS
]


def test_the_parser_finds_the_documented_snippets() -> None:
    """Positive control: a parser that finds nothing proves nothing."""
    assert len(_PAIRS) == _EXPECTED_TOTAL
    assert len(_OFFLINE) == _EXPECTED_OFFLINE
    assert len(_BARE) == 1
    assert len(_NETWORK) == 1


@pytest.mark.parametrize(
    ("fence", "expected"),
    _OFFLINE,
    ids=[f"line{fence.start_line}" for fence, _ in _OFFLINE],
)
def test_offline_snippets_reproduce_their_pasted_output(
    fence: Fence, expected: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each offline snippet's stdout equals its pasted block exactly.

    Run from a temporary directory so a misrouted snippet that writes
    files can never pollute the checkout.
    """
    monkeypatch.chdir(tmp_path)
    assert _run(fence) == expected


def test_bare_expression_snippets_hold() -> None:
    """A block with no pasted output is a claim: each comparison must hold.

    Every claim must be a genuine comparison, and the claim count is
    pinned — so demoting one to a comment, an assignment, or a bare
    value fails loud instead of leaving zero assertions.
    """
    assert _BARE != []
    claims = 0
    for fence in _BARE:
        namespace: dict[str, object] = {}
        for node in ast.parse(fence.code).body:
            if isinstance(node, ast.Expr):
                assert isinstance(node.value, ast.Compare), ast.unparse(node)
                expression = compile(
                    ast.Expression(node.value), f"README.md:{fence.start_line}", "eval"
                )
                assert eval(expression, namespace), ast.unparse(node)
                claims += 1
            else:
                statement = ast.Module(body=[node], type_ignores=[])
                exec(
                    compile(statement, f"README.md:{fence.start_line}", "exec"),
                    namespace,
                )
    assert claims == _EXPECTED_BARE_CLAIMS


@pytest.mark.network
def test_the_quickstart_reproduces_its_pasted_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The front-page example runs against the real release, values pinned."""
    (fence, expected) = _NETWORK[0]
    assert expected is not None
    monkeypatch.chdir(tmp_path)
    assert _run(fence) == expected
