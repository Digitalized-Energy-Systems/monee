"""Execute the testcode blocks of selected docs pages, in page order, and
check each printed output against the page's testoutput block, mirroring the
Sphinx doctest builder so the docs cannot drift from the code."""

import contextlib
import doctest
import io
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parents[2] / "docs" / "source"

_DIRECTIVE_RE = re.compile(r"^(\s*)\.\.\s+(testcode|testoutput)::\s*$")
_OPTION_FLAGS = {
    "+SKIP": doctest.SKIP,
    "+ELLIPSIS": doctest.ELLIPSIS,
    "+NORMALIZE_WHITESPACE": doctest.NORMALIZE_WHITESPACE,
}


@dataclass
class DocBlock:
    kind: str
    line: int
    body: str
    flags: int = 0
    expected: "DocBlock | None" = field(default=None, repr=False)


def parse_blocks(path: Path) -> list[DocBlock]:
    blocks = []
    lines = path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        match = _DIRECTIVE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        base_indent, kind = len(match.group(1)), match.group(2)
        start_line = i + 1
        i += 1
        raw = []
        while i < len(lines):
            line = lines[i]
            if line.strip() and (len(line) - len(line.lstrip())) <= base_indent:
                break
            raw.append(line)
            i += 1
        flags = 0
        content = []
        in_fields = True
        for line in raw:
            stripped = line.strip()
            if in_fields and stripped.startswith(":options:"):
                for option in stripped.removeprefix(":options:").split():
                    flags |= _OPTION_FLAGS[option]
                continue
            if stripped:
                in_fields = False
            content.append(line)
        body = textwrap.dedent("\n".join(content)).strip("\n")
        blocks.append(DocBlock(kind=kind, line=start_line, body=body, flags=flags))

    paired = []
    for block in blocks:
        if block.kind == "testoutput":
            assert paired and paired[-1].expected is None, (
                f"{path.name}:{block.line}: testoutput without testcode"
            )
            paired[-1].expected = block
        else:
            paired.append(block)
    return paired


def run_page(page: str):
    path = DOCS / page
    blocks = parse_blocks(path)
    assert blocks, f"no testcode blocks found in {page}"
    checker = doctest.OutputChecker()
    namespace = {}
    for block in blocks:
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(block.body, f"{page}:{block.line}", "exec"), namespace)
        expected = block.expected
        if expected is None or expected.flags & doctest.SKIP:
            continue
        got = stdout.getvalue()
        want = expected.body + "\n"
        assert checker.check_output(want, got, expected.flags), (
            f"{page}:{expected.line}: output mismatch\n"
            f"--- expected ---\n{want}--- got ---\n{got}"
        )


@pytest.mark.parametrize(
    "page",
    [
        "quickstart.rst",
        "tutorials/03_coupled_network.rst",
        "how-to/load_shedding.rst",
        "how-to/economic_dispatch.rst",
        "how-to/express_structures.rst",
        pytest.param(
            "how-to/district_heating.rst",
            marks=pytest.mark.skipif(
                not (DOCS / "how-to" / "district_heating.rst").exists(),
                reason="page not yet authored (dh-docs batch)",
            ),
        ),
    ],
)
def test_doc_page_blocks(page):
    run_page(page)
