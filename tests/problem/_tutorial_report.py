"""The report a tutorial page shows, and its comparison with the one its
example prints."""

import re
import textwrap

_DECIMAL = re.compile(r"-?\d+\.\d+")


def tutorial_report(page):
    """The page's first ``.. code-block:: text``, dedented."""
    lines = page.read_text(encoding="utf-8").splitlines()
    start = lines.index(".. code-block:: text") + 2
    block = []
    for line in lines[start:]:
        if line and not line.startswith("   "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block)).strip("\n")


def _text(line):
    return " ".join(_DECIMAL.sub("#", line).split())


def assert_report_matches(shown, printed):
    """Every line equal, except that a decimal number may differ by one unit
    in its last printed digit: a value within solver noise of a rounding
    boundary rounds either way depending on the CasADi build. Runs of spaces
    count as one, since such a flip can widen a column by a digit."""
    shown_lines, printed_lines = shown.splitlines(), printed.splitlines()
    assert len(shown_lines) == len(printed_lines), (shown, printed)
    for s, p in zip(shown_lines, printed_lines):
        assert _text(s) == _text(p), (s, p)
        for a, b in zip(_DECIMAL.findall(s), _DECIMAL.findall(p)):
            unit = 10.0 ** -len(a.split(".")[1])
            assert abs(float(a) - float(b)) <= 1.01 * unit, (s, p)
