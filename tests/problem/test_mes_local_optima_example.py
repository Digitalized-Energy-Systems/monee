"""Runs examples/mes_local_optima.py and requires every plausibility check it
performs to pass, and the report printed in its tutorial page to be the one
it produces, so neither can drift from the library."""

import contextlib
import importlib.util
import io
import sys
import textwrap
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "mes_local_optima.py"
TUTORIAL = ROOT / "docs" / "source" / "tutorials" / "mes_local_optima.rst"


def _load_example():
    spec = importlib.util.spec_from_file_location("mes_local_optima", EXAMPLE)
    module = importlib.util.module_from_spec(spec)
    # dataclasses resolves the defining module through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def solved():
    example = _load_example()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return example, example.run()


def _tutorial_report():
    lines = TUTORIAL.read_text(encoding="utf-8").splitlines()
    start = lines.index(".. code-block:: text") + 2
    block = []
    for line in lines[start:]:
        if line and not line.startswith("   "):
            break
        block.append(line)
    return textwrap.dedent("\n".join(block)).strip("\n")


def test_mes_local_optima_example_is_plausible(solved):
    example, (results, checks) = solved

    assert list(results.day["SCIP"]) == list(range(len(example.ed.EL_PRICE)))
    assert list(results.sweep["IPOPT"]) == [
        (cable_mva, t)
        for cable_mva in example.SWEEP_CABLE_MVA
        for t in example.SWEEP_HOURS
    ]
    assert checks.failed == []


def test_tutorial_shows_the_report_the_example_prints(solved):
    example, run = solved
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        example.print_report(*run)

    assert _tutorial_report() == printed.getvalue().strip("\n")
