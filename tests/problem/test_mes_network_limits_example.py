"""Runs examples/mes_network_limits.py and requires every plausibility check
it performs to pass, and the report printed in its tutorial page to be the
one it produces, so neither can drift from the library."""

import contextlib
import importlib.util
import io
import sys
import warnings
from pathlib import Path

import pytest

from tests.problem._tutorial_report import assert_report_matches, tutorial_report

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "mes_network_limits.py"
TUTORIAL = ROOT / "docs" / "source" / "tutorials" / "mes_network_limits.rst"


def _load_example():
    spec = importlib.util.spec_from_file_location("mes_network_limits", EXAMPLE)
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


def test_mes_network_limits_example_is_plausible(solved):
    example, (days, checks) = solved

    assert list(days) == list(example.DAYS)
    assert all(len(day.hours) == len(example.ed.EL_PRICE) for day in days.values())
    assert checks.failed == []


def test_tutorial_shows_the_report_the_example_prints(solved):
    example, run = solved
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        example.print_report(*run)

    assert_report_matches(tutorial_report(TUTORIAL), printed.getvalue().strip("\n"))
