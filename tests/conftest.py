"""Suite-wide pytest configuration.

The CI installs the pip ``gurobipy`` wheel so the native-Gurobi backend gets
exercised. That wheel ships a size-limited (licence-free) licence capped at
2000 variables / 2000 constraints. Most test models are tiny and fit, but a few
larger ones (replicated districts, the restoration benchmark, simbench MES) do
not, and Gurobi then raises a ``GurobiError`` ("Model too large for size-limited
license"). That is an environment limitation, not a test failure, so any test
that trips it is skipped rather than failed. A machine with a full Gurobi
licence runs those tests normally.
"""

from __future__ import annotations

import pytest

# Gurobi error code GRB.Error.SIZE_LIMIT_EXCEEDED.
_GUROBI_SIZE_LIMIT_ERRNO = 10010
# Message fragments Gurobi/Pyomo surface when the size-limited licence is hit.
_SIZE_LIMIT_NEEDLES = ("size-limited license", "model too large")


def _iter_exception_chain(exc):
    """Yield ``exc`` and every ``__cause__`` / ``__context__`` behind it, so a
    size-limit error wrapped by Pyomo (or anything else) is still recognised."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _is_gurobi_size_limit(exc) -> bool:
    for e in _iter_exception_chain(exc):
        if (
            type(e).__name__ == "GurobiError"
            and getattr(e, "errno", None) == _GUROBI_SIZE_LIMIT_ERRNO
        ):
            return True
        message = str(e).lower()
        if any(needle in message for needle in _SIZE_LIMIT_NEEDLES):
            return True
    return False


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    """Turn a Gurobi size-limited-licence overflow into a skip wherever it
    surfaces during a test, instead of letting it fail the run."""
    outcome = yield
    excinfo = getattr(outcome, "excinfo", None)
    if excinfo is not None and _is_gurobi_size_limit(excinfo[1]):
        outcome.force_exception(
            pytest.skip.Exception(
                "Gurobi size-limited (licence-free) wheel: model exceeds the "
                "2000 variable/constraint cap; install a full Gurobi licence to "
                "run this test."
            )
        )
