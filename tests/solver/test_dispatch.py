"""
Tests for :mod:`monee.solver.dispatch` — the (solver, backend) resolution layer.

These tests focus on the *routing* logic, not the underlying solvers.  We
inspect the type of the returned solver instance rather than running it.
"""

import pytest

from monee.simulation.multi_period import (
    GekkoMultiPeriodSolver,
    PyomoMultiPeriodSolver,
)
from monee.solver import (
    GEKKO_SOLVERS,
    GEKKOSolver,
    PyomoSolver,
    resolve_multi_period_solver,
    resolve_solver,
)
from monee.solver.dispatch import _auto_backend

# Auto-routing


def test_auto_backend_gekko_for_apopt_bpopt_ipopt():
    assert _auto_backend("apopt") == "gekko"
    assert _auto_backend("bpopt") == "gekko"
    assert _auto_backend("ipopt") == "gekko"


def test_auto_backend_pyomo_for_unknown_names():
    assert _auto_backend("gurobi") == "pyomo"
    assert _auto_backend("scip") == "pyomo"
    assert _auto_backend("glpk") == "pyomo"
    assert _auto_backend("definitely_not_a_solver") == "pyomo"


# resolve_solver — defaults & instances


def test_resolve_default_returns_gekko_ipopt():
    s = resolve_solver()
    assert isinstance(s, GEKKOSolver)
    assert s.solver == GEKKO_SOLVERS["ipopt"]


def test_resolve_instance_returned_unchanged():
    inst = GEKKOSolver(solver=2)
    assert resolve_solver(inst) is inst


def test_resolve_instance_with_backend_raises():
    with pytest.raises(ValueError, match="backend= cannot be specified"):
        resolve_solver(GEKKOSolver(), backend="pyomo")


# resolve_solver — strings


def test_resolve_string_apopt_routes_to_gekko():
    s = resolve_solver("apopt")
    assert isinstance(s, GEKKOSolver)
    assert s.solver == GEKKO_SOLVERS["apopt"]


def test_resolve_string_ipopt_routes_to_gekko_by_default():
    s = resolve_solver("ipopt")
    assert isinstance(s, GEKKOSolver)
    assert s.solver == GEKKO_SOLVERS["ipopt"]


def test_resolve_explicit_backend_pyomo_overrides_dual():
    """``solver="ipopt", backend="pyomo"`` → Pyomo+IPOPT (if installed)."""
    pyo_env = pytest.importorskip("pyomo.environ")
    if not pyo_env.SolverFactory("ipopt").available(exception_flag=False):
        pytest.skip("IPOPT not installed for Pyomo on this system.")
    s = resolve_solver("ipopt", backend="pyomo")
    assert isinstance(s, PyomoSolver)
    assert s._solver_name == "ipopt"


def test_resolve_pyomo_known_but_unavailable_raises():
    """A Pyomo plugin name that exists but whose executable isn't installed."""
    pyo_env = pytest.importorskip("pyomo.environ")
    # Find a known plugin name that is NOT available.
    known = list(pyo_env.SolverFactory.__dict__["_cls"])
    unavailable = [
        n
        for n in known
        if not n.startswith("_")
        and not pyo_env.SolverFactory(n).available(exception_flag=False)
    ]
    if not unavailable:
        pytest.skip("All Pyomo solvers happen to be available — cannot test path.")
    with pytest.raises(ValueError, match="not available"):
        resolve_solver(unavailable[0], backend="pyomo")


def test_resolve_pyomo_unknown_name_raises():
    with pytest.raises(ValueError, match="no solver plugin named"):
        resolve_solver("definitely_not_a_solver", backend="pyomo")


def test_resolve_gekko_unknown_name_raises():
    with pytest.raises(ValueError, match="GEKKO has no solver"):
        resolve_solver("gurobi", backend="gekko")


def test_resolve_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_solver("ipopt", backend="cplex")


# resolve_multi_period_solver — same shape


def test_resolve_multi_period_default_returns_gekko():
    s = resolve_multi_period_solver()
    assert isinstance(s, GekkoMultiPeriodSolver)


def test_resolve_multi_period_instance_returned_unchanged():
    inst = GekkoMultiPeriodSolver()
    assert resolve_multi_period_solver(inst) is inst


def test_resolve_multi_period_string_apopt_routes_to_gekko():
    s = resolve_multi_period_solver("apopt")
    assert isinstance(s, GekkoMultiPeriodSolver)
    assert s._solver_int == GEKKO_SOLVERS["apopt"]


def test_resolve_multi_period_pyomo_unknown_raises():
    with pytest.raises(ValueError, match="no solver plugin named"):
        resolve_multi_period_solver("definitely_not_a_solver", backend="pyomo")


def test_resolve_multi_period_routes_pyomo_for_gurobi():
    """If Pyomo+Gurobi is available, ``solver='gurobi'`` should route to it."""
    pyo_env = pytest.importorskip("pyomo.environ")
    if not pyo_env.SolverFactory("gurobi").available(exception_flag=False):
        pytest.skip("Gurobi not installed for Pyomo on this system.")
    s = resolve_multi_period_solver("gurobi")
    assert isinstance(s, PyomoMultiPeriodSolver)
    assert s._solver_name == "gurobi"
