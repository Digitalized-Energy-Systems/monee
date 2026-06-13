"""Routing tests for :mod:`monee.solver.dispatch` - the (solver, backend) resolution layer."""

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
from monee.solver.gurobipy import GurobipySolver


def _unavailable_pyomo_solvers(pyo_env):
    """Return known Pyomo plugin names whose executables are not available."""
    known = list(pyo_env.SolverFactory.__dict__["_cls"])
    return [
        n
        for n in known
        if not n.startswith("_")
        and not pyo_env.SolverFactory(n).available(exception_flag=False)
    ]


# Auto-routing


def test_auto_backend_gekko_for_apopt_bpopt_ipopt():
    # WHEN
    apopt_backend = _auto_backend("apopt")
    bpopt_backend = _auto_backend("bpopt")
    ipopt_backend = _auto_backend("ipopt")

    # THEN
    assert apopt_backend == "gekko"
    assert bpopt_backend == "gekko"
    assert ipopt_backend == "gekko"


def test_auto_backend_pyomo_for_unknown_names():
    # WHEN
    gurobi_backend = _auto_backend("gurobi")
    scip_backend = _auto_backend("scip")
    glpk_backend = _auto_backend("glpk")
    unknown_backend = _auto_backend("definitely_not_a_solver")

    # THEN
    assert gurobi_backend == "pyomo"
    assert scip_backend == "pyomo"
    assert glpk_backend == "pyomo"
    assert unknown_backend == "pyomo"


# resolve_solver - defaults & instances


def test_resolve_default_returns_gekko_ipopt():
    # WHEN
    s = resolve_solver()

    # THEN
    assert isinstance(s, GEKKOSolver)
    assert s.solver == GEKKO_SOLVERS["ipopt"]


def test_resolve_instance_returned_unchanged():
    # GIVEN
    inst = GEKKOSolver(solver=2)

    # WHEN
    resolved = resolve_solver(inst)

    # THEN
    assert resolved is inst


def test_resolve_instance_with_backend_raises():
    # GIVEN
    inst = GEKKOSolver()

    # WHEN / THEN
    with pytest.raises(ValueError, match="backend= cannot be specified"):
        resolve_solver(inst, backend="pyomo")


# resolve_solver - strings


def test_resolve_string_apopt_routes_to_gekko():
    # WHEN
    s = resolve_solver("apopt")

    # THEN
    assert isinstance(s, GEKKOSolver)
    assert s.solver == GEKKO_SOLVERS["apopt"]


def test_resolve_string_ipopt_routes_to_gekko_by_default():
    # WHEN
    s = resolve_solver("ipopt")

    # THEN
    assert isinstance(s, GEKKOSolver)
    assert s.solver == GEKKO_SOLVERS["ipopt"]


def test_resolve_explicit_backend_pyomo_overrides_dual():
    # GIVEN
    pyo_env = pytest.importorskip("pyomo.environ")
    if not pyo_env.SolverFactory("ipopt").available(exception_flag=False):
        pytest.skip("IPOPT not installed for Pyomo on this system.")

    # WHEN
    s = resolve_solver("ipopt", backend="pyomo")

    # THEN
    assert isinstance(s, PyomoSolver)
    assert s._solver_name == "ipopt"


def test_resolve_pyomo_known_but_unavailable_raises():
    # GIVEN
    pyo_env = pytest.importorskip("pyomo.environ")
    unavailable = _unavailable_pyomo_solvers(pyo_env)
    if not unavailable:
        pytest.skip("All Pyomo solvers happen to be available - cannot test path.")

    # WHEN / THEN
    with pytest.raises(ValueError, match="not available"):
        resolve_solver(unavailable[0], backend="pyomo")


def test_resolve_pyomo_unknown_name_raises():
    # WHEN / THEN
    with pytest.raises(ValueError, match="no solver plugin named"):
        resolve_solver("definitely_not_a_solver", backend="pyomo")


def test_resolve_gekko_unknown_name_raises():
    # WHEN / THEN
    with pytest.raises(ValueError, match="GEKKO has no solver"):
        resolve_solver("gurobi", backend="gekko")


def test_resolve_unknown_backend_raises():
    # WHEN / THEN
    with pytest.raises(ValueError, match="Unknown backend"):
        resolve_solver("ipopt", backend="cplex")


# resolve_solver - gurobipy backend


def test_resolve_gurobipy_backend_with_gurobi_name():
    # WHEN
    s = resolve_solver("gurobi", backend="gurobipy")

    # THEN
    assert isinstance(s, GurobipySolver)


def test_resolve_gurobipy_backend_without_solver_name():
    # backend alone is enough; gurobipy provides only 'gurobi'.
    # WHEN
    s = resolve_solver(backend="gurobipy")

    # THEN
    assert isinstance(s, GurobipySolver)


def test_resolve_gurobipy_backend_rejects_other_solver_name():
    # WHEN / THEN
    with pytest.raises(ValueError, match="only provides the 'gurobi' solver"):
        resolve_solver("ipopt", backend="gurobipy")


def test_resolve_gurobi_still_routes_to_pyomo_by_default():
    # Auto-routing is unchanged: 'gurobi' without an explicit backend stays Pyomo.
    # GIVEN
    pyo_env = pytest.importorskip("pyomo.environ")
    if not pyo_env.SolverFactory("gurobi").available(exception_flag=False):
        pytest.skip("Gurobi not installed for Pyomo on this system.")

    # WHEN
    s = resolve_solver("gurobi")

    # THEN
    assert isinstance(s, PyomoSolver)
    assert s._solver_name == "gurobi"


# resolve_multi_period_solver - same shape


def test_resolve_multi_period_default_returns_gekko():
    # WHEN
    s = resolve_multi_period_solver()

    # THEN
    assert isinstance(s, GekkoMultiPeriodSolver)


def test_resolve_multi_period_instance_returned_unchanged():
    # GIVEN
    inst = GekkoMultiPeriodSolver()

    # WHEN
    resolved = resolve_multi_period_solver(inst)

    # THEN
    assert resolved is inst


def test_resolve_multi_period_string_apopt_routes_to_gekko():
    # WHEN
    s = resolve_multi_period_solver("apopt")

    # THEN
    assert isinstance(s, GekkoMultiPeriodSolver)
    assert s._solver_int == GEKKO_SOLVERS["apopt"]


def test_resolve_multi_period_pyomo_unknown_raises():
    # WHEN / THEN
    with pytest.raises(ValueError, match="no solver plugin named"):
        resolve_multi_period_solver("definitely_not_a_solver", backend="pyomo")


def test_resolve_multi_period_routes_pyomo_for_gurobi():
    # GIVEN
    pyo_env = pytest.importorskip("pyomo.environ")
    if not pyo_env.SolverFactory("gurobi").available(exception_flag=False):
        pytest.skip("Gurobi not installed for Pyomo on this system.")

    # WHEN
    s = resolve_multi_period_solver("gurobi")

    # THEN
    assert isinstance(s, PyomoMultiPeriodSolver)
    assert s._solver_name == "gurobi"


def test_resolve_multi_period_gurobipy_backend_raises():
    # The native Gurobi backend is single-period only.
    # WHEN / THEN
    with pytest.raises(ValueError, match="single-period only"):
        resolve_multi_period_solver("gurobi", backend="gurobipy")
