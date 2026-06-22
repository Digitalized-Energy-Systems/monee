"""Tests for Pyomo infeasibility diagnostic tools."""

import pyomo.environ as pyo
import pytest

import monee.model as mm
from monee.model import Network
from monee.model.child import ExtPowerGrid, PowerLoad
from monee.model.node import Bus
from monee.problem.core import Constraints, OptimizationProblem
from monee.simulation.multi_period import run_multi_period
from monee.simulation.timeseries import TimeseriesData
from monee.solver.infeasibility.pyo import (
    InfeasibilityReport,
    _parse_var_name,
    _var_display_name,
    collect_constraint_residuals,
    collect_variable_bound_violations,
    collect_variables_at_bounds,
    diagnose_infeasibility,
)
from monee.solver.pyo import PyomoSolver
from tests.util import solver_available

# These tests exercise the diagnostics against a real solver run via raw
# pyo.SolverFactory, which needs the standalone executable (the pyscipopt
# bridge has different no-solution write-back semantics).
requires_scip = pytest.mark.skipif(
    not solver_available("scip"), reason="scip executable not installed"
)
requires_ipopt = pytest.mark.skipif(
    not solver_available("ipopt"), reason="ipopt executable not installed"
)


def test_parse_var_name_single_period():
    # WHEN
    result = _parse_var_name("child_3__p_mw")

    # THEN
    assert result == {"cat": "child", "id": 3, "t": None, "attr": "p_mw"}


def test_parse_var_name_multi_period():
    # WHEN
    result = _parse_var_name("child_3_t2__p_mw")

    # THEN
    assert result == {"cat": "child", "id": 3, "t": 2, "attr": "p_mw"}


def test_parse_var_name_branch():
    # WHEN
    result = _parse_var_name("branch_7__loading_from_pu")

    # THEN
    assert result == {
        "cat": "branch",
        "id": 7,
        "t": None,
        "attr": "loading_from_pu",
    }


def test_parse_var_name_unrecognized():
    # WHEN
    weird = _parse_var_name("something_weird")
    short = _parse_var_name("x")

    # THEN
    assert weird is None
    assert short is None


def test_var_display_name():
    # WHEN
    single = _var_display_name("child_3__p_mw")
    multi = _var_display_name("child_3_t2__p_mw")
    unknown = _var_display_name("unknown")

    # THEN
    assert single == "child[3].p_mw"
    assert multi == "child[3].p_mw (t=2)"
    assert unknown == "unknown"


def _infeasible_model():
    """Build a Pyomo model with conflicting constraints."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5), initialize=3)
    m.y = pyo.Var(bounds=(0, 5), initialize=3)
    m.cons = pyo.ConstraintList()
    m.cons.add(m.x + m.y >= 9)
    m.cons.add(m.x + m.y <= 4)
    m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
    return m


@requires_scip
def test_collect_constraint_residuals():
    # GIVEN
    m = _infeasible_model()
    solver = pyo.SolverFactory("scip")
    solver.solve(m)

    # WHEN
    residuals = collect_constraint_residuals(m, tol=1e-4)

    # THEN
    assert len(residuals) > 0

    # residuals sorted by magnitude (descending)
    if len(residuals) > 1:
        assert residuals[0].residual >= residuals[1].residual


def test_collect_variable_bound_violations():
    # GIVEN
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5), initialize=10)
    m.cons = pyo.ConstraintList()
    m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)

    # WHEN
    violations = collect_variable_bound_violations(m, tol=1e-4)

    # THEN
    assert len(violations) == 1
    assert violations[0].violation == pytest.approx(5.0, abs=0.1)
    assert violations[0].display_name == "x"


def test_collect_variables_at_bounds():
    # GIVEN
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10), initialize=0)
    m.y = pyo.Var(bounds=(0, 10), initialize=5)
    m.z = pyo.Var(bounds=(0, 10), initialize=10)
    m.cons = pyo.ConstraintList()
    m.obj = pyo.Objective(expr=m.x + m.y + m.z, sense=pyo.minimize)

    # WHEN
    at_bounds = collect_variables_at_bounds(m, tol=1e-4)

    # THEN
    names = {v["name"] for v in at_bounds}
    assert "x" in names
    assert "z" in names
    assert "y" not in names


@requires_scip
def test_diagnose_infeasibility_report():
    # GIVEN
    m = _infeasible_model()
    solver = pyo.SolverFactory("scip")
    solver.solve(m)

    # WHEN
    report = diagnose_infeasibility(m, solver_name="scip", compute_mis_flag=False)

    # THEN
    assert isinstance(report, InfeasibilityReport)

    summary = report.summary()
    assert "Violated constraints" in summary or "No violated constraints" in summary
    assert isinstance(repr(report), str)


@requires_scip
def test_diagnose_infeasibility_with_mis():
    # GIVEN
    m = _infeasible_model()
    solver = pyo.SolverFactory("scip")
    solver.solve(m)

    # WHEN
    report = diagnose_infeasibility(m, solver_name="scip", compute_mis_flag=True)

    # THEN
    # MIS should identify the two conflicting constraints
    assert len(report.mis_constraints) >= 2

    summary = report.summary()
    assert "Minimal Infeasible Subsystem" in summary


@requires_ipopt
def test_infeasible_monee_pyomo_solve():

    # GIVEN
    _LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=2.0, q_mvar=0.0))
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)

    # infeasible: ext-grid must supply 2 MW but is bounded to [0.1, 0.5]
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).equation(lambda m: m.p_mw >= 0.1).equation(
        lambda m: m.p_mw <= 0.5
    )
    prob.constraints = cons

    # WHEN
    # ipopt detects nonlinear infeasibility much faster than SCIP
    solver = PyomoSolver(solver_name="ipopt")
    result = solver.solve(net, optimization_problem=prob)

    # THEN
    assert not result.success
    assert hasattr(result, "infeasibility_report")
    assert isinstance(result.infeasibility_report, InfeasibilityReport)


def test_infeasible_multi_period_pyomo():

    # GIVEN
    _LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=1.0, q_mvar=0.0))
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 3.0])

    # impossible ext-grid bounds
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).equation(lambda m: m.p_mw >= 0.1).equation(
        lambda m: m.p_mw <= 0.5
    )
    prob.constraints = cons

    # WHEN / THEN
    # ipopt detects nonlinear infeasibility much faster than SCIP
    with pytest.raises(
        RuntimeError,
        match="CasADi/IPOPT multi-period solve did not converge \(return status: 'Infeasible_Problem_Detected'\)\.",
    ):
        run_multi_period(
            net,
            td,
            steps=2,
            optimization_problem=prob,
            solver="ipopt",
            dt_h=1.0,
        )
