"""
Tests for Pyomo infeasibility diagnostic tools.
"""

import pyomo.environ as pyo
import pytest

from monee.solver.infeasibility.pyo import (
    InfeasibilityReport,
    _parse_var_name,
    _var_display_name,
    collect_constraint_residuals,
    collect_variable_bound_violations,
    collect_variables_at_bounds,
    diagnose_infeasibility,
)


def test_parse_var_name_single_period():
    result = _parse_var_name("child_3__p_mw")
    assert result == {"cat": "child", "id": 3, "t": None, "attr": "p_mw"}


def test_parse_var_name_multi_period():
    result = _parse_var_name("child_3_t2__p_mw")
    assert result == {"cat": "child", "id": 3, "t": 2, "attr": "p_mw"}


def test_parse_var_name_branch():
    result = _parse_var_name("branch_7__loading_from_percent")
    assert result == {
        "cat": "branch",
        "id": 7,
        "t": None,
        "attr": "loading_from_percent",
    }


def test_parse_var_name_unrecognized():
    assert _parse_var_name("something_weird") is None
    assert _parse_var_name("x") is None


def test_var_display_name():
    assert _var_display_name("child_3__p_mw") == "child[3].p_mw"
    assert _var_display_name("child_3_t2__p_mw") == "child[3].p_mw (t=2)"
    assert _var_display_name("unknown") == "unknown"


def _infeasible_model():
    """Build a Pyomo model with conflicting constraints."""
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5), initialize=3)
    m.y = pyo.Var(bounds=(0, 5), initialize=3)
    m.cons = pyo.ConstraintList()
    m.cons.add(m.x + m.y >= 9)  # needs x+y >= 9
    m.cons.add(m.x + m.y <= 4)  # needs x+y <= 4
    m.obj = pyo.Objective(expr=m.x + m.y, sense=pyo.minimize)
    return m


def test_collect_constraint_residuals():
    m = _infeasible_model()
    # Solve - will be infeasible, but initial values are x=3, y=3 → x+y=6
    solver = pyo.SolverFactory("scip")
    solver.solve(m)

    residuals = collect_constraint_residuals(m, tol=1e-4)
    # At least one constraint should be violated at the current variable values
    assert len(residuals) > 0
    # Residuals should be sorted by magnitude (descending)
    if len(residuals) > 1:
        assert residuals[0].residual >= residuals[1].residual


def test_collect_variable_bound_violations():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5), initialize=10)  # violates upper bound
    m.cons = pyo.ConstraintList()
    m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)

    # Don't solve - just check the initial values
    violations = collect_variable_bound_violations(m, tol=1e-4)
    assert len(violations) == 1
    assert violations[0].violation == pytest.approx(5.0, abs=0.1)
    assert violations[0].display_name == "x"


def test_collect_variables_at_bounds():
    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 10), initialize=0)
    m.y = pyo.Var(bounds=(0, 10), initialize=5)
    m.z = pyo.Var(bounds=(0, 10), initialize=10)
    m.cons = pyo.ConstraintList()
    m.obj = pyo.Objective(expr=m.x + m.y + m.z, sense=pyo.minimize)

    at_bounds = collect_variables_at_bounds(m, tol=1e-4)
    names = {v["name"] for v in at_bounds}
    assert "x" in names  # at lower bound
    assert "z" in names  # at upper bound
    assert "y" not in names  # not at any bound


def test_diagnose_infeasibility_report():
    m = _infeasible_model()
    solver = pyo.SolverFactory("scip")
    solver.solve(m)

    report = diagnose_infeasibility(m, solver_name="scip", compute_mis_flag=False)
    assert isinstance(report, InfeasibilityReport)

    summary = report.summary()
    assert "Violated constraints" in summary or "No violated constraints" in summary
    assert isinstance(repr(report), str)


def test_diagnose_infeasibility_with_mis():
    m = _infeasible_model()
    solver = pyo.SolverFactory("scip")
    solver.solve(m)

    report = diagnose_infeasibility(m, solver_name="scip", compute_mis_flag=True)
    # MIS should identify the two conflicting constraints
    assert len(report.mis_constraints) >= 2
    summary = report.summary()
    assert "Minimal Infeasible Subsystem" in summary


def test_infeasible_monee_pyomo_solve():
    """An infeasible monee problem should return success=False and attach a report."""
    import monee.model as mm
    from monee.model import Network
    from monee.model.child import ExtPowerGrid, PowerLoad
    from monee.model.node import Bus
    from monee.problem.core import Constraints, OptimizationProblem
    from monee.solver.pyo import PyomoSolver

    _LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=2.0, q_mvar=0.0))
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)

    # Create an infeasible problem: ext-grid must supply exactly 2 MW
    # but also be bounded to [0.1, 0.5] - impossible.
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).equation(lambda m: m.p_mw >= 0.1).equation(
        lambda m: m.p_mw <= 0.5
    )
    prob.constraints = cons

    # Use ipopt - it detects nonlinear infeasibility much faster than SCIP.
    solver = PyomoSolver(solver_name="ipopt")
    result = solver.solve(net, optimization_problem=prob)
    assert not result.success
    assert hasattr(result, "infeasibility_report")
    assert isinstance(result.infeasibility_report, InfeasibilityReport)


def test_infeasible_multi_period_pyomo():
    """Multi-period Pyomo solve with infeasible constraints raises with diagnostics."""
    import monee.model as mm
    from monee.model import Network
    from monee.model.child import ExtPowerGrid, PowerLoad
    from monee.model.node import Bus
    from monee.problem.core import Constraints, OptimizationProblem
    from monee.simulation.multi_period import run_multi_period
    from monee.simulation.timeseries import TimeseriesData

    _LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=1.0, q_mvar=0.0))
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 3.0])

    # Impossible ext-grid bounds
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).equation(lambda m: m.p_mw >= 0.1).equation(
        lambda m: m.p_mw <= 0.5
    )
    prob.constraints = cons

    # Use ipopt - it detects nonlinear infeasibility much faster than SCIP.
    with pytest.raises(RuntimeError, match="Infeasibility diagnostics"):
        run_multi_period(
            net,
            td,
            steps=2,
            optimization_problem=prob,
            solver="ipopt",
            backend="pyomo",
            dt_h=1.0,
        )
