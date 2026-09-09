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


def test_summary_puts_causes_before_iterate_residuals():
    from monee.solver.infeasibility.pyo import BoundViolation, ConstraintResidual

    report = InfeasibilityReport(
        constraint_residuals=[
            ConstraintResidual(
                index="gas_pipe_0_weymouth",
                body_value=6.26,
                lower=0.0,
                upper=0.0,
                residual=6.2624,
                display_name="GasPipe[0].weymouth",
                carrier="gas",
            )
        ],
        bound_violations=[
            BoundViolation(
                name="branch_7_pressure_pu",
                value=0.85,
                lower=0.86,
                upper=1.1,
                violation=0.01,
                display_name="PressureRegulator[7].pressure_pu",
                carrier="gas",
            )
        ],
        relaxation_explanation=["ub of var branch_7_pressure_pu"],
        mis_constraints=["constraint: PressureRegulator[7].set_pressure"],
    )

    summary = report.summary()
    headers = [ln for ln in summary.splitlines() if ln.startswith("===")]
    assert headers == [
        "=== Minimal Infeasible Subsystem (1 elements) ===",
        "=== Infeasibility explanation: feasible after relaxing (1 elements) ===",
        "=== Variable bound violations (1 total) ===",
        "=== Violated constraints (1 total), residuals at the last iterate "
        "(not a solution) ===",
    ]
    assert "GasPipe[0].weymouth" in summary
    assert "PressureRegulator[7].pressure_pu" in summary


def test_summary_without_a_mis_still_leads_with_the_bound_violations():
    from monee.solver.infeasibility.pyo import BoundViolation, ConstraintResidual

    report = InfeasibilityReport(
        constraint_residuals=[
            ConstraintResidual(
                index="c1", body_value=1.0, lower=0.0, upper=0.0, residual=1.0
            )
        ],
        bound_violations=[
            BoundViolation(
                name="child_1_p_mw",
                value=3.0,
                lower=0.0,
                upper=2.0,
                violation=1.0,
                display_name="ExtPowerGrid[1].p_mw",
            )
        ],
        mis_message="scip not installed",
    )

    headers = [ln for ln in report.summary().splitlines() if ln.startswith("===")]
    assert headers[0].startswith("=== Minimal Infeasible Subsystem: not available")
    assert headers[1] == "=== Variable bound violations (1 total) ==="
    assert headers[2].startswith("=== Violated constraints (1 total)")


def test_summary_of_an_empty_report_keeps_the_no_violation_headers():
    summary = InfeasibilityReport().summary()
    assert "=== No variable bound violations ===" in summary
    assert "=== No violated constraints ===" in summary


def _residual(index, component, residual, carrier="water"):
    from monee.solver.infeasibility.pyo import ConstraintResidual

    return ConstraintResidual(
        index=index,
        body_value=residual,
        lower=0.0,
        upper=0.0,
        residual=residual,
        display_name=f"eq of {component}",
        carrier=carrier,
        component=component,
    )


def _many_violation_report():
    return InfeasibilityReport(
        constraint_residuals=[
            _residual("node_44_eq_0", "Junction (node 44, water)", 312.1),
            _residual("node_44_eq_1", "Junction (node 44, water)", 285.7),
            _residual("node_49_eq_0", "Junction (node 49, water)", 159.4),
            _residual("branch_1_2_0_eq_4", "GasPipe (branch 1->2, gas)", 6.3, "gas"),
            _residual("branch_1_2_0_eq_5", "GasPipe (branch 1->2, gas)", 6.2, "gas"),
            _residual("branch_1_2_0_eq_6", "GasPipe (branch 1->2, gas)", 6.1, "gas"),
        ],
        mis_message="no MIS",
    )


def test_violation_hotspots_rank_components_by_violated_equation_count():
    # GIVEN / WHEN
    spots = _many_violation_report().violation_hotspots()

    # THEN  (three violated equations beat two, despite the smaller residuals)
    assert [s["component"] for s in spots] == [
        "GasPipe (branch 1->2, gas)",
        "Junction (node 44, water)",
        "Junction (node 49, water)",
    ]
    assert spots[0]["count"] == 3
    assert spots[1]["max_residual"] == pytest.approx(312.1)


def test_summary_puts_the_hotspot_ranking_before_the_raw_residuals():
    # GIVEN / WHEN
    summary = _many_violation_report().summary()
    headers = [ln for ln in summary.splitlines() if ln.startswith("===")]

    # THEN
    assert headers[0].startswith("=== Minimal Infeasible Subsystem: not available")
    assert headers[1] == (
        "=== Violation hotspots (6 violated equations across 3 components) ==="
    )
    assert headers.index("=== No variable bound violations ===") > 1
    assert any(h.startswith("=== Violated constraints (6 total)") for h in headers)
    assert "GasPipe (branch 1->2, gas): 3 violated eq(s)" in summary


def test_summary_hotspot_list_is_capped():
    from monee.solver.infeasibility.pyo import ConstraintResidual

    # GIVEN  (20 distinct components, one violated equation each)
    report = InfeasibilityReport(
        constraint_residuals=[
            ConstraintResidual(
                index=f"node_{i}_eq_0",
                body_value=1.0,
                lower=0.0,
                upper=0.0,
                residual=1.0,
                component=f"Bus (node {i}, power)",
            )
            for i in range(20)
        ]
    )

    # WHEN
    lines = report.summary(max_items=5).splitlines()

    # THEN
    assert "  ... and 15 more components" in lines


def test_summary_skips_the_hotspot_section_for_a_single_component():
    # GIVEN / WHEN
    summary = InfeasibilityReport(
        constraint_residuals=[_residual("node_1_eq_0", "Bus (node 1, power)", 1.0)]
    ).summary()

    # THEN
    assert "Violation hotspots" not in summary


def test_no_mis_reason_names_the_per_solve_cap_when_the_solves_time_out():
    from monee.solver.infeasibility.pyo import _no_mis_reason

    # GIVEN / WHEN
    reason = _no_mis_reason(
        "... This model is likely unstable\n",
        "scip",
        False,
        ["maxTimeLimit", "maxTimeLimit", "optimal"],
        10.0,
    )

    # THEN
    assert "2 of the 3 internal MIS solves did not reach optimality" in reason
    assert "2x maxTimeLimit" in reason
    assert "per-solve cap is 10 s" in reason
    assert "mis_time_limit" in reason
    assert "numerically unstable" not in reason


def test_no_mis_reason_keeps_the_instability_diagnosis_when_solves_were_optimal():
    from monee.solver.infeasibility.pyo import _no_mis_reason

    # GIVEN / WHEN
    reason = _no_mis_reason(
        "... This model is likely unstable\n", "scip", False, ["optimal"], 10.0
    )

    # THEN
    assert "numerically unstable" in reason
    assert "mis_time_limit" not in reason


def test_recording_solver_records_terminations_and_proxies_attributes():
    from monee.solver.infeasibility.pyo import _RecordingSolver

    class _Results:
        class solver:
            termination_condition = "maxTimeLimit"

    class _Stub:
        options = {"limits/time": 10.0}

        def solve(self, *_a, **_kw):
            return _Results()

        def available(self, exception_flag=False):
            return True

    # GIVEN
    rec = _RecordingSolver(_Stub())

    # WHEN
    rec.solve("model")
    rec.solve("model")

    # THEN
    assert rec.terminations == ["maxTimeLimit", "maxTimeLimit"]
    assert rec.available(exception_flag=False) is True
    assert rec.options == {"limits/time": 10.0}


def test_mis_time_limit_reaches_the_solver_options(monkeypatch):
    from monee.solver.infeasibility import pyo as pyo_diag

    captured = {}

    class _Stub:
        options: dict = {}

        def available(self, exception_flag=False):
            captured.update(self.options)
            return False

    monkeypatch.setattr(pyo_diag, "_make_mis_solver", lambda _name: _Stub())

    # WHEN
    report = pyo_diag.diagnose_infeasibility(
        pyo.ConcreteModel(), solver_name="scip", mis_time_limit=42.0
    )

    # THEN
    assert captured == {"limits/time": 42.0}
    assert "not available" in report.mis_message
