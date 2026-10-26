"""Physical-language infeasibility reporting: name translation, carrier
grouping, MIS failure reasons and the CasADi failure diagnostics."""

import pyomo.environ as pyo
import pytest

import monee.express as mx
import monee.model as mm
from monee.problem.core import Constraints, Objectives, OptimizationProblem
from monee.simulation import TimeseriesData, run_multi_period
from monee.solver.casadi import CasADiSolveError
from monee.solver.infeasibility.pyo import (
    ComponentIndex,
    _describe_error,
    _parse_con_name,
    _parse_var_name,
    diagnose_infeasibility,
)

LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)


def _two_bus_net():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net, base_kv=1)
    b1 = mx.create_bus(net, base_kv=1)
    mx.create_line(net, b0, b1, **LINE)
    ext_id = mx.create_ext_power_grid(net, b0, name="grid connection")
    load_id = mx.create_power_load(net, b1, p_mw=1.0, q_mvar=0.0)
    return net, ext_id, load_id


def test_parse_con_name():
    assert _parse_con_name("node_6_eq_1") == {
        "cat": "node",
        "id": 6,
        "t": None,
        "i": 1,
    }
    assert _parse_con_name("child_0_t3_eq_2") == {
        "cat": "child",
        "id": 0,
        "t": 3,
        "i": 2,
    }
    assert _parse_con_name("branch_3_4_0_eq_13") == {
        "cat": "branch",
        "id": (3, 4, 0),
        "t": None,
        "i": 13,
    }
    assert _parse_con_name("something_else") is None


def test_parse_var_name_branch_tuple_id():
    assert _parse_var_name("branch_3_4_0__mass_flow_kgs") == {
        "cat": "branch",
        "id": (3, 4, 0),
        "t": None,
        "attr": "mass_flow_kgs",
    }


def test_component_index_translates_names():
    net, _ext_id, _load_id = _two_bus_net()
    index = ComponentIndex(net)

    display, carrier, period = index.constraint_display("node_0_eq_0")
    assert "Bus" in display
    assert "[node_0_eq_0]" in display
    assert carrier is not None
    assert period is None

    display, carrier, _period = index.var_display("child_0__p_mw")
    assert "ExtPowerGrid" in display
    assert "grid connection" in display
    assert display.endswith(".p_mw")


def test_report_groups_by_carrier_with_reading_guide():
    net, _ext_id, _load_id = _two_bus_net()
    j0 = mx.create_water_junction(net)
    j1 = mx.create_water_junction(net)
    mx.create_water_pipe(net, j0, j1, diameter_m=0.15, length_m=50)

    pm = pyo.ConcreteModel()
    pm.x = pyo.Var(initialize=5.0)
    pm.node_0_eq_0 = pyo.Constraint(expr=pm.x == 0.0)
    pm.node_2_eq_0 = pyo.Constraint(expr=pm.x == 4.0)
    pm.obj = pyo.Objective(expr=pm.x)

    report = diagnose_infeasibility(pm, compute_mis_flag=False, network=net)
    summary = report.summary()
    assert "Reading guide:" in summary
    assert "--- carrier:" in summary
    assert "By carrier:" in summary
    carriers = {r.carrier for r in report.constraint_residuals}
    assert len(carriers) == 2
    assert report.mis_message == "not requested (compute_mis_flag=False)"
    assert "Minimal Infeasible Subsystem: not available" in summary


def test_report_first_violated_period():
    net, _ext_id, _load_id = _two_bus_net()
    pm = pyo.ConcreteModel()
    pm.x = pyo.Var(initialize=5.0)
    pm.node_0_t2_eq_0 = pyo.Constraint(expr=pm.x == 0.0)
    pm.node_0_t1_eq_0 = pyo.Constraint(expr=pm.x == 4.0)
    pm.obj = pyo.Objective(expr=pm.x)

    report = diagnose_infeasibility(pm, compute_mis_flag=False, network=net)
    assert report.first_violated_period() == 1
    assert "First violated period: t=1." in report.summary()


def test_mis_failure_reason_never_empty(monkeypatch):
    import pyomo.contrib.iis as iis_mod

    def _boom(*_a, **_k):
        raise AssertionError()

    monkeypatch.setattr(iis_mod, "compute_infeasibility_explanation", _boom)

    pm = pyo.ConcreteModel()
    pm.x = pyo.Var(initialize=5.0, bounds=(0, 1))
    pm.cons = pyo.ConstraintList()
    pm.cons.add(pm.x >= 3)
    pm.obj = pyo.Objective(expr=pm.x)

    report = diagnose_infeasibility(pm, compute_mis_flag=True)
    assert report.mis_constraints == []
    assert report.mis_message is not None
    assert report.mis_message.startswith("AssertionError")


def test_parse_mis_output_relaxation_and_mis_sections():
    from monee.solver.infeasibility.pyo import _parse_mis_output

    output = (
        "Model unknown may be infeasible. A feasible solution was found with "
        "only the following variable bounds relaxed:\n"
        "\tlb of var child_0__p_mw by 0.305\n"
        "Constraints / bounds in MIS:\n"
        "\tconstraint: node_1_eq_0\n"
        "Constraints / bounds in guards for stability:\n"
        "\tconstraint: node_2_eq_0\n"
    )
    mis_lines, relax_lines = _parse_mis_output(output)
    assert mis_lines == ["constraint: node_1_eq_0"]
    assert relax_lines == ["lb of var child_0__p_mw by 0.305"]


def test_relaxation_explanation_rendered_and_translated():
    from monee.solver.infeasibility.pyo import InfeasibilityReport, _annotate_report

    net, _ext_id, _load_id = _two_bus_net()
    report = InfeasibilityReport(
        relaxation_explanation=["lb of var child_0__p_mw by 0.305"],
        mis_message="no MIS",
    )
    _annotate_report(report, ComponentIndex(net))
    summary = report.summary()
    assert "Infeasibility explanation: feasible after relaxing" in summary
    assert "ExtPowerGrid" in summary
    assert "not available (no MIS)" in summary


def test_describe_error_empty_message():
    try:
        raise ValueError()
    except ValueError as e:
        desc = _describe_error(e)
    assert desc.startswith("ValueError")
    assert "raised at" in desc


def test_failure_diagnostics_max_iter_flags_likely_infeasible():
    import casadi as ca
    import numpy as np

    from monee.solver.casadi import _failure_diagnostics

    x = ca.SX.sym("x")
    msg, info = _failure_diagnostics(
        {"return_status": "Maximum_Iterations_Exceeded"},
        x,
        np.array([1.0]),
        ca.vertcat(x),
        ca.DM([0.0]),
        ca.DM([0.0]),
        [],
    )
    assert "likely infeasible" in msg
    assert "iteration limit" in msg
    assert info["max_residual"] == pytest.approx(1.0)


def _infeasible_mp_problem():
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(mm.ExtPowerGrid).equation(lambda m: m.p_mw >= -2.5).equation(
        lambda m: m.p_mw <= 0.5
    )
    prob.constraints = cons
    obj = Objectives()
    obj.select(lambda m: isinstance(m, mm.ExtPowerGrid)).calculate(
        lambda models: sum(m.p_mw**2 for m in models)
    )
    prob.objectives = obj
    return prob


def test_casadi_multi_period_infeasible_names_first_period():
    net, _ext_id, load_id = _two_bus_net()
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 3.5, 1.0, 3.8])

    with pytest.raises(CasADiSolveError) as exc_info:
        run_multi_period(
            net,
            td,
            optimization_problem=_infeasible_mp_problem(),
            solver="ipopt",
            dt_h=1.0,
        )
    err = exc_info.value
    assert err.first_infeasible_period == 1
    assert err.max_residual is not None and err.max_residual > 1e-4
    assert "First infeasible period: t=1" in str(err)
    assert "solver='scip'" in str(err)


def test_casadi_single_period_infeasible_mentions_pyomo_report():
    net, _ext_id, load_id = _two_bus_net()
    for child in net.childs:
        if child.id == load_id:
            child.model.p_mw = 3.5

    import monee

    # Batch solver-ergonomics item S1: the failed single-period solve returns a
    # result carrying the report; strict=True still raises it.
    result = monee.run_energy_flow_optimization(net, _infeasible_mp_problem())
    assert result.success is False
    assert "result.infeasibility_report" in result.infeasibility_report.message

    net2, _ext_id2, load_id2 = _two_bus_net()
    for child in net2.childs:
        if child.id == load_id2:
            child.model.p_mw = 3.5
    with pytest.raises(CasADiSolveError) as exc_info:
        monee.run_energy_flow_optimization(net2, _infeasible_mp_problem(), strict=True)
    assert "result.infeasibility_report" in str(exc_info.value)


def test_multi_period_user_objective_split():
    net, ext_id, load_id = _two_bus_net()
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.5])
    ef = [100.0, 200.0]
    td.add_objective_data(ext_id, "ef", ef)

    prob = OptimizationProblem()
    obj = Objectives()
    obj.select(lambda m: isinstance(m, mm.ExtPowerGrid)).calculate(
        lambda models: sum(-m.ef * m.p_mw for m in models)
    )
    prob.objectives = obj

    result = run_multi_period(net, td, optimization_problem=prob, dt_h=1.0)
    assert result.success
    ext_p = result.get_result_for_id(ext_id, "p_mw", mm.ExtPowerGrid)
    recomputed = sum(-e * p for e, p in zip(ef, ext_p.values))
    assert result.user_objective == pytest.approx(recomputed, rel=1e-6)
    assert result.aux_objective == pytest.approx(
        result.objective - result.user_objective
    )
