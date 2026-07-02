"""Solver-backend regression tests.

Covers, per backend:
  Pyomo    - log10 (not ln) + exp_impl threading for the smooth Swamee-Jain
             friction; failed solves must not clobber the input network;
             per-solver option inheritance for interface-variant names.
  Gurobipy - SolverResult carries solver_status/termination_condition;
             INF_OR_UNBD disambiguation; failed solves must not clobber the
             input network; timeseries driver guards (lookback, name/compound
             series) and per-step failure raising.
  CasADi   - run() honours the per-step failure contract; solver_options
             override; negative-scale bound ordering; simulation mode label.
  GEKKO    - run-directory cleanup; withdrawn Var name/integer preservation.
"""

import importlib.util
import math
import os
import types

import pytest

import monee.model as mm
from monee.model import Var
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.node import Bus
from monee.problem.core import Constraints, OptimizationProblem
from monee.simulation import TimeseriesData

HAVE_CASADI = importlib.util.find_spec("casadi") is not None
requires_casadi = pytest.mark.skipif(not HAVE_CASADI, reason="casadi missing")


def _scip_available() -> bool:
    from monee.solver.pyo import _classic_scip_available, _pyscipopt_available

    return _classic_scip_available() or _pyscipopt_available()


requires_scip = pytest.mark.skipif(
    not _scip_available(), reason="no SCIP (executable or pyscipopt) available"
)


def _gurobipy_available() -> bool:
    try:
        import gurobipy as gp

        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.addVar()
        m.optimize()
        return True
    except Exception:
        return False


requires_gurobi = pytest.mark.skipif(
    not _gurobipy_available(), reason="gurobipy not available"
)


def _gas_net():
    """Three junctions, two pipes: flows are fixed by mass balance, so nodal
    pressures isolate the friction correlation (Swamee-Jain log10 wiring)."""
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=1))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.6))])
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1000, roughness_m=0.01), g0, g1)
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1500, roughness_m=0.01), g0, g2)
    return pn


def _el_net(formulation, load_name=None):
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    lid = net.child(PowerLoad(p_mw=1.0, q_mvar=0.1), name=load_name)
    b1 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[lid, net.child(PowerGenerator(p_mw=0.5, q_mvar=0))],
    )
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
        b0,
        b1,
    )
    net.apply_formulation(formulation)
    return net, lid


def _infeasible_ext_grid_problem():
    prob = OptimizationProblem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).equation(lambda m: m.p_mw >= 0.1).equation(
        lambda m: m.p_mw <= 0.05
    )
    prob.constraints = cons
    return prob


def _var_snapshot(net):
    """{(cat, id): {attr: value}} for every Var on the network's models."""
    snap = {}

    def grab(model):
        return {k: v.value for k, v in model.__dict__.items() if isinstance(v, Var)}

    for node in net.nodes:
        snap[("node", node.id)] = grab(node.model)
        for child in net.childs_by_ids(node.child_ids):
            snap[("child", child.id)] = grab(child.model)
    for branch in net.branches:
        snap[("branch", branch.id)] = grab(branch.model)
    for compound in net.compounds:
        snap[("compound", compound.id)] = grab(compound.model)
    return snap


def test_pyomo_builds_nonlinear_friction_without_keyerror(monkeypatch):
    """friction_model='nonlinear' requires exp_impl to be threaded through the
    Pyomo branch pass; a missing impl surfaces as a KeyError during the build.
    Build-only: the solve is stubbed out."""
    from pyomo.opt import SolverStatus, TerminationCondition

    from monee.model.formulation import make_gas_nlp_formulation
    from monee.solver.pyo import PyomoSolver

    class _DummySolver:
        options = {}

        def warm_start_capable(self):
            return False

        def solve(self, model, **kwargs):
            return types.SimpleNamespace(
                solver=types.SimpleNamespace(
                    status=SolverStatus.ok,
                    termination_condition=TerminationCondition.optimal,
                )
            )

    monkeypatch.setattr(
        PyomoSolver, "_make_solver", staticmethod(lambda name: _DummySolver())
    )

    net = _gas_net()
    net.apply_formulation(make_gas_nlp_formulation(friction_model="nonlinear"))
    result = PyomoSolver("scip").solve(net)
    assert result is not None


@requires_casadi
@requires_scip
def test_pyomo_nonlinear_friction_matches_casadi():
    """Cross-backend parity on the friction-sensitive pressures: ln instead of
    log10 in Swamee-Jain would skew the friction factor by ~2.3x."""
    from monee.model.formulation import make_gas_nlp_formulation
    from monee.solver.casadi import CasADiSolver
    from monee.solver.pyo import PyomoSolver

    nc = _gas_net()
    nc.apply_formulation(make_gas_nlp_formulation(friction_model="nonlinear"))
    r_casadi = CasADiSolver().solve(nc)

    np_ = _gas_net()
    np_.apply_formulation(make_gas_nlp_formulation(friction_model="nonlinear"))
    r_pyomo = PyomoSolver("scip").solve(np_)

    assert r_casadi.success and r_pyomo.success
    p_c = r_casadi.dataframes["Junction"]["pressure_pu"]
    p_p = r_pyomo.dataframes["Junction"]["pressure_pu"]
    for a, b in zip(p_c, p_p):
        assert math.isclose(a, b, abs_tol=1e-4)
    f_c = r_casadi.dataframes["GasPipe"]["friction"]
    f_p = r_pyomo.dataframes["GasPipe"]["friction"]
    for a, b in zip(f_c, f_p):
        assert math.isclose(a, b, abs_tol=1e-4)


@requires_gurobi
def test_gurobipy_time_limit_abort_reports_termination_condition():
    from monee.model.formulation import EL_MISOCP_FORMULATION
    from monee.solver.gurobipy import GurobipySolver

    net, _ = _el_net(EL_MISOCP_FORMULATION)
    result = GurobipySolver(params={"TimeLimit": 0}).solve(net)

    assert result.termination_condition == "maxTimeLimit"
    assert result.solver_status == "aborted"


@requires_gurobi
def test_gurobipy_optimal_reports_ok_optimal():
    from monee.model.formulation import EL_MISOCP_FORMULATION
    from monee.solver.gurobipy import GurobipySolver

    net, _ = _el_net(EL_MISOCP_FORMULATION)
    result = GurobipySolver().solve(net)

    assert result.success
    assert result.solver_status == "ok"
    assert result.termination_condition == "optimal"


@requires_gurobi
def test_gurobipy_failed_solve_leaves_input_network_untouched():
    from monee.model.formulation import EL_MISOCP_FORMULATION
    from monee.solver.gurobipy import GurobipySolver

    net, _ = _el_net(EL_MISOCP_FORMULATION)
    before = _var_snapshot(net)

    result = GurobipySolver().solve(
        net, optimization_problem=_infeasible_ext_grid_problem()
    )

    assert not result.success
    assert result.termination_condition == "infeasible"
    assert _var_snapshot(net) == before


@requires_scip
def test_pyomo_failed_solve_leaves_input_network_untouched():
    from monee.model.formulation import EL_MISOCP_FORMULATION
    from monee.solver.pyo import PyomoSolver

    net, _ = _el_net(EL_MISOCP_FORMULATION)
    before = _var_snapshot(net)

    result = PyomoSolver("scip").solve(
        net, optimization_problem=_infeasible_ext_grid_problem()
    )

    assert not result.success
    assert _var_snapshot(net) == before


@requires_casadi
def test_casadi_timeseries_run_raises_on_step_failure(monkeypatch):
    from monee.model.formulation import EL_NLP_FORMULATION
    from monee.solver import casadi as cmod
    from monee.solver.casadi import CasADiSolveError, CasADiTimeseries

    net, lid = _el_net(EL_NLP_FORMULATION)
    td = TimeseriesData()
    td.add_child_series(lid, "p_mw", [0.5, 1.0])

    real_nlpsol = cmod.ca.nlpsol

    def fake_nlpsol(name, plugin, prob, opts):
        solver = real_nlpsol(name, plugin, prob, opts)

        class _Wrapped:
            def __call__(self, **kw):
                return solver(**kw)

            def stats(self):
                s = dict(solver.stats())
                s["success"] = False
                return s

        return _Wrapped()

    monkeypatch.setattr(cmod.ca, "nlpsol", fake_nlpsol)

    ts = CasADiTimeseries(net, td, simulation=False)
    with pytest.raises(CasADiSolveError, match="step 0"):
        ts.run()
    assert ts.last_step_success is False


@requires_gurobi
def test_gurobipy_timeseries_run_raises_on_step_failure():
    from monee.model.formulation import EL_MISOCP_FORMULATION
    from monee.solver.gurobipy import GurobipySolveError, GurobipyTimeseries

    net, lid = _el_net(EL_MISOCP_FORMULATION)
    td = TimeseriesData()
    td.add_child_series(lid, "p_mw", [1.0, 1.5])

    ts = GurobipyTimeseries(net, td, params={"TimeLimit": 0})
    try:
        with pytest.raises(GurobipySolveError, match="step 0"):
            ts.run()
        assert ts.last_step_success is False
        assert ts.last_status_ok is False
    finally:
        ts.dispose()


@requires_gurobi
def test_param_step_state_lookback_and_none_guard():
    """_ParamStepState.get rejects step arguments other than -1 and seeds
    None/NaN initial values as 0.0 parameters."""
    import gurobipy as gp

    from monee.solver.gurobipy import _ParamStepState

    gm = gp.Model()
    gm.setParam("OutputFlag", 0)
    try:
        state = _ParamStepState(
            gm, {("c1", "e"): None, ("c2", "e"): float("nan"), ("c3", "e"): 2.5}
        )
        v_none = state.get("c1", "e")
        v_nan = state.get("c2", "e", step=-1)
        v_ok = state.get("c3", "e")
        gm.update()
        assert (v_none.LB, v_none.UB) == (0.0, 0.0)
        assert (v_nan.LB, v_nan.UB) == (0.0, 0.0)
        assert (v_ok.LB, v_ok.UB) == (2.5, 2.5)

        with pytest.raises(NotImplementedError):
            state.get("c1", "e", step=-2)
        with pytest.raises(NotImplementedError):
            state.get("c1", "e", step=0)
    finally:
        gm.dispose()


def test_collect_constraint_residuals_includes_named_constraints():
    """The residual scan covers named (attribute) constraints, not just the
    anonymous ``pm.cons`` list."""
    import pyomo.environ as pyo

    from monee.solver.infeasibility.pyo import collect_constraint_residuals

    m = pyo.ConcreteModel()
    m.x = pyo.Var(bounds=(0, 5), initialize=0)
    m.cons = pyo.ConstraintList()
    m.cons.add(m.x >= 3)  # violated at x=0 (anonymous list)
    # named physics-style constraint, as attached by PyomoSolver._add_equation
    m.node_1_eq_0 = pyo.Constraint(expr=m.x >= 4)  # violated at x=0
    m.obj = pyo.Objective(expr=m.x, sense=pyo.minimize)

    residuals = collect_constraint_residuals(m, tol=1e-4)

    names = {str(r.index) for r in residuals}
    assert "node_1_eq_0" in names
    assert any("cons" in n for n in names)


def test_gekko_solve_cleans_run_directory_and_preserves_names():
    """GEKKO removes its APMonitor run directory after the solve, and withdrawn
    Vars keep their original monee names/integer flags via the var_meta
    registry instead of the mangled GEKKO NAME."""
    import monee.solver.gekko as gmod
    from monee.solver.gekko import GEKKOSolver

    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=0.5))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    pn.branch(mm.GasPipe(diameter_m=0.3, length_m=100, roughness_m=0.01), g0, g1)

    paths = []
    real_gekko = gmod.GEKKO

    def spy(*args, **kwargs):
        m = real_gekko(*args, **kwargs)
        paths.append(m._path)
        return m

    gmod.GEKKO = spy
    try:
        result = GEKKOSolver(solver=3).solve(pn)
    finally:
        gmod.GEKKO = real_gekko

    assert result.success
    assert paths and not os.path.exists(paths[0])

    junction = result.network.nodes[0].model
    assert junction.pressure_squared_pu.name == "pressure_sq_pu"


@requires_casadi
def test_casadi_solver_options_are_honoured():
    """Per-instance solver_options are merged over the module defaults: a
    1-iteration cap cannot converge, whereas the defaults (3000 iterations)
    solve the model fine."""
    from monee.model.formulation import EL_NLP_FORMULATION
    from monee.solver.casadi import CasADiSolveError, CasADiSolver

    net, _ = _el_net(EL_NLP_FORMULATION)
    with pytest.raises(CasADiSolveError):
        CasADiSolver(solver_options={"ipopt.max_iter": 1}).solve(net)

    net2, _ = _el_net(EL_NLP_FORMULATION)
    assert CasADiSolver().solve(net2).success


@requires_casadi
def test_casadi_merged_opts_defaults_and_overrides():
    from monee.solver.casadi import _IPOPT_OPTS, _merged_ipopt_opts

    merged = _merged_ipopt_opts({"ipopt.tol": 1e-9})
    assert merged["ipopt.tol"] == 1e-9
    assert merged["ipopt.max_iter"] == _IPOPT_OPTS["ipopt.max_iter"]
    assert _IPOPT_OPTS["ipopt.tol"] == 1e-6  # module defaults not mutated

    debug = _merged_ipopt_opts(None, debug=True)
    assert debug["ipopt.print_level"] > 0


@requires_gurobi
def test_gurobipy_timeseries_rejects_name_addressed_series():
    """GurobipyTimeseries fails loudly on series it cannot honour instead of
    silently freezing them at their static values."""
    from monee.model.formulation import EL_MISOCP_FORMULATION
    from monee.solver.gurobipy import GurobipyTimeseries

    net, _ = _el_net(EL_MISOCP_FORMULATION, load_name="ts_load")
    td = TimeseriesData()
    td.add_child_series_by_name("ts_load", "p_mw", [1.0, 1.5])

    with pytest.raises(ValueError, match="name-addressed"):
        GurobipyTimeseries(net, td)


@requires_gurobi
def test_gurobipy_inf_or_unbd_is_disambiguated_to_infeasible():
    """INF_OR_UNBD is disambiguated via a DualReductions=0 re-solve."""
    from monee.solver.gurobipy import GurobipySolver, _require_gurobipy

    solver = GurobipySolver()
    gp, GRB, nlfunc = _require_gurobipy()
    solver._gp, solver._GRB, solver._nlfunc = gp, GRB, nlfunc

    m = gp.Model()
    m.setParam("OutputFlag", 0)
    try:
        x = m.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="x")
        y = m.addVar(lb=0, ub=1, name="y")
        m.update()
        m.addConstr(y >= 2, name="conflict")
        m.setObjective(x, GRB.MINIMIZE)
        m.optimize()
        assert m.Status == GRB.INF_OR_UNBD  # precondition: status is inconclusive

        success, report, status_str, tc_str = solver._classify(m, phase_label="test")

        assert success is False
        assert tc_str == "infeasible"
        assert report is not None
        assert "conflict" in report.summary() or "y" in report.summary()
    finally:
        m.dispose()


def test_dispatch_unknown_backend_message_lists_casadi():
    from monee.solver.dispatch import resolve_solver

    with pytest.raises(ValueError, match="casadi"):
        resolve_solver("ipopt", backend="nonsense")


def test_pyomo_per_solver_options_inherited_by_interface_variants():
    from monee.solver.pyo import PER_SOLVER_OPTIONS, _effective_solver_options

    base = PER_SOLVER_OPTIONS["gurobi"]
    for variant in ("gurobi", "appsi_gurobi", "gurobi_direct", "gurobi_persistent"):
        assert _effective_solver_options(variant) == base
    assert _effective_solver_options("appsi_highs") == {}
    # exact-name entries take precedence over the normalized base entry
    PER_SOLVER_OPTIONS["gurobi_direct"] = {"MIPGap": 0.5}
    try:
        assert _effective_solver_options("gurobi_direct")["MIPGap"] == 0.5
        assert (
            _effective_solver_options("gurobi_direct")["TimeLimit"] == base["TimeLimit"]
        )
    finally:
        del PER_SOLVER_OPTIONS["gurobi_direct"]


@requires_casadi
def test_casadi_inject_reorders_bounds_for_negative_scale():
    from monee.solver.casadi import CasADiSolver

    class _Model:
        pass

    model = _Model()
    model.v = Var(value=1.0, min=-4.0, max=2.0, scale=-2.0)

    solver = CasADiSolver()
    solver._reg = []
    solver._inject(model, None, None)

    (entry,) = solver._reg
    assert entry["lb"] <= entry["ub"]
    assert entry["lb"] == -1.0 and entry["ub"] == 2.0  # (-4,2)/-2 -> (2,-1) sorted
    assert entry["x0"] == -0.5
    assert entry["integer"] is False


@requires_casadi
def test_casadi_equation_warns_on_false_sentinel(caplog):
    import logging

    from monee.solver.casadi import CasModel

    m = CasModel()
    with caplog.at_level(logging.WARNING, logger="monee.solver.casadi"):
        m.Equation(False)
        m.Equation(True)
    assert len(m.cons) == 0
    assert sum("always-false" in r.message for r in caplog.records) == 1


@requires_casadi
def test_casadi_simulation_mode_reported():
    from monee.model.formulation import EL_NLP_FORMULATION
    from monee.solver.casadi import CasADiSolver

    net, _ = _el_net(EL_NLP_FORMULATION)
    result = CasADiSolver().solve(net, simulation=True)
    assert result.success
    assert result.mode_used == "simulation"

    net2, _ = _el_net(EL_NLP_FORMULATION)
    assert CasADiSolver().solve(net2).mode_used == "optimization"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
