"""Tests for the in-process CasADi/IPOPT backend (:mod:`monee.solver.casadi`).

CasADi is an optional backend; every test skips cleanly when it is not
installed. Correctness is pinned against the GEKKO backend (the reference NLP
solver) rather than hard-coded numbers, so these stay valid as formulations
evolve.
"""

import contextlib
import logging
import math

import numpy as np
import pytest

import monee
import monee.express as mx
import monee.model as mm
from monee import TimeseriesData
from monee.model import Network, Var
from monee.model.branch import PowerLine
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.grid import PowerGrid
from monee.model.node import Bus
from monee.solver import GEKKOSolver
from monee.solver.dispatch import resolve_multi_period_solver, resolve_solver

ca = pytest.importorskip("casadi")

from monee.solver import CasADiSolver, CasADiTimeseries  # noqa: E402
from monee.solver.casadi import CasADiSolveError  # noqa: E402


def _two_line_net(vm=1.0, controllable_gen=False):
    pn = Network(PowerGrid(name="power", sn_mva=1))
    node_0 = pn.node(
        Bus(base_kv=1),
        child_ids=[
            pn.child(PowerGenerator(p_mw=Var(1) if controllable_gen else 1, q_mvar=0))
        ],
        grid=mm.EL,
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=vm, va_degree=0))],
        grid=mm.EL,
    )
    node_2 = pn.node(
        Bus(base_kv=1), child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))], grid=mm.EL
    )
    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_1,
    )
    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_2,
    )
    return pn


def _vm(result):
    return result.dataframes["Bus"]["vm_pu"].to_numpy()


# --------------------------------------------------------------------------- #
# Dispatch routing
# --------------------------------------------------------------------------- #
def test_resolve_casadi_backend():
    s = resolve_solver(backend="casadi")
    assert isinstance(s, CasADiSolver)


def test_resolve_casadi_backend_with_ipopt_name():
    s = resolve_solver("ipopt", backend="casadi")
    assert isinstance(s, CasADiSolver)


def test_resolve_casadi_backend_rejects_other_solver_name():
    with pytest.raises(ValueError, match="only provides IPOPT"):
        resolve_solver("gurobi", backend="casadi")


def test_casadi_does_not_inherit_gekko():
    # The backends share equation-assembly via the core mixin, not inheritance.
    assert not issubclass(CasADiSolver, GEKKOSolver)


# --------------------------------------------------------------------------- #
# Single-shot correctness vs GEKKO
# --------------------------------------------------------------------------- #
def test_power_flow_matches_gekko():
    pn = _two_line_net()
    rg = GEKKOSolver().solve(pn, simulation=True)
    rc = CasADiSolver().solve(pn, simulation=True)

    assert rc.success
    assert np.nanmax(np.abs(_vm(rg) - _vm(rc))) < 1e-5


def test_power_flow_via_monee_solve_backend():
    pn = _two_line_net()
    result = monee.solve(pn, backend="casadi", simulation=True)
    assert result.success
    assert len(result.dataframes) == 5


def test_opf_objective_matches_gekko():
    # Minimise the slack injection on a controllable generator network.
    def build():
        pn = _two_line_net(controllable_gen=True)
        pn.objective(lambda net: net.childs[1].model.vars["p_mw"])
        return pn

    rg = GEKKOSolver().solve(build())
    rc = CasADiSolver().solve(build())

    assert rc.success
    assert math.isclose(
        rc.dataframes["ExtPowerGrid"]["p_mw"][0],
        rg.dataframes["ExtPowerGrid"]["p_mw"][0],
        rel_tol=1e-4,
    )


def test_empty_step_state_solves():
    # A plain (non-temporal) network with a step_state still solves: the temporal
    # passes simply contribute no equations.
    pn = _two_line_net()
    from monee.simulation.step_state import StepState

    result = CasADiSolver().solve(pn, step_state=StepState(), simulation=True)
    assert result.success


# --------------------------------------------------------------------------- #
# Timeseries graph reuse
# --------------------------------------------------------------------------- #
def _load_profile_td(net, n_steps):
    profile = 0.6 + 0.4 * np.sin(np.linspace(0, 2 * np.pi, n_steps))
    td = TimeseriesData()
    for child in net.childs:
        if isinstance(child.model, mm.PowerLoad):
            td.add_child_series(
                child.id, "p_mw", (float(child.model.p_mw) * profile).tolist()
            )
    return td


def test_casadi_timeseries_matches_per_step_gekko():
    n = 5
    net = _two_line_net()
    td = _load_profile_td(net, n)

    ts = CasADiTimeseries(net, td, simulation=True)
    reuse = np.array([d["Bus"]["vm_pu"].to_numpy() for d in ts.run()])

    res_g = monee.run_timeseries(net, _load_profile_td(net, n), solver=GEKKOSolver())
    gekko = res_g.get_result_for(mm.Bus, "vm_pu").to_numpy()

    assert reuse.shape == gekko.shape
    assert np.nanmax(np.abs(reuse - gekko)) < 1e-4


def test_run_timeseries_casadi_fast_path_equivalent():
    n = 5
    net = _two_line_net()

    res_c = monee.run_timeseries(
        net, _load_profile_td(net, n), backend="casadi", simulation=True
    )
    res_g = monee.run_timeseries(net, _load_profile_td(net, n), solver=GEKKOSolver())

    assert res_c.failed_steps == []
    assert len(res_c.step_results) == n
    vc = res_c.get_result_for(mm.Bus, "vm_pu").to_numpy()
    vg = res_g.get_result_for(mm.Bus, "vm_pu").to_numpy()
    assert np.nanmax(np.abs(vc - vg)) < 1e-4


def test_run_timeseries_falls_back_when_hooks_present(monkeypatch):
    # Step hooks observe temporal state the reuse path doesn't maintain, so the
    # standard per-step loop must be used instead of the CasADi fast path.
    import monee.simulation.timeseries as tsmod

    calls = {"reuse": 0}
    orig = tsmod._run_casadi_reuse

    def _spy(*a, **k):
        calls["reuse"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(tsmod, "_run_casadi_reuse", _spy)

    n = 3
    net = _two_line_net()
    seen = []
    monee.run_timeseries(
        net,
        _load_profile_td(net, n),
        backend="casadi",
        simulation=True,
        step_hooks=[lambda *a: seen.append(a[1])],
    )
    assert calls["reuse"] == 0  # fast path skipped because hooks are present
    assert seen == [0, 1, 2]


def test_run_timeseries_casadi_fast_path_forwards_solver_options(monkeypatch):
    # The reuse driver must honour the resolved solver's per-instance IPOPT
    # options: a 1-iteration cap cannot converge, so the first step raises.
    import monee.simulation.timeseries as tsmod
    from monee.solver.casadi import CasADiSolveError

    calls = {"reuse": 0}
    orig = tsmod._run_casadi_reuse

    def _spy(*a, **k):
        calls["reuse"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(tsmod, "_run_casadi_reuse", _spy)

    net = _two_line_net()
    with pytest.raises(CasADiSolveError):
        monee.run_timeseries(
            net,
            _load_profile_td(net, 3),
            solver=CasADiSolver(solver_options={"ipopt.max_iter": 1}),
            simulation=True,
        )
    assert calls["reuse"] == 1


# --------------------------------------------------------------------------- #
# Temporal coupling (storage) and multi-period
# --------------------------------------------------------------------------- #
def _el_net_with_storage(e_initial=5.0, e_max=10.0, p_max_mw=2.0):
    # Two-bus net (ext-grid at b0, load + battery at b1). Keeping the component
    # count low ensures the battery's child id exceeds every node id, so the
    # multi-period (comp_id, attr) state lookups resolve to the battery and not a
    # same-id bus (node/child ids share a 0-based namespace).
    from monee.model.child import PowerLoad
    from monee.model.storage import ElectricStorage

    net = Network(PowerGrid(name="power", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=2.0, q_mvar=0.0))
    storage_id = net.child(
        ElectricStorage(e_mwh_initial=e_initial, e_mwh_max=e_max, p_max_mw=p_max_mw),
        name="storage",
    )
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id, storage_id])
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
        b0,
        b1,
    )
    return net, storage_id


def test_temporal_timeseries_storage_matches_gekko():
    from monee.model.storage import ElectricStorage  # noqa: F401

    dispatch = [1.0, -0.5, 0.8]

    net_c, sid_c = _el_net_with_storage()
    td_c = TimeseriesData()
    td_c.add_child_series(sid_c, "p_mw", dispatch)
    res_c = monee.run_timeseries(
        net_c, td_c, steps=3, backend="casadi", simulation=True
    )

    net_g, sid_g = _el_net_with_storage()
    td_g = TimeseriesData()
    td_g.add_child_series(sid_g, "p_mw", dispatch)
    res_g = monee.run_timeseries(net_g, td_g, steps=3)

    assert res_c.failed_steps == []
    e_c = res_c.get_result_for_id(sid_c, "e_mwh").to_numpy()
    e_g = res_g.get_result_for_id(sid_g, "e_mwh").to_numpy()
    assert np.nanmax(np.abs(e_c - e_g)) < 1e-3

    # Inter-step SoC invariant e[t] == e[t-1] + dt_h * p[t] must hold (proves the
    # temporal per-step path ran, not the memory-less fast path).
    p_c = res_c.get_result_for_id(sid_c, "p_mw").to_numpy()
    for t in range(1, 3):
        assert abs(e_c[t] - (e_c[t - 1] + p_c[t])) < 1e-3


def test_memoryless_network_uses_reuse_fast_path(monkeypatch):
    # A plain power flow with id-addressed load series must take the build-once
    # graph-reuse fast path (the headline speedup), not the per-step loop.
    import monee.simulation.timeseries as tsmod

    calls = {"reuse": 0}
    orig = tsmod._run_casadi_reuse

    def _spy(*a, **k):
        calls["reuse"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(tsmod, "_run_casadi_reuse", _spy)

    net = _two_line_net()
    monee.run_timeseries(
        net, _load_profile_td(net, 4), backend="casadi", simulation=True
    )
    assert calls["reuse"] == 1


def test_temporal_network_skips_reuse_fast_path(monkeypatch):
    import monee.simulation.timeseries as tsmod

    calls = {"reuse": 0}
    orig = tsmod._run_casadi_reuse

    def _spy(*a, **k):
        calls["reuse"] += 1
        return orig(*a, **k)

    monkeypatch.setattr(tsmod, "_run_casadi_reuse", _spy)

    net, sid = _el_net_with_storage()
    td = TimeseriesData()
    td.add_child_series(sid, "p_mw", [1.0, -0.5, 0.8])
    monee.run_timeseries(net, td, steps=3, backend="casadi", simulation=True)
    assert calls["reuse"] == 0  # storage = temporal coupling -> per-step loop


def test_resolve_multi_period_casadi_backend():
    from monee.solver import CasADiMultiPeriodSolver

    s = resolve_multi_period_solver(backend="casadi")
    assert isinstance(s, CasADiMultiPeriodSolver)


def test_resolve_multi_period_casadi_rejects_other_name():
    with pytest.raises(ValueError, match="only provides IPOPT"):
        resolve_multi_period_solver("gurobi", backend="casadi")


def test_multi_period_controllable_storage_soc_recursion():
    # The plain controllable-storage problem has no cost (objective 0), so the
    # dispatch is non-unique - assert the physical invariants the solution must
    # satisfy rather than a specific (solver-dependent) trajectory.
    from monee.model.storage import ElectricStorage
    from monee.problem import OptimizationProblem
    from monee.simulation.multi_period import run_multi_period

    e_initial, e_max = 4.0, 8.0
    net, sid = _el_net_with_storage(e_initial=e_initial, e_max=e_max)
    prob = OptimizationProblem()
    prob.controllable_storages()
    res = run_multi_period(
        net, TimeseriesData(), steps=3, optimization_problem=prob, backend="casadi"
    )

    assert res.success
    e = res.get_result_for(ElectricStorage, "e_mwh")[sid].to_numpy()
    p = res.get_result_for(ElectricStorage, "p_mw")[sid].to_numpy()
    assert sid in res.get_result_for(ElectricStorage, "p_mw").columns
    # SoC within bounds and obeying the inter-period recursion e[t]=e[t-1]+dt*p[t]
    # (dt_h = 1.0); e[0] = e_initial + dt*p[0].
    assert (e >= -1e-6).all() and (e <= e_max + 1e-6).all()
    assert abs(e[0] - (e_initial + p[0])) < 1e-3
    for t in range(1, 3):
        assert abs(e[t] - (e[t - 1] + p[t])) < 1e-3


def test_multi_period_initial_and_terminal_state():
    from monee.model.storage import ElectricStorage
    from monee.problem import OptimizationProblem
    from monee.simulation.multi_period import run_multi_period

    net, sid = _el_net_with_storage(e_initial=5.0, e_max=10.0)
    prob = OptimizationProblem()
    prob.controllable_storages()
    res = run_multi_period(
        net,
        TimeseriesData(),
        steps=3,
        optimization_problem=prob,
        backend="casadi",
        initial_state={(sid, "e_mwh"): 5.0},
        terminal_state={(sid, "e_mwh"): 5.0},
    )
    assert res.success
    e = res.get_result_for(ElectricStorage, "e_mwh")[sid].to_numpy()
    # Terminal SoC pinned to 5.0.
    assert abs(e[-1] - 5.0) < 1e-2


# --------------------------------------------------------------------------- #
# Warm-start hint
# --------------------------------------------------------------------------- #
def test_merged_ipopt_opts_warm_overlay_and_user_override():
    from monee.solver.casadi import _IPOPT_WARM_OPTS, _merged_ipopt_opts

    warm = _merged_ipopt_opts(None, warm=True)
    for key, val in _IPOPT_WARM_OPTS.items():
        assert warm[key] == val
    assert warm["ipopt.constr_viol_tol"] == 1e-6
    cold = _merged_ipopt_opts(None, warm=False)
    assert "ipopt.mu_init" not in cold
    assert "ipopt.constr_viol_tol" not in cold
    # explicit user options win over the warm overlay
    user = _merged_ipopt_opts({"ipopt.mu_init": 0.05}, warm=True)
    assert user["ipopt.mu_init"] == 0.05
    # the warm feasibility cap follows a tighter user tol; explicit wins
    tight = _merged_ipopt_opts({"ipopt.tol": 1e-8}, warm=True)
    assert tight["ipopt.constr_viol_tol"] == 1e-8
    explicit = _merged_ipopt_opts({"ipopt.constr_viol_tol": 1e-4}, warm=True)
    assert explicit["ipopt.constr_viol_tol"] == 1e-4


def test_casadi_warm_start_hint_is_one_shot_and_matches_cold_solve():
    solver = CasADiSolver()
    net = _two_line_net()

    cold = solver.solve(net)  # persists the solution into net (warm values)
    solver.set_warm_start_hint(True)
    warm = solver.solve(net)
    assert solver._warm_start_hint is False  # consumed by the solve

    assert cold.success and warm.success
    np.testing.assert_allclose(_vm(warm), _vm(cold), rtol=1e-6)


def _single_gas_pipe_net(mass_flow_kgs=0.12):
    gas = mm.create_gas_grid("gas", type="lgas")
    pn = Network()
    g0 = pn.node(mm.Junction(), grid=gas, child_ids=[pn.child(mm.ExtHydrGrid())])
    g1 = pn.node(
        mm.Junction(),
        grid=gas,
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=mass_flow_kgs))],
    )
    pn.branch(mm.GasPipe(diameter_m=0.15, length_m=400, temperature_ext_k=300), g0, g1)
    return pn


def test_relaxed_gas_formulation_warns_on_casadi():
    # GIVEN the epigraph-relaxed gas formulation solved on IPOPT, which relaxes
    # its direction binary.
    from monee.solver.core import reset_validation_summary_counts

    reset_validation_summary_counts()
    with pytest.warns(UserWarning, match="Result validation"):
        result = monee.run_energy_flow(
            _single_gas_pipe_net(), formulation="convex_miqcqp"
        )

    # THEN the untightened relaxation is reported instead of silently returning
    # a pressure drop that is several times the Weymouth value.
    assert result.success
    assert any(w.category == "relaxation" for w in result.warnings)


def test_default_gas_formulation_does_not_warn(caplog):
    with caplog.at_level(logging.WARNING, logger="monee.solver.casadi"):
        result = monee.run_energy_flow(_single_gas_pipe_net())

    assert not [w for w in result.warnings if w.category == "relaxation"]


def _unidirectional_water_ring(n=6):
    net = mm.Network()
    pipe_kwargs = dict(
        diameter_m=0.2,
        length_m=250,
        insulation_thickness_m=0.05,
        lambda_insulation_w_per_m_k=0.03,
        temperature_ext_k=278.15,
        unidirectional=True,
    )
    supply = [mx.create_water_junction(net) for _ in range(n)]
    return_ = [mx.create_water_junction(net) for _ in range(n)]
    for i in range(n):
        mx.create_water_pipe(net, supply[i], supply[(i + 1) % n], **pipe_kwargs)
        mx.create_water_pipe(net, return_[i], return_[(i + 1) % n], **pipe_kwargs)
    for i in range(1, n):
        mx.create_heat_exchanger(net, supply[i], return_[i], q_mw=0.3)
    mx.create_water_ext_grid(net, supply[0], t_k=363.15)
    mx.create_consume_hydr_grid(net, return_[0])
    return net


def test_squared_mass_flow_stays_tied_to_the_flow():
    # GIVEN a ring of unidirectional pipes, which pins every positive flow to 0.
    net = _unidirectional_water_ring()

    # WHEN
    try:
        result = monee.run_energy_flow(net)
    except CasADiSolveError:
        # The pinned ring has no physical solution; reporting that is correct.
        return

    # THEN the Darcy epigraph variable is never larger than the flow allows, so
    # no pressure drop can be fabricated on a pipe that carries no flow.
    pipes = result.get(mm.WaterPipe)
    squared = pipes["mass_flow_pos_kgs_squared"]
    flow = pipes["mass_flow_pos_kgs"].clip(lower=0)
    assert (squared <= (flow + 1e-3) ** 2 + 1e-2).all()


def _two_bus_net(load_mw=2.0):
    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net, base_kv=20)
    bus_1 = mx.create_bus(net, base_kv=20)
    mx.create_line(net, bus_0, bus_1, 800, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=load_mw, q_mvar=0.0)
    return net, bus_1


def test_solve_reports_termination_metadata_and_residuals():
    net, _ = _two_bus_net()

    result = monee.run_energy_flow(net)

    assert result.solver_status == "ok"
    assert result.termination_condition == "Solve_Succeeded"
    assert result.residuals is not None
    assert result.residuals < 1e-6


def test_acceptable_level_stop_is_reported_as_a_warning(caplog):
    net, _ = _two_bus_net()
    # Ask for a tolerance IPOPT cannot reach, so it stops at the acceptable
    # level instead - which stats["success"] alone cannot distinguish.
    solver = CasADiSolver(
        solver_options={
            "ipopt.tol": 1e-16,
            "ipopt.acceptable_tol": 1e-6,
            "ipopt.acceptable_iter": 2,
        }
    )

    with caplog.at_level(logging.WARNING, logger="monee.solver.casadi"):
        result = solver.solve(net)

    assert result.termination_condition == "Solved_To_Acceptable_Level"
    assert result.solver_status == "warning"
    assert any("Solved_To_Acceptable_Level" in r.message for r in caplog.records)


def test_non_converged_solve_is_retried_with_an_adaptive_barrier(monkeypatch, caplog):
    calls = []

    class _FakeSolver:
        def __init__(self, status):
            self._status = status

        def __call__(self, **_kwargs):
            return {"x": 0.0, "f": 0.0}

        def stats(self):
            return {
                "success": self._status == "Solve_Succeeded",
                "return_status": self._status,
            }

    def fake_nlpsol(name, _plugin, _nlp, opts):
        calls.append((name, opts))
        return _FakeSolver("Solve_Succeeded")

    from monee.solver import casadi as casadi_backend

    monkeypatch.setattr(casadi_backend.ca, "nlpsol", fake_nlpsol)

    with caplog.at_level(logging.WARNING, logger="monee.solver.casadi"):
        _sol, stats = casadi_backend._solve_with_retry(
            _FakeSolver("Restoration_Failed"), {}, {}, {}, "monee_test", "solve"
        )

    assert stats["return_status"] == "Solve_Succeeded"
    assert calls[0][1]["ipopt.mu_strategy"] == "adaptive"
    messages = [r.getMessage() for r in caplog.records]
    assert any("retrying automatically" in m for m in messages)
    # The notice must not read like a failed run: it says a recovery step is
    # under way and a closing line reports that the retry converged.
    assert any("not a failed run" in m for m in messages)
    assert any("recovered at return status 'Solve_Succeeded'" in m for m in messages)


def test_multi_period_retry_notice_carries_the_window_index(monkeypatch, caplog):
    import monee.express as mx_
    from monee.solver import casadi as casadi_backend

    net = mx_.create_multi_energy_network()
    bus_0 = mx_.create_bus(net, base_kv=20)
    bus_1 = mx_.create_bus(net, base_kv=20)
    mx_.create_line(net, bus_0, bus_1, 800, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx_.create_ext_power_grid(net, bus_0)
    mx_.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    labels = []
    original = casadi_backend._solve_with_retry

    def spy(solver, nlp, opts, args, name, what, x0_alt=None):
        labels.append(what)
        return original(solver, nlp, opts, args, name, what, x0_alt=x0_alt)

    monkeypatch.setattr(casadi_backend, "_solve_with_retry", spy)

    solver = casadi_backend.CasADiMultiPeriodSolver()
    solver.solve_multi_period(net, steps=2, dt_h=1.0)
    solver.solve_multi_period(net, steps=2, dt_h=1.0)

    assert labels[0].startswith("multi-period solve 1 (T=2 periods; window 1")
    assert labels[1].startswith("multi-period solve 2 (T=2 periods; window 2")


def test_simulation_warns_about_a_custom_child_var_without_equations(caplog):
    @mm.model
    class _FreeVarChild(mm.ChildModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.p_mw = Var(1.0, min=0.5, max=3.0, name="p_mw")
            self.q_mvar = 0.0

        def equations(self, grid, node, **kwargs):
            return []

    net, bus_1 = _two_bus_net()
    mx.create_el_child(net, _FreeVarChild(), node_id=bus_1)

    with caplog.at_level(logging.WARNING, logger="monee.solver.casadi"):
        result = monee.run_energy_flow(net)

    assert result.mode_used == "simulation"
    assert any(
        w.category == "dof" and "not square" in w.message for w in result.warnings
    )


def _infeasible_electric_net():
    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net, base_kv=20)
    bus_1 = mx.create_bus(net, base_kv=20)
    mx.create_line(net, bus_0, bus_1, 800, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0, max_import_mw=0.5)
    mx.create_power_load(net, bus_1, p_mw=2.0, q_mvar=0.0)
    return net


def test_infeasible_electric_net_message_carries_no_compound_hint():
    net = _infeasible_electric_net()

    # Batch solver-ergonomics item S1: the failure is reported on the result.
    result = monee.run_energy_flow(net)
    assert result.success is False
    report = result.infeasibility_report
    assert report.numerics_failure is False  # a verdict, not a breakdown

    message = report.message
    assert "Infeasible_Problem_Detected" in message
    assert "compound" not in message.lower()
    assert "flat start" not in message.lower()
    # The compound-specific retry hint must stay absent; the generic pointer
    # to the Pyomo structured report is expected on every failed solve.
    assert "result.infeasibility_report" in message
    assert "diagnose_infeasibility" in message


def test_solver_options_kwarg_passes_through_the_functional_api(caplog):
    from monee.problem import OptimizationProblem

    net, _ = _two_bus_net()
    result = monee.run_energy_flow_optimization(
        net,
        OptimizationProblem(),
        backend="casadi",
        solver_options={"ipopt.print_level": 0},
    )
    assert result.success

    net2, _ = _two_bus_net()
    with caplog.at_level(logging.WARNING, logger="monee.solver.casadi"):
        with contextlib.suppress(CasADiSolveError):
            monee.run_energy_flow_optimization(
                net2,
                OptimizationProblem(),
                backend="casadi",
                solver_options={"ipopt.max_iter": 3},
            )
    assert any("Maximum_Iterations_Exceeded" in r.message for r in caplog.records)


def test_solve_solver_options_merge_over_instance_options():
    net, _ = _two_bus_net()
    solver = CasADiSolver(solver_options={"ipopt.max_iter": 3})
    assert solver.solve(net, solver_options={"ipopt.max_iter": 3000}).success
