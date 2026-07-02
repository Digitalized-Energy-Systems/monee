"""Tests for the in-process CasADi/IPOPT backend (:mod:`monee.solver.casadi`).

CasADi is an optional backend; every test skips cleanly when it is not
installed. Correctness is pinned against the GEKKO backend (the reference NLP
solver) rather than hard-coded numbers, so these stay valid as formulations
evolve.
"""

import math

import numpy as np
import pytest

import monee
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
