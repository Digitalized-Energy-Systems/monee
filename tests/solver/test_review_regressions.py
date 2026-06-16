"""Regression tests for three issues surfaced by the orchestrated review.

Each test is written to FAIL on the pre-fix code and PASS after the fix:
  1. CasADiTimeseries silently dropped name-addressed / compound series.
  2. The CasADi default no longer raised on non-convergence (failure-contract
     divergence from GEKKO).
  3. The gurobipy build-once reuse path diverged from rebuild for an LTC
     first_step_steady_state network (step-0 structural switch).
"""

import pytest

import monee.model as mm
from monee import LumpedThermalCapacitance, TimeseriesData, run_energy_flow
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.formulation import EL_NLP_FORMULATION, make_heat_convex_milp_formulation
from monee.model.node import Bus


def _gurobipy_available() -> bool:
    try:
        import gurobipy  # noqa: F401

        from monee.solver.gurobipy import GurobipySolver  # noqa: F401

        return True
    except Exception:
        return False


def _el_net(load_name="ts_load"):
    """ext-grid - line - (named load + gen). Converges as a plain power flow."""
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1), grid=mm.EL,
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    lid = net.child(PowerLoad(p_mw=1.0, q_mvar=0.1), name=load_name)
    b1 = net.node(
        Bus(base_kv=1), grid=mm.EL,
        child_ids=[lid, net.child(PowerGenerator(p_mw=0.5, q_mvar=0))],
    )
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1), b0, b1
    )
    net.apply_formulation(EL_NLP_FORMULATION)
    return net, lid


def _water_ltc_net(first_step_steady_state):
    net = mm.Network()
    n0 = net.node(mm.Junction(), mm.WATER, child_ids=[net.child(mm.ExtHydrGrid(t_k=356))])
    n1 = net.node(
        mm.Junction(), mm.WATER, child_ids=[net.child(mm.Source(mass_flow_kgs=5, t_k=356))]
    )
    sink = net.child(mm.Sink(mass_flow_kgs=10))
    n2 = net.node(mm.Junction(), mm.WATER, child_ids=[sink])
    pipe = dict(diameter_m=0.15, length_m=100)
    net.branch(mm.WaterPipe(**pipe), n0, n1)
    net.branch(mm.WaterPipe(**pipe), n1, n2)
    net.branch(mm.WaterPipe(**pipe), n2, n0)
    net.add_extension(
        LumpedThermalCapacitance(first_step_steady_state=first_step_steady_state)
    )
    net.apply_formulation(make_heat_convex_milp_formulation())
    return net, sink


# --------------------------------------------------------------------------- #
# Issue 1: CasADiTimeseries must declare name-addressed series as parameters.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("casadi") is None, reason="casadi missing"
)
def test_casadi_timeseries_declares_name_addressed_series():
    from monee.solver.casadi import CasADiTimeseries

    net, lid = _el_net()
    td = TimeseriesData()
    td.add_child_series_by_name("ts_load", "p_mw", [0.5, 1.0, 1.5])

    ts = CasADiTimeseries(net, td, formulation=None, simulation=False, steps=3)

    load_model = ts._network.child_by_id(lid).model
    declared = {(id(m), k) for (m, k, _psx, _series) in ts._params}
    # Pre-fix: only id-addressed dicts were scanned, so the by-name series for
    # "ts_load".p_mw was never turned into a CasADi parameter -> dropped.
    assert (id(load_model), "p_mw") in declared

    # Behavioural: the by-name series is actually applied per step (pre-fix the
    # load would stay at its static 1.0 for every step).
    applied = []
    for t in range(3):
        ts.step_result(t)
        applied.append(float(mm.value(load_model.p_mw)))
    assert applied == [0.5, 1.0, 1.5]


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("casadi") is None, reason="casadi missing"
)
def test_casadi_timeseries_name_series_wins_over_id():
    """When the same attr has both an id- and a name-addressed series, the
    name series wins (one parameter, matching TimeseriesData.apply_to_child)."""
    from monee.solver.casadi import CasADiTimeseries

    net, lid = _el_net()
    td = TimeseriesData()
    td.add_child_series(lid, "p_mw", [9.0, 9.0, 9.0])  # id series
    td.add_child_series_by_name("ts_load", "p_mw", [0.5, 1.0, 1.5])  # name series

    ts = CasADiTimeseries(net, td, formulation=None, simulation=False, steps=3)
    load_model = ts._network.child_by_id(lid).model
    # exactly one parameter for (load, p_mw) - no duplicate from id + name
    assert sum(1 for (m, k, _, _) in ts._params if m is load_model and k == "p_mw") == 1
    applied = []
    for t in range(3):
        ts.step_result(t)
        applied.append(float(mm.value(load_model.p_mw)))
    assert applied == [0.5, 1.0, 1.5]  # name series wins


# --------------------------------------------------------------------------- #
# Issue 2: a non-converged CasADi solve must RAISE (uniform with GEKKO), not
# silently return success=False.
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    __import__("importlib").util.find_spec("casadi") is None, reason="casadi missing"
)
def test_casadi_raises_on_nonconvergence(monkeypatch):
    from monee.solver import casadi as cmod
    from monee.solver.casadi import CasADiSolveError, CasADiSolver

    net, _ = _el_net()

    # Force IPOPT's success flag false while still returning a valid iterate, so
    # the only thing under test is the convergence-handling contract.
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

    with pytest.raises(CasADiSolveError):
        run_energy_flow(net, solver=CasADiSolver())


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("casadi") is None, reason="casadi missing"
)
def test_run_timeseries_honours_failure_under_casadi(monkeypatch):
    """The failure contract holds end-to-end via run_timeseries (CasADi reuse
    step_result path): on_step_error='raise' propagates, 'skip' records failures."""
    from monee import run_timeseries
    from monee.solver import casadi as cmod
    from monee.solver.casadi import CasADiSolver

    net, lid = _el_net()
    td = TimeseriesData()
    td.add_child_series(lid, "p_mw", [0.5, 1.0, 1.5])  # id series -> CasADi reuse path

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

    with pytest.raises(Exception):
        run_timeseries(net, td, solver=CasADiSolver(), on_step_error="raise")

    res = run_timeseries(net, td, solver=CasADiSolver(), on_step_error="skip")
    assert len(res.failed_steps) == 3


# --------------------------------------------------------------------------- #
# Issue 3: the gurobipy reuse fast path must NOT engage for an LTC
# first_step_steady_state net (it can't reproduce the step-0 structural switch).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _gurobipy_available(), reason="gurobipy not available")
def test_gurobipy_reuse_gated_out_for_ltc_steady_state():
    from monee.simulation.timeseries import _gurobipy_reuse_eligible
    from monee.solver.gurobipy import GurobipySolver

    td = TimeseriesData()

    net_ss, sink_ss = _water_ltc_net(first_step_steady_state=True)
    td_ss = TimeseriesData()
    td_ss.add_child_series(sink_ss, "mass_flow_kgs", [10.0, 10.0])
    # first_step_steady_state -> rebuild loop only (build-once can't switch step-0).
    assert (
        _gurobipy_reuse_eligible(
            net_ss, GurobipySolver(), None, [], td_ss, {}, None
        )
        is False
    )

    net_ok, sink_ok = _water_ltc_net(first_step_steady_state=False)
    td_ok = TimeseriesData()
    td_ok.add_child_series(sink_ok, "mass_flow_kgs", [10.0, 10.0])
    # Control: plain LTC temporal coupling stays eligible (the gurobipy reuse
    # path supports it) - so the gate is specific, not a blanket LTC exclusion.
    assert (
        _gurobipy_reuse_eligible(
            net_ok, GurobipySolver(), None, [], td_ok, {}, None
        )
        is True
    )
