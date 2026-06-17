"""Tests for the native gurobipy solver backend (:mod:`monee.solver.gurobipy`).

Covers the two formulation families the backend supports - smooth NLP
(``*_nlp``, via ``gurobipy.nlfunc`` + general nonlinear constraints) and the
(MI)QCQP / MISOCP / PWL convex path - plus the IIS infeasibility diagnostics
and the lexicographic objective handling.

The solve-based tests need the ``gurobipy`` package, but *not* a Gurobi licence
file: the pip wheel ships a built-in size-limited licence (2000 variables /
2000 constraints) that needs no setup, and every model here is tiny (the largest
is the urban-district MISOCP at ~164 variables / ~128 constraints). So the only
thing gating these tests is whether ``gurobipy`` is installed; they are skipped
wholesale when it is not.
"""

import math

import pytest

import monee.model as mm
import monee.solver as ms
from monee.model.formulation import (
    EL_MISOCP_FORMULATION,
    GAS_NONCONVEX_MIQCQP_FORMULATION,
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
    make_gas_nlp_formulation,
    make_heat_nlp_formulation,
)
from tests.util import create_g2h_net, create_water_loop

GEKKO_IPOPT = 3
FRICTION_MODELS = ["constant", "pwl", "nonlinear", "hybrid"]


def _gurobipy_available() -> bool:
    """True iff gurobipy imports and a (any) licence solves a trivial model.

    The pip wheel's built-in size-limited licence satisfies this probe, so no
    Gurobi licence file is required - the tests run anywhere gurobipy installs.
    """
    try:
        import gurobipy as gp

        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.addVar()
        m.optimize()
        return True
    except Exception:
        return False


HAVE_GUROBI = _gurobipy_available()
requires_gurobi = pytest.mark.skipif(
    not HAVE_GUROBI,
    reason="gurobipy not installed (or its licence rejected a trivial solve)",
)


def _gas_only_net():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=1))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.6))])
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1000, roughness_m=0.01), g0, g1)
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1500, roughness_m=0.01), g0, g2)
    return pn


def _build_misocp_net():
    from monee.network import create_urban_district_net

    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net


def _make_shedding_problem(lex_objectives=False):
    from monee.problem.min_load_shedding import create_min_load_shedding_problem

    return create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
        lex_objectives=lex_objectives,
    )


# --------------------------------------------------------------------------- #
# Pure backend logic (no gurobipy package / licence required)
# --------------------------------------------------------------------------- #
# The backend imports gurobipy lazily (only inside ``_require_gurobipy``), so its
# parameter handling and value-scrubbing helpers can be imported and exercised
# without the package or a licence. These run everywhere, including CI where
# gurobipy is not installed.


def test_params_merge_over_defaults_without_mutating_defaults():
    # GIVEN
    from monee.solver.gurobipy import DEFAULT_GUROBI_PARAMS, GurobipySolver

    # WHEN
    solver = GurobipySolver(params={"TimeLimit": 60, "FeasibilityTol": 1e-7})

    # THEN
    assert solver._params["TimeLimit"] == 60  # caller overrides the default 300
    assert solver._params["MIPGap"] == DEFAULT_GUROBI_PARAMS["MIPGap"]  # default kept
    assert solver._params["FeasibilityTol"] == 1e-7  # extra param merged in
    assert "FeasibilityTol" not in DEFAULT_GUROBI_PARAMS  # module default untouched


def test_var_value_returns_zero_when_no_solution_loaded():
    # GIVEN
    from monee.solver.gurobipy import GurobipySolver

    class _NoSolutionVar:
        @property
        def X(self):
            raise RuntimeError("Unable to retrieve attribute 'X'")

    # WHEN / THEN
    assert GurobipySolver._var_value(_NoSolutionVar()) == 0.0


def test_var_value_scrubs_nan():
    # GIVEN
    from monee.solver.gurobipy import GurobipySolver

    class _NanVar:
        X = float("nan")

    # WHEN / THEN
    assert GurobipySolver._var_value(_NanVar()) == 0.0


def test_var_value_passes_through_finite_solution():
    # GIVEN
    from monee.solver.gurobipy import GurobipySolver

    class _GoodVar:
        X = 3.5

    # WHEN / THEN
    assert GurobipySolver._var_value(_GoodVar()) == 3.5


def test_sanitize_name_strips_gurobi_unsafe_chars():
    # GIVEN
    from monee.solver.gurobipy import GurobipySolver

    # WHEN
    safe = GurobipySolver._sanitize_name("node (3), p_mw [UB]")

    # THEN
    assert safe == "node_3_p_mw_UB"


# --------------------------------------------------------------------------- #
# Report formatting (no solver / licence required)
# --------------------------------------------------------------------------- #


def test_iis_report_summary_lists_constraints_and_bounds():
    # GIVEN
    from monee.solver.gurobipy import GurobiIISReport

    report = GurobiIISReport(
        constraints=["node_3_eq_0", "branch_1_eq_2"], bounds=["child_2__p_mw [UB]"]
    )

    # WHEN
    summary = report.summary(max_items=10)

    # THEN
    assert "node_3_eq_0" in summary
    assert "child_2__p_mw [UB]" in summary
    assert "Constraints in IIS (2)" in summary
    assert "Variable bounds in IIS (1)" in summary


def test_iis_report_summary_handles_empty():
    # GIVEN
    from monee.solver.gurobipy import GurobiIISReport

    # WHEN
    summary = GurobiIISReport(constraints=[], bounds=[]).summary()

    # THEN
    assert "empty IIS" in summary


# --------------------------------------------------------------------------- #
# Minimal solve (smallest model; proves license-free operation)
# --------------------------------------------------------------------------- #


def _minimal_gas_net():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=0.5))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    pn.branch(mm.GasPipe(diameter_m=0.3, length_m=100, roughness_m=0.01), g0, g1)
    return pn


@requires_gurobi
def test_minimal_solve_runs_under_size_limited_license():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    net = _minimal_gas_net()
    net.apply_formulation(make_gas_nlp_formulation())

    # WHEN
    result = GurobipySolver().solve(net)

    # THEN
    assert result.success
    # Mass conservation on a single pipe: the ext grid balances the 0.5 kg/s source.
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], 0.5, abs_tol=1e-3
    )


# --------------------------------------------------------------------------- #
# Smooth NLP family
# --------------------------------------------------------------------------- #


@requires_gurobi
@pytest.mark.parametrize("friction_model", FRICTION_MODELS)
def test_smooth_mes_solves(friction_model):
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    network = create_g2h_net()
    network.apply_formulation(make_gas_nlp_formulation(friction_model=friction_model))
    network.apply_formulation(make_heat_nlp_formulation(friction_model=friction_model))

    # WHEN
    result = GurobipySolver().solve(network)

    # THEN
    assert result.success
    assert math.isfinite(result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0])

    # No spurious bidirectional flow (smooth complementarity holds).
    gas_pipes = result.dataframes["GasPipe"]
    for pos, neg in zip(gas_pipes["mass_flow_pos_kgs"], gas_pipes["mass_flow_neg_kgs"]):
        assert min(abs(pos), abs(neg)) < 1e-2


@requires_gurobi
def test_smooth_gas_pressure_within_bounds():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    network = create_g2h_net()
    network.apply_formulation(make_gas_nlp_formulation(friction_model="constant"))
    network.apply_formulation(make_heat_nlp_formulation(friction_model="constant"))

    # WHEN
    result = GurobipySolver().solve(network)

    # THEN
    assert result.success
    junctions = result.dataframes["Junction"]
    assert (junctions["pressure_squared_pu"] >= -1e-6).all()
    assert (junctions["pressure_squared_pu"] <= 2 + 1e-6).all()
    # The pressure_pu reporting intermediate was materialised and read back.
    assert junctions["pressure_pu"].notna().any()


@requires_gurobi
@pytest.mark.parametrize("friction_model", FRICTION_MODELS)
def test_smooth_gas_matches_gekko_ipopt(friction_model):
    """The general-nonlinear formulation must reproduce the reference IPOPT
    solution (validates nlfunc / log10 friction wiring)."""
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    ng = _gas_only_net()
    ng.apply_formulation(make_gas_nlp_formulation(friction_model=friction_model))
    ne = _gas_only_net()
    ne.apply_formulation(make_gas_nlp_formulation(friction_model=friction_model))

    # WHEN
    r_gurobi = GurobipySolver().solve(ng)
    r_gekko = ms.GEKKOSolver(solver=GEKKO_IPOPT).solve(ne)

    # THEN
    assert r_gurobi.success and r_gekko.success
    assert math.isclose(
        r_gurobi.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0],
        r_gekko.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0],
        abs_tol=1e-3,
    )


@requires_gurobi
def test_smooth_heat_loop_solves_and_conserves_mass():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    net, _, _, _ = create_water_loop(source_t_k=330)
    net.apply_formulation(make_heat_nlp_formulation())

    # WHEN
    result = GurobipySolver().solve(net)

    # THEN
    assert result.success
    # Loop carries the 5 kg/s source through to the 10 kg/s sink.
    flows = list(result.dataframes["WaterPipe"]["mass_flow_kgs"])
    assert any(abs(f) > 1e-3 for f in flows)


# --------------------------------------------------------------------------- #
# (MI)QCQP / MISOCP convex path
# --------------------------------------------------------------------------- #


@requires_gurobi
def test_misocp_optimization_matches_pyomo_gurobi():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    prob = _make_shedding_problem()

    # WHEN
    r_native = GurobipySolver().solve(_build_misocp_net(), optimization_problem=prob)
    r_pyomo = ms.PyomoSolver("gurobi").solve(
        _build_misocp_net(), optimization_problem=_make_shedding_problem()
    )

    # THEN
    assert r_native.success and r_pyomo.success
    # Same model, same solver - objectives must agree.
    assert math.isclose(
        r_native.objective, r_pyomo.objective, rel_tol=1e-4, abs_tol=1e-4
    )


@requires_gurobi
def test_nonconvex_miqcqp_solves():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    net = create_g2h_net()
    net.apply_formulation(GAS_NONCONVEX_MIQCQP_FORMULATION)
    net.apply_formulation(HEAT_NONCONVEX_MIQCQP_FORMULATION)

    # WHEN
    result = GurobipySolver().solve(net)

    # THEN
    assert result.success


@requires_gurobi
def test_lexicographic_matches_single_phase_on_unstressed_net():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN: unstressed net is feasible with ~no shedding in either mode.
    r_plain = GurobipySolver().solve(
        _build_misocp_net(), optimization_problem=_make_shedding_problem(False)
    )
    r_lex = GurobipySolver().solve(
        _build_misocp_net(), optimization_problem=_make_shedding_problem(True)
    )

    # THEN
    assert r_plain.success and r_lex.success
    assert math.isclose(r_plain.objective, r_lex.objective, rel_tol=1e-3, abs_tol=1e-3)


@requires_gurobi
def test_lexicographic_uses_native_priorities_when_linear():
    """The MISOCP shedding objectives are linear, so lex must take the native
    ``setObjectiveN`` path (single solve) rather than the two-phase fallback."""
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    solver = GurobipySolver()
    calls = {"native": 0, "two_phase": 0}
    orig_native = solver._solve_lexicographic_native
    orig_two = solver._solve_lexicographic_two_phase

    def spy_native(*a, **k):
        calls["native"] += 1
        return orig_native(*a, **k)

    def spy_two(*a, **k):
        calls["two_phase"] += 1
        return orig_two(*a, **k)

    solver._solve_lexicographic_native = spy_native
    solver._solve_lexicographic_two_phase = spy_two

    # WHEN
    result = solver.solve(
        _build_misocp_net(), optimization_problem=_make_shedding_problem(True)
    )

    # THEN
    assert result.success
    assert calls["native"] == 1
    assert calls["two_phase"] == 0


@requires_gurobi
def test_lexicographic_native_handles_quadratic_user_objective():
    """With ext grids on, the user objective carries a quadratic slack term.
    Native ``setObjectiveN`` is linear-only, so the backend must hoist the
    quadratic onto an auxiliary and still take the native path (not two-phase)."""
    from monee.problem.min_load_shedding import create_min_load_shedding_problem
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN: include_ext_grids=True => quadratic ext-slack in the user objective.
    prob = create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        include_ext_grids=True,
        lex_objectives=True,
    )
    solver = GurobipySolver()
    calls = {"native": 0, "two_phase": 0}
    on, ot = solver._solve_lexicographic_native, solver._solve_lexicographic_two_phase
    solver._solve_lexicographic_native = lambda *a, **k: (
        calls.__setitem__("native", calls["native"] + 1) or on(*a, **k)
    )
    solver._solve_lexicographic_two_phase = lambda *a, **k: (
        calls.__setitem__("two_phase", calls["two_phase"] + 1) or ot(*a, **k)
    )

    # WHEN
    result = solver.solve(_build_misocp_net(), optimization_problem=prob)

    # THEN
    assert result.success
    assert calls["native"] == 1
    assert calls["two_phase"] == 0


# --------------------------------------------------------------------------- #
# Simulation mode + warm-start round-trip
# --------------------------------------------------------------------------- #


@requires_gurobi
def test_simulation_gas_only_matches_optimization_path():
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    ref = _gas_only_net()
    ref.apply_formulation(make_gas_nlp_formulation())
    sim = _gas_only_net()
    sim.apply_formulation(make_gas_nlp_formulation())

    # WHEN
    ref_res = GurobipySolver().solve(ref)
    sim_res = GurobipySolver().solve(sim, simulation=True)

    # THEN
    assert ref_res.success and sim_res.success
    assert math.isclose(
        ref_res.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0],
        sim_res.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0],
        abs_tol=1e-3,
    )


@requires_gurobi
def test_resolve_solver_accepts_instance():
    """The backend plugs into the public solve path as a concrete instance."""
    from monee.solver.dispatch import resolve_solver
    from monee.solver.gurobipy import GurobipySolver

    # GIVEN
    solver = GurobipySolver()

    # WHEN
    resolved = resolve_solver(solver)

    # THEN
    assert resolved is solver


# --------------------------------------------------------------------------- #
# Infeasibility diagnostics
# --------------------------------------------------------------------------- #


@requires_gurobi
def test_infeasible_model_reports_iis():
    """A directly contradictory model is classified as failed with an IIS
    report (exercises GurobipySolver._classify / _compute_iis)."""
    from monee.solver.gurobipy import GurobipySolver, _require_gurobipy

    # GIVEN
    solver = GurobipySolver()
    gp, GRB, nlfunc = _require_gurobipy()
    solver._gp, solver._GRB, solver._nlfunc = gp, GRB, nlfunc
    m = gp.Model()
    m.setParam("OutputFlag", 0)
    x = m.addVar(lb=0, ub=1, name="x_var")
    m.update()
    m.addConstr(x >= 2, name="impossible_c")

    # WHEN
    m.optimize()
    success, report = solver._classify(m, phase_label="test")

    # THEN
    assert success is False
    assert report is not None
    summary = report.summary()
    assert "impossible_c" in summary or "x_var" in summary


@requires_gurobi
def test_ltc_timeseries_on_gurobi_backend_does_not_crash():
    """Regression: a timeseries solve with the LumpedThermalCapacitance extension
    on the Gurobi backend used to crash in ``ext.activate_timeseries`` - it warm-
    started ``node.model.t_pu.value = prev`` after ``t_pu`` had become a gurobipy
    Var (which has no ``.value``). The warm start now routes through
    ``set_initial_value``, which writes ``.Start`` for gurobipy vars."""
    from monee.model import LumpedThermalCapacitance
    from monee.model.formulation import make_heat_nlp_formulation
    from monee.simulation import TimeseriesData, run_timeseries
    from monee.solver.gurobipy import GurobipySolver
    from tests.util import child_id_by_type

    # GIVEN a water loop with thermal capacitance, on a Gurobi-compatible heat NLP
    net, _, _, _ = create_water_loop(source_t_k=356)
    net.add_extension(LumpedThermalCapacitance())
    net.apply_formulation(make_heat_nlp_formulation())
    sink_id = child_id_by_type(net, mm.Sink)
    td = TimeseriesData()
    td.add_child_series(sink_id, "mass_flow_kgs", [10.0, 6.0, 8.0, 12.0])

    # WHEN  (previously raised AttributeError: 'gurobipy.Var' has no 'value')
    result = run_timeseries(net, td, solver=GurobipySolver())

    # THEN every step solves and the carried junction temperature stays finite
    assert not result.failed_steps
    t_last = result.get_result_for(mm.Junction, "t_pu").to_numpy()[-1]
    assert math.isfinite(float(t_last.max()))


# --------------------------------------------------------------------------- #
# Timeseries build-once model-reuse fast path
# --------------------------------------------------------------------------- #

# A no-op step hook makes a run ineligible for the fast path (forces the
# per-step rebuild loop), so it serves as the "rebuild" reference.
_NOOP_HOOK = lambda *a, **k: None  # noqa: E731


def _el_two_bus_net():
    """ext-grid at b0, a load + generator at b1 on a single MISOCP line."""
    from monee.model.child import PowerGenerator, PowerLoad
    from monee.model.node import Bus

    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[net.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    load_id = net.child(PowerLoad(p_mw=1.0, q_mvar=0.0))
    b1 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[load_id, net.child(PowerGenerator(p_mw=0.5, q_mvar=0))],
    )
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
        b0,
        b1,
    )
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net, load_id


def _el_net_with_storage(e_initial=5.0, e_max=10.0, p_max_mw=2.0):
    from monee.model.child import PowerLoad
    from monee.model.node import Bus
    from monee.model.storage import ElectricStorage

    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[net.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
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
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net, storage_id


@requires_gurobi
def test_run_timeseries_gurobipy_fast_path_matches_rebuild(monkeypatch):
    """The build-once fast path engages for a plain (memory-less) gurobipy run
    and yields identical results to the per-step rebuild loop."""
    import numpy as np

    import monee
    import monee.simulation.timeseries as tsmod
    from monee.simulation import TimeseriesData
    from monee.solver.gurobipy import GurobipySolver

    n = 4
    calls = {"reuse": 0}
    orig = tsmod._run_gurobipy_reuse
    monkeypatch.setattr(
        tsmod,
        "_run_gurobipy_reuse",
        lambda *a, **k: calls.__setitem__("reuse", calls["reuse"] + 1) or orig(*a, **k),
    )

    net, load_id = _el_two_bus_net()
    profile = [1.0, 1.5, 0.8, 1.2]

    td_fast = TimeseriesData()
    td_fast.add_child_series(load_id, "p_mw", profile)
    res_fast = monee.run_timeseries(net, td_fast, solver=GurobipySolver())

    net2, load_id2 = _el_two_bus_net()
    td_slow = TimeseriesData()
    td_slow.add_child_series(load_id2, "p_mw", profile)
    res_slow = monee.run_timeseries(
        net2, td_slow, solver=GurobipySolver(), step_hooks=[_NOOP_HOOK]
    )

    assert calls["reuse"] == 1  # fast path engaged for the first (no-hook) run
    assert res_fast.failed_steps == [] and res_slow.failed_steps == []
    vf = res_fast.get_result_for(mm.Bus, "vm_pu").to_numpy()
    vs = res_slow.get_result_for(mm.Bus, "vm_pu").to_numpy()
    assert vf.shape == vs.shape
    assert np.nanmax(np.abs(vf - vs)) < 1e-6


@requires_gurobi
def test_run_timeseries_gurobipy_fast_path_handles_storage_coupling(monkeypatch):
    """Unlike the CasADi backend, the gurobipy fast path *stays engaged* for a
    temporally-coupled network (storage SoC): the carried state is wired as a
    per-step parameter, reproducing the rebuild loop exactly (including the
    inter-step SoC invariant)."""
    import numpy as np

    import monee
    import monee.simulation.timeseries as tsmod
    from monee.simulation import TimeseriesData
    from monee.solver.gurobipy import GurobipySolver

    calls = {"reuse": 0}
    orig = tsmod._run_gurobipy_reuse
    monkeypatch.setattr(
        tsmod,
        "_run_gurobipy_reuse",
        lambda *a, **k: calls.__setitem__("reuse", calls["reuse"] + 1) or orig(*a, **k),
    )

    dispatch = [1.0, -0.5, 0.8, 0.3]

    net, sid = _el_net_with_storage()
    td = TimeseriesData()
    td.add_child_series(sid, "p_mw", dispatch)
    res_fast = monee.run_timeseries(net, td, solver=GurobipySolver())

    net2, sid2 = _el_net_with_storage()
    td2 = TimeseriesData()
    td2.add_child_series(sid2, "p_mw", dispatch)
    res_slow = monee.run_timeseries(
        net2, td2, solver=GurobipySolver(), step_hooks=[_NOOP_HOOK]
    )

    assert (
        calls["reuse"] == 1
    )  # temporal coupling does NOT disable the gurobi fast path
    assert res_fast.failed_steps == []
    e_fast = res_fast.get_result_for_id(sid, "e_mwh").to_numpy()
    e_slow = res_slow.get_result_for_id(sid2, "e_mwh").to_numpy()
    assert np.nanmax(np.abs(e_fast - e_slow)) < 1e-6

    # Inter-step SoC invariant e[t] == e[t-1] + dt_h * p[t] (dt_h = 1.0): proves
    # the carried state was actually coupled, not just re-bounded.
    p_fast = res_fast.get_result_for_id(sid, "p_mw").to_numpy()
    for t in range(1, len(dispatch)):
        assert abs(e_fast[t] - (e_fast[t - 1] + p_fast[t])) < 1e-6


@requires_gurobi
def test_run_timeseries_gurobipy_falls_back_when_hooks_present(monkeypatch):
    """Step hooks observe per-step state, so the standard per-step loop must run
    instead of the build-once fast path."""
    import monee
    import monee.simulation.timeseries as tsmod
    from monee.simulation import TimeseriesData
    from monee.solver.gurobipy import GurobipySolver

    calls = {"reuse": 0}
    orig = tsmod._run_gurobipy_reuse
    monkeypatch.setattr(
        tsmod,
        "_run_gurobipy_reuse",
        lambda *a, **k: calls.__setitem__("reuse", calls["reuse"] + 1) or orig(*a, **k),
    )

    net, load_id = _el_two_bus_net()
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.5, 0.8])
    seen = []
    monee.run_timeseries(
        net,
        td,
        solver=GurobipySolver(),
        step_hooks=[lambda *a: seen.append(a[1])],
    )

    assert calls["reuse"] == 0  # fast path skipped because hooks are present
    assert seen == [0, 1, 2]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
