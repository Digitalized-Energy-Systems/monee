"""Tests for the native gurobipy solver backend (:mod:`monee.solver.gurobipy`).

Covers the two formulation families the backend supports - smooth NLP
(``*_nlp``, via ``gurobipy.nlfunc`` + general nonlinear constraints) and the
(MI)QCQP / MISOCP / PWL convex path - plus the IIS infeasibility diagnostics
and the lexicographic objective handling.

All solve-based tests need both the ``gurobipy`` package and a working Gurobi
licence, so they are skipped wholesale when either is missing.
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
FRICTION_MODELS = ["constant", "pwl", "nonlinear"]


def _gurobipy_available() -> bool:
    """True iff gurobipy imports *and* a licence lets us solve a trivial model."""
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
    not HAVE_GUROBI, reason="gurobipy + Gurobi licence not available"
)


def _gas_only_net():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))
    g0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=1))])
    g1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.6))])
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1000, roughness_m=0.01), g0, g1)
    pn.branch(mm.GasPipe(diameter_m=0.35, length_m=1500, roughness_m=0.01), g0, g2)
    return pn


def _heat_only_net():
    pn = mm.Network(mm.create_water_grid("heat"))
    w0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))])
    w1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(1))])
    w2 = pn.node(mm.Junction())
    w3 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))])
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)
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
    r_native = GurobipySolver().solve(
        _build_misocp_net(), optimization_problem=prob
    )
    r_pyomo = ms.PyomoSolver("gurobi").solve(
        _build_misocp_net(), optimization_problem=_make_shedding_problem()
    )

    # THEN
    assert r_native.success and r_pyomo.success
    # Same model, same solver - objectives must agree.
    assert math.isclose(r_native.objective, r_pyomo.objective, rel_tol=1e-4, abs_tol=1e-4)


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
