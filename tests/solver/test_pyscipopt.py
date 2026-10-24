"""Tests for the pyscipopt bridge (:class:`monee.solver.pyo.PyscipoptSolver`).

The bridge drives SCIP through the ``pyscipopt`` Python bindings: it writes the
Pyomo model to an AMPL ``.nl`` file, solves it with the in-process SCIP model,
and reads the solution back onto the Pyomo Vars. :class:`PyomoSolver` falls back
to it automatically when the standalone ``scip``/``scipampl`` executable is
absent but the bindings are installed.

``pyscipopt`` is a core monee dependency, so the solve-based tests run wherever
the suite does. The status-mapping and selection tests need no SCIP at all.
"""

import math
import types

import pytest
from pyomo.opt import SolverStatus, TerminationCondition

import monee.solver.pyo as pyo_mod
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.model.grid import PowerGrid
from monee.network import create_urban_district_net
from monee.problem.min_load_shedding import create_min_load_shedding_problem
from monee.solver import PyomoSolver
from monee.solver.pyo import PyscipoptSolver, _pyscipopt_available

HAVE_PYSCIPOPT = _pyscipopt_available()
requires_pyscipopt = pytest.mark.skipif(
    not HAVE_PYSCIPOPT, reason="pyscipopt bindings not installed"
)


def _shedding_problem():
    return create_min_load_shedding_problem(
        bounds_vm=(0.5, 1.5),
        bounds_pressure=(0.5, 1.5),
        bounds_t=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
    )


# --------------------------------------------------------------------------- #
# SCIP status -> Pyomo (status, termination_condition) mapping (no solve)
# --------------------------------------------------------------------------- #


def test_build_result_optimal():
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("optimal", has_solution=True)

    # THEN
    assert result.solver.status == SolverStatus.ok
    assert result.solver.termination_condition == TerminationCondition.optimal


def test_build_result_infeasible():
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("infeasible", has_solution=False)

    # THEN
    assert result.solver.status == SolverStatus.warning
    assert result.solver.termination_condition == TerminationCondition.infeasible


def test_build_result_unbounded():
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("unbounded", has_solution=False)

    # THEN
    assert result.solver.status == SolverStatus.warning
    assert result.solver.termination_condition == TerminationCondition.unbounded


def test_build_result_limit_with_incumbent_is_usable():
    """A limit (time/node/...) that still produced a feasible incumbent maps to
    aborted+maxTimeLimit, which ``_classify_solve_result`` treats as success."""
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("timelimit", has_solution=True)

    # THEN
    assert result.solver.status == SolverStatus.aborted
    assert result.solver.termination_condition == TerminationCondition.maxTimeLimit


def test_build_result_gaplimit_is_optimal_within_tolerance():
    """A gap-limit stop means the incumbent meets the configured relative gap:
    reported ok/optimal (converged within tolerance), distinguishable from a
    real timeout's aborted/maxTimeLimit."""
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("gaplimit", has_solution=True)

    # THEN
    assert result.solver.status == SolverStatus.ok
    assert result.solver.termination_condition == TerminationCondition.optimal


def test_build_result_gaplimit_without_incumbent_is_a_failure():
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("gaplimit", has_solution=False)

    # THEN no solution to read back, but no infeasibility verdict either
    assert result.solver.status == SolverStatus.aborted
    assert result.solver.termination_condition == TerminationCondition.other
    assert result.solver.has_solution is False


def _scip_executable_result(message):
    """What Pyomo's interface to the scip executable returns for a stop at a
    limit: ok/other, with SCIP's own message."""
    return types.SimpleNamespace(
        solver=types.SimpleNamespace(
            status=SolverStatus.ok,
            termination_condition=TerminationCondition.other,
            message=message,
        )
    )


def test_scip_executable_gap_limit_reads_optimal_like_the_bridge():
    # GIVEN
    result = _scip_executable_result("gap limit reached")

    # WHEN
    success, _, _, tc_str = pyo_mod._classify_solve_result(
        result, None, "scip", phase_label="test"
    )

    # THEN
    assert success
    assert tc_str == str(TerminationCondition.optimal)


def test_scip_executable_other_limits_stay_other():
    # GIVEN
    result = _scip_executable_result("memory limit reached")

    # WHEN
    _, _, _, tc_str = pyo_mod._classify_solve_result(
        result, None, "scip", phase_label="test"
    )

    # THEN
    assert tc_str == str(TerminationCondition.other)


# --------------------------------------------------------------------------- #
# Capabilities + backend selection
# --------------------------------------------------------------------------- #


def test_pyscipopt_solver_is_not_warmstart_capable():
    # GIVEN / WHEN / THEN
    assert PyscipoptSolver().warm_start_capable() is False


@requires_pyscipopt
def test_pyscipopt_solver_reports_available():
    # GIVEN / WHEN / THEN
    assert PyscipoptSolver().available() is True


@requires_pyscipopt
def test_make_solver_routes_scip_to_bridge_without_classic_exe(monkeypatch):
    """With no standalone scip on PATH but the bindings present, ``scip`` resolves
    to the pyscipopt bridge rather than a SolverFactory wrapper."""
    # GIVEN
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)

    # WHEN
    solver = PyomoSolver._make_solver("scip")

    # THEN
    assert isinstance(solver, PyscipoptSolver)


def test_make_solver_prefers_classic_scip_exe_when_present(monkeypatch):
    """When the standalone executable exists, the SolverFactory path is used so
    the bridge's .nl round-trip is avoided."""
    # GIVEN
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: True)

    # WHEN
    solver = PyomoSolver._make_solver("scip")

    # THEN
    assert not isinstance(solver, PyscipoptSolver)


def test_make_solver_non_scip_name_is_never_the_bridge(monkeypatch):
    # GIVEN: even with no classic scip exe, a non-scip name must not hit the bridge.
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)

    # WHEN
    solver = PyomoSolver._make_solver("highs")

    # THEN
    assert not isinstance(solver, PyscipoptSolver)


# --------------------------------------------------------------------------- #
# Real solve through the bridge
# --------------------------------------------------------------------------- #


@requires_pyscipopt
def test_bridge_solves_misocp_load_shedding(monkeypatch):
    """End-to-end: a MISOCP load-shedding solve forced through the pyscipopt
    bridge converges and writes finite values back onto the model."""
    # GIVEN: force the bridge even if a classic scip exe happens to be on PATH.
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)

    # WHEN
    result = PyomoSolver("scip").solve(net, optimization_problem=_shedding_problem())

    # THEN
    assert result.success
    assert result.objective is not None and math.isfinite(result.objective)
    # Solution was read back: bus voltages are populated and physical.
    vm = result.dataframes["Bus"]["vm_pu"]
    assert vm.notna().all()
    assert (vm > 0).all()


def test_scip_preset_ships_a_default_relative_gap():
    """The SCIP preset carries limits/gap so exploratory MILPs stop at the
    incumbent instead of grinding on the optimality proof."""
    assert pyo_mod.PER_SOLVER_OPTIONS["scip"]["limits/gap"] == 1e-4


def test_solver_result_importable_from_monee_solver():
    from monee.solver import SolverResult
    from monee.solver.core import SolverResult as core_solver_result

    assert SolverResult is core_solver_result


@requires_pyscipopt
def test_solver_options_reach_scip_through_public_chain(monkeypatch):
    """Per-call solver_options passed to run_energy_flow_optimization reach the
    SCIP bridge, merged over instance options and the module preset; a None
    value removes a preset key."""
    # GIVEN
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)
    seen: dict = {}
    orig_solve = PyscipoptSolver.solve

    def spy(self, model, **kwargs):
        seen.update(self.options)
        return orig_solve(self, model, **kwargs)

    monkeypatch.setattr(PyscipoptSolver, "solve", spy)
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    solver = PyomoSolver("scip", solver_options={"limits/gap": 0.02})

    # WHEN
    import monee

    result = monee.run_energy_flow_optimization(
        net,
        _shedding_problem(),
        solver=solver,
        solver_options={"limits/gap": 0.01, "limits/time": None},
    )

    # THEN: per-call wins over instance, instance over preset; None removes.
    assert result.success
    assert seen["limits/gap"] == 0.01
    assert "limits/time" not in seen
    assert seen["misc/allowstrongdualreds"] is False


@requires_pyscipopt
def test_bridge_writeback_matches_a_second_identical_solve(monkeypatch):
    """The .nl-name-matching write-back is deterministic: two identical solves
    through the bridge agree on the objective (guards the Windows CRLF name-strip
    path that previously left every Var unmatched)."""
    # GIVEN
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)

    def _solve():
        net = create_urban_district_net()
        for grid in net.grids:
            if isinstance(grid, PowerGrid):
                grid.sn_mva = 100
        net.apply_formulation(EL_MISOCP_FORMULATION)
        return PyomoSolver("scip").solve(net, optimization_problem=_shedding_problem())

    # WHEN
    r1, r2 = _solve(), _solve()

    # THEN
    assert r1.success and r2.success
    assert math.isclose(r1.objective, r2.objective, rel_tol=1e-3, abs_tol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


@requires_pyscipopt
def test_multi_period_scip_routes_to_the_bridge_without_classic_exe(monkeypatch):
    """Regression: the Pyomo multi-period solver called ``pyo.SolverFactory``
    directly, so ``run_multi_period(..., solver="scip")`` failed with "No
    executable found for solver 'scip'" where the single-period path falls back
    to the pyscipopt bridge."""
    import monee.express as mx
    from monee.simulation import TimeseriesData, run_multi_period

    # GIVEN no standalone scip executable
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, b1, p_mw=0.1, q_mvar=0.0, name="load")
    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", [0.1, 0.5])

    # WHEN
    result = run_multi_period(net, td, steps=2, dt_h=1.0, solver="scip")

    # THEN
    assert result.success
    assert result.backend_used == "pyomo"


def test_build_result_limit_without_incumbent_is_a_failure_not_a_verdict():
    """Regression: a SCIP time limit without an incumbent was reported as
    'infeasible' (and a MIS computed), a false verdict about the model.
    gurobipy reports the same stop as aborted/maxTimeLimit without success."""
    # GIVEN
    result = PyscipoptSolver._build_result("timelimit", has_solution=False)

    # WHEN
    success, report, status, tc = pyo_mod._classify_solve_result(
        result, None, "scip", phase_label="test"
    )

    # THEN
    assert result.solver.termination_condition == TerminationCondition.maxTimeLimit
    assert result.solver.status == SolverStatus.aborted
    assert success is False
    assert report is None


def test_build_result_infeasible_or_unbounded_keeps_the_infeasibility_verdict():
    """Regression: SCIP's 'inforunbd' is a proof about the model, not a limit
    stop; it fell into the no-incumbent limit branch and lost the diagnosis."""
    result = PyscipoptSolver._build_result("inforunbd", has_solution=False)

    assert result.solver.termination_condition == TerminationCondition.infeasible


def test_build_result_limit_with_incumbent_is_still_a_witness():
    result = PyscipoptSolver._build_result("timelimit", has_solution=True)

    success, _report, _status, _tc = pyo_mod._classify_solve_result(
        result, None, "scip", phase_label="test"
    )

    assert success is True


# --------------------------------------------------------------------------- #
# Limit stops report the incumbent and the dual bound
# --------------------------------------------------------------------------- #


def _nonconvex_shedding(solver_options, net_factory):
    import monee

    return monee.run_energy_flow_optimization(
        net_factory(),
        create_min_load_shedding_problem(
            bounds_vm=(0.9, 1.1),
            bounds_pressure=(0.5, 1.5),
            bounds_t=(0.5, 1.5),
            include_ext_grids=False,
            include_storages=False,
        ),
        solver=PyomoSolver("scip"),
        formulation="nonconvex_miqcqp",
        solver_options=solver_options,
    )


@requires_pyscipopt
def test_scip_time_limit_reports_the_dual_bound():
    """Regression: a time-limited SCIP solve gave no figure at all, so a study
    could not check that the bound brackets another solver's optimum."""
    from monee.network.mes import create_mv_multi_cigre

    # GIVEN a nonconvex shedding problem SCIP needs about a minute to prove
    # WHEN
    result = _nonconvex_shedding({"limits/time": 1.0}, create_mv_multi_cigre)

    # THEN
    stats = result.solver_stats
    assert result.termination_condition == "maxTimeLimit"
    assert math.isfinite(stats["bound"])
    if result.success:
        assert stats["obj"] >= stats["bound"]
    else:
        assert math.isnan(stats["obj"])


@requires_pyscipopt
def test_scip_limit_stop_returns_the_incumbent_with_bound_and_gap():
    from monee.network.mes import create_monee_benchmark_net

    # GIVEN SCIP stopped at its first solution, far from the bound
    # WHEN
    result = _nonconvex_shedding({"limits/solutions": 1}, create_monee_benchmark_net)

    # THEN
    stats = result.solver_stats
    assert result.success
    assert result.termination_condition == "maxTimeLimit"
    assert stats["obj"] == pytest.approx(result.objective)
    assert stats["bound"] < stats["obj"]
    assert stats["gap"] == pytest.approx(
        (stats["obj"] - stats["bound"]) / abs(stats["obj"])
    )
    assert stats["gap"] > 0.5


@requires_pyscipopt
def test_scip_optimum_reports_a_closed_gap():
    # GIVEN / WHEN
    result = _nonconvex_shedding({}, create_urban_district_net)

    # THEN
    assert result.termination_condition == "optimal"
    assert result.solver_stats["obj"] == pytest.approx(result.objective)
    assert result.solver_stats["gap"] < 1e-6
