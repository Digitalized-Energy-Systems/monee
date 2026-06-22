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

import pytest
from pyomo.opt import SolverStatus, TerminationCondition

import monee.solver.pyo as pyo_mod
from monee.model.formulation import EL_MISOCP_FORMULATION
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
    """A limit (time/node/gap) that still produced a feasible incumbent maps to
    aborted+maxTimeLimit, which ``_classify_solve_result`` treats as success."""
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("timelimit", has_solution=True)

    # THEN
    assert result.solver.status == SolverStatus.aborted
    assert result.solver.termination_condition == TerminationCondition.maxTimeLimit


def test_build_result_limit_without_incumbent_is_infeasible():
    """A limit hit with no incumbent must not be read back as a solution, so it
    is reported as infeasible to surface a diagnostic instead."""
    # GIVEN / WHEN
    result = PyscipoptSolver._build_result("timelimit", has_solution=False)

    # THEN
    assert result.solver.status == SolverStatus.warning
    assert result.solver.termination_condition == TerminationCondition.infeasible


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


@requires_pyscipopt
def test_bridge_writeback_matches_a_second_identical_solve(monkeypatch):
    """The .nl-name-matching write-back is deterministic: two identical solves
    through the bridge agree on the objective (guards the Windows CRLF name-strip
    path that previously left every Var unmatched)."""
    # GIVEN
    monkeypatch.setattr(pyo_mod, "_classic_scip_available", lambda: False)

    def _solve():
        net = create_urban_district_net()
        net.apply_formulation(EL_MISOCP_FORMULATION)
        return PyomoSolver("scip").solve(net, optimization_problem=_shedding_problem())

    # WHEN
    r1, r2 = _solve(), _solve()

    # THEN
    assert r1.success and r2.success
    assert math.isclose(r1.objective, r2.objective, rel_tol=1e-6, abs_tol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
