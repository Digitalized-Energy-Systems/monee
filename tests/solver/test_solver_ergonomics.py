"""Batch solver-ergonomics: failure reporting (S1) and ``solver='auto'`` (S2).

Each test fails on the pre-batch code: S1's default was to raise, and
``solver='auto'`` did not resolve at all.
"""

import importlib.util

import pytest

import monee.model as mm
from monee import run_energy_flow
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.formulation import EL_NLP_FORMULATION
from monee.model.node import Bus
from monee.solver import resolve_multi_period_solver, resolve_solver
from monee.solver.dispatch import AutoSolver

requires_casadi = pytest.mark.skipif(
    importlib.util.find_spec("casadi") is None, reason="casadi missing"
)


def _el_net():
    """ext-grid - line - (load + gen); converges as a plain power flow."""
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    b1 = net.node(
        Bus(base_kv=1),
        grid=mm.EL,
        child_ids=[
            net.child(PowerLoad(p_mw=1.0, q_mvar=0.1)),
            net.child(PowerGenerator(p_mw=0.5, q_mvar=0)),
        ],
    )
    net.branch(
        mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
        b0,
        b1,
    )
    net.apply_formulation(EL_NLP_FORMULATION)
    return net


def _force_ipopt_failure(monkeypatch, return_status="Restoration_Failed"):
    """Make every CasADi solve report a numerics failure with a real iterate."""
    from monee.solver import casadi as cmod

    real_nlpsol = cmod.ca.nlpsol

    def fake_nlpsol(name, plugin, prob, opts):
        solver = real_nlpsol(name, plugin, prob, opts)

        class _Wrapped:
            def __call__(self, **kw):
                return solver(**kw)

            def stats(self):
                s = dict(solver.stats())
                s["success"] = False
                s["return_status"] = return_status
                return s

        return _Wrapped()

    monkeypatch.setattr(cmod.ca, "nlpsol", fake_nlpsol)


# --------------------------------------------------------------------------- #
# S1: a numerics failure is reported on the result; strict=True still raises.
# --------------------------------------------------------------------------- #
@requires_casadi
def test_failed_solve_returns_a_result_with_the_diagnostics(monkeypatch):
    from monee.solver.casadi import CasADiFailureReport

    _force_ipopt_failure(monkeypatch)
    result = run_energy_flow(_el_net())

    assert result.success is False
    assert result.solver_status == "error"
    assert result.termination_condition == "Restoration_Failed"
    assert result.objective is None
    assert isinstance(result.infeasibility_report, CasADiFailureReport)
    report = result.infeasibility_report
    assert report.numerics_failure is True
    assert "Restoration_Failed" in report.message
    assert report.summary() == report.message


@requires_casadi
def test_strict_still_raises_the_solve_error(monkeypatch):
    from monee.solver.casadi import CasADiSolveError

    _force_ipopt_failure(monkeypatch)
    with pytest.raises(CasADiSolveError) as excinfo:
        run_energy_flow(_el_net(), strict=True)

    assert excinfo.value.report is not None
    assert excinfo.value.report.return_status == "Restoration_Failed"


@requires_casadi
def test_failed_solve_leaves_the_callers_network_untouched(monkeypatch):
    net = _el_net()
    before = float(mm.value(net.branches[0].model.length_m))

    _force_ipopt_failure(monkeypatch)
    result = run_energy_flow(net)

    assert result.success is False
    assert float(mm.value(net.branches[0].model.length_m)) == before
    # The frames exist and report inputs, not a solution.
    assert "Bus" in result.dataframes


@requires_casadi
def test_a_proven_infeasibility_is_not_classified_as_numerics(monkeypatch):
    _force_ipopt_failure(monkeypatch, return_status="Infeasible_Problem_Detected")
    result = run_energy_flow(_el_net())

    assert result.success is False
    assert result.infeasibility_report.numerics_failure is False


def test_is_numerics_failure_separates_breakdowns_from_verdicts():
    from monee.solver.casadi import is_numerics_failure

    assert is_numerics_failure("Restoration_Failed")
    assert is_numerics_failure("Error_In_Step_Computation")
    assert is_numerics_failure("Maximum_Iterations_Exceeded")
    assert not is_numerics_failure("Infeasible_Problem_Detected")
    assert not is_numerics_failure("Not_Enough_Degrees_Of_Freedom")


# --------------------------------------------------------------------------- #
# S2: solver='auto' retries a numerics failure once on SCIP.
# --------------------------------------------------------------------------- #
def test_auto_resolves_to_the_auto_solver():
    assert isinstance(resolve_solver("auto"), AutoSolver)
    assert isinstance(resolve_solver("AUTO"), AutoSolver)


def test_auto_rejects_an_explicit_backend():
    with pytest.raises(ValueError, match="picks the back-end itself"):
        resolve_solver("auto", backend="casadi")


def test_auto_is_rejected_for_multi_period():
    with pytest.raises(ValueError, match="single-period only"):
        resolve_multi_period_solver("auto")


@requires_casadi
def test_auto_falls_back_to_scip_on_a_numerics_failure(monkeypatch):
    _force_ipopt_failure(monkeypatch)
    result = run_energy_flow(_el_net(), solver="auto")

    assert result.success is True
    assert result.backend_used == "pyomo"
    assert result.solver_used == "scip"
    assert result.fallback_from == "casadi/ipopt"
    assert any(w.category == "backend_fallback" for w in result.warnings)


@requires_casadi
def test_auto_keeps_a_first_attempt_result_and_marks_no_fallback():
    result = run_energy_flow(_el_net(), solver="auto")

    assert result.success is True
    assert result.backend_used == "casadi"
    assert result.fallback_from is None
    assert not any(w.category == "backend_fallback" for w in result.warnings)


@requires_casadi
def test_auto_does_not_retry_a_proven_infeasibility(monkeypatch):
    calls = []
    real_resolve = resolve_solver

    solver = AutoSolver()
    monkeypatch.setattr(
        solver, "_fallback_solver", lambda: calls.append(1) or real_resolve("scip")
    )
    _force_ipopt_failure(monkeypatch, return_status="Infeasible_Problem_Detected")

    result = solver.solve(_el_net())

    assert result.success is False
    assert result.fallback_from is None
    assert calls == []


@requires_casadi
def test_auto_delegates_the_backend_labels_to_the_primary():
    solver = AutoSolver()

    assert solver.backend_name == "casadi"
    assert solver.solver_name == "ipopt"
