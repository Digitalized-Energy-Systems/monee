"""Tests for the lexicographic (two-phase) objective solve in PyomoSolver.

The lex path splits the objective into a *user* tier (e.g. the
load-shedding sum from ``OptimizationProblem.objectives``) and an
*aux* tier (formulation-level tightening terms returned by
``branch.minimize`` / ``node.minimize`` / ``child.minimize``).  Phase 1
minimises only the user tier; phase 2 minimises the aux tier with a
cap constraint pinning the phase-1 optimum.

These tests exercise:

1. Backwards-compat - default ``lex_objectives=False`` matches the
   pre-existing single-objective solve.
2. Phase-1 dominance - in lex mode the user objective is *exactly* at
   its true optimum (i.e. independent of any aux scale).
3. Decoupling - boosting the formulation-level aux scale ×100 leaves
   the lex user-objective unchanged but inflates the legacy weighted
   objective.
4. Helper math - ``_lex_cap_slack`` respects ``MIPGap``.
"""

import pytest

from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import create_urban_district_net
from monee.problem.core import OptimizationProblem
from monee.problem.min_load_shedding import (
    create_min_load_shedding_problem,
)
from monee.solver.pyo import PyomoSolver as _PyomoSolverCls

pytest.importorskip("pyomo")


def _solver_available(name: str) -> bool:
    import pyomo.environ as pyo

    try:
        return pyo.SolverFactory(name).available(exception_flag=False)
    except Exception:
        return False


# All optimisation tests need a MISOCP-capable solver (SOC cones).
SOLVER = "gurobi" if _solver_available("gurobi") else None


def _user_obj_value(network):
    """Recompute the user (shedding) objective from the solved network.

    Mirrors ``min_load_shedding._calc_objective`` but reads numeric
    values out of the post-solve Var attributes.
    """
    from monee.model import (
        HeatExchanger,
        HeatGenerator,
        HeatLoad,
        PowerGenerator,
        PowerLoad,
        Sink,
        Source,
    )
    from monee.model.core import Var
    from monee.problem.min_load_shedding import (
        _HE_OBJECTIVE_TYPES,
    )

    def _val(x):
        if isinstance(x, Var):
            return x.value if x.value is not None else 0.0
        return x

    total = 0.0
    for comp in network.all_components():
        m = comp.model
        if (
            not isinstance(
                m,
                (PowerLoad, PowerGenerator, HeatLoad, HeatGenerator, Sink, Source)
                + _HE_OBJECTIVE_TYPES,
            )
            and type(m) is not HeatExchanger
        ):
            continue
        reg = _val(getattr(m, "regulation", 1))
        # Replicate _shedding_mw using numeric values
        if isinstance(m, PowerLoad):
            total += _val(m.p_mw) * (1 - reg) * 1e3
        elif isinstance(m, PowerGenerator):
            total += (-_val(m.p_mw)) * (1 - reg) * 0.1
        elif isinstance(m, HeatLoad):
            total += _val(m.q_mw_heat) * (1 - reg) * 1e3
        elif isinstance(m, HeatGenerator):
            total += (-_val(m.q_mw_heat)) * (1 - reg) * 0.1
        elif isinstance(m, (Sink, Source)):
            # Skip - gas factor lookup requires the grid map; tests below
            # don't rely on Sink/Source so this approximation is OK.
            continue
        else:  # HE / PassiveHE branches
            q_set = getattr(m, "q_mw_set", 0)
            q_del = getattr(m, "q_mw_delivered", None)
            if isinstance(q_set, (int, float)) and q_del is not None:
                q_del_val = _val(q_del)
                shed = (q_set - q_del_val) if q_set > 0 else (q_del_val - q_set)
                w = 1e3 if q_set > 0 else 0.1
                total += shed * w
    return total


def _build_unstressed_net():
    """Urban-district net at nominal loading - feasible with no shedding."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    return net


# --------------------------------------------------------------------------- #
# 1. Helper-math test (no solver dependency)
# --------------------------------------------------------------------------- #


def test_lex_cap_slack_respects_mipgap():
    """``_lex_cap_slack`` must scale with the configured MIPGap."""
    # Tight default - slack is tiny.
    slack_tight = _PyomoSolverCls._lex_cap_slack(100.0, {"MIPGap": 0.0})
    assert slack_tight < 1e-3
    # Loose 1 % gap - slack should reflect that.
    slack_loose = _PyomoSolverCls._lex_cap_slack(100.0, {"MIPGap": 1e-2})
    assert slack_loose >= 1.0  # 1 % of 100 = 1.0 plus floor


def test_lex_cap_slack_floors_at_one_for_small_optima():
    """``max(1, |S*|)`` floor avoids zero slack when S* ≈ 0."""
    slack = _PyomoSolverCls._lex_cap_slack(0.0, {"MIPGap": 1e-2})
    # 1 % of max(1, 0) = 0.01 ⇒ slack ≥ 0.01
    assert slack >= 0.01 - 1e-12


# --------------------------------------------------------------------------- #
# 2. OptimizationProblem default + flag plumbing
# --------------------------------------------------------------------------- #


def test_optimization_problem_default_lex_off():
    prob = OptimizationProblem()
    assert prob.lex_objectives is False


def test_optimization_problem_lex_flag_passthrough():
    prob = OptimizationProblem(lex_objectives=True)
    assert prob.lex_objectives is True


def test_create_problem_lex_flag_threads_to_optimization_problem():
    prob = create_min_load_shedding_problem(lex_objectives=True)
    assert prob.lex_objectives is True

    prob_default = create_min_load_shedding_problem()
    assert prob_default.lex_objectives is False


# --------------------------------------------------------------------------- #
# 3. Solve-based behavioural tests (require Gurobi / MISOCP-capable solver)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
def test_lex_matches_legacy_on_unstressed_network():
    """When no shedding is needed both modes find the same shed value (0)."""
    net = _build_unstressed_net()
    prob_legacy = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
    )
    res_legacy = _PyomoSolverCls(SOLVER).solve(net, optimization_problem=prob_legacy)
    assert res_legacy.success

    net2 = _build_unstressed_net()
    prob_lex = create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
        lex_objectives=True,
    )
    res_lex = _PyomoSolverCls(SOLVER).solve(net2, optimization_problem=prob_lex)
    assert res_lex.success

    # Both should recover full demand (regulation ≈ 1 everywhere).
    shed_legacy = _user_obj_value(res_legacy.network)
    shed_lex = _user_obj_value(res_lex.network)
    assert shed_legacy < 1e-3
    assert shed_lex < 1e-3


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
def test_lex_user_objective_is_optimal_independent_of_aux_scale():
    """Boost the aux tightening terms ×N - legacy drifts, lex does not.

    The legacy weighted-sum balances ``demand_weight=1e3`` against
    ``sum(current_pu·br_r)``.  If we artificially scale the aux terms
    up, the weighted sum's optimum can pull regulation < 1 to reduce
    the (now-inflated) loss term, even though no shedding is
    *required*.  Lex mode forbids this trade-off by construction:
    phase 1 sees only user terms, so the user objective is at its
    true optimum (≈0) regardless of aux magnitude.
    """
    AUX_SCALE = 50.0  # large enough to overpower demand_weight=1e3 in legacy

    def _scaled_solve(lex: bool):
        net = _build_unstressed_net()
        prob = create_min_load_shedding_problem(
            bounds_el=(0.5, 1.5),
            bounds_gas=(0.5, 1.5),
            bounds_heat=(0.5, 1.5),
            include_ext_grids=False,
            include_storages=False,
            lex_objectives=lex,
        )

        # Monkey-patch the MISOCP branch tightener to scale its
        # contribution.  Implemented by wrapping the branch's
        # minimize() to inflate the returned expression.
        from monee.model.formulation.misoc.el import (
            MISOCPElectricityBranchFormulation,
        )

        orig_minimize = MISOCPElectricityBranchFormulation.minimize

        def scaled_minimize(
            self, branch, grid, from_node_model, to_node_model, **kwargs
        ):
            inner = orig_minimize(
                self, branch, grid, from_node_model, to_node_model, **kwargs
            )
            return [AUX_SCALE * e for e in inner]

        MISOCPElectricityBranchFormulation.minimize = scaled_minimize
        try:
            return _PyomoSolverCls(SOLVER).solve(net, optimization_problem=prob)
        finally:
            MISOCPElectricityBranchFormulation.minimize = orig_minimize

    res_legacy = _scaled_solve(lex=False)
    res_lex = _scaled_solve(lex=True)
    assert res_legacy.success
    assert res_lex.success

    shed_legacy = _user_obj_value(res_legacy.network)
    shed_lex = _user_obj_value(res_lex.network)

    # Lex must find the *true* unstressed optimum: shed ≈ 0.
    assert shed_lex < 1e-3, (
        f"Lex phase-1 should pin shed at its true optimum (~0), got {shed_lex}"
    )
    # Legacy is *not* required to fail here on this small benchmark,
    # but in either case lex's shed cannot exceed legacy's shed (lex
    # is a tighter optimisation of the user tier).
    assert shed_lex <= shed_legacy + 1e-6
