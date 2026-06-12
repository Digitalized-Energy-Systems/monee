"""Tests for the lexicographic (two-phase) objective solve in PyomoSolver."""

import pytest

from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.network import create_urban_district_net
from monee.problem.core import OptimizationProblem
from monee.problem.min_load_shedding import (
    create_min_load_shedding_problem,
)
from monee.solver.pyo import PyomoSolver as _PyomoSolverCls
from tests.util import solver_available as _solver_available

pytest.importorskip("pyomo")


# All optimisation tests need a MISOCP-capable solver (SOC cones).
SOLVER = "gurobi" if _solver_available("gurobi") else None


def _user_obj_value(network):
    """Recompute the user (shedding) objective post-solve, mirroring min_load_shedding._calc_objective."""
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
            # Skip - gas factor lookup needs the grid map; tests don't rely on it
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
    net.apply_formulation(EL_MISOCP_FORMULATION)
    return net


def _make_shedding_problem(lex_objectives=False):
    """Min-load-shedding problem with the relaxed bounds used by these tests."""
    return create_min_load_shedding_problem(
        bounds_el=(0.5, 1.5),
        bounds_gas=(0.5, 1.5),
        bounds_heat=(0.5, 1.5),
        include_ext_grids=False,
        include_storages=False,
        lex_objectives=lex_objectives,
    )


def _solve_with_scaled_aux(lex: bool, aux_scale: float):
    """Solve the unstressed net with the MISOCP el-branch aux terms scaled by aux_scale."""
    from monee.model.formulation.miqcqp.convex.el import (
        MISOCPElectricityBranchFormulation,
    )

    net = _build_unstressed_net()
    prob = _make_shedding_problem(lex_objectives=lex)

    orig_minimize = MISOCPElectricityBranchFormulation.minimize

    def scaled_minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        inner = orig_minimize(
            self, branch, grid, from_node_model, to_node_model, **kwargs
        )
        return [aux_scale * e for e in inner]

    MISOCPElectricityBranchFormulation.minimize = scaled_minimize
    try:
        return _PyomoSolverCls(SOLVER).solve(net, optimization_problem=prob)
    finally:
        MISOCPElectricityBranchFormulation.minimize = orig_minimize


# --------------------------------------------------------------------------- #
# 1. Helper-math tests (no solver dependency)
# --------------------------------------------------------------------------- #


def test_lex_cap_slack_respects_mipgap():
    # GIVEN
    user_optimum = 100.0

    # WHEN
    slack_tight = _PyomoSolverCls._lex_cap_slack(user_optimum, {"MIPGap": 0.0})
    slack_loose = _PyomoSolverCls._lex_cap_slack(user_optimum, {"MIPGap": 1e-2})

    # THEN
    assert slack_tight < 1e-3

    # 1 % of 100 = 1.0 plus floor
    assert slack_loose >= 1.0


def test_lex_cap_slack_floors_at_one_for_small_optima():
    # GIVEN
    user_optimum = 0.0

    # WHEN
    slack = _PyomoSolverCls._lex_cap_slack(user_optimum, {"MIPGap": 1e-2})

    # THEN
    # max(1, |S*|) floor avoids zero slack: 1 % of max(1, 0) = 0.01
    assert slack >= 0.01 - 1e-12


# --------------------------------------------------------------------------- #
# 2. OptimizationProblem default + flag plumbing
# --------------------------------------------------------------------------- #


def test_optimization_problem_default_lex_off():
    # GIVEN

    # WHEN
    prob = OptimizationProblem()

    # THEN
    assert prob.lex_objectives is False


def test_optimization_problem_lex_flag_passthrough():
    # GIVEN

    # WHEN
    prob = OptimizationProblem(lex_objectives=True)

    # THEN
    assert prob.lex_objectives is True


def test_create_problem_lex_flag_threads_to_optimization_problem():
    # GIVEN

    # WHEN
    prob = create_min_load_shedding_problem(lex_objectives=True)
    prob_default = create_min_load_shedding_problem()

    # THEN
    assert prob.lex_objectives is True
    assert prob_default.lex_objectives is False


# --------------------------------------------------------------------------- #
# 3. Solve-based behavioural tests (require Gurobi / MISOCP-capable solver)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
def test_lex_matches_legacy_on_unstressed_network():
    # GIVEN
    net_legacy = _build_unstressed_net()
    prob_legacy = _make_shedding_problem(lex_objectives=False)
    net_lex = _build_unstressed_net()
    prob_lex = _make_shedding_problem(lex_objectives=True)

    # WHEN
    res_legacy = _PyomoSolverCls(SOLVER).solve(
        net_legacy, optimization_problem=prob_legacy
    )
    res_lex = _PyomoSolverCls(SOLVER).solve(net_lex, optimization_problem=prob_lex)

    # THEN
    assert res_legacy.success
    assert res_lex.success

    # Both modes recover full demand (regulation ~ 1 everywhere, shed ~ 0)
    shed_legacy = _user_obj_value(res_legacy.network)
    shed_lex = _user_obj_value(res_lex.network)
    assert shed_legacy < 1e-3
    assert shed_lex < 1e-3


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
def test_lex_user_objective_is_optimal_independent_of_aux_scale():
    # GIVEN
    aux_scale = 50.0

    # WHEN
    res_legacy = _solve_with_scaled_aux(lex=False, aux_scale=aux_scale)
    res_lex = _solve_with_scaled_aux(lex=True, aux_scale=aux_scale)

    # THEN
    assert res_legacy.success
    assert res_lex.success

    # Lex phase 1 sees only user terms, so shed sits at its true optimum (~0)
    shed_legacy = _user_obj_value(res_legacy.network)
    shed_lex = _user_obj_value(res_lex.network)
    assert shed_lex < 1e-3, (
        f"Lex phase-1 should pin shed at its true optimum (~0), got {shed_lex}"
    )

    # Lex's shed cannot exceed legacy's (lex is a tighter user-tier optimisation)
    assert shed_lex <= shed_legacy + 1e-6
