"""Tests for the deterministic auto-priority-floor on ``demand_weight``."""

import pytest

from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.network import create_restoration_benchmark, create_urban_district_net
from monee.problem.min_load_shedding import (
    _aux_objective_upper_bound,
    create_min_load_shedding_problem,
)
from monee.solver.pyo import PyomoSolver
from tests.util import solver_available as _solver_available

SOLVER = "gurobi" if _solver_available("gurobi") else None


# --------------------------------------------------------------------------- #
# 1. Helper math - no solver dependency
# --------------------------------------------------------------------------- #


def test_aux_upper_bound_finite_and_positive_on_urban():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)

    # WHEN
    a_max = _aux_objective_upper_bound(net)

    # THEN
    assert a_max > 0
    assert a_max < 1e9


def test_aux_upper_bound_scales_with_network_size():
    # GIVEN
    small = create_urban_district_net()
    small.apply_formulation(EL_MISOCP_FORMULATION)
    big = create_restoration_benchmark(misocp=True)

    # WHEN
    bound_small = _aux_objective_upper_bound(small)
    bound_big = _aux_objective_upper_bound(big)

    # THEN
    assert bound_big >= bound_small


def test_aux_upper_bound_zero_on_bare_network():
    # GIVEN
    import monee.model as mm

    net = mm.Network(mm.create_power_grid("p"))
    net.node(mm.Bus(base_kv=20))

    # WHEN
    a_max = _aux_objective_upper_bound(net)

    # THEN
    # a network with no branches/junctions contributes 0
    assert a_max == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Plumbing - flag passthrough & default-on behaviour
# --------------------------------------------------------------------------- #


def test_auto_priority_floor_on_by_default():
    # GIVEN
    prob_default = create_min_load_shedding_problem()

    # WHEN
    prob_off = create_min_load_shedding_problem(auto_priority_floor=False)

    # THEN
    # auto_priority_floor (default True) registers exactly one extra
    # appliable (the hook) compared to an explicitly disabled problem
    assert (
        len(prob_default._controllable_appliables)
        == len(prob_off._controllable_appliables) + 1
    )


def test_auto_priority_floor_raises_low_user_weight():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    # The hook caps the bound with the problem's default max_line_loading
    # (1.5, active because check_lp defaults to True); the capped
    # bound stays valid since the loading constraint bounds current_pu_squared.
    a_max = _aux_objective_upper_bound(net, max_line_loading=1.5)
    prob = create_min_load_shedding_problem(
        demand_weight=1.0,
        generator_weight=1e-4,
        auto_priority_floor=True,
        priority_safety_factor=10.0,
    )

    # WHEN
    prob._apply(net)

    # THEN
    from monee.model import PowerLoad

    probe = _probe_model(net, PowerLoad)
    assert probe is not None, "expected a PowerLoad in the urban net"

    # user demand_weight=1 must be bumped to >= α·A_max
    expected_floor = 10.0 * a_max
    weight = _objective_weight(prob, probe)
    assert weight == pytest.approx(max(1.0, expected_floor), rel=1e-9)


def test_auto_priority_floor_honours_user_floor_when_higher():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    a_max = _aux_objective_upper_bound(net)
    user_demand = 100.0 * a_max  # well above α·A_max=10·A_max
    prob = create_min_load_shedding_problem(
        demand_weight=user_demand,
        generator_weight=0.1,
        auto_priority_floor=True,
        priority_safety_factor=10.0,
    )

    # WHEN
    prob._apply(net)

    # THEN
    from monee.model import PowerLoad

    probe = _probe_model(net, PowerLoad)

    # a large user-supplied demand_weight must not be lowered
    weight = _objective_weight(prob, probe)
    assert weight == pytest.approx(user_demand, rel=1e-9)


def test_auto_priority_floor_preserves_demand_generator_ratio():
    # GIVEN
    net = create_urban_district_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    user_ratio = 1e-4  # generator / demand
    prob = create_min_load_shedding_problem(
        demand_weight=1.0,
        generator_weight=user_ratio,
        auto_priority_floor=True,
        priority_safety_factor=10.0,
    )

    # WHEN
    prob._apply(net)

    # THEN
    from monee.model import PowerGenerator, PowerLoad

    load_probe = _probe_model(net, PowerLoad)
    gen_probe = _probe_model(net, PowerGenerator)
    assert load_probe is not None and gen_probe is not None

    # scaling demand also scales generator so the ratio survives
    w_load = _objective_weight(prob, load_probe)
    w_gen = _objective_weight(prob, gen_probe)
    assert w_gen / w_load == pytest.approx(user_ratio, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. Behavioural test - auto floor protects weighted mode under aux scaling
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
def test_auto_floor_keeps_user_objective_optimal_under_aux_scale():
    # GIVEN
    AUX_SCALE = 50.0

    from monee.model.formulation.miqcqp.convex.el import (
        MISOCPElectricityBranchFormulation,
    )

    orig_minimize = MISOCPElectricityBranchFormulation.minimize

    def scaled_minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        inner = orig_minimize(
            self, branch, grid, from_node_model, to_node_model, **kwargs
        )
        return [AUX_SCALE * e for e in inner]

    MISOCPElectricityBranchFormulation.minimize = scaled_minimize
    try:
        net = create_urban_district_net()
        net.apply_formulation(EL_MISOCP_FORMULATION)
        prob = create_min_load_shedding_problem(
            bounds_vm=(0.5, 1.5),
            bounds_pressure=(0.5, 1.5),
            bounds_t=(0.5, 1.5),
            include_ext_grids=False,
            include_storages=False,
            demand_weight=1.0,
            generator_weight=1e-4,
            auto_priority_floor=True,
            priority_safety_factor=10.0,
        )

        # WHEN
        res = PyomoSolver(SOLVER).solve(net, optimization_problem=prob)

        # THEN
        assert res.success

        # on an unstressed network shed ≈ 0; a dominating aux would inflate it
        shed_total = _total_power_load_shed(res.network)
        assert shed_total < 1e-3, (
            f"auto-floor failed to dominate aux×{AUX_SCALE} - "
            f"shed = {shed_total}, expected ~0"
        )
    finally:
        MISOCPElectricityBranchFormulation.minimize = orig_minimize


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def network_models(network):
    return [c.model for c in network.all_components()]


def _probe_model(network, model_type):
    """Return the first model of the given type in the network, or None."""
    return next((m for m in network_models(network) if isinstance(m, model_type)), None)


def _objective_weight(prob, model):
    """Recover a model's weight via the objective's data_attacher closure.

    ``_data_attacher`` returns ``(weight, gas_factor, cp_rated)``; the
    weight is element 0.
    """
    return prob.objectives._objectives[0]._data_attacher(model)[0]


def _total_power_load_shed(network):
    """Sum p_mw * (1 - regulation) over all PowerLoad models."""
    from monee.model import PowerLoad
    from monee.model.core import Var

    shed_total = 0.0
    for comp in network.all_components():
        m = comp.model
        reg = getattr(m, "regulation", 1)
        reg_val = reg.value if isinstance(reg, Var) else reg
        if isinstance(m, PowerLoad) and reg_val is not None:
            shed_total += m.p_mw * (1 - reg_val)
    return shed_total
