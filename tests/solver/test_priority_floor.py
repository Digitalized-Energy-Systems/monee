"""Tests for the deterministic auto-priority-floor on ``demand_weight``.

The floor protects the weighted-sum solve from network-size growth:
the formulation-level tightening objective scales with branch count,
so a fixed ``demand_weight=1e3`` may be over-run on large networks.
``auto_priority_floor=True`` computes an upper bound on the aux
contribution from the actual network and sets the demand weight to
``α · A_max`` (defaulting to ``α=10``).
"""

import pytest

from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network import create_restoration_benchmark, create_urban_district_net
from monee.problem.min_load_shedding import (
    _aux_objective_upper_bound,
    create_min_load_shedding_problem,
)
from monee.solver.pyo import PyomoSolver


def _solver_available(name: str) -> bool:
    import pyomo.environ as pyo

    try:
        return pyo.SolverFactory(name).available(exception_flag=False)
    except Exception:
        return False


SOLVER = "gurobi" if _solver_available("gurobi") else None


# --------------------------------------------------------------------------- #
# 1. Helper math - no solver dependency
# --------------------------------------------------------------------------- #


def test_aux_upper_bound_finite_and_positive_on_urban():
    """The bound is well-defined and non-trivial on a real network."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    a_max = _aux_objective_upper_bound(net)
    assert a_max > 0
    assert a_max < 1e9


def test_aux_upper_bound_scales_with_network_size():
    """Larger nets must have larger (or equal) bounds."""
    small = create_urban_district_net()
    small.apply_formulation(MISOCP_NETWORK_FORMULATION)
    big = create_restoration_benchmark(misocp=True)
    assert _aux_objective_upper_bound(big) >= _aux_objective_upper_bound(small)


def test_aux_upper_bound_zero_on_bare_network():
    """A network with no branches/junctions contributes 0."""
    import monee.model as mm

    net = mm.Network(mm.create_power_grid("p"))
    # Single bus, no branches.
    net.node(mm.Bus(base_kv=20))
    assert _aux_objective_upper_bound(net) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# 2. Plumbing - flag passthrough & opt-in default
# --------------------------------------------------------------------------- #


def test_auto_priority_floor_off_by_default():
    """Default behaviour is unchanged - opt-in only.

    Compare the number of registered ``_controllable_appliables``
    between a default problem and one with ``auto_priority_floor=True``;
    the latter must have exactly one more entry (the auto-floor hook).
    """
    prob_default = create_min_load_shedding_problem()
    prob_auto = create_min_load_shedding_problem(auto_priority_floor=True)
    assert (
        len(prob_auto._controllable_appliables)
        == len(prob_default._controllable_appliables) + 1
    )


def test_auto_priority_floor_raises_low_user_weight():
    """User demand_weight=1 must be bumped to ≥ α·A_max on a real network."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    a_max = _aux_objective_upper_bound(net)

    prob = create_min_load_shedding_problem(
        demand_weight=1.0,
        generator_weight=1e-4,
        auto_priority_floor=True,
        priority_safety_factor=10.0,
    )
    prob._apply(net)
    # The mutable weight dict is captured by the closure; we can
    # peek at it via the objectives expression by re-invoking
    # _aux_objective_upper_bound and asserting the floor relation
    # implicitly via weight_fn.  Indirect path:
    from monee.model import PowerLoad

    # Iterate models, find one PowerLoad, then call weight_fn-like
    # via the data_attacher: ``_calc_objective`` uses ``data[0]``.
    # We just compare the demand weight to the expected floor.
    expected_floor = 10.0 * a_max
    # Look up the demand weight via the objective's data_attacher.
    # The Objectives class stores a single Objective with a
    # _data_attacher pointing at the dict-bundling closure.  Drive
    # it on a PowerLoad to recover its weight.
    objective = prob.objectives._objectives[0]
    # Find any PowerLoad in the network as a probe.
    probe = next((m for m in network_models(net) if isinstance(m, PowerLoad)), None)
    assert probe is not None, "expected a PowerLoad in the urban net"
    weight, _ = objective._data_attacher(probe)
    assert weight == pytest.approx(max(1.0, expected_floor), rel=1e-9)


def test_auto_priority_floor_honours_user_floor_when_higher():
    """A large user-supplied demand_weight must NOT be lowered."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    a_max = _aux_objective_upper_bound(net)
    user_demand = 100.0 * a_max  # well above α·A_max=10·A_max

    prob = create_min_load_shedding_problem(
        demand_weight=user_demand,
        generator_weight=0.1,
        auto_priority_floor=True,
        priority_safety_factor=10.0,
    )
    prob._apply(net)

    from monee.model import PowerLoad

    probe = next((m for m in network_models(net) if isinstance(m, PowerLoad)), None)
    objective = prob.objectives._objectives[0]
    weight, _ = objective._data_attacher(probe)
    assert weight == pytest.approx(user_demand, rel=1e-9)


def test_auto_priority_floor_preserves_demand_generator_ratio():
    """Scaling demand also scales generator so the ratio survives."""
    net = create_urban_district_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)

    user_ratio = 1e-4  # generator / demand
    prob = create_min_load_shedding_problem(
        demand_weight=1.0,
        generator_weight=user_ratio,
        auto_priority_floor=True,
        priority_safety_factor=10.0,
    )
    prob._apply(net)

    from monee.model import PowerGenerator, PowerLoad

    load_probe = next(
        (m for m in network_models(net) if isinstance(m, PowerLoad)), None
    )
    gen_probe = next(
        (m for m in network_models(net) if isinstance(m, PowerGenerator)),
        None,
    )
    assert load_probe is not None and gen_probe is not None

    objective = prob.objectives._objectives[0]
    w_load, _ = objective._data_attacher(load_probe)
    w_gen, _ = objective._data_attacher(gen_probe)
    assert w_gen / w_load == pytest.approx(user_ratio, rel=1e-9)


# --------------------------------------------------------------------------- #
# 3. Behavioural test - auto floor protects weighted mode under aux scaling
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(SOLVER is None, reason="Gurobi (MISOCP) not available")
def test_auto_floor_keeps_user_objective_optimal_under_aux_scale():
    """With auto_priority_floor=True, weighted mode is robust to aux scaling.

    Counterpart to ``test_lex_user_objective_is_optimal_independent_of_aux_scale``
    (which exercises lex mode).  Here we test the *weighted* mode
    with the auto-priority floor enabled: scaling the MISOCP aux
    term ×50 must not pull the user objective off its true optimum
    on an unstressed network.
    """
    AUX_SCALE = 50.0

    from monee.model.formulation.misoc.el import (
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
        net.apply_formulation(MISOCP_NETWORK_FORMULATION)
        prob = create_min_load_shedding_problem(
            bounds_el=(0.5, 1.5),
            bounds_gas=(0.5, 1.5),
            bounds_heat=(0.5, 1.5),
            include_ext_grids=False,
            include_storages=False,
            demand_weight=1.0,  # tiny - would be insufficient without auto
            generator_weight=1e-4,
            auto_priority_floor=True,
            priority_safety_factor=10.0,
        )
        res = PyomoSolver(SOLVER).solve(net, optimization_problem=prob)
        assert res.success

        # On an unstressed network the user objective (shed) at the
        # optimum is 0.  If the aux had dominated, regulation
        # would have been pulled below 1 to reduce losses,
        # inflating shed.  Verify shed ≈ 0.
        shed_total = 0.0
        for comp in res.network.all_components():
            m = comp.model
            from monee.model import PowerLoad
            from monee.model.core import Var

            reg = getattr(m, "regulation", 1)
            reg_val = reg.value if isinstance(reg, Var) else reg
            if isinstance(m, PowerLoad) and reg_val is not None:
                shed_total += m.p_mw * (1 - reg_val)
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
