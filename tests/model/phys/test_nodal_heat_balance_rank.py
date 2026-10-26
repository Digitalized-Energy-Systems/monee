"""The nodal heat balance must not leave T_n unconstrained at zero mass flow.

``Node.calc_signed_heat_flow`` builds the balance from terms that are each
multiplied by a mass flow, then adds a conduction-style regulariser scaled by
``grid.node_heat_reg_kgs`` to keep d(heat_bal)/dT_n non-zero when those terms
vanish. That attribute was read via ``getattr(..., 0.0)`` and **no grid class
declared it**, so the regulariser was unreachable and a junction with no flow
had its temperature pinned only by the ``t_pu in [0.3, 2.0]`` Var box.

Formulations are pinned explicitly rather than left to the network default,
because the defect is formulation-specific: ``HEAT_NONCONVEX_MIQCQP`` (the
member of ``DEFAULT_SIMULATION_FORMULATION``, and what a simulated net solves)
degenerates, while the McCormick heat MILP reformulates the balance and never
exposes it.
"""

import pytest

import monee.model as mm
import monee.solver as ms
from monee.model.formulation.bundles import (
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
    make_heat_convex_milp_formulation,
)

T_REF_K = 356.0
VAR_FLOOR_T_K = 0.3 * T_REF_K  # 106.8


def _net(dead_leg_flow, node_heat_reg_kgs, formulation):
    grid = mm.create_water_grid("water")
    grid.node_heat_reg_kgs = node_heat_reg_kgs
    net = mm.Network(grid)
    src = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.ExtHydrGrid(pressure_pu=1.0, t_k=T_REF_K))],
    )
    live = net.node(mm.Junction(), child_ids=[net.child(mm.Sink(mass_flow_kgs=0.1))])
    dead = net.node(
        mm.Junction(), child_ids=[net.child(mm.Sink(mass_flow_kgs=dead_leg_flow))]
    )
    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=100), src, live)
    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=100), src, dead)
    net.apply_formulation(formulation)
    return net, live, dead


def _t_k(net, node_id):
    node = next(n for n in net.nodes if n.id == node_id)
    vals = getattr(node.model, "values", {}) or {}
    if "t_k" in vals:
        return float(vals["t_k"])
    return float(vals["t_pu"]) * T_REF_K


def _solve(net):
    return getattr(ms.PyomoSolver().solve(net), "network", net)


def test_water_grid_declares_the_regulariser_the_balance_reads():
    """The bug was an attribute nobody defined; keep it declared and defaulted."""
    grid = mm.create_water_grid("water")
    assert hasattr(grid, "node_heat_reg_kgs")
    assert grid.node_heat_reg_kgs == 0.0, "default must preserve prior behaviour"


def test_zero_flow_junction_pins_at_the_var_floor_without_the_regulariser():
    """The defect itself, under the formulation a simulated net actually uses.

    106.8 K is not a temperature — it is ``0.3 * t_ref_k``, the Var box bound.
    """
    net, live, dead = _net(0.0, 0.0, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    solved = _solve(net)
    assert _t_k(solved, dead) == pytest.approx(VAR_FLOOR_T_K, abs=1e-3)
    # The determined leg is unaffected: local degeneracy, not a broken solve —
    # which is why it survived unnoticed.
    assert 340.0 < _t_k(solved, live) < T_REF_K


def test_regulariser_restores_a_physical_temperature_at_zero_flow():
    net, _, dead = _net(0.0, 1e-6, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    assert _t_k(_solve(net), dead) == pytest.approx(T_REF_K, abs=0.05)


def test_coefficient_must_clear_a_threshold_to_bite():
    """1e-8 leaves the node at the floor; the fix is not "any epsilon works"."""
    net, _, dead = _net(0.0, 1e-8, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    assert _t_k(_solve(net), dead) == pytest.approx(VAR_FLOOR_T_K, abs=1e-3)


def test_regulariser_barely_perturbs_a_determined_node():
    """+2.2e-5 K at 1e-6 — far below any band these readings are graded on."""
    base, live_b, _ = _net(0.0, 0.0, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    reg, live_r, _ = _net(0.0, 1e-6, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    assert abs(_t_k(_solve(reg), live_r) - _t_k(_solve(base), live_b)) < 1e-3


def test_a_flowing_leg_is_determined_either_way():
    """Control: with real flow the balance has rank and both agree."""
    off, _, dead_off = _net(0.1, 0.0, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    on, _, dead_on = _net(0.1, 1e-6, HEAT_NONCONVEX_MIQCQP_FORMULATION)
    t_off, t_on = _t_k(_solve(off), dead_off), _t_k(_solve(on), dead_on)
    assert t_off > 340.0 and abs(t_on - t_off) < 1e-3


def test_mccormick_heat_milp_never_exposes_the_degeneracy():
    """Why the oracle arm shows zero sub-ambient readings while the sim does."""
    milp = make_heat_convex_milp_formulation(
        num_partitions=4, include_heat_exchangers=False
    )
    for k_reg in (0.0, 1e-6):
        net, _, dead = _net(0.0, k_reg, milp)
        assert _t_k(_solve(net), dead) == pytest.approx(T_REF_K, abs=0.05)
