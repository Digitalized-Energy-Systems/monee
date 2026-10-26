"""Flow-direction repair for the unidirectional McCormick-DHS pipes.

The relaxation pins each pipe to its as-built ``from -> to`` direction, so a
junction whose inbound pipe is switched off can only be re-fed by reversing the
pipes below it. Without the repair the nodal mass balance conflicts with
``mass_flow_neg_kgs >= 0`` and the model is infeasible.
"""

import math

import monee.express as mx
import monee.model as mm
import monee.solver as ms
from monee.model.formulation import orient_unidirectional_water_pipes
from monee.model.formulation.milp.heat import REVERSE_ATTR, _branch_m_u


def _build_line(n_junctions: int = 4):
    """``ext -> j0 -> j1 -> ... -> jn``, every junction drawing mass."""
    net = mm.Network()
    juncs = [mx.create_water_junction(net) for _ in range(n_junctions)]
    for a, b in zip(juncs[:-1], juncs[1:]):
        mx.create_water_pipe(
            net, from_node_id=a, to_node_id=b, diameter_m=0.15, length_m=100.0
        )
    mx.create_water_ext_grid(net, juncs[0], t_k=360.0)
    for junc in juncs[1:]:
        mx.create_water_sink(net, junc, mass_flow_kgs=0.5)
    grid = list(net.grids)[0]
    grid.t_pu_min_env = 0.82
    grid.t_pu_max_env = 1.15
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))
    return net, juncs


def _reversed_ids(net):
    return [b.id for b in net.branches if getattr(b.model, REVERSE_ATTR, False)]


def test_design_orientation_is_left_untouched():
    # GIVEN a network whose as-built directions already supply every draw
    net, _ = _build_line()

    # WHEN
    flipped = orient_unidirectional_water_pipes(net)

    # THEN nothing is marked, so the model is unchanged
    assert flipped == []
    assert _reversed_ids(net) == []


def test_starved_junction_reverses_the_pipes_below_it():
    # GIVEN the first pipe is cut, so j1..j3 can only be fed backwards from j3
    net, juncs = _build_line()
    mx.create_water_pipe(
        net,
        from_node_id=juncs[0],
        to_node_id=juncs[-1],
        diameter_m=0.15,
        length_m=100.0,
    )
    net.branch_by_id((juncs[0], juncs[1], 0)).active = False

    # WHEN
    flipped = orient_unidirectional_water_pipes(net)

    # THEN only the pipes on the path from the surviving feed are reversed
    assert set(flipped) == {
        (juncs[1], juncs[2], 0),
        (juncs[2], juncs[3], 0),
    }


def test_severed_island_marks_nothing():
    # GIVEN a junction with a draw and no water path to any injector
    net, _ = _build_line()
    island = mx.create_water_junction(net)
    mx.create_water_sink(net, island, mass_flow_kgs=0.5)

    # WHEN / THEN no orientation helps, so none is invented
    assert orient_unidirectional_water_pipes(net) == []


def test_reversed_pipe_solves_and_carries_flow_backwards():
    # GIVEN the design orientation leaves the sink unsuppliable
    net = mm.Network()
    a, b = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_pipe(
        net, from_node_id=b, to_node_id=a, diameter_m=0.15, length_m=100.0
    )
    mx.create_water_ext_grid(net, a, t_k=360.0)
    mx.create_water_sink(net, b, mass_flow_kgs=0.5)
    grid = list(net.grids)[0]
    grid.t_pu_min_env = 0.82
    grid.t_pu_max_env = 1.15
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))

    # WHEN
    assert orient_unidirectional_water_pipes(net) == [(b, a, 0)]
    result = ms.PyomoSolver().solve(net)

    # THEN the pipe runs a -> b: the magnitude sits on the pos half and the
    # signed flow is positive against the stored orientation.
    assert result.success
    pipes = result.get(mm.WaterPipe)
    assert math.isclose(list(pipes["mass_flow_pos_kgs"])[0], 0.5, abs_tol=1e-6)
    assert math.isclose(list(pipes["mass_flow_kgs"])[0], 0.5, abs_tol=1e-6)


def test_reversed_pipe_sends_from_its_to_node():
    # GIVEN the same reversed single-pipe network
    net = mm.Network()
    a, b = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_pipe(
        net, from_node_id=b, to_node_id=a, diameter_m=0.15, length_m=100.0
    )
    mx.create_water_ext_grid(net, a, t_k=360.0)
    mx.create_water_sink(net, b, mass_flow_kgs=0.5)
    grid = list(net.grids)[0]
    grid.t_pu_min_env = 0.82
    grid.t_pu_max_env = 1.15
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))
    orient_unidirectional_water_pipes(net)

    # WHEN
    result = ms.PyomoSolver().solve(net)

    # THEN enthalpy leaves the pinned-temperature end (a) and arrives at b, so
    # b cannot come out hotter than the source.
    assert result.success
    juncs = result.get(mm.Junction)
    t_k = list(juncs["t_k"])
    assert math.isclose(t_k[0], 360.0, abs_tol=1e-3)
    assert t_k[1] <= t_k[0] + 1e-6


def test_reorientation_ignores_every_as_built_capacity_hint():
    # GIVEN pipes carrying their as-built downstream-demand hints, one of which
    # the repair will reverse
    net, juncs = _build_line()
    mx.create_water_pipe(
        net,
        from_node_id=juncs[0],
        to_node_id=juncs[-1],
        diameter_m=0.15,
        length_m=100.0,
    )
    net.branch_by_id((juncs[0], juncs[1], 0)).active = False
    for branch in net.branches:
        branch.model.m_U_design = 0.01
    grid = list(net.grids)[0]

    # WHEN
    assert orient_unidirectional_water_pipes(net)

    # THEN the hints stop biting grid-wide: reversing re-roots the supply tree,
    # so even an untouched pipe upstream of the reconnection carries a stale one.
    # The hint itself survives on the model - dropping it would be unrecoverable.
    assert all(b.model.m_U_design == 0.01 for b in net.branches)
    assert all(_branch_m_u(b.model, grid) > 0.01 for b in net.branches)


def test_intact_topology_keeps_the_capacity_hints():
    net, _ = _build_line()
    for branch in net.branches:
        branch.model.m_U_design = 0.01
    grid = list(net.grids)[0]

    assert orient_unidirectional_water_pipes(net) == []

    assert all(_branch_m_u(b.model, grid) == 0.01 for b in net.branches)


def test_a_later_intact_pass_restores_the_capacity_hints():
    # GIVEN a net whose hints the outage pass suppressed
    net, juncs = _build_line()
    mx.create_water_pipe(
        net,
        from_node_id=juncs[0],
        to_node_id=juncs[-1],
        diameter_m=0.15,
        length_m=100.0,
    )
    outage = net.branch_by_id((juncs[0], juncs[1], 0))
    outage.active = False
    for branch in net.branches:
        branch.model.m_U_design = 0.01
    grid = list(net.grids)[0]
    assert orient_unidirectional_water_pipes(net)

    # WHEN the pipe comes back
    outage.active = True

    # THEN the tightening returns with the design orientation
    assert orient_unidirectional_water_pipes(net) == []
    assert all(_branch_m_u(b.model, grid) == 0.01 for b in net.branches)


def _he_fed_mesh(cut=True):
    """Supply mesh whose demand hangs off heat exchangers rather than sinks -
    the shape ``create_restoration_benchmark`` builds."""
    net = mm.Network()
    hs = [mx.create_water_junction(net) for _ in range(3)]
    hr = mx.create_water_junction(net)
    for a, b in ((0, 1), (1, 2), (0, 2)):
        mx.create_water_pipe(
            net, from_node_id=hs[a], to_node_id=hs[b], diameter_m=0.25, length_m=100.0
        )
    mx.create_water_ext_grid(net, hs[0], t_k=358.0)
    mx.create_consume_hydr_grid(net, hr)
    mx.create_heat_exchanger(net, hs[1], hr, 0.03)
    mx.create_heat_exchanger(net, hs[2], hr, 0.02)
    grid = list(net.grids)[0]
    grid.t_pu_min_env = 0.82
    grid.t_pu_max_env = 1.15
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))
    if cut:
        net.branch_by_id((hs[0], hs[1], 0)).active = False
    return net, hs


def test_heat_exchanger_draw_counts_as_a_junction_to_supply():
    # GIVEN hs1 is fed only through an HE draw - no Sink child names the demand
    net, hs = _he_fed_mesh()

    # WHEN
    flipped = orient_unidirectional_water_pipes(net)

    # THEN the pipe below the cut is turned around and the case solves
    assert flipped == [(hs[1], hs[2], 0)]
    assert ms.PyomoSolver().solve(net).success


def test_he_fed_mesh_is_untouched_while_intact():
    net, _ = _he_fed_mesh(cut=False)

    assert orient_unidirectional_water_pipes(net) == []


def test_a_passive_heat_exchanger_is_not_a_bidirectional_edge():
    # GIVEN both branches out of the sink point back at the source; only the
    # pipe can be reversed, the passive HE is pinned to its stored direction
    net = mm.Network()
    a, b = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_pipe(
        net, from_node_id=b, to_node_id=a, diameter_m=0.15, length_m=100.0
    )
    net.branch(
        mm.PassiveHeatExchanger(q_mw=0.001, diameter_m=0.15, length_m=10.0), b, a
    )
    mx.create_water_ext_grid(net, a, t_k=360.0)
    mx.create_water_sink(net, b, mass_flow_kgs=0.5)
    grid = list(net.grids)[0]
    grid.t_pu_min_env = 0.82
    grid.t_pu_max_env = 1.15
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))

    # WHEN / THEN counting the HE as bidirectional would report b as supplied
    assert orient_unidirectional_water_pipes(net) == [(b, a, 0)]
    assert ms.PyomoSolver().solve(net).success


def test_a_second_scenario_on_one_network_matches_a_fresh_one():
    # GIVEN two contingencies evaluated against the same network object
    net, juncs = _build_line()
    mx.create_water_pipe(
        net,
        from_node_id=juncs[0],
        to_node_id=juncs[-1],
        diameter_m=0.15,
        length_m=100.0,
    )

    def only_off(branch_id):
        for branch in net.branches:
            branch.active = branch.id != branch_id
        return orient_unidirectional_water_pipes(net)

    assert only_off((juncs[0], juncs[1], 0))

    # WHEN a scenario the design orientation already covers runs afterwards
    flipped = only_off((juncs[0], juncs[-1], 0))

    # THEN the marks from the previous scenario are gone, not carried over
    assert flipped == []
    assert _reversed_ids(net) == []
    assert ms.PyomoSolver().solve(net).success


def test_without_the_repair_the_starved_junction_is_infeasible():
    # GIVEN the sink upstream of the only pipe's direction, repair skipped
    net = mm.Network()
    a, b = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_pipe(
        net, from_node_id=b, to_node_id=a, diameter_m=0.15, length_m=100.0
    )
    mx.create_water_ext_grid(net, a, t_k=360.0)
    mx.create_water_sink(net, b, mass_flow_kgs=0.5)
    net.apply_formulation(mm.make_heat_convex_milp_formulation(num_partitions=4))

    # WHEN / THEN the mass balance at b cannot be met with m >= 0
    result = ms.PyomoSolver().solve(net)
    assert not result.success
