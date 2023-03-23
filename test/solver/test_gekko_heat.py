import math

import monee.model as mm
import monee.solver as ms


def create_branching_two_pipe_heat_example():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.2))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.3))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_0,
        g_node_2,
    )
    return pn


def create_two_pipes_with_he_no_branching():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    g_node_1 = pn.node(mm.Junction())
    g_node_2 = pn.node(mm.Junction())
    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=1 * 10**-4, diameter_m=0.15),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_2,
        g_node_3,
    )
    return pn


def test_two_pipes_heat_network():
    heat_net = create_branching_two_pipe_heat_example()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.5)
    assert len(result.dataframes) == 4


def test_heat_exchanger():
    gas_net = create_two_pipes_with_he_no_branching()
    result = ms.GEKKOSolver().solve(gas_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.1)
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 359.224812)
    assert len(result.dataframes) == 5
