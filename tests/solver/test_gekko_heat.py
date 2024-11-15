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


def create_rect_he_heat_example():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=1))],
    )
    g_node_3 = pn.node(
        mm.Junction(),
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_3,
        g_node_2,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=-0.001, diameter_m=0.1),
        g_node_0,
        g_node_2,
    )

    pn.branch(
        mm.HeatExchanger(q_mw=-0.001, diameter_m=0.1),
        g_node_1,
        g_node_3,
    )
    return pn


def create_ext_branching_heat_example():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
    )
    g_node_2 = pn.node(
        mm.Junction(),
    )
    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=0.3))],
    )
    g_node_4 = pn.node(
        mm.Junction(),
    )
    g_node_5 = pn.node(
        mm.Junction(),
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_3,
        g_node_4,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_4,
        g_node_5,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.001, diameter_m=0.1),
        g_node_1,
        g_node_4,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.001, diameter_m=0.1),
        g_node_2,
        g_node_5,
    )
    return pn


def create_ext_branching_heat_example_t():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
    )
    g_node_1_1 = pn.node(
        mm.Junction(),
    )
    g_node_2 = pn.node(
        mm.Junction(),
    )
    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=0.3))],
    )
    g_node_4 = pn.node(
        mm.Junction(),
    )
    g_node_4_1 = pn.node(
        mm.Junction(),
    )
    g_node_5 = pn.node(
        mm.Junction(),
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_1,
        g_node_1_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_3,
        g_node_4,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_4,
        g_node_5,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_4,
        g_node_4_1,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.001, diameter_m=0.1),
        g_node_1,
        g_node_4,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.001, diameter_m=0.1),
        g_node_2,
        g_node_5,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.001, diameter_m=0.1),
        g_node_1_1,
        g_node_4_1,
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
        g_node_2,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_2,
        g_node_3,
    )
    return pn


def create_line_heating_with_dead_end():
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
    g_node_4 = pn.node(mm.Junction())

    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_2,
        g_node_3,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_3,
        g_node_4,
    )
    return pn


def create_circular_heating_net():
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
    g_node_4 = pn.node(mm.Junction())

    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_2,
        g_node_3,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_3,
        g_node_4,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        g_node_4,
        g_node_0,
    )
    return pn


def test_two_pipes_heat_network():
    heat_net = create_branching_two_pipe_heat_example()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.5)
    assert len(result.dataframes) == 4


def test_rect_he_heat_network():
    heat_net = create_rect_he_heat_example()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 1)
    assert len(result.dataframes) == 5


def test_ext_branching_pipes_heat_network():
    heat_net = create_ext_branching_heat_example()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.3)
    assert math.isclose(result.dataframes["Junction"]["pressure_pa"][4], 1000021.5258)
    assert math.isclose(result.dataframes["Junction"]["t_k"][4], 360.56008779)
    assert len(result.dataframes) == 5


def test_ext_branching_pipes_heat_network_t():
    heat_net = create_ext_branching_heat_example_t()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.3)
    assert math.isclose(result.dataframes["Junction"]["pressure_pa"][6], 1000024.8653)
    assert math.isclose(result.dataframes["Junction"]["t_k"][6], 361.36877208)
    assert len(result.dataframes) == 5


def test_heat_exchanger():
    heat_net = create_two_pipes_with_he_no_branching()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.1)
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 359.224812)
    assert len(result.dataframes) == 5


def test_dead_end():
    heat_net = create_line_heating_with_dead_end()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.1)
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 358.97637437)
    assert len(result.dataframes) == 4
