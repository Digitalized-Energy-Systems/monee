import math

import monee.model as mm
import monee.solver as ms


def create_branching_two_pipe_heat_example():
    pn = mm.Network()

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=0.2))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        mm.WATER,
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

def create_t_heat_grid_test():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_mid = pn.node(
        mm.Junction(),
    )
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=0.1))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.3))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_mid,
        g_node_0,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_1,
        g_node_mid,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_mid,
        g_node_2,
    )
    return pn

def create_circle_heat_grid_test():
    pn = mm.Network()

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Source(mass_flow=0.1))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=0.3))],
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
        g_node_2,
        g_node_0,
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
    )
    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=1))],
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
        child_ids=[pn.child(mm.Sink(mass_flow=0.3))],
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
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )

    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    g_node_4 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    g_node_5 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
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
        mm.HeatExchangerGenerator(q_mw=-0.001, diameter_m=0.1),
        g_node_1,
        g_node_4,
    )
    pn.branch(
        mm.HeatExchangerGenerator(q_mw=-0.001, diameter_m=0.1),
        g_node_2,
        g_node_5,
    )
    return pn


def create_two_pipes_with_he_no_branching():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=0.3))],
    )
    g_node_1 = pn.node(mm.Junction(), mm.WATER)
    g_node_2 = pn.node(mm.Junction(), mm.WATER)
    g_node_3 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=-0.01, diameter_m=0.15),
        g_node_2,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.168, length_m=200),
        g_node_2,
        g_node_3,
    )
    return pn


def create_line_heating_with_dead_end():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(), 
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    g_node_1 = pn.node(mm.Junction(), mm.WATER)
    g_node_2 = pn.node(mm.Junction(), mm.WATER)
    g_node_3 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_4 = pn.node(mm.Junction(), mm.WATER)

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
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    g_node_1 = pn.node(mm.Junction(), mm.WATER)
    g_node_2 = pn.node(mm.Junction(), mm.WATER)
    g_node_3 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    g_node_4 = pn.node(mm.Junction(), mm.WATER)

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

    print(result)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.5)
    assert len(result.dataframes) == 4

def test_t_heat_network():
    heat_net = create_t_heat_grid_test()
    result = ms.GEKKOSolver().solve(heat_net)

    print(result)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.2, rel_tol=1e-4)
    assert math.isclose(result.dataframes["Junction"]["t_k"][3], 345.58940343)
    assert len(result.dataframes) == 5


def test_circle_heat_network():
    heat_net = create_circle_heat_grid_test()
    result = ms.GEKKOSolver().solve(heat_net)

    print(result)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.2, rel_tol=1e-4)
    assert math.isclose(result.dataframes["Junction"]["t_k"][3], 345.58940343)
    assert len(result.dataframes) == 5


def test_ext_branching_pipes_heat_network():
    heat_net = create_ext_branching_heat_example()
    result = ms.GEKKOSolver().solve(heat_net)
    print(result)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.3)
    assert math.isclose(result.dataframes["Junction"]["pressure_pa"][4], 999978.47416)
    assert math.isclose(result.dataframes["Junction"]["t_k"][4], 283.1684032)
    assert len(result.dataframes) == 5


def test_ext_branching_pipes_heat_network_t():
    heat_net = create_ext_branching_heat_example_t()
    result = ms.GEKKOSolver().solve(heat_net)

    print(result)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.29495210986)
    assert math.isclose(result.dataframes["Junction"]["pressure_pa"][6], 999978.47416)
    assert math.isclose(result.dataframes["Junction"]["t_k"][6], 335.69253892)
    assert len(result.dataframes) == 5
 


def test_heat_exchanger():
    heat_net = create_two_pipes_with_he_no_branching()
    result = ms.GEKKOSolver().solve(heat_net)
    print(result)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.3)
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 348.85464268)
    assert math.isclose(result.dataframes["Junction"]["pressure_pa"][0], 999995.94365)
    assert len(result.dataframes) == 5


def test_dead_end():
    heat_net = create_line_heating_with_dead_end()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.1)
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 350)
    assert len(result.dataframes) == 4
