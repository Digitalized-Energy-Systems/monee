import math

import monee.model as mm
import monee.solver as ms


def create_branching_two_pipe_heat_example():
    pn = mm.Network()

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=30))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=3))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.56, length_m=100),
        g_node_1,
        g_node_0,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.56, length_m=100),
        g_node_0,
        g_node_2,
    )
    return pn


def create_t_heat_grid_test():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=0.1))],
    )
    g_node_mid = pn.node(
        mm.Junction(),
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.3))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.1, length_m=1000),
        g_node_0,
        g_node_mid,
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
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Source(mass_flow=5))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow=10))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.3, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.3, length_m=100),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.3, length_m=100),
        g_node_2,
        g_node_0,
    )
    return pn


def create_rect_he_heat_example():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
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
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=1))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=1))],
    )
    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=1))],
    )
    g_node_4 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=1))],
    )
    g_node_5 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=1))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.16, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.16, length_m=100),
        g_node_1,
        g_node_2,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.16, length_m=100),
        g_node_3,
        g_node_4,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.16, length_m=100),
        g_node_4,
        g_node_5,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.2, diameter_m=0.16),
        g_node_2,
        g_node_5,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.1, diameter_m=0.16),
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
    g_node_1 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=10))]
    )
    g_node_2 = pn.node(mm.Junction(), mm.WATER)
    g_node_3 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.05, diameter_m=0.15),
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
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
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
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -33)
    assert len(result.dataframes) == 4
    assert math.isclose(result.dataframes["Junction"]["pressure_pa"][2], 999999.60636)
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][2], 355.74529187, abs_tol=0.001
    )


def test_t_heat_network():
    heat_net = create_t_heat_grid_test()
    result = ms.GEKKOSolver().solve(heat_net)

    print(result)
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.2, rel_tol=1e-4
    )
    assert math.isclose(result.dataframes["Junction"]["t_k"][3], 332.37994134)
    assert len(result.dataframes) == 5


def test_circle_heat_network():
    heat_net = create_circle_heat_grid_test()
    result = ms.GEKKOSolver().solve(heat_net)

    print(result)
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][0], -5, rel_tol=1e-4
    )
    assert math.isclose(result.dataframes["Junction"]["t_k"][2], 367.18008246)
    assert len(result.dataframes) == 5


def test_heat_exchanger():
    heat_net = create_two_pipes_with_he_no_branching()
    result = ms.GEKKOSolver().solve(heat_net)

    print(result)
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.39834289356
    )
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 383.17457358)
    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][0], 999996.1481, abs_tol=0.001
    )
    assert len(result.dataframes) == 6


def test_dead_end():
    heat_net = create_line_heating_with_dead_end()
    result = ms.GEKKOSolver().solve(heat_net)

    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.1, rel_tol=1e-5
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][0], 343.43247132, abs_tol=0.01
    )
    assert len(result.dataframes) == 4


def create_supply_return_parallel_he():
    """Supply and return as 3-node chains.  One node pair is connected by two
    parallel heat exchangers; the other by a single one."""
    pn = mm.Network()

    s0 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))])
    s1 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow=1))])
    s2 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow=1))])

    r0 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=10))]
    )
    r1 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow=3))])
    r2 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow=3))])

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s0, s1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s1, s2)

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r0, r1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r1, r2)

    # Two parallel HEs between s1 and r1
    pn.branch(mm.HeatExchanger(q_mw=0.04, diameter_m=0.15), s1, r1)
    pn.branch(mm.HeatExchanger(q_mw=0.04, diameter_m=0.15), s1, r1)
    # Single HE between s2 and r2
    pn.branch(mm.HeatExchanger(q_mw=0.05, diameter_m=0.15), s2, r2)

    return pn


def create_supply_return_parallel_he_real(q_mw_coeff=1):
    """Supply and return as 3-node chains.  One node pair is connected by two
    parallel heat exchangers; the other by a single one."""
    pn = mm.Network()

    s0 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))])
    s1 = pn.node(mm.Junction(), mm.WATER)
    s2 = pn.node(mm.Junction(), mm.WATER)

    r0 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow=10))]
    )
    r1 = pn.node(mm.Junction(), mm.WATER)
    r2 = pn.node(mm.Junction(), mm.WATER)

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s0, s1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s1, s2)

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r0, r1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r1, r2)

    pn.branch(mm.HeatExchanger(q_mw=q_mw_coeff * 0.04, diameter_m=0.15), s1, r1)
    pn.branch(mm.HeatExchanger(q_mw=q_mw_coeff * 0.5, diameter_m=0.15), s2, r2)

    return pn


def test_supply_return_parallel_he():
    net = create_supply_return_parallel_he()
    result = ms.GEKKOSolver().solve(net)
    print(result)
    assert result.success


def test_supply_return_parallel_he_real():
    net = create_supply_return_parallel_he_real(q_mw_coeff=1)
    result = ms.PyomoSolver().solve(net)

    assert result.success
    assert math.isclose(result.get(mm.Junction)["t_k"][3], 385.1, abs_tol=0.09)


def test_supply_return_parallel_he_real_loads():
    net = create_supply_return_parallel_he_real(q_mw_coeff=-1)
    result = ms.PyomoSolver().solve(net)

    assert result.success
    assert math.isclose(result.get(mm.Junction)["t_k"][3], 325.4, abs_tol=0.09)
