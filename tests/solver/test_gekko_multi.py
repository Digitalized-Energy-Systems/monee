import math

import monee.express as mx
import monee.model as mm
import monee.problem as mp
import monee.solver as ms


def create_two_line_example_with_2_pipe_example_p2g(source_flow=0.1):
    pn = mm.Network(mm.create_power_grid("power"))

    # POWER
    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=2, q_mvar=0)),
        ],
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0))],
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
    )

    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=source_flow))],
        grid=gas_grid,
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=0.2))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(diameter_m=0.5, length_m=100),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(diameter_m=0.5, length_m=150),
        g_node_0,
        g_node_2,
    )

    # MULTI
    pn.branch(
        mm.PowerToGas(efficiency=0.95, mass_flow_setpoint=0.05),
        el_node_0,
        g_node_0,
    )
    return pn


def create_g2h_net():
    pn = mm.Network()

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=1))],
        grid=gas_grid,
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_2,
    )

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    w_node_1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_3 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        w_node_0,
        w_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_3,
        w_node_2,
    )

    # multi
    mx.create_g2h(
        pn,
        gas_node_id=g_node_2,
        heat_node_id=w_node_2,
        heat_return_node_id=w_node_1,
        heat_energy_w=10000,
        diameter_m=0.4,
        efficiency=0.9,
    )

    return pn


def create_multi_chp():
    pn = mm.Network()

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    w_node_1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_3 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        w_node_0,
        w_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_3,
        w_node_2,
    )

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=1))],
        grid=gas_grid,
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_2,
    )

    # POWER
    power_grid = mm.create_power_grid("power")
    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0)),
        ],
        grid=power_grid,
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=power_grid,
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
        grid=power_grid,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.000007, x_ohm_per_m=0.000007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.000007, x_ohm_per_m=0.000007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )

    # multi
    pn.compound(
        mm.CHP(
            0.5,
            0.6,
            0.4,
            0.00005,
            regulation=0.5
        ),
        gas_node_id=g_node_2,
        heat_node_id=w_node_1,
        heat_return_node_id=w_node_2,
        power_node_id=el_node_2,
    )

    return pn


def create_in_line_p2h():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    w_node_1 = pn.node(mm.Junction())
    w_node_2 = pn.node(mm.Junction())
    w_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        w_node_1,
        w_node_0,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_2,
        w_node_3,
    )

    # POWER
    power_grid = mm.create_power_grid("power")
    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0)),
        ],
        grid=power_grid,
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=power_grid,
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
        grid=power_grid,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )

    # multi
    pn.compound(
        mm.PowerToHeat(0.1, 0.15, 300, 1),
        power_node_id=el_node_2,
        heat_node_id=w_node_2,
        heat_return_node_id=w_node_1,
    )
    return pn


def create_generic_transfer_el():
    pn = mm.Network(mm.create_power_grid("power"))

    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0)),
        ],
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
    )
    el_node_3 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )
    pn.branch(
        mm.GenericTransferBranch(),
        el_node_2,
        el_node_3,
    )

    return pn


def create_generic_transfer_gas():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))

    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=1))],
    )
    g_node_1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    g_node_2 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))])
    g_node_3 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))])

    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=1000, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=1500, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_2,
    )
    pn.branch(mm.GenericTransferBranch(), g_node_2, g_node_3)
    return pn


def create_generic_transfer_heat():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    w_node_1 = pn.node(mm.Junction())
    w_node_2 = pn.node(mm.Junction())
    w_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        w_node_0,
        w_node_1,
    )
    pn.branch(mm.GenericTransferBranch(), w_node_1, w_node_2)
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_2,
        w_node_3,
    )

    return pn


def test_small_p2g_network():
    multi_energy_network = create_two_line_example_with_2_pipe_example_p2g()

    result = ms.GEKKOSolver().solve(multi_energy_network)
    print(result)
    assert len(result.dataframes) == 11
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -2.5219502259)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.05)


def test_in_line_p2h():
    multi_energy_network = create_in_line_p2h()

    result = ms.GEKKOSolver().solve(multi_energy_network)
    print(result)
    assert len(result.dataframes) == 12
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][0], 580.35906866, abs_tol=0.001
    )


def test_load_shedding_p2g_network():
    multi_energy_network = create_two_line_example_with_2_pipe_example_p2g(
        source_flow=1
    )
    load_shedding_problem = mp.create_load_shedding_optimization_problem(
        ext_grid_el_bounds=(0, 0), ext_grid_gas_bounds=(-0.0, 0.0)
    )

    result = ms.GEKKOSolver().solve(
        multi_energy_network, optimization_problem=load_shedding_problem
    )
    print(result)
    assert len(result.dataframes) == 11
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0, abs_tol=0.001
    )
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], 0, abs_tol=0.001)


def test_generic_transfer_el():
    multi_energy_network = create_generic_transfer_el()

    result = ms.GEKKOSolver().solve(multi_energy_network)

    assert len(result.dataframes) == 6


def test_generic_transfer_gas():
    multi_energy_network = create_generic_transfer_gas()

    result = ms.GEKKOSolver().solve(multi_energy_network)

    assert len(result.dataframes) == 6


def test_generic_transfer_heat():
    multi_energy_network = create_generic_transfer_heat()

    result = ms.GEKKOSolver().solve(multi_energy_network)

    assert len(result.dataframes) == 5


def test_simple_chp():
    multi_energy_network = create_multi_chp()

    result = ms.GEKKOSolver().solve(multi_energy_network)
    print(result)

    assert len(result.dataframes) == 14
    assert math.isclose(
        result.dataframes["ExtPowerGrid"]["p_mw"][0], -0.0059846779661, abs_tol=0.001
    )
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][1],
        -2.5e-05,
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][1], 354.58134681, abs_tol=0.001
    )


def test_simple_g2h():
    multi_energy_network = create_g2h_net()

    result = ms.GEKKOSolver().solve(multi_energy_network)
    print(result)
    assert len(result.dataframes) == 9
    assert math.isclose(result.dataframes["Junction"]["t_k"][3], 373.5426926)


def test_network_convenience_methods():
    multi_energy_network = create_multi_chp()

    multi_energy_network.activate_by_id(mm.Node, 0)
    multi_energy_network.activate_by_id(mm.Branch, (1, 0, 0))
    multi_energy_network.activate_by_id(mm.Compound, 0)
    multi_energy_network.activate_by_id(mm.Child, 0)
    m = multi_energy_network.all_models_with_grid()

    assert len(m) == 30

    assert multi_energy_network.has_child(0)
    multi_energy_network.remove_child(0)
    assert len(multi_energy_network.childs) == 7
    assert len(multi_energy_network.childs_by_type(mm.Sink)) == 1
    assert len(multi_energy_network.compounds_by_type(mm.CHP)) == 1
    assert len(multi_energy_network.branches_by_ids([(1, 0, 0)])) == 1
    assert multi_energy_network.get_branch_between(1, 0) is not None
    assert multi_energy_network.has_branch_between(1, 0)
    assert len(multi_energy_network.components_connected_to(5)) == 2
    assert len(multi_energy_network.branches_connected_to(5)) == 1
    assert len(multi_energy_network.compounds_connected_to(2)) == 1
