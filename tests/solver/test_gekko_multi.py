import math
import random

import monee.express as mx
import monee.model as mm
import monee.network.mes as mes
import monee.problem as mp
import monee.solver as ms
from monee import run_energy_flow, run_energy_flow_optimization
from monee.model import Bus, ExtPowerGrid, PowerGenerator, PowerLine, PowerLoad, Source


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
        -5.0097600702e-05,
        abs_tol=0.001,
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][1], 355.8977904, abs_tol=0.001
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


def create_four_line_example():
    random.seed(9002)
    pn = mm.Network()

    node_0 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerGenerator(p_mw=0.1, q_mvar=0, regulation=0.5))],
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    node_2 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerLoad(p_mw=0.1, q_mvar=0))],
    )
    node_3 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerLoad(p_mw=0.2, q_mvar=0))],
    )
    node_4 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerLoad(p_mw=0.2, q_mvar=0))],
    )
    node_5 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerGenerator(p_mw=0.3, q_mvar=0, regulation=0.5))],
    )
    node_6 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerGenerator(p_mw=0.2, q_mvar=0, regulation=0.5))],
    )

    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_1,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_1,
        node_2,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_1,
        node_5,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_2,
        node_3,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_3,
        node_4,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_3,
        node_6,
    )

    new_mes = pn.copy()

    # gas
    bus_to_gas_junc = mes.create_gas_net_for_power(pn, new_mes, 1)
    new_mes.childs_by_type(Source)[0].model.mass_flow = -10
    new_mes.childs_by_type(Source)[0].model.regulation = 1

    # heat
    bus_index_to_junction_index, bus_index_to_end_junction_index = (
        mes.create_heat_net_for_power(pn, new_mes, 0)
    )
    new_water_junc = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc,
        mass_flow=0.075,
    )
    new_water_junc_2 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_2,
        mass_flow=0.075,
    )
    mx.create_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc,
        to_node_id=new_water_junc_2,
        diameter_m=0.20,
        q_mw=0.001,
    )
    new_water_junc_3 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_3,
        mass_flow=0.075,
    )
    mx.create_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc_2,
        to_node_id=new_water_junc_3,
        diameter_m=0.20,
        q_mw=0.001,
    )

    mx.create_p2g(
        new_mes,
        from_node_id=node_4,
        to_node_id=bus_to_gas_junc[node_4],
        efficiency=0.7,
        mass_flow_setpoint=0.005,
        regulation=0,
    )
    mx.create_chp(
        new_mes,
        power_node_id=node_1,
        heat_node_id=bus_index_to_junction_index[node_0],
        heat_return_node_id=new_water_junc,
        gas_node_id=bus_to_gas_junc[node_3],
        mass_flow_setpoint=0.0005,
        diameter_m=0.3,
        efficiency_power=0.5,
        efficiency_heat=0.5,
    )
    mx.create_g2p(
        new_mes,
        from_node_id=bus_to_gas_junc[node_1],
        to_node_id=node_1,
        efficiency=0.9,
        p_mw_setpoint=0.3,
        regulation=0,
    )
    mx.create_g2p(
        new_mes,
        from_node_id=bus_to_gas_junc[node_6],
        to_node_id=node_6,
        efficiency=0.9,
        p_mw_setpoint=1.5,
        regulation=0,
    )
    new_mes.branch(
        PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            parallel=1,
            backup=True,
        ),
        node_4,
        node_0,
    )
    new_mes.branch(
        PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            parallel=1,
            backup=True,
        ),
        node_6,
        node_2,
    )

    return new_mes


BOUND_EL = ("vm_pu", 1, 0.1)
BOUND_GAS = ("pressure_pu", 1, 0.1)
BOUND_HEAT = ("t_pu", 1, 0.1)


def test_load_shedding_four_lines():
    net_multi = create_four_line_example()

    print(run_energy_flow(net_multi))

    bounds_el = (
        BOUND_EL[1] * (1 - BOUND_EL[2]),
        BOUND_EL[1] * (1 + BOUND_EL[2]),
    )
    bounds_heat = (
        BOUND_HEAT[1] * (1 - BOUND_HEAT[2]),
        BOUND_HEAT[1] * (1 + BOUND_HEAT[2]),
    )
    bounds_gas = (
        BOUND_GAS[1] * (1 - BOUND_GAS[2]),
        BOUND_GAS[1] * (1 + BOUND_GAS[2]),
    )
    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(-0.01, -0.01),
        ext_grid_gas_bounds=(-0.01, 0.01),
        debug=True,
    )

    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    assert mp.calc_general_resilience_performance(result.network) == (0, 0, 0)
    assert result is not None


""" def test_simbench_ls_optimization():
    random.seed(42)

    net_simbench = obtain_simbench_net("1-LV-urban6--2-no_sw")
    for child in net_simbench.childs_by_type(mm.PowerGenerator):
        child.model.p_mw = child.model.p_mw * 4
    cp_density_coeff = 1.5
    net_multi = mn.generate_mes_based_on_power_net(
        net_simbench,
        heat_deployment_rate=1,
        gas_deployment_rate=1,
        p2g_density=0.1 * cp_density_coeff,
        p2h_density=0.2 * cp_density_coeff,
        chp_density=0.2 * cp_density_coeff,
    )
    net_multi.deactivate_by_id(mm.Branch, (226, 227, 0))

    print(run_energy_flow(net_multi))

    bounds_el = (
        BOUND_EL[1] * (1 - BOUND_EL[2]),
        BOUND_EL[1] * (1 + BOUND_EL[2]),
    )
    bounds_heat = (
        BOUND_HEAT[1] * (1 - BOUND_HEAT[2]),
        BOUND_HEAT[1] * (1 + BOUND_HEAT[2]),
    )
    bounds_gas = (
        BOUND_GAS[1] * (1 - BOUND_GAS[2]),
        BOUND_GAS[1] * (1 + BOUND_GAS[2]),
    )
    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
    )

    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    print(result)
    assert False
    assert result.dataframes["ExtPowerGrid"]["p_mw"][0] == -0.25
 """
