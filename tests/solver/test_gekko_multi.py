import math
import random

import monee.model as mm
import monee.network as mn
import monee.problem as mp
import monee.solver as ms
from monee import run_energy_flow, run_energy_flow_optimization
from monee.io.from_simbench import obtain_simbench_net
from monee.problem.load_shedding import create_load_shedding_optimization_problem

BOUND_EL = ("vm_pu", 1, 0.2)
BOUND_GAS = ("pressure_pa", 500000, 0.3)
BOUND_HEAT = ("t_k", 352, 0.1)


def create_two_line_example_with_2_pipe_example_p2g(source_flow=0.1):
    pn = mm.Network(mm.create_power_grid("power"))

    # POWER
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
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=gas_grid
    )

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

    # MULTI
    pn.branch(
        mm.PowerToGas(efficiency=0.95, mass_flow_setpoint=0.1),
        el_node_0,
        g_node_0,
    )
    return pn


def create_multi_chp():
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
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_2,
        w_node_3,
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
        mm.CHP(
            0.15,
            1,
            1,
            0.1,
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
        mm.PowerToHeat(0.1, 0.015, 300, 1, in_line_operation=True),
        power_node=el_node_2,
        heat_node=w_node_2,
        heat_return_node=w_node_1,
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

    assert len(result.dataframes) == 11
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -0.090487525893)
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.8)


def test_in_line_p2h():
    multi_energy_network = create_in_line_p2h()

    result = ms.GEKKOSolver().solve(multi_energy_network)

    assert len(result.dataframes) == 12
    assert math.isclose(result.dataframes["Junction"]["t_k"][0], 598.005423)


def test_load_shedding_p2g_network():
    multi_energy_network = create_two_line_example_with_2_pipe_example_p2g(
        source_flow=1
    )
    load_shedding_problem = create_load_shedding_optimization_problem()

    result = ms.GEKKOSolver().solve(
        multi_energy_network, optimization_problem=load_shedding_problem
    )

    assert len(result.dataframes) == 11
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0)
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], 0)


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
        result.dataframes["ExtHydrGrid"]["mass_flow"][1], -0.033016318428
    )
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -0.091089923543)


def test_simbench_ls_optimization():
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
