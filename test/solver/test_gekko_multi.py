import math

import monee.model as mm
import monee.solver as ms
import monee.network as mn
from monee import run_energy_flow
from monee.problem.load_shedding import create_load_shedding_optimization_problem
from monee.io.from_simbench import obtain_simbench_net


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
        gas_node=g_node_2,
        heat_node=w_node_1,
        heat_return_node=w_node_2,
        power_node=el_node_2,
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
    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], 0.86103583375)
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -0.086315875428)


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


"""
def test_chp_deactivation_in_simbench():
    # GIVEN
    net = obtain_simbench_net("1-LV-urban6--2-no_sw")
    net_multi = mn.generate_mes_based_on_power_net(
        net, heat_deployment_rate=0.1, gas_deployment_rate=0.3
    )

    import matplotlib.pyplot as plt
    import networkx as nx

    nx.draw_networkx(net_multi._network_internal, node_size=10, font_size=5)
    plt.savefig("abc.pdf")

    # WHEN
    pre_result = run_energy_flow(net_multi)
    print(pre_result)
    net_multi.deactivate(net_multi.compounds_by_type(mm.CHP)[0])
    post_result = run_energy_flow(net_multi)

    # THEN
    assert False
"""
