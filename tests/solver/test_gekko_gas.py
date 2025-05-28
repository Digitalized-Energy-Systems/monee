import math

import monee.model as mm
import monee.solver as ms


def create_two_pipes_no_branching():
    pn = mm.Network()

    # GAS
    g_node_0 = pn.node(
        mm.Junction(),
        mm.GAS,
        child_ids=[pn.child(mm.ExtHydrGrid())],
    )
    g_node_1 = pn.node(mm.Junction(), mm.GAS)
    g_node_2 = pn.node(
        mm.Junction(),
        mm.GAS,
        child_ids=[pn.child(mm.Sink(mass_flow=0.2))],
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
            diameter_m=0.75, length_m=2000, temperature_ext_k=300, roughness=0.01
        ),
        g_node_1,
        g_node_2,
    )
    return pn


def create_two_pipes_gas_example():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))

    # GAS
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=0.1))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid())],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.2))],
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
            diameter_m=0.75, length_m=2000, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_2,
    )
    return pn


def test_two_pipes_gas_network():
    gas_net = create_two_pipes_gas_example()
    result = ms.GEKKOSolver().solve(gas_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.1)
    assert len(result.dataframes) == 5


def test_two_pipes_line_gas_network():
    gas_net = create_two_pipes_no_branching()
    result = ms.GEKKOSolver().solve(gas_net)

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow"][0], -0.2)
    assert len(result.dataframes) == 4
