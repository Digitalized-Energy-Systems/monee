import math

import monee.model as mm
import monee.solver as ms


def create_two_pipes_no_branching():
    pn = mm.Network()

    pn.activate_grid(grid=mm.GAS)
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
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.2))],
    )

    pn.branch(
        mm.GasPipe(diameter_m=0.75, length_m=100, temperature_ext_k=300),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(diameter_m=0.75, length_m=2000, temperature_ext_k=300),
        g_node_1,
        g_node_2,
    )
    return pn


def create_two_pipes_gas_example():
    pn = mm.Network(mm.create_gas_grid("gas", type="lgas"))

    pn.activate_grid(grid=mm.GAS)
    # GAS
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow_kgs=0.2))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid())],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.4))],
    )

    pn.branch(
        mm.GasPipe(diameter_m=1, length_m=2000),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(diameter_m=0.3, length_m=200),
        g_node_0,
        g_node_2,
    )
    return pn


def create_branching_gas_net():
    pn = mm.Network()

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow_kgs=0.1))],
        grid=gas_grid,
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))], grid=gas_grid
    )
    g_node_3 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=100, temperature_ext_k=300, roughness_m=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness_m=0.01
        ),
        g_node_0,
        g_node_2,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness_m=0.01
        ),
        g_node_2,
        g_node_3,
    )
    return pn


def test_two_pipes_gas_network():
    # GIVEN
    gas_net = create_two_pipes_gas_example()

    # WHEN
    result = ms.GEKKOSolver().solve(gas_net)

    # THEN
    assert result.success

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -0.2)

    # analytic constant-friction (Swamee-Jain Re->inf) Weymouth solution;
    # wider tolerance on node 2 absorbs the documented epigraph relaxation slack
    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][2], 999977.43912, abs_tol=5.0
    )
    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][0], 999999.89303, abs_tol=0.01
    )

    assert len(result.dataframes) == 5


def test_two_pipes_line_gas_network():
    # GIVEN
    gas_net = create_two_pipes_no_branching()

    # WHEN
    result = ms.GEKKOSolver().solve(gas_net)

    # THEN
    assert result.success

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -0.2)
    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][2], 999999.51291, abs_tol=0.001
    )

    assert len(result.dataframes) == 4


def test_branching_gas_network():
    # GIVEN
    gas_net = create_branching_gas_net()

    # WHEN
    result = ms.GEKKOSolver().solve(gas_net)

    # THEN
    assert result.success

    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][2], 999999.86212, abs_tol=0.01
    )

    assert len(result.dataframes) == 5


def _single_pipe_gas_net(gas_type):
    """ext-grid - pipe - sink on a gas grid of ``gas_type``."""
    pn = mm.Network()
    gas = mm.create_gas_grid("gas", type=gas_type)
    g0 = pn.node(mm.Junction(), grid=gas, child_ids=[pn.child(mm.ExtHydrGrid())])
    g1 = pn.node(
        mm.Junction(), grid=gas, child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.2))]
    )
    pn.branch(mm.GasPipe(diameter_m=0.3, length_m=5000, temperature_ext_k=300), g0, g1)
    return pn


def test_lgas_vs_methane_molar_mass_is_live():
    # GIVEN the same pipe on the realistic L-gas (M = 18.1 g/mol) and on the
    # lighter methane / H-gas (M = 16.5 g/mol).
    res_lgas = ms.GEKKOSolver().solve(_single_pipe_gas_net("lgas"))
    res_methane = ms.GEKKOSolver().solve(_single_pipe_gas_net("methane"))

    # THEN both solve, and the gas's molar mass is honoured: here lgas has the
    # LARGER molar mass (18.1 > 16.5 g/mol, the N2/CO2-diluted L-gas of this
    # config - not the colloquial "L-gas is lighter"), so its specific gas
    # constant R/M is smaller and it loses LESS pressure than methane over the
    # identical pipe. Confirms R_specific is derived from the grid
    # (= universal_gas_constant / molar_mass), not hardcoded.
    assert res_lgas.success and res_methane.success
    p_lgas = res_lgas.dataframes["Junction"]["pressure_pa"][1]
    p_methane = res_methane.dataframes["Junction"]["pressure_pa"][1]
    assert p_lgas > p_methane
    assert math.isclose(p_lgas, 999863.356, abs_tol=1.0)
    assert math.isclose(p_methane, 999849.361, abs_tol=1.0)
