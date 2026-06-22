import math

import monee.model as mm
import monee.solver as ms
from monee import run_energy_flow
from monee.model.formulation import make_heat_nlp_formulation
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER as CP
from tests.util import create_water_loop


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
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=30))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=3))],
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
        # t_k types the injected stream; without it the junction temperature
        # is structurally underdetermined - APOPT fails on the rank-deficient
        # system and SCIP/IPOPT return arbitrary, mutually disagreeing values.
        child_ids=[pn.child(mm.Source(mass_flow_kgs=0.1, t_k=340))],
    )
    g_node_mid = pn.node(
        mm.Junction(),
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.3))],
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
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow_kgs=1))],
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
        mm.HeatExchanger(q_mw=-0.001),
        g_node_0,
        g_node_2,
    )

    pn.branch(
        mm.HeatExchanger(q_mw=-0.001),
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
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))],
    )
    g_node_2 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))],
    )
    g_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))],
    )
    g_node_4 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))],
    )
    g_node_5 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow_kgs=1))],
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
        mm.HeatExchanger(q_mw=0.2),
        g_node_2,
        g_node_5,
    )
    pn.branch(
        mm.HeatExchanger(q_mw=0.1),
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
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.3))],
    )
    g_node_1 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow_kgs=10))],
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
        mm.HeatExchanger(q_mw=0.05),
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
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))],
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
        child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.1))],
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
    # GIVEN
    heat_net = create_branching_two_pipe_heat_example()

    # WHEN
    result = ms.PyomoSolver().solve(heat_net)
    print(result)

    # THEN
    assert result.success

    assert math.isclose(result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -33)
    assert len(result.dataframes) == 4
    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][2], 999999.84300, abs_tol=0.001
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][2], 355.74529187, abs_tol=0.001
    )


def test_t_heat_network():
    # GIVEN
    heat_net = create_t_heat_grid_test()

    # WHEN
    result = ms.GEKKOSolver().solve(heat_net)
    print(result)

    # THEN
    assert result.success

    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -0.2, rel_tol=1e-4
    )
    # 356 K slack and 340 K source mix at mid and cool toward the sink; with
    # the source temperature typed, APOPT and SCIP agree on this value to 5
    # decimals (the old 332.38 stemmed from the underdetermined system).
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][3], 334.50727, abs_tol=0.001
    )
    assert len(result.dataframes) == 5


def test_circle_heat_network():
    # GIVEN
    heat_net, _, _, _ = create_water_loop(source_t_k=340)

    # WHEN
    result = ms.GEKKOSolver().solve(heat_net)
    print(result)

    # THEN
    assert result.success

    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -5, rel_tol=1e-4
    )
    # Sink junction mixes the 356 K slack and 340 K source streams (minus pipe
    # losses), so it lies between the injection temperatures. The old 367.18 K
    # exceeded every injection temperature - an artifact of the structurally
    # underdetermined source temperature. APOPT/SCIP now agree within 5e-3 K.
    assert math.isclose(result.dataframes["Junction"]["t_k"][2], 347.917, abs_tol=0.01)
    assert len(result.dataframes) == 5


def test_heat_exchanger():
    # GIVEN
    heat_net = create_two_pipes_with_he_no_branching()

    # WHEN
    result = ms.GEKKOSolver().solve(heat_net)
    print(result)

    # THEN
    assert result.success

    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -0.39834289356
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][0], 383.17457358, abs_tol=0.01
    )
    assert math.isclose(
        result.dataframes["Junction"]["pressure_pa"][0], 999997.65278, abs_tol=0.01
    )
    assert len(result.dataframes) == 6


def test_dead_end():
    # GIVEN
    heat_net = create_line_heating_with_dead_end()

    # WHEN
    result = ms.GEKKOSolver().solve(heat_net)

    # THEN
    assert result.success

    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][0], -0.1, rel_tol=1e-5
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][0], 343.40404, abs_tol=0.01
    )
    assert len(result.dataframes) == 4


def create_supply_return_parallel_he():
    """Supply/return 3-node chains; one node pair joined by two parallel HEs, the other by one."""
    pn = mm.Network()

    s0 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))])
    s1 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))]
    )
    s2 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))]
    )

    r0 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow_kgs=10))],
    )
    r1 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow_kgs=3))]
    )
    r2 = pn.node(
        mm.Junction(), mm.WATER, child_ids=[pn.child(mm.Sink(mass_flow_kgs=3))]
    )

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s0, s1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s1, s2)

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r0, r1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r1, r2)

    # Two parallel HEs between s1 and r1
    pn.branch(mm.HeatExchanger(q_mw=0.04), s1, r1)
    pn.branch(mm.HeatExchanger(q_mw=0.04), s1, r1)
    # Single HE between s2 and r2
    pn.branch(mm.HeatExchanger(q_mw=0.05), s2, r2)

    return pn


def create_supply_return_parallel_he_real(q_mw_coeff=1):
    """Supply/return 3-node chains with single heat exchangers between node pairs."""
    pn = mm.Network()

    s0 = pn.node(mm.Junction(), mm.WATER, child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))])
    s1 = pn.node(mm.Junction(), mm.WATER)
    s2 = pn.node(mm.Junction(), mm.WATER)

    r0 = pn.node(
        mm.Junction(),
        mm.WATER,
        child_ids=[pn.child(mm.ConsumeHydrGrid(mass_flow_kgs=10))],
    )
    r1 = pn.node(mm.Junction(), mm.WATER)
    r2 = pn.node(mm.Junction(), mm.WATER)

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s0, s1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), s1, s2)

    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r0, r1)
    pn.branch(mm.WaterPipe(diameter_m=0.56, length_m=100), r1, r2)

    pn.branch(mm.HeatExchanger(q_mw=q_mw_coeff * 0.04), s1, r1)
    pn.branch(mm.HeatExchanger(q_mw=q_mw_coeff * 0.5), s2, r2)

    return pn


def test_supply_return_parallel_he():
    # GIVEN
    net = create_supply_return_parallel_he()

    # WHEN
    result = ms.GEKKOSolver().solve(net)
    print(result)

    # THEN
    assert result.success


def test_supply_return_parallel_he_real():
    # GIVEN
    net = create_supply_return_parallel_he_real(q_mw_coeff=1)

    # WHEN
    result = ms.PyomoSolver().solve(net)

    # THEN
    assert result.success

    assert math.isclose(result.get(mm.Junction)["t_k"][3], 385.1, abs_tol=0.09)


def test_supply_return_parallel_he_real_loads():
    # GIVEN
    net = create_supply_return_parallel_he_real(q_mw_coeff=-1)

    # WHEN
    result = ms.PyomoSolver().solve(net)

    # THEN
    assert result.success

    assert math.isclose(result.get(mm.Junction)["t_k"][3], 325.4, abs_tol=0.09)


def test_passive_he_minor_loss_matches_analytic():

    q_mw, mdot, zeta, dia = 0.1, 2.0, 5.0, 0.1
    grid = mm.create_water_grid("heat")
    net = mm.Network()
    a = net.node(
        mm.Junction(), grid=grid, child_ids=[net.child(mm.ExtHydrGrid(t_k=350))]
    )
    b = net.node(
        mm.Junction(), grid=grid, child_ids=[net.child(mm.Sink(mass_flow_kgs=mdot))]
    )
    net.branch(
        mm.PassiveHeatExchanger(q_mw=q_mw, diameter_m=dia, loss_coefficient=zeta), a, b
    )
    net.apply_formulation(make_heat_nlp_formulation(friction_model="constant"))

    result = run_energy_flow(net)
    print(result)
    assert result.success

    j = result.dataframes["Junction"]
    t_in, t_out = j["t_k"].iloc[0], j["t_k"].iloc[1]
    dp = j["pressure_pa"].iloc[0] - j["pressure_pa"].iloc[1]

    rho = grid.fluid_density_kg_per_m3
    v = mdot / (rho * math.pi / 4 * dia**2)
    assert math.isclose(t_out, t_in + q_mw * 1e6 / (CP * mdot), rel_tol=1e-4)
    assert math.isclose(dp, zeta * rho / 2 * v**2, rel_tol=1e-3)


def test_passive_he_zero_loss_coefficient_is_lossless():

    net = mm.Network()
    a = net.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[net.child(mm.ExtHydrGrid(t_k=350))]
    )
    b = net.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[net.child(mm.Sink(mass_flow_kgs=2.0))],
    )
    net.branch(
        mm.PassiveHeatExchanger(q_mw=0.01, diameter_m=0.1, loss_coefficient=0.0), a, b
    )
    net.apply_formulation(make_heat_nlp_formulation(friction_model="constant"))

    result = run_energy_flow(net)
    assert result.success
    j = result.dataframes["Junction"]
    assert math.isclose(
        j["pressure_pa"].iloc[0], j["pressure_pa"].iloc[1], abs_tol=1e-3
    )
