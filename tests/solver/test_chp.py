"""Tests for the CHP compound and CHPControlNode."""

import math

import monee.model as mm
import monee.solver as ms
from monee.model.formulation import MISOCP_NETWORK_FORMULATION


def _build_chp_network(
    efficiency_power=0.6,
    efficiency_heat=0.4,
    mass_flow_setpoint=0.001,
    regulation=1.0,
    diameter_m=0.3,
):
    """Minimal three-grid (power / heat / gas) network with one CHP unit."""
    pn = mm.Network()

    # Heat (water) grid
    w0 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[
            pn.child(mm.Sink(mass_flow=0.5)),
            pn.child(mm.ExtHydrGrid(t_k=359)),
        ],
    )
    w1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ConsumeHydrGrid(1))]
    )
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)

    # Gas grid
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g0 = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Source(mass_flow=1))]
    )
    g1 = pn.node(mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Sink(mass_flow=1))]
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g2,
    )

    # Power grid
    power_grid = mm.create_power_grid("power")
    p0 = pn.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0))],
    )
    p1 = pn.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    p2 = pn.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
    )
    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=7e-6, x_ohm_per_m=7e-6, parallel=1),
        p0,
        p1,
    )
    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=7e-6, x_ohm_per_m=7e-6, parallel=1),
        p0,
        p2,
    )

    # CHP
    pn.compound(
        mm.CHP(
            diameter_m,
            efficiency_power,
            efficiency_heat,
            mass_flow_setpoint,
            regulation=regulation,
        ),
        gas_node_id=g2,
        heat_node_id=w1,
        heat_return_node_id=w2,
        power_node_id=p2,
    )
    return pn


def test_chp_basic_solve():
    """Solve with Pyomo MISOCP and verify voltages, power direction, thermal output, and gas pressures."""
    net = _build_chp_network()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    result = ms.PyomoSolver().solve(net)
    assert len(result.dataframes) == 15

    bus_df = result.dataframes["Bus"]
    cn_df = result.dataframes["CHPControlNode"]
    jct_df = result.dataframes["Junction"]

    # Slack bus (ExtPowerGrid) is fixed at 1.0 pu; others within +/-10%
    assert all(0.9 <= v <= 1.1 for v in bus_df["vm_pu"])
    slack_row = result.dataframes["ExtPowerGrid"].iloc[0]
    slack_bus_id = int(slack_row["node_id"])
    slack_vm = bus_df.loc[bus_df["id"] == slack_bus_id, "vm_pu"].iloc[0]
    assert math.isclose(slack_vm, 1.0, abs_tol=1e-6)

    # CHP injects electrical power (negative by load-sign convention)
    assert cn_df["el_mw"].iloc[0] < 0
    # ExtPowerGrid absorbs the CHP surplus
    assert result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0] > 0

    # heat_mw is negative (heat leaves the CHP)
    assert cn_df["heat_mw"].iloc[0] < 0
    # w1 (supply tap, index 1) is cooler than w2 (return tap, index 2)
    # because the CHP heats water flowing through its SubHE from w1 to w2
    t_supply = jct_df["t_k"].iloc[1]
    t_return = jct_df["t_k"].iloc[2]
    assert t_supply < t_return
    assert 330 < t_supply < 400
    assert 330 < t_return < 400

    # Gas pressures near 1.0 pu
    assert all(0.99 < p <= 1.01 for p in jct_df["pressure_pu"] if p > 0)


def test_chp_compound_structure():
    net = _build_chp_network()
    chps = net.compounds_by_type(mm.CHP)
    assert len(chps) == 1
    assert len(chps[0].subcomponents) == 5


def test_chp_energy_balance_invariant():
    """el_mw / eff_p == heat_mw / eff_h since both are driven by the same gas input."""
    eff_p, eff_h = 0.6, 0.4
    net = _build_chp_network(
        efficiency_power=eff_p, efficiency_heat=eff_h, mass_flow_setpoint=0.001
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    el_mw = cn["el_mw"].iloc[0]
    heat_mw = cn["heat_mw"].iloc[0]
    assert math.isclose(el_mw / eff_p, heat_mw / eff_h, rel_tol=1e-4)


def test_chp_regulation_linear_scaling():
    """Halving regulation halves el_mw."""
    net_full = _build_chp_network(regulation=1.0)
    net_half = _build_chp_network(regulation=0.5)
    r_full = ms.GEKKOSolver().solve(net_full)
    r_half = ms.GEKKOSolver().solve(net_half)
    el_full = r_full.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    el_half = r_half.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_half, 0.5 * el_full, rel_tol=1e-4)


def test_chp_efficiency_ratio():
    """el_mw scales with efficiency_power; heat_mw scales with efficiency_heat."""
    net_a = _build_chp_network(
        efficiency_power=0.7, efficiency_heat=0.2, mass_flow_setpoint=0.001
    )
    net_b = _build_chp_network(
        efficiency_power=0.3, efficiency_heat=0.6, mass_flow_setpoint=0.001
    )
    ra = ms.GEKKOSolver().solve(net_a)
    rb = ms.GEKKOSolver().solve(net_b)
    el_a = ra.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    el_b = rb.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    heat_a = ra.dataframes["CHPControlNode"]["heat_mw"].iloc[0]
    heat_b = rb.dataframes["CHPControlNode"]["heat_mw"].iloc[0]
    assert math.isclose(el_a / el_b, 0.7 / 0.3, rel_tol=1e-4)
    assert math.isclose(heat_a / heat_b, 0.2 / 0.6, rel_tol=1e-4)


def test_chp_mass_flow_linearity():
    """Doubling mass_flow_setpoint doubles el_mw."""
    net_lo = _build_chp_network(mass_flow_setpoint=0.0005)
    net_hi = _build_chp_network(mass_flow_setpoint=0.001)
    r_lo = ms.GEKKOSolver().solve(net_lo)
    r_hi = ms.GEKKOSolver().solve(net_hi)
    el_lo = r_lo.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    el_hi = r_hi.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_hi, 2.0 * el_lo, rel_tol=1e-4)


def test_chp_absolute_values():
    """Verify el_mw and heat_mw against the analytical formula.

    For lgas (HHV = 15.3 kWh/kg), regulation=1, mass_flow=0.001 kg/s:
      el_mw  = -eff_p * mass * 3.6 * HHV = -0.033048 MW
      heat_mw = -eff_h * mass * 3.6 * HHV = -0.022032 MW
    """
    HHV = 15.3
    eff_p, eff_h, mf = 0.6, 0.4, 0.001
    expected_el = -eff_p * mf * 3.6 * HHV
    expected_heat = -eff_h * mf * 3.6 * HHV

    net = _build_chp_network(
        efficiency_power=eff_p,
        efficiency_heat=eff_h,
        mass_flow_setpoint=mf,
        regulation=1.0,
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    assert math.isclose(cn["el_mw"].iloc[0], expected_el, rel_tol=1e-4)
    assert math.isclose(cn["heat_mw"].iloc[0], expected_heat, rel_tol=1e-4)


def test_chp_misocp_formulation():
    """MISOCP formulation produces consistent vm_pu/vm_pu_squared, energy balance, and matches GEKKO voltages."""
    net = _build_chp_network(mass_flow_setpoint=0.001)
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    result = ms.PyomoSolver().solve(net)

    assert len(result.dataframes) == 15

    bus_df = result.dataframes["Bus"]
    for vm_sq in bus_df["vm_pu_squared"]:
        assert 0.0 <= vm_sq <= 3.0

    for _, row in bus_df.iterrows():
        assert math.isclose(row["vm_pu"], math.sqrt(row["vm_pu_squared"]), rel_tol=1e-3)

    cn = result.dataframes["CHPControlNode"]
    eff_p, eff_h = 0.6, 0.4
    el_mw = cn["el_mw"].iloc[0]
    heat_mw = cn["heat_mw"].iloc[0]
    assert math.isclose(el_mw / eff_p, heat_mw / eff_h, rel_tol=1e-3)

    # MISOCP and GEKKO should give the same voltage profile
    gekko_net = _build_chp_network(mass_flow_setpoint=0.001)
    gekko_result = ms.GEKKOSolver().solve(gekko_net)
    gekko_vm = sorted(gekko_result.dataframes["Bus"]["vm_pu"].tolist())
    misocp_vm = sorted(result.dataframes["Bus"]["vm_pu"].tolist())
    for v_ac, v_socp in zip(gekko_vm, misocp_vm):
        assert math.isclose(v_ac, v_socp, abs_tol=1e-4)


def test_chp_heat_dominated():
    """High thermal efficiency: heat output should be eff_h/eff_p times electrical output."""
    eff_p, eff_h = 0.15, 0.75
    net = _build_chp_network(
        efficiency_power=eff_p, efficiency_heat=eff_h, mass_flow_setpoint=0.0001
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    el_mw = cn["el_mw"].iloc[0]
    heat_mw = cn["heat_mw"].iloc[0]
    assert math.isclose(abs(heat_mw), abs(el_mw) * (eff_h / eff_p), rel_tol=1e-4)


def test_chp_power_dominated():
    """High electrical efficiency: same ratio check as heat-dominated but inverted."""
    eff_p, eff_h = 0.8, 0.1
    net = _build_chp_network(
        efficiency_power=eff_p, efficiency_heat=eff_h, mass_flow_setpoint=0.001
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    el_mw = cn["el_mw"].iloc[0]
    heat_mw = cn["heat_mw"].iloc[0]
    assert math.isclose(abs(heat_mw), abs(el_mw) * (eff_h / eff_p), rel_tol=1e-4)
