"""Comprehensive tests for the CHP compound and CHPControlNode."""

import math

import monee.express as mx
import monee.model as mm
import monee.solver as ms
from monee.model.formulation import MISOCP_NETWORK_FORMULATION

# ── Shared network builder ────────────────────────────────────────────────────


def _build_chp_network(
    efficiency_power=0.6,
    efficiency_heat=0.4,
    mass_flow_setpoint=0.001,
    regulation=1.0,
    diameter_m=0.3,
):
    """Minimal three-grid (power / heat / gas) network with one CHP unit."""
    pn = mm.Network()

    # ── Heat (water) grid ─────────────────────────────────────────────
    w0 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.Sink(mass_flow=0.5))]
    )
    w1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))]
    )
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)

    # ── Gas grid ──────────────────────────────────────────────────────
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
            diameter_m=0.75, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g2,
    )

    # ── Power grid ────────────────────────────────────────────────────
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

    # ── CHP ───────────────────────────────────────────────────────────
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


# ── 1. Basic solve: GEKKO, full regulation ────────────────────────────────────


def test_chp_basic_solve():
    net = _build_chp_network()
    result = ms.GEKKOSolver().solve(net)
    assert len(result.dataframes) == 14

    bus_df = result.dataframes["Bus"]
    cn_df = result.dataframes["CHPControlNode"]
    jct_df = result.dataframes["Junction"]

    # ── Voltages ──────────────────────────────────────────────────────
    # Slack bus (ExtPowerGrid) is fixed at 1.0 pu; others within ±10 %
    assert all(0.9 <= v <= 1.1 for v in bus_df["vm_pu"])
    # Slack bus va_degree = 0 by convention
    slack_row = result.dataframes["ExtPowerGrid"].iloc[0]
    slack_bus_id = int(slack_row["node_id"])
    slack_vm = bus_df.loc[bus_df["id"] == slack_bus_id, "vm_pu"].iloc[0]
    assert math.isclose(slack_vm, 1.0, abs_tol=1e-6)

    # ── Power direction ───────────────────────────────────────────────
    # CHP injects electrical power (negative by load-sign convention)
    assert cn_df["el_mw"].iloc[0] < 0
    # ExtPowerGrid absorbs the CHP surplus (positive = absorbing)
    assert result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0] > 0

    # ── Thermal output ────────────────────────────────────────────────
    # CHP extracts energy from gas → heat_w is negative (heat leaves the CHP)
    assert cn_df["heat_w"].iloc[0] < 0
    # Supply temperature > return temperature: heat is delivered to the loop
    # Water junctions: w1 (supply side) and w2 (return side) are index 1 & 2
    t_supply = jct_df["t_k"].iloc[1]  # w1 – supply tap
    t_return = jct_df["t_k"].iloc[2]  # w2 – return tap
    assert t_supply > t_return
    # Both temperatures are physically plausible for district heat (330–400 K)
    assert 330 < t_supply < 400
    assert 330 < t_return < 400

    # ── Gas pressures ─────────────────────────────────────────────────
    # Pressure near 1.0 pu; tiny drop along 150 m pipe at 0.001 kg/s
    assert all(
        0.99 < p <= 1.01 for p in jct_df["pressure_pu"] if p > 0
    )  # gas junctions have pressure; water junctions use t


# ── 2. Network structure: compound has exactly 5 subcomponents ────────────────


def test_chp_compound_structure():
    net = _build_chp_network()
    chps = net.compounds_by_type(mm.CHP)
    assert len(chps) == 1
    # create() adds: CHPControlNode(node) + gas branch + heat-in branch
    #               + SubHE(heat-out branch) + power branch  →  5
    assert len(chps[0].subcomponents) == 5


# ── 3. Energy balance: el_mw / eff_p  ≡  heat_w / (eff_h × 10⁶) ─────────────


def test_chp_energy_balance_invariant():
    eff_p, eff_h = 0.6, 0.4
    net = _build_chp_network(
        efficiency_power=eff_p, efficiency_heat=eff_h, mass_flow_setpoint=0.001
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    el_mw = cn["el_mw"].iloc[0]
    heat_w = cn["heat_w"].iloc[0]
    # Both driven by the same gas_kgps × regulation × (3.6 × HHV) factor
    assert math.isclose(el_mw / eff_p, heat_w / (eff_h * 1e6), rel_tol=1e-4)


# ── 4. Regulation = 0 → no electrical or thermal output ──────────────────────


def test_chp_regulation_zero():
    net = _build_chp_network(regulation=0.0)
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    assert math.isclose(cn["el_mw"].iloc[0], 0.0, abs_tol=1e-6)
    assert math.isclose(cn["heat_w"].iloc[0], 0.0, abs_tol=1.0)
    # With no CHP output the slack must inject power to cover generator line
    # losses (instead of absorbing CHP surplus) → p_mw < 0
    assert result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0] < 0


# ── 5. Regulation linearity: halving regulation halves el_mw ─────────────────


def test_chp_regulation_linear_scaling():
    net_full = _build_chp_network(regulation=1.0)
    net_half = _build_chp_network(regulation=0.5)
    r_full = ms.GEKKOSolver().solve(net_full)
    r_half = ms.GEKKOSolver().solve(net_half)
    el_full = r_full.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    el_half = r_half.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_half, 0.5 * el_full, rel_tol=1e-4)


# ── 6. Efficiency ratio: higher eff_p → more electrical, less heat ────────────


def test_chp_efficiency_ratio():
    """el_mw scales with efficiency_power; heat_w scales with efficiency_heat."""
    # Use small mass_flow_setpoint to keep heat output within pipe capacity
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
    heat_a = ra.dataframes["CHPControlNode"]["heat_w"].iloc[0]
    heat_b = rb.dataframes["CHPControlNode"]["heat_w"].iloc[0]
    # Same mass flow setpoint → ratios between runs equal efficiency ratios
    assert math.isclose(el_a / el_b, 0.7 / 0.3, rel_tol=1e-4)
    assert math.isclose(heat_a / heat_b, 0.2 / 0.6, rel_tol=1e-4)


# ── 7. Mass-flow setpoint doubles el_mw (linear in gas_kgps) ─────────────────


def test_chp_mass_flow_linearity():
    net_lo = _build_chp_network(mass_flow_setpoint=0.0005)
    net_hi = _build_chp_network(mass_flow_setpoint=0.001)
    r_lo = ms.GEKKOSolver().solve(net_lo)
    r_hi = ms.GEKKOSolver().solve(net_hi)
    el_lo = r_lo.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    el_hi = r_hi.dataframes["CHPControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_hi, 2.0 * el_lo, rel_tol=1e-4)


# ── 8. Absolute values match physics formula ──────────────────────────────────


def test_chp_absolute_values():
    """
    For lgas (HHV = 15.3 kWh/kg), regulation=1, mass_flow=0.001 kg/s:
      el_mw  = -eff_p × mass × 3.6 × HHV
             = -0.6 × 0.001 × 3.6 × 15.3 = -0.033048 MW
      heat_w = -eff_h × mass × 3.6 × HHV × 1e6
             = -0.4 × 0.001 × 3.6 × 15.3 × 1e6 = -22032 W
    """
    HHV = 15.3  # kWh/kg for lgas
    eff_p, eff_h, mf = 0.6, 0.4, 0.001
    expected_el = -eff_p * mf * 3.6 * HHV  # MW
    expected_heat = -eff_h * mf * 3.6 * HHV * 1e6  # W

    net = _build_chp_network(
        efficiency_power=eff_p,
        efficiency_heat=eff_h,
        mass_flow_setpoint=mf,
        regulation=1.0,
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    assert math.isclose(cn["el_mw"].iloc[0], expected_el, rel_tol=1e-4)
    assert math.isclose(cn["heat_w"].iloc[0], expected_heat, rel_tol=1e-4)


# ── 9. MISOCP formulation (Pyomo solver) ─────────────────────────────────────


def test_chp_misocp_formulation():
    net = _build_chp_network(mass_flow_setpoint=0.001)
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    result = ms.PyomoSolver().solve(net)

    assert len(result.dataframes) == 14

    # MISOCP adds vm_pu_squared; all values must be in [0, 3]
    bus_df = result.dataframes["Bus"]
    for vm_sq in bus_df["vm_pu_squared"]:
        assert 0.0 <= vm_sq <= 3.0

    # vm_pu ≈ sqrt(vm_pu_squared) for each bus
    for _, row in bus_df.iterrows():
        assert math.isclose(row["vm_pu"], math.sqrt(row["vm_pu_squared"]), rel_tol=1e-3)

    # Energy balance still holds under MISOCP
    cn = result.dataframes["CHPControlNode"]
    eff_p, eff_h = 0.6, 0.4
    el_mw = cn["el_mw"].iloc[0]
    heat_w = cn["heat_w"].iloc[0]
    assert math.isclose(el_mw / eff_p, heat_w / (eff_h * 1e6), rel_tol=1e-3)

    # MISOCP and AC power flow should give identical voltage profile
    gekko_net = _build_chp_network(mass_flow_setpoint=0.001)
    gekko_result = ms.GEKKOSolver().solve(gekko_net)
    gekko_vm = sorted(gekko_result.dataframes["Bus"]["vm_pu"].tolist())
    misocp_vm = sorted(result.dataframes["Bus"]["vm_pu"].tolist())
    for v_ac, v_socp in zip(gekko_vm, misocp_vm):
        assert math.isclose(v_ac, v_socp, abs_tol=1e-4)


# ── 10. Express API builds an equivalent network ─────────────────────────────


def test_chp_express_api():
    pn = mm.Network()

    # Heat grid
    w0 = mx.create_water_junction(pn)
    w1 = mx.create_water_junction(pn)
    w2 = mx.create_water_junction(pn)
    w3 = mx.create_water_junction(pn)
    mx.create_ext_hydr_grid(pn, w3, t_k=359)
    mx.create_sink(pn, w0, mass_flow=0.5)
    mx.create_water_pipe(pn, w0, w1, diameter_m=0.15, length_m=100)
    mx.create_water_pipe(pn, w3, w2, diameter_m=0.15, length_m=200)

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
            diameter_m=0.75, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g2,
    )

    # Power grid
    p0 = mx.create_bus(pn)
    p1 = mx.create_bus(pn)
    p2 = mx.create_bus(pn)
    mx.create_power_generator(pn, p0, p_mw=1, q_mvar=0)
    mx.create_ext_power_grid(pn, p1)
    mx.create_power_load(pn, p2, p_mw=1, q_mvar=0)
    mx.create_line(pn, p0, p1, length_m=1000, r_ohm_per_m=7e-6, x_ohm_per_m=7e-6)
    mx.create_line(pn, p0, p2, length_m=1000, r_ohm_per_m=7e-6, x_ohm_per_m=7e-6)

    # CHP via express API
    mx.create_chp(
        pn,
        power_node_id=p2,
        heat_node_id=w1,
        heat_return_node_id=w2,
        gas_node_id=g2,
        diameter_m=0.3,
        efficiency_power=0.6,
        efficiency_heat=0.4,
        mass_flow_setpoint=0.001,
        regulation=1.0,
    )

    result = ms.GEKKOSolver().solve(pn)
    assert len(result.dataframes) == 14
    assert len(pn.compounds_by_type(mm.CHP)) == 1


# ── 11. Two CHP units in parallel ────────────────────────────────────────────


def test_chp_two_units():
    pn = mm.Network()

    # Heat grid (supply + return for both CHPs)
    w0 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.Sink(mass_flow=0.5))]
    )
    w1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))]
    )
    w4 = pn.node(mm.Junction(), grid=mm.WATER_KEY)  # second CHP supply tap
    w5 = pn.node(mm.Junction(), grid=mm.WATER_KEY)  # second CHP return tap
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w0, w4)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w2)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w3, w5)

    # Gas grid
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g0 = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Source(mass_flow=1))]
    )
    g1 = pn.node(mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.ExtHydrGrid())])
    g2 = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Sink(mass_flow=1))]
    )
    g3 = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Sink(mass_flow=1))]
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g2,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.75, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g0,
        g3,
    )

    # Power grid
    power_grid = mm.create_power_grid("power")
    p0 = pn.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[pn.child(mm.PowerGenerator(p_mw=2, q_mvar=0))],
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

    # Two identical CHP units feeding the same power bus
    pn.compound(
        mm.CHP(0.3, 0.6, 0.4, 0.001, regulation=1.0),
        gas_node_id=g2,
        heat_node_id=w1,
        heat_return_node_id=w2,
        power_node_id=p2,
    )
    pn.compound(
        mm.CHP(0.3, 0.6, 0.4, 0.001, regulation=1.0),
        gas_node_id=g3,
        heat_node_id=w4,
        heat_return_node_id=w5,
        power_node_id=p2,
    )

    result = ms.GEKKOSolver().solve(pn)
    assert len(pn.compounds_by_type(mm.CHP)) == 2

    # Both CHPControlNodes produce identical el_mw (same setpoint + params)
    cn_df = result.dataframes["CHPControlNode"]
    assert len(cn_df) == 2
    assert math.isclose(cn_df["el_mw"].iloc[0], cn_df["el_mw"].iloc[1], rel_tol=1e-3)


# ── 12. Heat-dominated CHP (high thermal efficiency) ─────────────────────────


def test_chp_heat_dominated():
    eff_p, eff_h = 0.15, 0.75
    net = _build_chp_network(
        efficiency_power=eff_p, efficiency_heat=eff_h, mass_flow_setpoint=0.0001
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    el_mw = cn["el_mw"].iloc[0]
    heat_w = cn["heat_w"].iloc[0]
    # Heat output should be 5× electrical output (in consistent units)
    assert math.isclose(abs(heat_w) / 1e6, abs(el_mw) * (eff_h / eff_p), rel_tol=1e-4)


# ── 13. Power-dominated CHP (high electrical efficiency) ─────────────────────


def test_chp_power_dominated():
    eff_p, eff_h = 0.8, 0.1
    net = _build_chp_network(
        efficiency_power=eff_p, efficiency_heat=eff_h, mass_flow_setpoint=0.001
    )
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["CHPControlNode"]
    el_mw = cn["el_mw"].iloc[0]
    heat_w = cn["heat_w"].iloc[0]
    assert math.isclose(abs(heat_w) / 1e6, abs(el_mw) * (eff_h / eff_p), rel_tol=1e-4)


# ── 14. MISOCP: known reference values (regression) ──────────────────────────


def test_chp_misocp_regression():
    """Reference values from the existing Pyomo test suite."""
    pn = mm.Network()

    # Water grid
    w0 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.Sink(mass_flow=1))]
    )
    w1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w3 = pn.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))]
    )
    pn.branch(mm.WaterPipe(diameter_m=0.35, length_m=100), w0, w1)
    pn.branch(mm.WaterPipe(diameter_m=0.35, length_m=200), w3, w2)

    # Gas grid
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g0 = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Source(mass_flow=0.1))]
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

    pn.compound(
        mm.CHP(0.5, 0.6, 0.4, 0.0, regulation=0.5),
        gas_node_id=g2,
        heat_node_id=w1,
        heat_return_node_id=w2,
        power_node_id=p2,
    )

    pn.apply_formulation(MISOCP_NETWORK_FORMULATION)
    result = ms.PyomoSolver().solve(pn)

    assert len(result.dataframes) == 14
    assert math.isclose(
        result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0],
        -0.006264089217161262,
        abs_tol=0.001,
    )
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"].iloc[1], -0.9, abs_tol=0.01
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"].iloc[1], 357.924809287306, abs_tol=0.1
    )
