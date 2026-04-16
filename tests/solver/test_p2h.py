"""Comprehensive tests for the PowerToHeat (P2H) compound."""

import math

import monee.express as mx
import monee.model as mm
import monee.solver as ms
from monee.model.formulation import MISOCP_NETWORK_FORMULATION


def _build_p2h_network(
    heat_energy_w=20_000,
    efficiency=1.0,
    diameter_m=0.15,
):
    """Two-grid (power + heat/water) network with one P2H unit."""
    pn = mm.Network(mm.create_water_grid("heat"))

    w0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=0.1))])
    w1 = pn.node(mm.Junction())
    w2 = pn.node(mm.Junction())
    w3 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))])
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w1, w0)
    pn.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w2, w3)

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
        mm.PowerLine(length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        p0,
        p1,
    )
    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        p0,
        p2,
    )

    pn.compound(
        mm.PowerToHeat(heat_energy_w, diameter_m, 300, efficiency),
        power_node_id=p2,
        heat_node_id=w2,
        heat_return_node_id=w1,
    )
    return pn


def test_p2h_basic_solve():
    net = _build_p2h_network()
    result = ms.GEKKOSolver().solve(net)
    assert len(result.dataframes) == 12

    cn = result.dataframes["PowerToHeatControlNode"]
    jct = result.dataframes["Junction"]
    bus = result.dataframes["Bus"]

    # P2H consumes power (positive by load-sign convention)
    assert cn["el_mw"].iloc[0] > 0
    # Without CHP to compensate, P2H adds extra load → ExtPowerGrid injects
    assert result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0] < 0

    # P2H delivers heat to the water loop (positive heat_w)
    assert cn["heat_w"].iloc[0] > 0
    # Water temperature at the heated junction must exceed the return junction
    t_heated = jct["t_k"].iloc[1]  # w2 vicinity (P2H output)
    t_return = jct["t_k"].iloc[2]  # w2 input (return side)
    assert t_heated > t_return
    # Plausible district-heat range (350–430 K)
    assert 350 < t_heated < 430
    assert 340 < t_return < 400

    # Slack bus fixed at 1.0; others may drop due to P2H + load on 7e-5 Ω/m lines
    slack_vm = bus.loc[
        bus["id"] == result.dataframes["ExtPowerGrid"]["node_id"].iloc[0], "vm_pu"
    ].iloc[0]
    assert math.isclose(slack_vm, 1.0, abs_tol=1e-6)
    assert all(v > 0.85 for v in bus["vm_pu"])


def test_p2h_compound_structure():
    net = _build_p2h_network()
    p2hs = net.compounds_by_type(mm.PowerToHeat)
    assert len(p2hs) == 1
    # create() adds: PowerToHeatControlNode(node)
    #              + GenericTransferBranch(power → control)
    #              + GenericTransferBranch(control → heat_return)
    #              + SubHE(heat_node → control)  →  4
    assert len(p2hs[0].subcomponents) == 4


def test_p2h_energy_balance():
    net = _build_p2h_network(heat_energy_w=20_000, efficiency=0.8)
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(
        cn["heat_w"].iloc[0], 0.8 * cn["el_mw"].iloc[0] * 1e6, rel_tol=1e-4
    )


def test_p2h_perfect_efficiency():
    heat_w = 15_000
    net = _build_p2h_network(heat_energy_w=heat_w, efficiency=1.0)
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(cn["el_mw"].iloc[0], heat_w / 1e6, rel_tol=1e-4)
    assert math.isclose(cn["heat_w"].iloc[0], heat_w, rel_tol=1e-4)


def test_p2h_efficiency_linearity():
    """Same heat output at η=1.0 needs twice the electricity at η=0.5."""
    net_hi = _build_p2h_network(heat_energy_w=10_000, efficiency=1.0)
    net_lo = _build_p2h_network(heat_energy_w=10_000, efficiency=0.5)
    r_hi = ms.GEKKOSolver().solve(net_hi)
    r_lo = ms.GEKKOSolver().solve(net_lo)
    el_hi = r_hi.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    el_lo = r_lo.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_lo, 2.0 * el_hi, rel_tol=1e-4)


def test_p2h_heat_setpoint_linearity():
    net_lo = _build_p2h_network(heat_energy_w=10_000, efficiency=0.9)
    net_hi = _build_p2h_network(heat_energy_w=20_000, efficiency=0.9)
    r_lo = ms.GEKKOSolver().solve(net_lo)
    r_hi = ms.GEKKOSolver().solve(net_hi)
    el_lo = r_lo.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    el_hi = r_hi.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_hi, 2.0 * el_lo, rel_tol=1e-4)


def test_p2h_absolute_values():
    """
    heat_energy_w=20_000 W, efficiency=0.85:
      el_mw = heat_energy_w / (efficiency × 10⁶) = 20_000 / (0.85 × 10⁶) ≈ 0.023529 MW
      heat_w = heat_energy_w = 20_000 W
    """
    heat_w = 20_000
    eff = 0.85
    expected_el = heat_w / (eff * 1e6)

    net = _build_p2h_network(heat_energy_w=heat_w, efficiency=eff)
    result = ms.GEKKOSolver().solve(net)
    cn = result.dataframes["PowerToHeatControlNode"]

    assert math.isclose(cn["el_mw"].iloc[0], expected_el, rel_tol=1e-4)
    assert math.isclose(cn["heat_w"].iloc[0], heat_w, rel_tol=1e-4)


def test_p2h_temperature_rise():
    """
    Q = ṁ × cp × ΔT  →  ΔT = Q / (ṁ × cp)
    With ṁ = 0.1 kg/s, cp_water = 4186 J/(kg·K), heat_w = 20_000 W:
      ΔT ≈ 20_000 / (0.1 × 4186) ≈ 47.8 K
    """
    heat_w = 20_000
    mass_flow = 0.1
    cp_water = 4186
    expected_dt = heat_w / (mass_flow * cp_water)  # ≈ 47.8 K

    net = _build_p2h_network(heat_energy_w=heat_w, efficiency=1.0)
    result = ms.GEKKOSolver().solve(net)
    jct = result.dataframes["Junction"]

    # Heated junction (P2H supply output) vs return junction
    t_supply = jct["t_k"].iloc[1]
    t_return = jct["t_k"].iloc[2]
    actual_dt = t_supply - t_return
    assert math.isclose(actual_dt, expected_dt, rel_tol=0.02)


def test_p2h_misocp_formulation():
    net = _build_p2h_network()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    result = ms.PyomoSolver().solve(net)

    assert len(result.dataframes) == 12

    # MISOCP adds vm_pu_squared; must satisfy sqrt relation
    bus_df = result.dataframes["Bus"]
    assert "vm_pu_squared" in bus_df.columns
    for _, row in bus_df.iterrows():
        assert 0.0 <= row["vm_pu_squared"] <= 3.0
        assert math.isclose(row["vm_pu"], math.sqrt(row["vm_pu_squared"]), rel_tol=1e-3)

    # Physics consistent with GEKKO
    gekko_result = ms.GEKKOSolver().solve(_build_p2h_network())
    cn_pyo = result.dataframes["PowerToHeatControlNode"]
    cn_gkk = gekko_result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(cn_pyo["el_mw"].iloc[0], cn_gkk["el_mw"].iloc[0], rel_tol=1e-3)
    assert math.isclose(
        cn_pyo["heat_w"].iloc[0], cn_gkk["heat_w"].iloc[0], rel_tol=1e-3
    )


def test_p2h_misocp_regression():
    """Regression against values from the existing Pyomo test suite."""
    net = _build_p2h_network(heat_energy_w=20_000, efficiency=1.0)
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)
    result = ms.PyomoSolver().solve(net)

    assert len(result.dataframes) == 12
    assert math.isclose(
        result.dataframes["Junction"]["t_k"].iloc[0], 394.13290124571745, abs_tol=0.01
    )


def test_p2h_express_api():
    net = mm.Network()

    # Power grid
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    bus_2 = mx.create_bus(net)
    mx.create_power_generator(net, bus_0, p_mw=1, q_mvar=0)
    mx.create_ext_power_grid(net, bus_1)
    mx.create_power_load(net, bus_2, p_mw=1, q_mvar=0)
    mx.create_line(net, bus_0, bus_1, length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_line(net, bus_0, bus_2, length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    # Heat grid
    water_grid = mm.create_water_grid("heat")
    w0 = net.node(
        mm.Junction(), grid=water_grid, child_ids=[net.child(mm.Sink(mass_flow=0.1))]
    )
    w1 = net.node(mm.Junction(), grid=water_grid)
    w2 = net.node(mm.Junction(), grid=water_grid)
    w3 = net.node(
        mm.Junction(), grid=water_grid, child_ids=[net.child(mm.ExtHydrGrid(t_k=356))]
    )
    net.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w1, w0)
    net.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w2, w3)

    # P2H via express API
    mx.create_p2h(
        net,
        power_node_id=bus_2,
        heat_node_id=w2,
        heat_return_node_id=w1,
        heat_energy_w=20_000,
        diameter_m=0.15,
        efficiency=1.0,
    )

    result = ms.GEKKOSolver().solve(net)
    assert len(result.dataframes) == 12
    assert len(net.compounds_by_type(mm.PowerToHeat)) == 1
    cn = result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(cn["heat_w"].iloc[0], 20_000, rel_tol=1e-4)


def test_p2h_two_units():
    """Two P2H units on separate power buses, sharing the same water loop."""
    pn = mm.Network(mm.create_water_grid("heat"))

    # Shared heat grid: single supply/return loop
    w0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=0.2))])
    w1 = pn.node(mm.Junction())
    w2 = pn.node(mm.Junction())
    w3 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))])
    pn.branch(mm.WaterPipe(diameter_m=0.2, length_m=100), w1, w0)
    pn.branch(mm.WaterPipe(diameter_m=0.2, length_m=200), w2, w3)

    # Power grid — two load buses, each hosting one P2H unit
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
        child_ids=[pn.child(mm.PowerLoad(p_mw=0.5, q_mvar=0))],
    )
    p3 = pn.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[pn.child(mm.PowerLoad(p_mw=0.5, q_mvar=0))],
    )
    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        p0,
        p1,
    )
    pn.branch(
        mm.PowerLine(length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        p0,
        p2,
    )
    pn.branch(
        mm.PowerLine(length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        p0,
        p3,
    )

    # Two identical P2H units on different power buses, same water loop
    pn.compound(
        mm.PowerToHeat(10_000, 0.15, 300, 1.0),
        power_node_id=p2,
        heat_node_id=w2,
        heat_return_node_id=w1,
    )
    pn.compound(
        mm.PowerToHeat(10_000, 0.15, 300, 1.0),
        power_node_id=p3,
        heat_node_id=w2,
        heat_return_node_id=w1,
    )

    result = ms.GEKKOSolver().solve(pn)
    assert len(pn.compounds_by_type(mm.PowerToHeat)) == 2

    # Both control nodes produce identical heat (same setpoint + efficiency)
    cn_df = result.dataframes["PowerToHeatControlNode"]
    assert len(cn_df) == 2
    assert math.isclose(cn_df["heat_w"].iloc[0], cn_df["heat_w"].iloc[1], rel_tol=1e-3)
    # Combined heat equals sum of individual setpoints
    assert math.isclose(cn_df["heat_w"].sum(), 20_000, rel_tol=1e-4)


def test_p2h_cop_analogy():
    """
    With η < 1 the unit wastes electricity; same heat at η=0.6 needs 1/0.6× el.
    The ratio el_mw(η=0.6) / el_mw(η=1.0) should equal 1/0.6 ≈ 1.667.
    """
    net_ref = _build_p2h_network(heat_energy_w=10_000, efficiency=1.0)
    net_low = _build_p2h_network(heat_energy_w=10_000, efficiency=0.6)
    r_ref = ms.GEKKOSolver().solve(net_ref)
    r_low = ms.GEKKOSolver().solve(net_low)
    el_ref = r_ref.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    el_low = r_low.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_low / el_ref, 1.0 / 0.6, rel_tol=1e-4)


def test_p2h_no_gas_interaction():
    """
    A P2H compound must not add any gas-domain component to the network.
    """
    net = _build_p2h_network()
    gas_types = {"GasPipe", "GasGrid", "Source", "Sink", "ExtHydrGrid", "Junction"}
    # None of the P2H subcomponents should belong to a gas grid
    for compound in net.compounds_by_type(mm.PowerToHeat):
        for sub in compound.subcomponents:
            grid = getattr(sub, "grid", None)
            if grid is not None:
                assert not isinstance(grid, mm.GasGrid), (
                    f"P2H subcomponent {sub} unexpectedly on a gas grid"
                )
