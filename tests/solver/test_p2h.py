"""Tests for the PowerToHeat (P2H) compound."""

import math

import monee.model as mm
import monee.solver as ms
from monee.model.formulation import EL_MISOCP_FORMULATION


def _build_p2h_network(
    heat_mw=0.020,
    efficiency=1.0,
    diameter_m=0.15,
):
    """Two-grid (power + heat/water) network with one P2H unit."""
    pn = mm.Network(mm.create_water_grid("heat"))

    w0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ConsumeHydrGrid(0.1))])
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
        mm.PowerToHeat(heat_mw, diameter_m, 300, efficiency),
        power_node_id=p2,
        heat_node_id=w2,
        heat_return_node_id=w1,
    )
    return pn


def test_p2h_basic_solve():
    # GIVEN
    net = _build_p2h_network()

    # WHEN
    result = ms.GEKKOSolver().solve(net)

    # THEN
    assert result.success
    assert len(result.dataframes) == 12

    cn = result.dataframes["PowerToHeatControlNode"]
    jct = result.dataframes["Junction"]
    bus = result.dataframes["Bus"]

    # P2H consumes power; without CHP to compensate, ExtPowerGrid injects
    assert cn["el_mw"].iloc[0] > 0
    assert result.dataframes["ExtPowerGrid"]["p_mw"].iloc[0] < 0

    # P2H delivers heat (injection convention: negative heat_mw means heat leaves
    # the unit): heated junction (w2) hotter than return (w1), in district-heat range
    assert cn["heat_mw"].iloc[0] < 0
    t_heated = jct["t_k"].iloc[1]
    t_return = jct["t_k"].iloc[2]
    assert t_heated > t_return
    assert 350 < t_heated < 430
    assert 340 < t_return < 400

    # Slack bus fixed at 1.0; others may drop due to P2H + load
    slack_vm = bus.loc[
        bus["id"] == result.dataframes["ExtPowerGrid"]["node_id"].iloc[0], "vm_pu"
    ].iloc[0]
    assert math.isclose(slack_vm, 1.0, abs_tol=1e-6)
    assert all(v > 0.85 for v in bus["vm_pu"])


def test_p2h_compound_structure():
    # GIVEN
    net = _build_p2h_network()

    # WHEN
    p2hs = net.compounds_by_type(mm.PowerToHeat)

    # THEN
    assert len(p2hs) == 1

    # control node + 2 transfer branches + SubHE = 4 subcomponents
    assert len(p2hs[0].subcomponents) == 4


def test_p2h_energy_balance():
    # GIVEN
    net = _build_p2h_network(heat_mw=0.020, efficiency=0.8)

    # WHEN
    result = ms.GEKKOSolver().solve(net)

    # THEN
    assert result.success

    cn = result.dataframes["PowerToHeatControlNode"]
    # heat_mw is generator-signed (negative = injection): |heat| = efficiency * el
    assert math.isclose(cn["heat_mw"].iloc[0], -0.8 * cn["el_mw"].iloc[0], rel_tol=1e-4)


def test_p2h_perfect_efficiency():
    # GIVEN
    heat_mw = 0.015
    net = _build_p2h_network(heat_mw=heat_mw, efficiency=1.0)

    # WHEN
    result = ms.GEKKOSolver().solve(net)

    # THEN
    assert result.success

    cn = result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(cn["el_mw"].iloc[0], heat_mw, rel_tol=1e-4)
    # heat_mw is generator-signed (negative = injection)
    assert math.isclose(cn["heat_mw"].iloc[0], -heat_mw, rel_tol=1e-4)


def test_p2h_efficiency_linearity():
    # GIVEN
    net_hi = _build_p2h_network(heat_mw=0.010, efficiency=1.0)
    net_lo = _build_p2h_network(heat_mw=0.010, efficiency=0.5)

    # WHEN
    r_hi = ms.GEKKOSolver().solve(net_hi)
    r_lo = ms.GEKKOSolver().solve(net_lo)

    # THEN
    assert r_hi.success
    assert r_lo.success

    # same heat output at η=1.0 needs twice the electricity at η=0.5
    el_hi = r_hi.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    el_lo = r_lo.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_lo, 2.0 * el_hi, rel_tol=1e-4)


def test_p2h_heat_setpoint_linearity():
    # GIVEN
    net_lo = _build_p2h_network(heat_mw=0.010, efficiency=0.9)
    net_hi = _build_p2h_network(heat_mw=0.020, efficiency=0.9)

    # WHEN
    r_lo = ms.GEKKOSolver().solve(net_lo)
    r_hi = ms.GEKKOSolver().solve(net_hi)

    # THEN
    assert r_lo.success
    assert r_hi.success

    el_lo = r_lo.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    el_hi = r_hi.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_hi, 2.0 * el_lo, rel_tol=1e-4)


def test_p2h_absolute_values():
    # GIVEN
    heat_mw = 0.020
    eff = 0.85
    expected_el = heat_mw / eff
    net = _build_p2h_network(heat_mw=heat_mw, efficiency=eff)

    # WHEN
    result = ms.GEKKOSolver().solve(net)

    # THEN
    assert result.success

    # el_mw = heat_mw / efficiency; heat_mw = -heat_mw (injection)
    cn = result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(cn["el_mw"].iloc[0], expected_el, rel_tol=1e-4)
    assert math.isclose(cn["heat_mw"].iloc[0], -heat_mw, rel_tol=1e-4)


def test_p2h_misocp_formulation():
    # GIVEN
    net = _build_p2h_network()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    net_ref = _build_p2h_network()

    # WHEN
    result = ms.PyomoSolver().solve(net)
    gekko_result = ms.GEKKOSolver().solve(net_ref)

    # THEN
    assert result.success
    assert gekko_result.success
    assert len(result.dataframes) == 12

    # MISOCP adds vm_pu_squared; must satisfy sqrt relation
    bus_df = result.dataframes["Bus"]
    assert "vm_pu_squared" in bus_df.columns
    for _, row in bus_df.iterrows():
        assert 0.0 <= row["vm_pu_squared"] <= 3.0
        assert math.isclose(row["vm_pu"], math.sqrt(row["vm_pu_squared"]), rel_tol=1e-3)

    # physics consistent with GEKKO
    cn_pyo = result.dataframes["PowerToHeatControlNode"]
    cn_gkk = gekko_result.dataframes["PowerToHeatControlNode"]
    assert math.isclose(cn_pyo["el_mw"].iloc[0], cn_gkk["el_mw"].iloc[0], rel_tol=1e-3)
    assert math.isclose(
        cn_pyo["heat_mw"].iloc[0], cn_gkk["heat_mw"].iloc[0], rel_tol=1e-3
    )


def test_p2h_cop_analogy():
    # GIVEN
    net_ref = _build_p2h_network(heat_mw=0.010, efficiency=1.0)
    net_low = _build_p2h_network(heat_mw=0.010, efficiency=0.6)

    # WHEN
    r_ref = ms.GEKKOSolver().solve(net_ref)
    r_low = ms.GEKKOSolver().solve(net_low)

    # THEN
    assert r_ref.success
    assert r_low.success

    # same heat at η=0.6 needs 1/0.6× the electricity of η=1.0
    el_ref = r_ref.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    el_low = r_low.dataframes["PowerToHeatControlNode"]["el_mw"].iloc[0]
    assert math.isclose(el_low / el_ref, 1.0 / 0.6, rel_tol=1e-4)


def test_p2h_no_gas_interaction():
    # GIVEN
    net = _build_p2h_network()

    # WHEN
    compounds = net.compounds_by_type(mm.PowerToHeat)

    # THEN
    # no P2H subcomponent may belong to a gas grid
    for compound in compounds:
        for sub in compound.subcomponents:
            grid = getattr(sub, "grid", None)
            if grid is not None:
                assert not isinstance(grid, mm.GasGrid), (
                    f"P2H subcomponent {sub} unexpectedly on a gas grid"
                )
