import math

import monee.model as mm
from monee.solver import PyomoSolver
from tests.util import assert_control_node_solved as _assert_control_node_solved
from tests.util import assert_junction_nan as _assert_junction_nan
from tests.util import assert_junction_solved as _assert_junction_solved

_LINE = dict(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1)
_PIPE = dict(diameter_m=0.15, length_m=100)
_GAS_PIPE = dict(diameter_m=0.3, length_m=100, temperature_ext_k=300, roughness=0.01)


def _power_grid(pn):
    """Return (grid, slack_id, gen_id, load_id) for a small three-bus power grid."""
    g = mm.create_power_grid("power")
    p_slack = pn.node(
        mm.Bus(base_kv=1),
        grid=g,
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    p_gen = pn.node(
        mm.Bus(base_kv=1),
        grid=g,
        child_ids=[pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0))],
    )
    p_load = pn.node(
        mm.Bus(base_kv=1),
        grid=g,
        child_ids=[pn.child(mm.PowerLoad(p_mw=0.2, q_mvar=0))],
    )
    pn.branch(mm.PowerLine(**_LINE), p_slack, p_gen)
    return g, p_slack, p_gen, p_load


def _deactivatable_power_line(pn, src, dst, deactivate):
    bid = pn.branch(mm.PowerLine(**_LINE), src, dst)
    if deactivate:
        pn.branch_by_id(bid).active = False


def _heat_grid_main(pn):
    """j_ext(ExtHydrGrid) → j_hub → j_sink(Sink); j_hub anchors isolated heat clusters. Returns (grid, j_ext, j_hub)."""
    water_grid = mm.create_water_grid("heat")
    j_ext = pn.node(
        mm.Junction(), grid=water_grid, child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))]
    )
    j_hub = pn.node(mm.Junction(), grid=water_grid)
    j_sink = pn.node(
        mm.Junction(), grid=water_grid, child_ids=[pn.child(mm.Sink(mass_flow=0.05))]
    )
    pn.branch(mm.WaterPipe(**_PIPE), j_ext, j_hub)
    pn.branch(mm.WaterPipe(**_PIPE), j_hub, j_sink)
    return water_grid, j_ext, j_hub


def _heat_cluster(pn, grid, j_anchor, deactivate_bridge):
    """j_a → j_b bridged to j_anchor; a deactivated bridge cuts both off from ExtHydrGrid. Returns (j_a, j_b)."""
    j_a = pn.node(mm.Junction(), grid=grid)
    j_b = pn.node(mm.Junction(), grid=grid)
    pn.branch(mm.WaterPipe(**_PIPE), j_a, j_b)
    bridge_id = pn.branch(mm.WaterPipe(**_PIPE), j_anchor, j_a)
    if deactivate_bridge:
        pn.branch_by_id(bridge_id).active = False
    return j_a, j_b


def _gas_grid(pn, *, isolated_junction=False):
    """j_hub(ExtHydrGrid) → j_mid → j_main(Source) → [bridge] → j_iso. Returns (j_hub, j_main, j_iso)."""
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    j_hub = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.ExtHydrGrid())]
    )
    j_mid = pn.node(mm.Junction(), grid=gas_grid)
    j_main = pn.node(
        mm.Junction(), grid=gas_grid, child_ids=[pn.child(mm.Source(mass_flow=1))]
    )
    j_iso = pn.node(mm.Junction(), grid=gas_grid)
    pn.branch(mm.GasPipe(**_GAS_PIPE), j_hub, j_mid)
    pn.branch(mm.GasPipe(**_GAS_PIPE), j_mid, j_main)
    bridge_id = pn.branch(mm.GasPipe(**_GAS_PIPE), j_main, j_iso)
    if isolated_junction:
        pn.branch_by_id(bridge_id).active = False
    return j_hub, j_main, j_iso


def _heat_grid_full(pn):
    """j_ext(ExtHydrGrid) → j_supply → j_return(Sink). Returns (grid, j_ext, j_supply, j_return)."""
    water_grid = mm.create_water_grid("heat")
    j_ext = pn.node(
        mm.Junction(), grid=water_grid, child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))]
    )
    j_supply = pn.node(mm.Junction(), grid=water_grid)
    j_return = pn.node(
        mm.Junction(), grid=water_grid, child_ids=[pn.child(mm.Sink(mass_flow=0.05))]
    )
    pn.branch(mm.WaterPipe(**_PIPE), j_ext, j_supply)
    pn.branch(mm.WaterPipe(**_PIPE), j_supply, j_return)
    return water_grid, j_ext, j_supply, j_return


def _assert_bus_nan(result, bus_id, label):
    df = result.dataframes["Bus"]
    vm = df.loc[df["id"] == bus_id, "vm_pu"].iloc[0]
    assert math.isnan(vm), f"{label}: expected NaN vm_pu, got {vm}"


def _assert_bus_solved(result, bus_id, label, *, expected_vm=None):
    df = result.dataframes["Bus"]
    vm = df.loc[df["id"] == bus_id, "vm_pu"].iloc[0]
    assert not math.isnan(vm), f"{label}: bus must be solved (got NaN)"
    if expected_vm is not None:
        assert math.isclose(vm, expected_vm, abs_tol=1e-4), (
            f"{label}: expected vm_pu≈{expected_vm}, got {vm}"
        )


def _solve(net):
    return PyomoSolver().solve(net, exclude_unconnected_nodes=True)


def _build_p2h_power_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_supply, j_return = _heat_grid_full(pn)
    _, p_slack, p_gen, p_p2h = _power_grid(pn)
    _deactivatable_power_line(pn, p_gen, p_p2h, deactivate)
    pn.compound(
        mm.PowerToHeat(0.02, 0.15, 300, 0.9),
        power_node_id=p_p2h,
        heat_node_id=j_supply,
        heat_return_node_id=j_return,
    )
    return pn, p_slack, p_p2h, j_supply


def test_p2h_power_bus_isolated():
    # GIVEN
    pn, p_slack, p_p2h, j_supply = _build_p2h_power_isolated(deactivate=True)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    # disconnected power bus is NaN'd; control node stays active as thermal junction
    _assert_bus_nan(result, p_p2h, "p_p2h")
    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H bus isolated")

    # rest of the network stays intact
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_supply, "j_supply")


def test_p2h_power_bus_connected_baseline():
    # GIVEN
    pn, p_slack, _, _ = _build_p2h_power_isolated(deactivate=False)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H baseline")
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)


def _build_p2h_heat_isolated(deactivate: bool):
    pn = mm.Network()
    grid, _, j_main = _heat_grid_main(pn)
    j_heat_a, j_heat_b = _heat_cluster(pn, grid, j_main, deactivate_bridge=deactivate)
    _, p_slack, p_gen, p_p2h = _power_grid(pn)
    pn.branch(mm.PowerLine(**_LINE), p_gen, p_p2h)
    pn.compound(
        mm.PowerToHeat(0.02, 0.15, 300, 0.9),
        power_node_id=p_p2h,
        heat_node_id=j_heat_a,
        heat_return_node_id=j_heat_b,
    )
    return pn, p_slack, j_main, j_heat_a, j_heat_b


def test_p2h_heat_junctions_isolated():
    # GIVEN
    pn, p_slack, j_main, j_heat_a, j_heat_b = _build_p2h_heat_isolated(deactivate=True)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    # disconnected heat cluster is NaN'd; control node stays active, heat contribution zeroed
    _assert_junction_nan(result, j_heat_a, "j_heat_a")
    _assert_junction_nan(result, j_heat_b, "j_heat_b")
    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H heat isolated")

    # rest of the network stays intact
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_main, "j_main")


def test_p2h_heat_junctions_connected_baseline():
    # GIVEN
    pn, _, _, j_heat_a, _ = _build_p2h_heat_isolated(deactivate=False)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H heat baseline")
    _assert_junction_solved(result, j_heat_a, "j_heat_a")


def _build_chp_gas_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_gas_chp = _gas_grid(pn, isolated_junction=deactivate)
    _, _, j_heat_supply, j_heat_return = _heat_grid_full(pn)
    _, p_slack, p_gen, p_chp = _power_grid(pn)
    pn.branch(mm.PowerLine(**_LINE), p_gen, p_chp)
    pn.compound(
        mm.CHP(0.3, 0.6, 0.4, 0.001),
        gas_node_id=j_gas_chp,
        heat_node_id=j_heat_supply,
        heat_return_node_id=j_heat_return,
        power_node_id=p_chp,
    )
    return pn, p_slack, j_gas_chp, j_heat_supply


def test_chp_gas_junction_isolated():
    # GIVEN
    pn, p_slack, j_gas_chp, j_heat_supply = _build_chp_gas_isolated(deactivate=True)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    # disconnected gas junction is NaN'd; control node stays active, gas + heat contribution zeroed
    _assert_junction_nan(result, j_gas_chp, "j_gas_chp")
    _assert_control_node_solved(result, "CHPControlNode", "CHP gas isolated")

    # rest of the network stays intact
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_heat_supply, "j_heat_supply")


def test_chp_gas_junction_connected_baseline():
    # GIVEN
    pn, _, j_gas_chp, _ = _build_chp_gas_isolated(deactivate=False)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    _assert_control_node_solved(result, "CHPControlNode", "CHP baseline")
    _assert_junction_solved(result, j_gas_chp, "j_gas_chp")


def _build_chp_power_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_gas_chp = _gas_grid(pn)
    _, _, j_heat_supply, j_heat_return = _heat_grid_full(pn)
    _, p_slack, p_gen, p_chp = _power_grid(pn)
    _deactivatable_power_line(pn, p_gen, p_chp, deactivate)
    pn.compound(
        mm.CHP(0.3, 0.6, 0.4, 0.001),
        gas_node_id=j_gas_chp,
        heat_node_id=j_heat_supply,
        heat_return_node_id=j_heat_return,
        power_node_id=p_chp,
    )
    return pn, p_slack, p_chp, j_heat_supply, j_gas_chp


def test_chp_power_bus_isolated():
    # GIVEN
    pn, p_slack, p_chp, j_heat_supply, j_gas_chp = _build_chp_power_isolated(
        deactivate=True
    )

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    # disconnected power bus is NaN'd; control node stays active, contribution zeroed
    _assert_bus_nan(result, p_chp, "p_chp")
    _assert_control_node_solved(result, "CHPControlNode", "CHP bus isolated")

    # rest of the network stays intact
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_heat_supply, "j_heat_supply")
    _assert_junction_solved(result, j_gas_chp, "j_gas_chp")


def _build_g2h_gas_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_gas_g2h = _gas_grid(pn, isolated_junction=deactivate)
    _, _, j_heat_supply, j_heat_return = _heat_grid_full(pn)
    pn.compound(
        mm.GasToHeat(0.02, 0.15, 300, 0.9),
        gas_node_id=j_gas_g2h,
        heat_node_id=j_heat_supply,
        heat_return_node_id=j_heat_return,
    )
    return pn, j_gas_g2h, j_heat_supply


def test_g2h_gas_junction_isolated():
    # GIVEN
    pn, j_gas_g2h, j_heat_supply = _build_g2h_gas_isolated(deactivate=True)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    # disconnected gas junction is NaN'd; control node stays active, heat contribution zeroed
    _assert_junction_nan(result, j_gas_g2h, "j_gas_g2h")
    _assert_control_node_solved(
        result, "GasToHeatControlNode", "GasToHeat gas isolated"
    )

    _assert_junction_solved(result, j_heat_supply, "j_heat_supply")


def test_g2h_gas_junction_connected_baseline():
    # GIVEN
    pn, j_gas_g2h, _ = _build_g2h_gas_isolated(deactivate=False)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    _assert_control_node_solved(result, "GasToHeatControlNode", "GasToHeat baseline")
    _assert_junction_solved(result, j_gas_g2h, "j_gas_g2h")


def _build_g2h_heat_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_gas_g2h = _gas_grid(pn)
    grid, _, j_heat_main = _heat_grid_main(pn)
    j_heat_a, j_heat_b = _heat_cluster(
        pn, grid, j_heat_main, deactivate_bridge=deactivate
    )
    pn.compound(
        mm.GasToHeat(0.02, 0.15, 300, 0.9),
        gas_node_id=j_gas_g2h,
        heat_node_id=j_heat_a,
        heat_return_node_id=j_heat_b,
    )
    return pn, j_gas_g2h, j_heat_main, j_heat_a, j_heat_b


def test_g2h_heat_junctions_isolated():
    # GIVEN
    pn, j_gas_g2h, j_heat_main, j_heat_a, j_heat_b = _build_g2h_heat_isolated(
        deactivate=True
    )

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    # disconnected heat cluster is NaN'd; control node stays active, heat contribution zeroed
    _assert_junction_nan(result, j_heat_a, "j_heat_a")
    _assert_junction_nan(result, j_heat_b, "j_heat_b")
    _assert_control_node_solved(
        result, "GasToHeatControlNode", "GasToHeat heat isolated"
    )

    # rest of the network stays intact
    _assert_junction_solved(result, j_heat_main, "j_heat_main")
    _assert_junction_solved(result, j_gas_g2h, "j_gas_g2h")


def test_g2h_heat_junctions_connected_baseline():
    # GIVEN
    pn, j_gas_g2h, _, j_heat_a, _ = _build_g2h_heat_isolated(deactivate=False)

    # WHEN
    result = _solve(pn)

    # THEN
    assert result.success

    _assert_control_node_solved(
        result, "GasToHeatControlNode", "GasToHeat heat baseline"
    )
    _assert_junction_solved(result, j_heat_a, "j_heat_a")
    _assert_junction_solved(result, j_gas_g2h, "j_gas_g2h")
