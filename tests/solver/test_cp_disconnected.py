import math

import monee.model as mm
from monee.solver import PyomoSolver

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
    """j_ext(ExtHydrGrid) → j_hub → j_sink(Sink). Returns (grid, j_ext, j_hub).

    Three junctions are used instead of two: Pyomo requires at least one
    intermediate (pass-through) junction to converge on mixed heat+power networks.
    ``j_hub`` serves as the attachment point for isolated heat clusters.
    """
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
    """
    Create j_a → j_b, bridged to j_anchor via *grid*.  When the bridge is
    deactivated neither junction has a path to ExtHydrGrid and both enter
    ``ignored_nodes``.  Returns (j_a, j_b).
    """
    j_a = pn.node(mm.Junction(), grid=grid)
    j_b = pn.node(mm.Junction(), grid=grid)
    pn.branch(mm.WaterPipe(**_PIPE), j_a, j_b)
    bridge_id = pn.branch(mm.WaterPipe(**_PIPE), j_anchor, j_a)
    if deactivate_bridge:
        pn.branch_by_id(bridge_id).active = False
    return j_a, j_b


def _gas_grid(pn, *, isolated_junction=False):
    """
    j_hub(ExtHydrGrid) → j_mid → j_main(Source) → [bridge] → j_iso.
    Three junctions remain active even when j_iso is cut off (Pyomo requires
    at least 3 active gas nodes to converge on mixed multi-domain networks).
    Returns (j_hub, j_main, j_iso).
    """
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


def _assert_junction_nan(result, jct_id, label):
    # t_pu is a Var on Junction and gets NaN'd by inject_nans for ignored nodes.
    # t_k is Intermediate (derived via IntermediateEq) and is not NaN'd directly.
    df = result.dataframes["Junction"]
    t_pu = df.loc[df["id"] == jct_id, "t_pu"].iloc[0]
    assert math.isnan(t_pu), f"{label}: expected NaN t_pu, got {t_pu}"


def _assert_junction_solved(result, jct_id, label):
    df = result.dataframes["Junction"]
    t_pu = df.loc[df["id"] == jct_id, "t_pu"].iloc[0]
    assert not math.isnan(t_pu), f"{label}: junction must be solved (got NaN)"


def _assert_control_node_nan(result, model_name, label):
    """Check that the control node row has NaN t_pu (a Var on all CP control nodes)."""
    df = result.dataframes[model_name]
    t_pu = df["t_pu"].iloc[0]
    assert math.isnan(t_pu), f"{label}: {model_name}.t_pu must be NaN (got {t_pu})"


def _assert_control_node_solved(result, model_name, label):
    df = result.dataframes[model_name]
    t_pu = df["t_pu"].iloc[0]
    assert not math.isnan(t_pu), f"{label}: {model_name}.t_pu must be solved (got NaN)"


def _solve(net):
    return PyomoSolver().solve(net, exclude_unconnected_nodes=True)


def _build_p2h_power_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_supply, j_return = _heat_grid_full(pn)
    _, p_slack, p_gen, p_p2h = _power_grid(pn)
    _deactivatable_power_line(pn, p_gen, p_p2h, deactivate)
    pn.compound(
        mm.PowerToHeat(10_000, 0.15, 300, 0.9),
        power_node_id=p_p2h,
        heat_node_id=j_supply,
        heat_return_node_id=j_return,
    )
    return pn, p_slack, p_p2h, j_supply


def test_p2h_power_bus_isolated():
    """P2H: power_node (Bus) disconnected → compound deactivated (heat_mw=0), heat grid intact.

    The control node stays active as a thermal junction (set_active zeroes the
    heat contribution without removing the node from the thermal grid).
    """
    pn, p_slack, p_p2h, j_supply = _build_p2h_power_isolated(deactivate=True)
    result = _solve(pn)
    print(result)
    _assert_bus_nan(result, p_p2h, "p_p2h")
    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H bus isolated")
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_supply, "j_supply")


def test_p2h_power_bus_connected_baseline():
    """P2H: all nodes connected → compound solves normally."""
    pn, p_slack, _, _ = _build_p2h_power_isolated(deactivate=False)
    result = _solve(pn)
    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H baseline")
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)


def _build_p2h_heat_isolated(deactivate: bool):
    pn = mm.Network()
    grid, _, j_main = _heat_grid_main(pn)
    j_heat_a, j_heat_b = _heat_cluster(pn, grid, j_main, deactivate_bridge=deactivate)
    _, p_slack, p_gen, p_p2h = _power_grid(pn)
    pn.branch(mm.PowerLine(**_LINE), p_gen, p_p2h)
    pn.compound(
        mm.PowerToHeat(10_000, 0.15, 300, 0.9),
        power_node_id=p_p2h,
        heat_node_id=j_heat_a,
        heat_return_node_id=j_heat_b,
    )
    return pn, p_slack, j_main, j_heat_a, j_heat_b


def test_p2h_heat_junctions_isolated():
    """P2H: heat_node + heat_return_node cluster disconnected → compound deactivated, power intact.

    The control node stays active as a thermal junction; heat contribution is zeroed.
    """
    pn, p_slack, j_main, j_heat_a, j_heat_b = _build_p2h_heat_isolated(deactivate=True)
    result = _solve(pn)
    _assert_junction_nan(result, j_heat_a, "j_heat_a")
    _assert_junction_nan(result, j_heat_b, "j_heat_b")
    _assert_control_node_solved(result, "PowerToHeatControlNode", "P2H heat isolated")
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_main, "j_main")


def test_p2h_heat_junctions_connected_baseline():
    """P2H: heat cluster connected → compound solves normally."""
    pn, _, _, j_heat_a, _ = _build_p2h_heat_isolated(deactivate=False)
    result = _solve(pn)
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
    """CHP: gas_node (Junction) disconnected → compound deactivated, power + heat intact.

    The control node stays active as a thermal/power junction; gas + heat contribution zeroed.
    """
    pn, p_slack, j_gas_chp, j_heat_supply = _build_chp_gas_isolated(deactivate=True)
    result = _solve(pn)
    _assert_junction_nan(result, j_gas_chp, "j_gas_chp")
    _assert_control_node_solved(result, "CHPControlNode", "CHP gas isolated")
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_heat_supply, "j_heat_supply")


def test_chp_gas_junction_connected_baseline():
    """CHP: all nodes connected → compound solves normally."""
    pn, _, j_gas_chp, _ = _build_chp_gas_isolated(deactivate=False)
    result = _solve(pn)
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
    """CHP: power_node (Bus) disconnected → compound deactivated, heat + gas intact.

    The control node stays active as a thermal junction; contribution zeroed via set_active.
    """
    pn, p_slack, p_chp, j_heat_supply, j_gas_chp = _build_chp_power_isolated(
        deactivate=True
    )
    result = _solve(pn)
    _assert_bus_nan(result, p_chp, "p_chp")
    _assert_control_node_solved(result, "CHPControlNode", "CHP bus isolated")
    _assert_bus_solved(result, p_slack, "p_slack", expected_vm=1.0)
    _assert_junction_solved(result, j_heat_supply, "j_heat_supply")
    _assert_junction_solved(result, j_gas_chp, "j_gas_chp")


def _build_g2h_gas_isolated(deactivate: bool):
    pn = mm.Network()
    _, _, j_gas_g2h = _gas_grid(pn, isolated_junction=deactivate)
    _, _, j_heat_supply, j_heat_return = _heat_grid_full(pn)
    pn.compound(
        mm.GasToHeat(20_000, 0.15, 300, 0.9),
        gas_node_id=j_gas_g2h,
        heat_node_id=j_heat_supply,
        heat_return_node_id=j_heat_return,
    )
    return pn, j_gas_g2h, j_heat_supply


def test_g2h_gas_junction_isolated():
    """GasToHeat: gas_node (Junction) disconnected → compound deactivated, heat grid intact.

    The control node stays active as a thermal junction; heat contribution zeroed via set_active.
    """
    pn, j_gas_g2h, j_heat_supply = _build_g2h_gas_isolated(deactivate=True)
    result = _solve(pn)
    _assert_junction_nan(result, j_gas_g2h, "j_gas_g2h")
    _assert_control_node_solved(
        result, "GasToHeatControlNode", "GasToHeat gas isolated"
    )
    _assert_junction_solved(result, j_heat_supply, "j_heat_supply")


def test_g2h_gas_junction_connected_baseline():
    """GasToHeat: all nodes connected → compound solves normally."""
    pn, j_gas_g2h, _ = _build_g2h_gas_isolated(deactivate=False)
    result = _solve(pn)
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
        mm.GasToHeat(20_000, 0.15, 300, 0.9),
        gas_node_id=j_gas_g2h,
        heat_node_id=j_heat_a,
        heat_return_node_id=j_heat_b,
    )
    return pn, j_gas_g2h, j_heat_main, j_heat_a, j_heat_b


def test_g2h_heat_junctions_isolated():
    """GasToHeat: heat_node + heat_return_node cluster disconnected → compound deactivated, gas intact.

    The control node stays active as a gas junction; heat contribution zeroed via set_active.
    """
    pn, j_gas_g2h, j_heat_main, j_heat_a, j_heat_b = _build_g2h_heat_isolated(
        deactivate=True
    )
    result = _solve(pn)
    _assert_junction_nan(result, j_heat_a, "j_heat_a")
    _assert_junction_nan(result, j_heat_b, "j_heat_b")
    _assert_control_node_solved(
        result, "GasToHeatControlNode", "GasToHeat heat isolated"
    )
    _assert_junction_solved(result, j_heat_main, "j_heat_main")
    _assert_junction_solved(result, j_gas_g2h, "j_gas_g2h")


def test_g2h_heat_junctions_connected_baseline():
    """GasToHeat: heat cluster connected → compound solves normally."""
    pn, j_gas_g2h, _, j_heat_a, _ = _build_g2h_heat_isolated(deactivate=False)
    result = _solve(pn)
    _assert_control_node_solved(
        result, "GasToHeatControlNode", "GasToHeat heat baseline"
    )
    _assert_junction_solved(result, j_heat_a, "j_heat_a")
    _assert_junction_solved(result, j_gas_g2h, "j_gas_g2h")
