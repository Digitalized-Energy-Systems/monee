import pytest

import monee
import monee.express as mx
import monee.model as mm


def _dead_end_heat_load_net():
    net = mm.Network(mm.create_water_grid("water"))
    j0 = mx.create_water_junction(net)
    j1 = mx.create_water_junction(net)
    j2 = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, j0)
    mx.create_water_pipe(net, j0, j1, diameter_m=0.2, length_m=100)
    mx.create_water_pipe(net, j1, j2, diameter_m=0.2, length_m=100)
    mx.create_sink(net, j1, mass_flow_kgs=1)
    mx.create_heat_load(net, j2, q_mw=0.1, name="my heat load")
    return net


def test_dead_end_heat_load_pruning_names_load_and_rule():
    with pytest.warns(UserWarning, match="my heat load"):
        result = monee.run_energy_flow(_dead_end_heat_load_net())

    assert result.success
    pruned = [w for w in result.warnings if w.category == "pruned"]
    assert len(pruned) == 1
    message = pruned[0].message
    assert "HeatLoad" in message
    assert "my heat load" in message
    assert "heat-only" in message
    assert "diagnose_infeasibility" in message


def test_islanded_component_pruning_names_rule():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.0)
    b2 = mx.create_bus(net)
    b3 = mx.create_bus(net)
    mx.create_line(net, b2, b3, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b3, p_mw=0.5, q_mvar=0.0, name="islanded load")

    with pytest.warns(UserWarning, match="grid-forming"):
        result = monee.run_energy_flow(net)

    assert result.success
    pruned = next(w for w in result.warnings if w.category == "pruned")
    assert "islanded load" in pruned.message
    assert "slack or grid-forming" in pruned.message


def test_unpruned_net_emits_no_pruned_entry():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.0)

    result = monee.run_energy_flow(net)

    assert result.success
    assert [w for w in result.warnings if w.category == "pruned"] == []


def test_a_branch_coupler_anchors_its_leaf_gas_junction():
    """Regression: the dead-end rule runs on a copy without multi-grid
    branches, so a P2G feeding a childless leaf junction was pruned together
    with the junction (success=True, only a 'pruned' warning). Compound ports
    were already exempt; the coupler's injection anchors the node the same way."""
    net = mm.Network()
    b0 = mx.create_bus(net, base_kv=20)
    mx.create_ext_power_grid(net, b0)
    g0 = mx.create_gas_junction(net)
    g1 = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, g0)
    mx.create_gas_pipe(net, g0, g1, 0.1, 1000)
    mx.create_p2g(net, b0, g1, efficiency=0.7, mass_flow_setpoint_kgs=0.005)

    result = monee.run_energy_flow(net)

    assert result.success
    assert [w for w in result.warnings if w.category == "pruned"] == []
    p2g = result.get(mm.PowerToGas).iloc[0]
    assert not p2g["ignored"]
    assert p2g["to_mass_flow_kgs"] == pytest.approx(-0.005)


def _gas_fed_boiler_net(sink_kgs=0.0, boiler_regulation=0.0):
    """A gas island whose city gate is out, feeding a boiler of a heat network."""
    net = mm.Network(
        el_model=mm.create_power_grid("power"),
        gas_model=mm.create_gas_grid("gas", type="lgas"),
        water_model=mm.create_water_grid("heat"),
    )
    g0, g1 = mx.create_gas_junction(net), mx.create_gas_junction(net)
    mx.create_gas_pipe(net, g0, g1, diameter_m=0.1, length_m=100)
    gate = mx.create_gas_ext_grid(net, g0)
    if sink_kgs:
        mx.create_sink(net, g1, mass_flow_kgs=sink_kgs, name="bakery")
    ret, sup = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_ext_grid(net, ret, t_k=333.15)
    mx.create_water_pipe(net, sup, ret, diameter_m=0.3, length_m=20)
    mx.create_g2h(
        net,
        gas_node_id=g1,
        cold_node_id=ret,
        hot_node_id=sup,
        heat_energy_mw=1.0,
        diameter_m=0.2,
        efficiency=0.9,
        regulation=boiler_regulation,
    )
    net.deactivate_by_id(mm.ExtHydrGrid, gate)
    return net, {g0, g1}


def test_source_less_island_without_injections_is_dropped_and_listed():
    net, gas_island = _gas_fed_boiler_net()

    result = monee.run_energy_flow(net, exclude_unconnected_nodes=True)

    assert result.success
    assert result.dropped_islands == [frozenset(gas_island)]


def test_dropped_island_names_its_loads_and_the_coupler_switched_off():
    net, gas_island = _gas_fed_boiler_net(sink_kgs=0.01, boiler_regulation=0.5)

    result = monee.run_energy_flow(net, exclude_unconnected_nodes=True)

    assert result.success
    assert result.dropped_islands == [frozenset(gas_island)]
    pruned = next(w for w in result.warnings if w.category == "pruned")
    assert "Sink(1) 'bakery'" in pruned.message
    assert "GasToHeat(0) [rule: coupler switched off" in pruned.message
    assert result.get(mm.GasToHeatControlNode)["heat_mw"].iloc[0] == pytest.approx(
        0.0, abs=1e-6
    )


def test_heat_network_with_every_unit_and_consumer_off_solves():
    """Regression (resilience V1, gas supply cut): every substation shed and
    every producer off. The off producers' heat ports still bridged the return
    pipes to the reference in the pruning graph, so the return stayed in the
    solve without a slack and IPOPT stopped at Not_Enough_Degrees_Of_Freedom."""
    net = mm.Network(
        el_model=mm.create_power_grid("power"),
        gas_model=mm.create_gas_grid("gas", type="lgas"),
        water_model=mm.create_water_grid("heat"),
    )
    gas = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas)
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    sup, mid, cold, ret_boiler, ret_hp = (
        mx.create_water_junction(net) for _ in range(5)
    )
    mx.create_water_ext_grid(net, sup, t_k=358.15)
    mx.create_water_pipe(net, sup, mid, diameter_m=0.15, length_m=300)
    substation = mx.create_heat_exchanger(net, mid, cold, q_mw=0.5)
    mx.create_water_pipe(net, ret_boiler, cold, diameter_m=0.15, length_m=300)
    mx.create_water_pipe(net, cold, ret_hp, diameter_m=0.15, length_m=300)
    boiler = mx.create_g2h(
        net,
        gas,
        cold_node_id=ret_boiler,
        hot_node_id=sup,
        heat_energy_mw=1.0,
        diameter_m=0.15,
        efficiency=0.9,
    )
    hp = mx.create_p2h(
        net,
        bus,
        cold_node_id=ret_hp,
        hot_node_id=sup,
        heat_energy_mw=1.0,
        diameter_m=0.15,
        efficiency=3.0,
    )
    net.deactivate_by_id(mm.Branch, substation)
    net.deactivate_by_id(mm.Compound, boiler)
    net.deactivate_by_id(mm.Compound, hp)

    result = monee.run_energy_flow(net, exclude_unconnected_nodes=True)

    assert result.success
    assert frozenset({ret_boiler, cold, ret_hp}) in result.dropped_islands
