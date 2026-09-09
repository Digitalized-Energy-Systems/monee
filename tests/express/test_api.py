import inspect

import pytest

from monee import mm, mx, run_energy_flow


def test_api_el():
    # GIVEN
    net = mm.Network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    bus_2 = mx.create_bus(net)

    mx.create_ext_power_grid(net, bus_1)
    mx.create_power_generator(net, bus_0, 1, 0)
    mx.create_power_load(net, bus_2, 1, 0)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
    mx.create_line(net, bus_0, bus_2, 200, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)

    junc_0 = mx.create_gas_junction(net)
    junc_1 = mx.create_gas_junction(net)
    junc_2 = mx.create_gas_junction(net)

    mx.create_gas_pipe(net, junc_0, junc_1, diameter_m=0.3, length_m=100)
    mx.create_gas_pipe(net, junc_0, junc_2, diameter_m=0.3, length_m=100)

    mx.create_ext_hydr_grid(net, junc_1)
    mx.create_sink(net, junc_2, 1)

    mx.create_p2g(net, bus_0, junc_0, efficiency=0.9, mass_flow_setpoint_kgs=0.1)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def test_api_el_super_ex():
    # GIVEN
    net = mx.create_multi_energy_network()

    mx.create_ext_power_grid(net, 1)
    mx.create_power_generator(net, 0, 1, 0)
    mx.create_power_load(net, 2, 1, 0)
    mx.create_line(net, 0, 1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
    mx.create_line(net, 0, 2, 200, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
    mx.create_gas_pipe(net, 3, 4, diameter_m=0.6, length_m=100)
    mx.create_gas_pipe(net, 3, 5, diameter_m=0.6, length_m=100)
    mx.create_ext_hydr_grid(net, 4)
    mx.create_sink(net, 5, 1)
    mx.create_sink(net, 3, 1)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def _build_api_example_index(with_p2h: bool):
    # change doc if you change this!
    net = mx.create_multi_energy_network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, 0.1, 0)

    junc_0 = mx.create_water_junction(net)
    junc_1 = mx.create_water_junction(net)
    junc_2 = mx.create_water_junction(net)

    mx.create_ext_hydr_grid(net, junc_0)
    mx.create_water_pipe(net, junc_0, junc_1, diameter_m=0.12, length_m=100)
    mx.create_sink(net, junc_2, mass_flow_kgs=1)

    if with_p2h:
        mx.create_p2h(
            net,
            bus_1,
            junc_1,
            junc_2,
            heat_energy_mw=0.100,
            diameter_m=0.1,
            efficiency=0.9,
        )
    return net


def test_api_example_index():
    # GIVEN
    net = _build_api_example_index(with_p2h=True)
    net_without_p2h = _build_api_example_index(with_p2h=False)

    # WHEN
    result = run_energy_flow(net)
    result_without_p2h = run_energy_flow(net_without_p2h)

    # THEN
    assert result.success
    assert result_without_p2h.success

    # The P2H draws heat_energy_mw / efficiency from the bus, so the external
    # grid has to import that much more (ext p_mw is negative for an import).
    ext_p_mw = result.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
    ext_p_mw_without_p2h = result_without_p2h.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
    assert ext_p_mw < ext_p_mw_without_p2h
    assert ext_p_mw_without_p2h - ext_p_mw == pytest.approx(0.100 / 0.9, rel=0.05)


def test_ext_power_grid_defaults():
    # WHEN
    sig = inspect.signature(mx.create_ext_power_grid)

    # THEN
    assert sig.parameters["p_mw"].default == 0
    assert sig.parameters["q_mvar"].default == 0


def test_trafo():
    # GIVEN
    net = mx.create_multi_energy_network()

    bus_hv = mx.create_bus(net, base_kv=110)
    bus_lv = mx.create_bus(net, base_kv=11)

    mx.create_ext_power_grid(net, bus_hv)
    mx.create_power_load(net, bus_lv, p_mw=1, q_mvar=0)
    mx.create_trafo(net, bus_lv, bus_hv, sn_trafo_mva=160)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def test_gas_domain_functions():
    # GIVEN
    # ext_grid and sink are called first so they auto-create the gas junctions
    net = mx.create_multi_energy_network()

    mx.create_gas_ext_grid(net, 0)
    mx.create_gas_sink(net, 1, mass_flow_kgs=0.1)
    mx.create_gas_pipe(net, 0, 1, diameter_m=0.1, length_m=100)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def test_gas_source_auto_node():
    # GIVEN
    # create_gas_source with a new node_id auto-creates a gas junction
    net = mx.create_multi_energy_network()

    mx.create_gas_source(net, 0, mass_flow_kgs=0.1)
    mx.create_gas_ext_grid(net, 1)
    mx.create_gas_pipe(net, 0, 1, diameter_m=0.1, length_m=100)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def test_water_domain_functions():
    # GIVEN
    # create_water_ext_grid and create_water_sink auto-create water junctions
    net = mx.create_multi_energy_network()

    mx.create_water_ext_grid(net, 0)
    mx.create_water_sink(net, 1, mass_flow_kgs=1)
    mx.create_water_pipe(net, 0, 1, diameter_m=0.12, length_m=100)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def test_water_source_auto_node():
    # GIVEN
    # source (0.5 kg/s) and sink (1.5 kg/s) share auto-created node 1, giving
    # 1 kg/s net consumption with forward pipe flow (0->1)
    net = mx.create_multi_energy_network()

    mx.create_water_ext_grid(net, 0)
    mx.create_water_source(net, 1, mass_flow_kgs=0.5)
    mx.create_water_sink(net, 1, mass_flow_kgs=1.5)
    mx.create_water_pipe(net, 0, 1, diameter_m=0.12, length_m=100)

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success

    assert result is not None


def test_create_el_branch():
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    bid = mx.create_el_branch(
        net,
        b0,
        b1,
        mm.PowerLine(
            length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1, max_i_ka=0.3
        ),
    )
    assert bid is not None
    assert net.branch_by_id(bid) is not None


def _heat_compound_nodes(net, cold, hot):
    control = {b.to_node_id for b in net.branches if b.from_node_id == cold} & {
        b.from_node_id for b in net.branches if b.to_node_id == hot
    }
    return control


@pytest.mark.parametrize("kind", ["p2h", "g2h", "chp"])
def test_heat_compound_flows_from_cold_node_to_hot_node(kind):
    # GIVEN
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    junc_gas = mx.create_gas_junction(net)
    cold = mx.create_water_junction(net)
    hot = mx.create_water_junction(net)

    # WHEN
    if kind == "p2h":
        mx.create_p2h(
            net,
            power_node_id=bus,
            cold_node_id=cold,
            hot_node_id=hot,
            heat_energy_mw=0.1,
            diameter_m=0.1,
            efficiency=0.9,
        )
    elif kind == "g2h":
        mx.create_g2h(
            net,
            gas_node_id=junc_gas,
            cold_node_id=cold,
            hot_node_id=hot,
            heat_energy_mw=0.1,
            diameter_m=0.1,
            efficiency=0.9,
        )
    else:
        mx.create_chp(
            net,
            power_node_id=bus,
            cold_node_id=cold,
            hot_node_id=hot,
            gas_node_id=junc_gas,
            diameter_m=0.1,
            efficiency_power=0.35,
            efficiency_heat=0.45,
            mass_flow_setpoint_kgs=0.05,
        )

    # THEN the water runs cold node -> control node -> hot node
    assert _heat_compound_nodes(net, cold, hot)
    assert not _heat_compound_nodes(net, hot, cold)


def test_heat_side_aliases_wire_like_the_original_names():
    # GIVEN
    def build(use_aliases):
        net = mx.create_multi_energy_network()
        bus = mx.create_bus(net)
        cold = mx.create_water_junction(net)
        hot = mx.create_water_junction(net)
        kwargs = (
            {"cold_node_id": cold, "hot_node_id": hot}
            if use_aliases
            else {"heat_node_id": cold, "heat_return_node_id": hot}
        )
        mx.create_p2h(
            net,
            power_node_id=bus,
            heat_energy_mw=0.1,
            diameter_m=0.1,
            efficiency=0.9,
            **kwargs,
        )
        return net

    # THEN
    assert [(b.from_node_id, b.to_node_id) for b in build(True).branches] == [
        (b.from_node_id, b.to_node_id) for b in build(False).branches
    ]


def test_heat_side_aliases_reject_ambiguous_and_missing_wiring():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    cold = mx.create_water_junction(net)
    hot = mx.create_water_junction(net)

    with pytest.raises(ValueError):
        mx.create_p2h(
            net,
            power_node_id=bus,
            heat_node_id=cold,
            cold_node_id=cold,
            hot_node_id=hot,
            heat_energy_mw=0.1,
            diameter_m=0.1,
            efficiency=0.9,
        )
    with pytest.raises(ValueError):
        mx.create_p2h(
            net,
            power_node_id=bus,
            cold_node_id=cold,
            heat_energy_mw=0.1,
            diameter_m=0.1,
            efficiency=0.9,
        )


def test_heat_compound_heats_the_water_towards_the_hot_node():
    # GIVEN a closed loop with the plant reference pinned on the cold side
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    cold = mx.create_water_junction(net)
    hot = mx.create_water_junction(net)
    mx.create_water_ext_grid(net, cold, t_k=330)
    mx.create_passive_heat_exchanger(net, hot, cold, q_mw=0.5, diameter_m=0.1)
    mx.create_p2h(
        net,
        power_node_id=bus,
        cold_node_id=cold,
        hot_node_id=hot,
        heat_energy_mw=0.5,
        diameter_m=0.1,
        efficiency=0.95,
    )

    # WHEN
    result = run_energy_flow(net)

    # THEN
    assert result.success
    temperatures = result.get(mm.Junction)["t_k"]
    assert temperatures.iloc[0] == pytest.approx(330, abs=1e-3)
    assert temperatures.iloc[1] > temperatures.iloc[0] + 25


def test_compound_creators_store_name():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    junc_gas = mx.create_gas_junction(net)
    cold = mx.create_water_junction(net)
    hot = mx.create_water_junction(net)

    chp_id = mx.create_chp(
        net,
        power_node_id=bus,
        cold_node_id=cold,
        hot_node_id=hot,
        gas_node_id=junc_gas,
        diameter_m=0.1,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.05,
        name="chp_1",
    )
    p2h_id = mx.create_p2h(
        net,
        power_node_id=bus,
        cold_node_id=cold,
        hot_node_id=hot,
        heat_energy_mw=0.1,
        diameter_m=0.1,
        efficiency=0.9,
        name="hp_1",
    )
    g2h_id = mx.create_g2h(
        net,
        gas_node_id=junc_gas,
        cold_node_id=cold,
        hot_node_id=hot,
        heat_energy_mw=0.1,
        diameter_m=0.1,
        efficiency=0.9,
        name="boiler_1",
    )
    chp_hg_id = mx.create_chp_hg(
        net,
        power_node_id=bus,
        heat_node_id=mx.create_water_junction(net),
        gas_node_id=junc_gas,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.05,
        name="chp_hg_1",
    )

    assert net.compound_by_id(chp_id).name == "chp_1"
    assert net.compound_by_id(p2h_id).name == "hp_1"
    assert net.compound_by_id(g2h_id).name == "boiler_1"
    assert net.compound_by_id(chp_hg_id).name == "chp_hg_1"


def test_hg_coupler_branch_creators_store_name():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    junc_gas = mx.create_gas_junction(net)
    heat_1 = mx.create_water_junction(net)
    heat_2 = mx.create_water_junction(net)

    p2h_id = mx.create_p2h_hg(
        net, bus, heat_1, heat_energy_mw=0.1, efficiency=0.9, name="p2h_hg_1"
    )
    g2h_id = mx.create_g2h_hg(
        net, junc_gas, heat_2, heat_energy_mw=0.1, efficiency=0.9, name="g2h_hg_1"
    )

    assert net.branch_by_id(p2h_id).name == "p2h_hg_1"
    assert net.branch_by_id(g2h_id).name == "g2h_hg_1"


def test_name_lands_on_the_container_not_on_the_model():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net, name="bus_a")
    load_id = mx.create_power_load(net, bus, p_mw=0.1, q_mvar=0.0, name="hp_feeder")

    load = net.child_by_id(load_id)

    assert load.name == "hp_feeder"
    assert net.node_by_id(bus).name == "bus_a"
    assert not hasattr(load.model, "name")
    with pytest.raises(AttributeError):
        load.model.name
