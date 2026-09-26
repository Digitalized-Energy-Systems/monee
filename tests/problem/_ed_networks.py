"""Small networks for the multi-carrier economic dispatch tests. Every builder
returns the network plus the ids its tests read back."""

import monee.express as mx
import monee.model as mm


def gas_merit_net(source_cost, gas_type="lgas"):
    net = mm.Network(gas_model=mm.create_gas_grid("gas", type=gas_type))
    j0, j1, j2 = (mx.create_gas_junction(net) for _ in range(3))
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, j1, j2, diameter_m=0.15, length_m=400)
    ext_id = mx.create_gas_ext_grid(net, j0, max_export_kgs=0)
    mx.create_gas_sink(net, j1, mass_flow_kgs=0.15)
    source_id = mx.create_gas_source(net, j2, mass_flow_kgs=0.1, cost=source_cost)
    return net, source_id, ext_id


def gas_pressure_net():
    net = mm.Network(gas_model=mm.create_gas_grid("gas"))
    j0, j1, j2 = (mx.create_gas_junction(net) for _ in range(3))
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, j1, j2, diameter_m=0.05, length_m=3000)
    mx.create_gas_ext_grid(net, j0, max_export_kgs=0)
    mx.create_gas_sink(net, j1, mass_flow_kgs=0.2)
    source_id = mx.create_gas_source(net, j2, mass_flow_kgs=0.2, cost=1.0)
    return net, source_id, j2


def heat_dhs_net(generator_cost):
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=200)
    seg = dhs.line(2, heat_exchanger_q_mw=[0.15, 0.2])
    plant_id, _ = dhs.attach_heat_plant(seg.supply.first, seg.return_.last, t_k=358.0)
    generator_id = mx.create_heat_generator(
        net, seg.supply.last, q_mw=0.2, cost=generator_cost
    )
    return net, generator_id, plant_id, seg.return_.last


def heat_loop_net(generator_cost):
    net = mm.Network()
    r, s, s2 = (mx.create_water_junction(net) for _ in range(3))
    slack_id = mx.create_water_ext_grid(net, r, t_k=330)
    hx_id = mx.create_heat_exchanger(net, r, s, q_mw=-0.6, cost=generator_cost)
    mx.create_water_pipe(net, s, s2, diameter_m=0.15, length_m=100)
    mx.create_passive_heat_exchanger(net, s2, r, q_mw=0.5, diameter_m=0.15)
    return net, hx_id, slack_id


def chp_hg_net():
    net = mx.create_multi_energy_network()
    bus0, bus1 = mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, bus0, bus1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus0)
    mx.create_power_load(net, bus1, p_mw=1.0, q_mvar=0.0)
    g0, g1 = mx.create_gas_junction(net), mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, g0)
    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    w0, w1 = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_ext_grid(net, w0)
    mx.create_water_pipe(net, w0, w1, diameter_m=0.1, length_m=100)
    mx.create_water_sink(net, w1, mass_flow_kgs=3.0)
    mx.create_heat_load(net, w1, q_mw=0.3)
    mx.create_chp_hg(
        net,
        bus1,
        w1,
        g1,
        efficiency_power=0.4,
        efficiency_heat=0.4,
        mass_flow_setpoint_kgs=0.01,
    )
    return net


def hg_couplers_net():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    gj = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gj)
    j0, j1 = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_ext_grid(net, j0)
    mx.create_water_pipe(net, j0, j1, diameter_m=0.1, length_m=100)
    mx.create_water_sink(net, j1, mass_flow_kgs=1.0)
    mx.create_heat_load(net, j1, q_mw=0.3)
    mx.create_p2h_hg(net, bus, j1, heat_energy_mw=0.2, efficiency=0.95)
    mx.create_g2h_hg(net, gj, j1, heat_energy_mw=0.2, efficiency=0.9)
    return net


def g2p_net():
    net = mx.create_multi_energy_network()
    b0, b1 = mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, b1, p_mw=1.0, q_mvar=0.0)
    g0, g1 = mx.create_gas_junction(net), mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, g0)
    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_sink(net, g1, mass_flow_kgs=0.01)
    mx.create_g2p(net, g1, b1, efficiency=0.5, p_mw_setpoint=0.8)
    return net


def gas_storage_net():
    net = mm.Network(gas_model=mm.create_gas_grid("gas"))
    j0, j1 = mx.create_gas_junction(net), mx.create_gas_junction(net)
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.15, length_m=400)
    ext_id = mx.create_gas_ext_grid(net, j0)
    mx.create_gas_sink(net, j1, mass_flow_kgs=0.1)
    storage_id = mx.create_gas_child(net, mm.GasStorage(0, 720, 0.1), j1)
    return net, storage_id, ext_id


def ramp_collision_net():
    """The generator's child id equals its bus id, the case a lookup by
    attribute alone resolves to the bus."""
    net = mx.create_multi_energy_network()
    b0, b1 = mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_ext_power_grid(net, b0)
    generator_id = mx.create_power_generator(net, b1, p_mw=2.0, q_mvar=0.0, cost=10)
    mx.create_power_load(net, b1, p_mw=2.0, q_mvar=0.0)
    assert generator_id == b1
    return net, generator_id


def docs_el_net():
    """The network of the problems/economic_dispatch worked example."""
    net = mx.create_multi_energy_network()
    b0, b1, b2 = mx.create_bus(net), mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, 200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_line(net, b1, b2, 200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_generator(net, b1, p_mw=0.6, q_mvar=0.0, cost=10, name="cheap")
    mx.create_power_generator(net, b2, p_mw=0.6, q_mvar=0.0, cost=40, name="pricey")
    mx.create_power_load(net, b2, p_mw=0.8, q_mvar=0.0, name="demand")
    return net
