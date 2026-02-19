import inspect

from monee import mm, mx, run_energy_flow


def test_api_el():
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

    mx.create_gas_pipe(net, junc_0, junc_1, diameter_m=0.1, length_m=100)
    mx.create_gas_pipe(net, junc_0, junc_2, diameter_m=0.1, length_m=100)

    mx.create_ext_hydr_grid(net, junc_1)
    mx.create_sink(net, junc_2, 1)

    mx.create_p2g(net, bus_0, junc_0, efficiency=0.9, mass_flow_setpoint=0.1)

    result = run_energy_flow(net)

    assert result is not None


def test_api_el_super_ex():
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

    result = run_energy_flow(net)

    assert result is not None


def test_api_example_index():
    # change doc if you change this!
    net = mx.create_multi_energy_network()

    # electricity grid
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, 0.1, 0)

    # water-based district heating grid
    junc_0 = mx.create_water_junction(net)
    junc_1 = mx.create_water_junction(net)
    junc_2 = mx.create_water_junction(net)

    mx.create_ext_hydr_grid(net, junc_0)
    mx.create_water_pipe(net, junc_0, junc_1, diameter_m=0.12, length_m=100)
    mx.create_sink(net, junc_2, mass_flow=1)

    # creating connection between el and water grid
    mx.create_p2h(
        net, bus_1, junc_1, junc_2, heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9
    )

    result = run_energy_flow(net)

    print(result)
    assert result is not None


def test_ext_power_grid_defaults():
    sig = inspect.signature(mx.create_ext_power_grid)
    assert sig.parameters["p_mw"].default == 0
    assert sig.parameters["q_mvar"].default == 0


def test_trafo():
    net = mx.create_multi_energy_network()

    bus_hv = mx.create_bus(net, base_kv=110)
    bus_lv = mx.create_bus(net, base_kv=11)

    mx.create_ext_power_grid(net, bus_hv)
    mx.create_power_load(net, bus_lv, p_mw=1, q_mvar=0)
    mx.create_trafo(net, bus_lv, bus_hv, sn_trafo_mva=160)

    result = run_energy_flow(net)

    assert result is not None


def test_gas_domain_functions():
    """create_gas_ext_grid and create_gas_sink auto-create junctions."""
    net = mx.create_multi_energy_network()

    # Call ext_grid and sink first so they trigger auto-node creation
    mx.create_gas_ext_grid(net, 0)
    mx.create_gas_sink(net, 1, mass_flow=0.1)
    mx.create_gas_pipe(net, 0, 1, diameter_m=0.1, length_m=100)

    result = run_energy_flow(net)

    assert result is not None


def test_gas_source_auto_node():
    """create_gas_source with a new node_id auto-creates a gas junction."""
    net = mx.create_multi_energy_network()

    mx.create_gas_source(net, 0, mass_flow=0.1)
    mx.create_gas_ext_grid(net, 1)
    mx.create_gas_pipe(net, 0, 1, diameter_m=0.1, length_m=100)

    result = run_energy_flow(net)

    assert result is not None


def test_water_domain_functions():
    """create_water_ext_grid and create_water_sink auto-create water junctions."""
    net = mx.create_multi_energy_network()

    mx.create_water_ext_grid(net, 0)
    mx.create_water_sink(net, 1, mass_flow=1)
    mx.create_water_pipe(net, 0, 1, diameter_m=0.12, length_m=100)

    result = run_energy_flow(net)

    assert result is not None


def test_water_source_auto_node():
    """create_water_source with a new node_id auto-creates a water junction.

    The source (0.5 kg/s) and a larger sink (1.5 kg/s) share node 1, giving
    a net consumption of 1 kg/s. The pipe carries forward flow (0→1) so the
    direction variable stays at its default initialization of 1.
    """
    net = mx.create_multi_energy_network()

    mx.create_water_ext_grid(net, 0)
    mx.create_water_source(net, 1, mass_flow=0.5)  # auto-creates junction 1
    mx.create_water_sink(net, 1, mass_flow=1.5)  # node 1 already exists
    mx.create_water_pipe(net, 0, 1, diameter_m=0.12, length_m=100)

    result = run_energy_flow(net)

    assert result is not None
