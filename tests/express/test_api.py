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
    assert False
    assert result is not None
