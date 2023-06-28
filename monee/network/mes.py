import monee.model as mm


def create_heat_net_for_power(net_power, target_net, heat_deployment_rate):
    heat_grid = mm.create_water_grid("heat")

    bus_index_to_junction_index = {}
    for i in power_net.bus.index:
        bus_coords = (power_net.bus_geodata["x"][i], power_net.bus_geodata["y"][i])
        junc_index_first = ppipes.create_junction(
            net_heat, pn_bar=REF_BAR, tfluid_k=REF_TEMP, geodata=bus_coords
        )
        junc_index_second = ppipes.create_junction(
            net_heat, pn_bar=REF_BAR, tfluid_k=REF_TEMP, geodata=bus_coords
        )
        bus_index_to_junction_index[i] = junc_index_first
        bus_index_to_junction_index[
            i + start_return_net_id_junctions
        ] = junc_index_second

    for i, from_bus, to_bus in zip(
        power_net.trafo.index,
        power_net.trafo["lv_bus"],
        power_net.trafo["hv_bus"],
    ):
        bus_coords = (
            power_net.bus_geodata["x"][from_bus],
            power_net.bus_geodata["y"][from_bus],
        )
        v_feed = ppipes.create_junction(
            net_heat, pn_bar=REF_BAR, tfluid_k=REF_TEMP, geodata=bus_coords
        )
        ppipes.create_pump(
            net_heat, bus_index_to_junction_index[from_bus], v_feed, std_type="P1"
        )
        ppipes.create_pipe_from_parameters(
            net_heat,
            from_junction=v_feed,
            to_junction=bus_index_to_junction_index[to_bus],
            length_km=0.1,
            diameter_m=0.30,
            alpha_w_per_m2k=2.4 * 10**-5,
            sections=1,
        )
        ppipes.create_pipe_from_parameters(
            net_heat,
            from_junction=bus_index_to_junction_index[
                from_bus + start_return_net_id_junctions
            ],
            to_junction=bus_index_to_junction_index[
                to_bus + start_return_net_id_junctions
            ],
            length_km=0.1,
            diameter_m=0.30,
            alpha_w_per_m2k=2.4 * 10**-5,
            sections=1,
        )

    for i, length, from_bus, to_bus in zip(
        power_net.line.index,
        power_net.line["length_km"],
        power_net.line["from_bus"],
        power_net.line["to_bus"],
    ):
        bus_coords = (
            power_net.bus_geodata["x"][from_bus],
            power_net.bus_geodata["y"][from_bus],
        )
        v_feed = ppipes.create_junction(
            net_heat, pn_bar=REF_BAR, tfluid_k=REF_TEMP, geodata=bus_coords
        )
        ppipes.create_pump(
            net_heat, bus_index_to_junction_index[from_bus], v_feed, std_type="P1"
        )
        ppipes.create_pipe_from_parameters(
            net_heat,
            from_junction=v_feed,
            to_junction=bus_index_to_junction_index[to_bus],
            length_km=length * (random.random() / 4 + (7 / 8)),
            diameter_m=0.20,
            alpha_w_per_m2k=2.4 * 10**-5,
        )
        ppipes.create_pipe_from_parameters(
            net_heat,
            from_junction=bus_index_to_junction_index[
                from_bus + start_return_net_id_junctions
            ],
            to_junction=bus_index_to_junction_index[
                to_bus + start_return_net_id_junctions
            ],
            length_km=length * (random.random() / 4 + (7 / 8)),
            diameter_m=0.20,
            alpha_w_per_m2k=2.4 * 10**-5,
        )

    for i in power_net.bus.index:
        deployment_c_value = random.random()
        if deployment_c_value < heat_deployment_rate:
            ppipes.create_heat_exchanger(
                net_heat,
                from_junction=bus_index_to_junction_index[i],
                to_junction=bus_index_to_junction_index[
                    i + start_return_net_id_junctions
                ],
                diameter_m=0.030,
                qext_w=1,  # 10**3 * (random.random() - 0.5),
            )

    ppipes.create_ext_grid(
        net_heat,
        junction=0,
        p_bar=REF_BAR,
        t_k=REF_TEMP,
        name="Grid Connection Heat",
    )
    return net_heat


def create_gas_net_for_power():
    pass


def create_p2h_in_combined_generated_network():
    pass


def create_chp_in_combined_generated_network():
    pass


def create_p2g_in_combined_generated_network():
    pass


def generate_mes_based_on_power_net(
    net_power: mm.Network,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.3,
    p2g_density=0.1,
    p2h_density=0.1,
):
    new_mes_net = net_power.copy()
    net_heat = create_heat_net_for_power(net_power, new_mes_net, heat_deployment_rate)
    net_gas = create_gas_net_for_power(net_power, new_mes_net, gas_deployment_rate)

    create_p2h_in_combined_generated_network(
        new_mes_net, net_power, net_heat, p2h_density
    )
    create_chp_in_combined_generated_network(
        new_mes_net, net_power, net_heat, net_gas, chp_density
    )
    create_p2g_in_combined_generated_network(
        new_mes_net, net_power, net_gas, p2g_density
    )

    return new_mes_net
