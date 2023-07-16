import monee.model as mm
import monee.express as mx
import random

REF_BAR = 5
REF_TEMP = 342


def create_heat_net_for_power(power_net, target_net, heat_deployment_rate):
    heat_grid = mm.create_water_grid("heat")

    bus_index_to_junction_index = {}
    for node in power_net.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=heat_grid)
        mx.create_junction(target_net, position=node.position, grid=heat_grid)

        # convention: return junction for a junction with id *i* has the id *i+1*
        bus_index_to_junction_index[node.id] = junc_id

    for branch in power_net.branches:
        from_node_id = bus_index_to_junction_index[branch.from_node.id]
        to_node_id = bus_index_to_junction_index[branch.to_node.id]
        mx.create_water_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=0.2,
            length_m=branch.model.length_m,
            temperature_ext_k=296.15,
            roughness=0.001,
            lambda_insulation_w_per_k=2.4 * 10**-5,
            grid=heat_grid,
        )
        mx.create_water_pipe(
            target_net,
            from_node_id=from_node_id + 1,
            to_node_id=to_node_id + 1,
            diameter_m=0.2,
            length_m=branch.model.length_m,
            temperature_ext_k=296.15,
            roughness=0.001,
            lambda_insulation_w_per_k=2.4 * 10**-5,
            grid=heat_grid,
        )

    for node in power_net.nodes:
        deployment_c_value = random.random()
        if deployment_c_value < heat_deployment_rate:
            mx.create_heat_exchanger(
                target_net,
                from_node_id=bus_index_to_junction_index[node.id],
                to_node_id=bus_index_to_junction_index[node.id] + 1,
                diameter_m=0.030,
                q_mw=10**3 * (random.random() - 0.5),
            )

    mx.create_ext_hydr_grid(
        target_net,
        node_id=0,
        pressure_pa=REF_BAR,
        t_k=REF_TEMP,
        name="Grid Connection Heat",
    )
    return target_net


def create_gas_net_for_power(power_net, target_net, gas_deployment_rate):
    gas_grid = mm.create_gas_grid("gas", "lgas")

    bus_index_to_junction_index = {}
    for node in power_net.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=gas_grid)
        bus_index_to_junction_index[node.id] = junc_id

    for branch in power_net.branches:
        from_node_id = bus_index_to_junction_index[branch.from_node.id]
        to_node_id = bus_index_to_junction_index[branch.to_node.id]
        mx.create_gas_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=0.125,
            length_m=branch.model.length_m,
            grid=gas_grid,
        )

    for node in power_net.nodes:
        deployment_c_value = random.random()
        if deployment_c_value < gas_deployment_rate:
            mx.create_sink(
                target_net,
                bus_index_to_junction_index[node.id],
                mass_flow=0.1,
            )

    mx.create_ext_hydr_grid(
        target_net,
        node_id=0,
        pressure_pa=REF_BAR,
        t_k=REF_TEMP,
        name="Grid Connection Gas",
    )
    return target_net


def create_p2h_in_combined_generated_network(
    new_mes_net, net_power, net_heat, p2h_density
):
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
