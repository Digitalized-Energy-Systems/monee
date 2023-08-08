import monee.model as mm
import monee.express as mx
import random
from geopy import distance

REF_PA = 500000
REF_TEMP = 352
DEFAULT_LENGTH = 0.1


def get_length(net: mm.Network, branch, node1_id, node2_id):
    if hasattr(branch.model, "length_m"):
        return branch.model.length_m

    node1 = net.node_by_id(node1_id)
    node2 = net.node_by_id(node2_id)

    if node1.position is None or node2.position is None:
        raise Exception("The branch length can't be read from the given network!")

    return distance.distance(node1.position, node2.position).m


def create_heat_net_for_power(power_net, target_net, heat_deployment_rate):
    heat_grid = mm.create_water_grid("heat")

    power_net_as_st = mm.to_spanning_tree(power_net)
    bus_index_to_junction_index = {}
    for node in power_net_as_st.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=heat_grid)
        mx.create_junction(target_net, position=node.position, grid=heat_grid)

        # convention: return junction for a junction with id *i* has the id *i+1*
        bus_index_to_junction_index[node.id] = junc_id

    for branch in power_net_as_st.branches:
        from_node_id = bus_index_to_junction_index[branch.from_node_id]
        to_node_id = bus_index_to_junction_index[branch.to_node_id]
        mx.create_water_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=0.015,
            length_m=get_length(target_net, branch, from_node_id, to_node_id),
            temperature_ext_k=296.15,
            roughness=0.001,
            lambda_insulation_w_per_k=2.4 * 10**-5,
            grid=heat_grid,
        )
        mx.create_water_pipe(
            target_net,
            from_node_id=from_node_id + 1,
            to_node_id=to_node_id + 1,
            diameter_m=0.015,
            length_m=get_length(target_net, branch, from_node_id, to_node_id),
            temperature_ext_k=296.15,
            roughness=0.001,
            lambda_insulation_w_per_k=2.4 * 10**-5,
            grid=heat_grid,
        )

    for node in power_net_as_st.nodes:
        deployment_c_value = random.random()
        if deployment_c_value < heat_deployment_rate:
            mx.create_heat_exchanger(
                target_net,
                from_node_id=bus_index_to_junction_index[node.id],
                to_node_id=bus_index_to_junction_index[node.id] + 1,
                diameter_m=0.0030,
                q_mw=0.01 * (random.random() - 0.5),
            )

    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()],
        pressure_pa=REF_PA,
        t_k=REF_TEMP,
        name="Grid Connection Heat",
    )
    mx.create_consume_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()] + 1,
        pressure_pa=REF_PA,
        mass_flow=10,
        name="Grid Connection Return Heat",
    )
    return bus_index_to_junction_index


def create_gas_net_for_power(power_net, target_net, gas_deployment_rate):
    gas_grid = mm.create_gas_grid("gas", "lgas")

    bus_index_to_junction_index = {}
    for node in power_net.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=gas_grid)
        bus_index_to_junction_index[node.id] = junc_id

    for branch in power_net.branches:
        from_node_id = bus_index_to_junction_index[branch.from_node_id]
        to_node_id = bus_index_to_junction_index[branch.to_node_id]
        mx.create_gas_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=0.125,
            length_m=get_length(target_net, branch, from_node_id, to_node_id),
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
        node_id=bus_index_to_junction_index[target_net.first_node()],
        pressure_pa=REF_PA,
        t_k=REF_TEMP,
        name="Grid Connection Gas",
    )
    return bus_index_to_junction_index


def create_p2h_in_combined_generated_network(
    new_mes_net, net_power, bus_to_heat_junc, p2h_density
):
    for power_node in net_power.nodes:
        heat_junc = bus_to_heat_junc[power_node.id]
        heat_return_junc = heat_junc + 1
        if random.random() <= p2h_density:
            mx.create_p2h(
                new_mes_net,
                power_node_id=power_node.id,
                heat_node_id=heat_junc,
                heat_return_node_id=heat_return_junc,
                heat_energy_mw=0.01 * (random.random() - 0.5),
                diameter_m=0.0030,
                efficiency=0.4 * random.random() * 0.5,
            )


def create_chp_in_combined_generated_network(
    new_mes_net, net_power, bus_to_heat_junc, bus_to_gas_junc, chp_density
):
    for power_node in net_power.nodes:
        heat_junc = bus_to_heat_junc[power_node.id]
        heat_return_junc = heat_junc + 1
        gas_junc = bus_to_gas_junc[power_node.id]
        efficiency = 0.8 + random.random() / 10
        if random.random() <= chp_density:
            mx.create_chp(
                new_mes_net,
                power_node_id=power_node.id,
                heat_node_id=heat_junc,
                heat_return_node_id=heat_return_junc,
                gas_node_id=gas_junc,
                mass_flow_setpoint=0.08 * (random.random() - 0.5),
                diameter_m=0.1,
                efficiency_power=efficiency / 2,
                efficiency_heat=efficiency / 2,
            )


def create_p2g_in_combined_generated_network(
    new_mes_net, net_power, bus_to_gas_junc, p2g_density
):
    for power_node in net_power.nodes:
        gas_junc = bus_to_gas_junc[power_node.id]
        if random.random() <= p2g_density:
            mx.create_p2g(
                new_mes_net,
                from_node_id=power_node.id,
                to_node_id=gas_junc,
                efficiency=0.7,
                mass_flow_setpoint=0.1,
            )


def generate_mes_based_on_power_net(
    net_power: mm.Network,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.2,
    p2g_density=0.02,
    p2h_density=0.1,
):
    new_mes_net = net_power.copy()
    bus_to_heat_junc = create_heat_net_for_power(
        net_power, new_mes_net, heat_deployment_rate
    )
    bus_to_gas_junc = create_gas_net_for_power(
        net_power, new_mes_net, gas_deployment_rate
    )

    """
    create_p2h_in_combined_generated_network(
        new_mes_net, net_power, bus_to_heat_junc, p2h_density
    )
    """
    create_chp_in_combined_generated_network(
        new_mes_net, net_power, bus_to_heat_junc, bus_to_gas_junc, chp_density
    )
    """

    create_p2g_in_combined_generated_network(
        new_mes_net, net_power, bus_to_gas_junc, p2g_density
    )
    """

    return new_mes_net
