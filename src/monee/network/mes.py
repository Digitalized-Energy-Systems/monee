import random

from geopy import distance

import monee.express as mx
import monee.model as mm

REF_PA = 1000000
REF_TEMP = 356
DEFAULT_LENGTH = 100


def get_length(
    net: mm.Network, branch, node1_id, node2_id, default_length=DEFAULT_LENGTH
):
    """
    No docstring provided.
    """
    if hasattr(branch.model, "length_m"):
        return branch.model.length_m
    node1 = net.node_by_id(node1_id)
    node2 = net.node_by_id(node2_id)

    if node1.position is None or node2.position is None:
        return default_length

    return distance.distance(node1.position, node2.position).m


def create_heat_net_for_power(
    power_net,
    target_net,
    heat_deployment_rate,
    mass_flow_rate=0.075,
    default_diameter_m=0.12,
    length_scale=1,
    default_length=DEFAULT_LENGTH,
    power_scale=1,
):
    """
    No docstring provided.
    """
    heat_grid = mm.create_water_grid("water")
    heat_grid.t_ref = REF_TEMP
    heat_grid.pressure_ref = REF_PA
    target_net.set_default_grid("water", heat_grid)

    power_net_as_st = mm.to_spanning_tree(power_net)
    bus_index_to_junction_index = {}
    bus_index_to_end_junction_index = {}

    for node in power_net_as_st.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=heat_grid)
        mx.create_sink(
            target_net,
            junc_id,
            mass_flow=mass_flow_rate + random.random() * mass_flow_rate / 10,
        )
        bus_index_to_junction_index[node.id] = junc_id
        bus_index_to_end_junction_index[node.id] = junc_id
        deployment_c_value = random.random()
        if deployment_c_value < heat_deployment_rate:
            bus_index_to_end_junction_index[node.id] = mx.create_junction(
                target_net, position=node.position, grid=heat_grid
            )
            mx.create_heat_exchanger(
                target_net,
                from_node_id=bus_index_to_junction_index[node.id],
                to_node_id=bus_index_to_end_junction_index[node.id],
                diameter_m=0.2,
                q_mw=(-1 if random.random() > 0.8 else 1)
                * -0.1
                * random.random()
                * power_scale,
            )
            mx.create_sink(
                target_net,
                bus_index_to_end_junction_index[node.id],
                mass_flow=mass_flow_rate + random.random() * mass_flow_rate / 10,
            )
    for branch in power_net_as_st.branches:
        from_node_id = bus_index_to_end_junction_index[branch.from_node_id]
        to_node_id = bus_index_to_junction_index[branch.to_node_id]
        mx.create_water_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=default_diameter_m,
            length_m=get_length(
                target_net,
                branch,
                from_node_id,
                to_node_id,
                default_length=default_length,
            )
            * length_scale,
            temperature_ext_k=296.15,
            roughness=0.001,
            grid=heat_grid,
        )
    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()],
        t_k=REF_TEMP,
        name="Grid Connection Heat",
    )
    return (bus_index_to_junction_index, bus_index_to_end_junction_index)


def create_gas_net_for_power(
    power_net,
    target_net: mm.Network,
    gas_deployment_rate,
    scaling=1,
    source_scaling=1,
    default_diameter_m=0.3,
    length_scale=1,
    default_length=100,
):
    """
    No docstring provided.
    """
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    gas_grid.pressure_ref = REF_PA
    gas_grid.t_ref = REF_TEMP

    target_net.set_default_grid("gas", gas_grid)

    power_net_as_st = mm.to_spanning_tree(power_net)
    bus_index_to_junction_index = {}
    for node in power_net_as_st.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=gas_grid)
        bus_index_to_junction_index[node.id] = junc_id
    for branch in power_net_as_st.branches:
        from_node_id = bus_index_to_junction_index[branch.from_node_id]
        to_node_id = bus_index_to_junction_index[branch.to_node_id]
        mx.create_gas_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=default_diameter_m * scaling,
            length_m=get_length(
                target_net,
                branch,
                from_node_id,
                to_node_id,
                default_length=default_length,
            )
            * length_scale,
            grid=gas_grid,
        )
    for node in power_net_as_st.nodes:
        deployment_c_value = random.random()
        if deployment_c_value <= gas_deployment_rate:
            mx.create_sink(
                target_net,
                bus_index_to_junction_index[node.id],
                mass_flow=round(0.1 + 0.5 * random.random() * scaling, 2),
            )
    mx.create_source(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()],
        mass_flow=10 * scaling * source_scaling,
    )
    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()],
        t_k=REF_TEMP,
        name="Grid Connection Gas",
    )
    return bus_index_to_junction_index


def create_p2h_in_combined_generated_network(
    new_mes_net: mm.Network,
    net_power,
    bus_to_heat_junc,
    end_bus_to_heat_junc,
    p2h_density,
):
    """
    No docstring provided.
    """
    for power_node in net_power.nodes:
        heat_junc = bus_to_heat_junc[power_node.id]
        heat_junc_two = end_bus_to_heat_junc[power_node.id]
        if random.random() <= p2h_density:
            if heat_junc != heat_junc_two and new_mes_net.has_branch_between(
                heat_junc, heat_junc_two
            ):
                new_mes_net.remove_branch_between(heat_junc, heat_junc_two)
                mx.create_p2h(
                    new_mes_net,
                    power_node_id=power_node.id,
                    heat_node_id=heat_junc_two,
                    heat_return_node_id=heat_junc,
                    heat_energy_w=15_000,
                    diameter_m=0.003,
                    efficiency=0.4 * 0.5 * 0.5,
                )


def create_chp_in_combined_generated_network(
    new_mes_net: mm.Network,
    net_power,
    bus_to_heat_junc,
    end_bus_to_heat_junc,
    bus_to_gas_junc,
    chp_density,
):
    """
    No docstring provided.
    """
    for power_node in net_power.nodes:
        heat_junc = bus_to_heat_junc[power_node.id]
        heat_junc_two = end_bus_to_heat_junc[power_node.id]
        gas_junc = bus_to_gas_junc[power_node.id]
        efficiency = 0.8 + random.random() / 10
        if random.random() <= chp_density:
            if heat_junc != heat_junc_two and new_mes_net.has_branch_between(
                heat_junc, heat_junc_two
            ):
                new_mes_net.remove_branch_between(heat_junc, heat_junc_two)
                mx.create_chp(
                    new_mes_net,
                    power_node_id=power_node.id,
                    heat_node_id=heat_junc_two,
                    heat_return_node_id=heat_junc,
                    gas_node_id=gas_junc,
                    mass_flow_setpoint=0.015 * random.random(),
                    diameter_m=0.035,
                    efficiency_power=efficiency / 2,
                    efficiency_heat=efficiency / 2,
                )


def create_p2g_in_combined_generated_network(
    new_mes_net, net_power, bus_to_gas_junc, p2g_density
):
    """
    No docstring provided.
    """
    for power_node in net_power.nodes:
        gas_junc = bus_to_gas_junc[power_node.id]
        if random.random() <= p2g_density:
            mx.create_p2g(
                new_mes_net,
                from_node_id=power_node.id,
                to_node_id=gas_junc,
                efficiency=0.7,
                mass_flow_setpoint=0.045 * random.random(),
            )


def generate_mes_based_on_power_net(
    net_power: mm.Network,
    heat_deployment_rate,
    gas_deployment_rate,
    chp_density=0.1,
    p2g_density=0.02,
    p2h_density=0.1,
):
    """
    No docstring provided.
    """
    new_mes_net = net_power.copy()
    bus_to_heat_junc, end_bus_to_heat_junc = create_heat_net_for_power(
        net_power, new_mes_net, heat_deployment_rate
    )
    bus_to_gas_junc = create_gas_net_for_power(
        net_power, new_mes_net, gas_deployment_rate
    )
    create_p2h_in_combined_generated_network(
        new_mes_net, net_power, bus_to_heat_junc, end_bus_to_heat_junc, p2h_density
    )
    create_chp_in_combined_generated_network(
        new_mes_net,
        net_power,
        bus_to_heat_junc,
        end_bus_to_heat_junc,
        bus_to_gas_junc,
        chp_density,
    )
    create_p2g_in_combined_generated_network(
        new_mes_net, net_power, bus_to_gas_junc, p2g_density
    )
    return new_mes_net


def create_monee_benchmark_net():
    random.seed(9002)
    pn = mm.Network(el_model=mm.PowerGrid(name="power", sn_mva=100))

    node_0 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.PowerGenerator(p_mw=10, q_mvar=0, regulation=0.5))],
    )
    node_1 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=10, q_mvar=0, vm_pu=1, va_radians=0))],
    )
    node_2 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.PowerLoad(p_mw=10, q_mvar=0))],
    )
    node_3 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.PowerLoad(p_mw=20, q_mvar=0))],
    )
    node_4 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.PowerLoad(p_mw=20, q_mvar=0))],
    )
    node_5 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.PowerGenerator(p_mw=30, q_mvar=0, regulation=0.5))],
    )
    node_6 = pn.node(
        mm.Bus(base_kv=120),
        mm.EL,
        child_ids=[pn.child(mm.GridFormingGenerator(p_mw_max=50, q_mvar_max=50))],
    )
    max_i_ka = 319
    pn.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        node_0,
        node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        node_1,
        node_2,
    )
    pn.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        node_1,
        node_5,
    )
    pn.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        node_2,
        node_3,
    )
    pn.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        node_3,
        node_4,
    )
    pn.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            max_i_ka=max_i_ka,
            parallel=1,
        ),
        node_3,
        node_6,
    )

    new_mes = pn.copy()

    # gas
    bus_to_gas_junc = create_gas_net_for_power(pn, new_mes, 1, scaling=1)

    # # # heat
    bus_index_to_junction_index, bus_index_to_end_junction_index = (
        create_heat_net_for_power(
            pn, new_mes, 1, mass_flow_rate=1, default_diameter_m=0.16
        )
    )
    new_water_junc = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc,
        mass_flow=0.05,
    )
    new_water_junc_2 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_2,
        mass_flow=0.05,
    )
    mx.create_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc,
        to_node_id=new_water_junc_2,
        diameter_m=0.20,
        q_mw=0.03,
    )
    new_water_junc_3 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_3,
        mass_flow=0.06,
    )
    mx.create_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc_2,
        to_node_id=new_water_junc_3,
        diameter_m=0.20,
        q_mw=0.03,
    )
    mx.create_p2g(
        new_mes,
        from_node_id=node_4,
        to_node_id=bus_to_gas_junc[node_4],
        efficiency=0.7,
        mass_flow_setpoint=1,
        regulation=0,
    )
    mx.create_chp(
        new_mes,
        power_node_id=node_1,
        heat_node_id=bus_index_to_junction_index[node_0],
        heat_return_node_id=new_water_junc,
        gas_node_id=bus_to_gas_junc[node_3],
        mass_flow_setpoint=0.005,
        diameter_m=0.1,
        efficiency_power=0.5,
        efficiency_heat=0.5,
        regulation=1,
    )
    mx.create_g2p(
        new_mes,
        from_node_id=bus_to_gas_junc[node_1],
        to_node_id=node_1,
        efficiency=0.9,
        p_mw_setpoint=2,
        regulation=0,
    )
    mx.create_g2p(
        new_mes,
        from_node_id=bus_to_gas_junc[node_6],
        to_node_id=node_6,
        efficiency=0.9,
        p_mw_setpoint=1,
        regulation=0,
    )
    new_mes.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            parallel=1,
            backup=True,
            on_off=0,
            max_i_ka=max_i_ka,
        ),
        node_4,
        node_0,
    )
    new_mes.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            parallel=1,
            backup=True,
            on_off=0,
            max_i_ka=max_i_ka,
        ),
        node_5,
        node_2,
    )
    return new_mes


def create_mv_multi_cigre():
    import pandapower.networks as pn

    from monee.io.from_pandapower import from_pandapower_net

    random.seed(9004)
    pnet = pn.create_cigre_network_mv(with_der="pv_wind")

    monee_net = from_pandapower_net(pnet)
    new_mes = monee_net.copy()
    create_gas_net_for_power(
        monee_net,
        new_mes,
        1,
        source_scaling=1,
        default_diameter_m=0.64,
        length_scale=0.001,
        default_length=100000,
    )
    create_heat_net_for_power(
        monee_net,
        new_mes,
        0.5,
        mass_flow_rate=25,
        default_diameter_m=0.68,
        power_scale=100,
        length_scale=0.001,
        default_length=100000,
    )

    mx.create_power_generator(new_mes, 5, 2, 1)
    mx.create_power_generator(new_mes, 6, 3, 1)

    mx.create_p2g(
        new_mes,
        from_node_id=4,
        to_node_id=21,
        efficiency=0.7,
        mass_flow_setpoint=0.5,
        regulation=0.1,
    )
    mx.create_chp(
        new_mes,
        power_node_id=2,
        heat_node_id=43,
        heat_return_node_id=44,
        gas_node_id=25,
        mass_flow_setpoint=0.5,
        diameter_m=0.3,
        efficiency_power=0.58,
        efficiency_heat=0.4,
        regulation=1,
        remove_existing_branch=True,
    )

    new_mes.branch(
        mm.PowerLine(
            length_m=100,
            r_ohm_per_m=0.007,
            x_ohm_per_m=0.007,
            parallel=1,
            backup=True,
            on_off=0,
            max_i_ka=319,
        ),
        5,
        2,
    )
    return new_mes
