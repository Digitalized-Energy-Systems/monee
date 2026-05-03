import random

from geopy import distance

import monee.express as mx
import monee.model as mm

REF_PA = 1000000
REF_TEMP = 356
DEFAULT_LENGTH = 100

# Higher heating value of the default low-pressure gas grid is 15.3 kWh/kg.
# 15.3 kWh/kg * 3.6 MJ/kWh = 55.08 MJ/kg, so 1 MW chemical power
# corresponds to (1 MJ/s) / (HHV [MJ/kg]) = 1/55.08 kg/s.
GAS_HHV_MJ_PER_KG = 15.3 * 3.6


def _node_power_load_mw(power_net: mm.Network, node):
    p_mw = 0.0
    for child in power_net.childs_by_ids(node.child_ids):
        if isinstance(child.model, mm.PowerLoad):
            p_mw += float(getattr(child.model, "p_mw", 0.0) or 0.0)
    return p_mw


def _node_power_gen_mw(power_net: mm.Network, node):
    """Local power-generation magnitude in MW (positive scalar).

    Sums dispatchable generation at the node from ``PowerGenerator`` and
    ``GridFormingGenerator``.  ``ExtPowerGrid`` is intentionally excluded:
    its ``p_mw`` is a free slack-balance ``Var`` whose value is meaningless
    until a solve has run, and using it as a sizing reference would make
    downstream CP capacities depend on whether anyone called
    ``run_energy_flow`` on the input network first.
    """
    p_mw = 0.0
    for child in power_net.childs_by_ids(node.child_ids):
        model = child.model
        if isinstance(model, mm.PowerGenerator):
            # PowerGenerator stores the negated magnitude internally.
            p_mw += abs(float(getattr(model, "p_mw", 0.0) or 0.0))
        elif isinstance(model, mm.GridFormingGenerator):
            # p_mw is a Var with bounds [-p_mw_max, +p_mw_max].
            p_mw += abs(float(model.p_mw.max or 0.0))
    return p_mw


def get_length(
    net: mm.Network, branch, node1_id, node2_id, default_length=DEFAULT_LENGTH
):
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


def create_gas_tree_net_for_power(
    power_net: mm.Network,
    target_net: mm.Network,
    gas_load_share=0.4,
    gas_gen_share=0.3,
    default_diameter_m=0.3,
    length_scale=1,
    default_length=DEFAULT_LENGTH,
    min_load_kgs=0.01,
    min_source_kgs=0.01,
    slack_node_id=None,
):
    """Create an acyclic gas grid mirroring the spanning tree of ``power_net``.

    Topology
    --------
    * One gas junction per power node, positioned at the corresponding bus.
    * Gas pipes mirror the spanning-tree branches of the power network, so the
      resulting gas grid is guaranteed to be non-cyclic.

    Capacity sizing
    ---------------
    For each power node we sum its ``PowerLoad`` magnitudes (in MW) and treat
    ``gas_load_share`` of that as gas demand to be covered chemically.  The
    equivalent mass flow is::

        kg/s = MW * gas_load_share / HHV[MJ/kg]

    Generation magnitudes (``PowerGenerator`` / ``GridFormingGenerator`` /
    ``ExtPowerGrid``) are mirrored as gas sources scaled by ``gas_gen_share``.
    The slack ``ExtHydrGrid`` is attached at ``slack_node_id`` (or the
    network's first node) so that any residual imbalance is absorbed.

    Returns
    -------
    dict[int, int]
        Mapping ``power_node_id -> gas_junction_id``.
    """
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    gas_grid.pressure_ref = REF_PA
    gas_grid.t_ref = REF_TEMP
    target_net.set_default_grid("gas", gas_grid)

    power_net_as_st = mm.to_spanning_tree(power_net)

    bus_index_to_junction_index = {}
    for node in power_net_as_st.nodes:
        bus_index_to_junction_index[node.id] = mx.create_junction(
            target_net, position=node.position, grid=gas_grid
        )

    for branch in power_net_as_st.branches:
        from_id = bus_index_to_junction_index[branch.from_node_id]
        to_id = bus_index_to_junction_index[branch.to_node_id]
        mx.create_gas_pipe(
            target_net,
            from_node_id=from_id,
            to_node_id=to_id,
            diameter_m=default_diameter_m,
            length_m=get_length(
                target_net,
                branch,
                from_id,
                to_id,
                default_length=default_length,
            )
            * length_scale,
            grid=gas_grid,
        )

    for node in power_net_as_st.nodes:
        p_load_mw = _node_power_load_mw(power_net, node)
        if p_load_mw > 0 and gas_load_share > 0:
            mass_flow = max(
                min_load_kgs, p_load_mw * gas_load_share / GAS_HHV_MJ_PER_KG
            )
            mx.create_sink(
                target_net,
                bus_index_to_junction_index[node.id],
                mass_flow=round(mass_flow, 4),
            )

        p_gen_mw = _node_power_gen_mw(power_net, node)
        if p_gen_mw > 0 and gas_gen_share > 0:
            mass_flow = max(
                min_source_kgs, p_gen_mw * gas_gen_share / GAS_HHV_MJ_PER_KG
            )
            mx.create_source(
                target_net,
                node_id=bus_index_to_junction_index[node.id],
                mass_flow=round(mass_flow, 4),
            )

    slack_node = (
        slack_node_id if slack_node_id is not None else power_net_as_st.first_node()
    )
    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[slack_node],
        t_k=REF_TEMP,
        name="Grid Connection Gas",
    )
    return bus_index_to_junction_index


def create_heat_supply_return_net_for_power(
    power_net: mm.Network,
    target_net: mm.Network,
    heat_load_share=0.5,
    heat_gen_share=0.3,
    default_diameter_m=0.12,
    return_diameter_m=None,
    length_scale=1,
    default_length=DEFAULT_LENGTH,
    return_length_m=None,
    min_load_mw=0.01,
    min_gen_mw=0.01,
    slack_node_id=None,
    node_based_heat_loads=False,
):
    """Create a supply/return DHS grid mirroring the spanning tree of ``power_net``.

    Topology
    --------
    * One **supply junction** per power node, mirroring its bus position.
    * Supply pipes mirror the spanning-tree branches, forming an acyclic
      supply network.
    * Exactly **one shared return junction** that collects every consumer
      return flow and feeds every generator suction line.
    * One closing return pipe from the shared return junction back to the
      slack supply junction so that the loop is hydraulically closed.
    * Linear heat exchangers (default ``HeatExchanger`` formulation) act as
      heat loads (supply -> return) and heat generators (return -> supply).

    Capacity sizing
    ---------------
    Heat capacities are derived from the power network: every power load
    contributes ``heat_load_share`` of its ``p_mw`` as a thermal demand,
    every power generator contributes ``heat_gen_share`` of its rated power
    as thermal generation.  This keeps the heat and gas grids dimensionally
    coherent with the underlying power grid.

    Returns
    -------
    tuple[dict, int]
        ``(power_node_id -> supply_junction_id, return_junction_id)``.
    """
    heat_grid = mm.create_water_grid("water")
    heat_grid.t_ref = REF_TEMP
    heat_grid.pressure_ref = REF_PA
    if node_based_heat_loads:
        # Tighten the McCormick-DHS bilinear envelopes both on the per-unit
        # node temperature and on the per-pipe mass-flow ceiling.  The
        # default ``[0.3, 2.0]`` τ-envelope and 5 m/s velocity cap together
        # yield a per-pipe LP relaxation gap of ~36 MW, which dominates the
        # solve.  Anchor τ to the typical supply/return band and pick a
        # 2 m/s design velocity (typical for DHS); ``f_max`` is essentially
        # unused since ``calc_max_mass_flow`` already caps each pipe via
        # ``π/4·D²·ρ·v_max_mps``.
        heat_grid.t_pu_min_env = 296.15 / REF_TEMP  # ≈ 0.832 (ambient)
        heat_grid.t_pu_max_env = 1.15
        heat_grid.v_max_mps = 2.0
    target_net.set_default_grid("water", heat_grid)

    power_net_as_st = mm.to_spanning_tree(power_net)

    # Orient supply pipes outward from the slack via BFS over the (undirected)
    # spanning tree.  Otherwise the matpower from/to convention can leave
    # supply junctions without an incoming pipe, which the standard
    # LinearHeatExchanger formulation tolerates (mass_flow_pos absorbs flow
    # reversal) but McCormick-DHS does not (it pins direction to from→to).
    import networkx as nx

    slack_root = (
        slack_node_id if slack_node_id is not None else power_net_as_st.first_node()
    )
    undirected = nx.Graph(power_net_as_st.graph)
    branch_lookup = {}
    for branch in power_net_as_st.branches:
        branch_lookup[(branch.from_node_id, branch.to_node_id)] = branch
        branch_lookup[(branch.to_node_id, branch.from_node_id)] = branch

    bfs_edges = list(nx.bfs_edges(undirected, source=slack_root))

    if node_based_heat_loads:
        # PRUNE the supply tree to the Steiner subtree connecting the slack
        # to every consumer bus.  Required for McCormick-DHS: a dead-end
        # pipe ending at a junction with no consumer forces ``mass_flow = 0``
        # AND ``H_in = H_out = 0`` there, which through the heat-loss eq
        # pins ``t_pu`` upstream to ambient and propagates ambient temperature
        # back along the tree, breaking every downstream consumer.
        consumer_buses = {
            node.id
            for node in power_net_as_st.nodes
            if _node_power_load_mw(power_net, node) > 0 and heat_load_share > 0
        }
        parent = {slack_root: None}
        for p, c in bfs_edges:
            parent[c] = p
        keep_nodes = {slack_root}
        for cb in consumer_buses:
            n = cb
            while n is not None and n not in keep_nodes:
                keep_nodes.add(n)
                n = parent.get(n)
    else:
        keep_nodes = {n.id for n in power_net_as_st.nodes}

    bus_index_to_supply_junction = {}
    for node in power_net_as_st.nodes:
        if node.id not in keep_nodes:
            continue
        bus_index_to_supply_junction[node.id] = mx.create_junction(
            target_net, position=node.position, grid=heat_grid
        )

    return_junction = mx.create_junction(
        target_net, position=None, grid=heat_grid, name="dhs_return"
    )

    for p, c in bfs_edges:
        if p not in keep_nodes or c not in keep_nodes:
            continue
        branch = branch_lookup[(p, c)]
        from_id = bus_index_to_supply_junction[p]
        to_id = bus_index_to_supply_junction[c]
        mx.create_water_pipe(
            target_net,
            from_node_id=from_id,
            to_node_id=to_id,
            diameter_m=default_diameter_m,
            length_m=get_length(
                target_net,
                branch,
                from_id,
                to_id,
                default_length=default_length,
            )
            * length_scale,
            temperature_ext_k=296.15,
            roughness=0.001,
            grid=heat_grid,
        )

    for node in power_net_as_st.nodes:
        if node.id not in bus_index_to_supply_junction:
            continue
        supply_id = bus_index_to_supply_junction[node.id]

        p_load_mw = _node_power_load_mw(power_net, node)
        if p_load_mw > 0 and heat_load_share > 0:
            q_mw = max(min_load_mw, p_load_mw * heat_load_share)
            if node_based_heat_loads:
                # Node-based HeatLoad child (uses ``q_w_heat`` on the junction).
                # Compatible with the McCormick-DHS formulation, which only
                # accounts for branch enthalpy via WaterPipe ``H_in_w/H_out_w``
                # and node-level ``q_w_heat``.  Pair every HeatLoad with a
                # water Sink that drains the consumer's design mass flow —
                # without it the consumer junction is a hydraulic dead-end
                # (incoming pipe pinned to zero flow by McCormick-DHS), which
                # makes the heat-load demand structurally infeasible.
                mx.create_heat_load(target_net, supply_id, q_w=q_mw * 1e6)
                # Design flow same as the LinearHX would compute internally:
                # m = q / (c · ΔT) with c = 4180 J/(kg·K), ΔT = 30 K default.
                m_design = q_mw * 1e6 / (4180.0 * 30.0)
                mx.create_water_sink(
                    target_net, supply_id, mass_flow=round(m_design, 6)
                )
            else:
                # Branch-based HeatExchanger consumer: bridges supply junction
                # to the shared return junction.  Compatible with the default
                # LinearHeatExchanger formulation but NOT with McCormick-DHS.
                mx.create_heat_exchanger(
                    target_net,
                    from_node_id=supply_id,
                    to_node_id=return_junction,
                    q_mw=q_mw,
                )

        # Per-bus heat-generator HXs (return → supply) are only meaningful
        # under formulations that allow flow reversal (e.g. plain
        # LinearHeatExchanger).  Under McCormick-DHS the rooted-outward
        # supply tree pins every pipe's flow direction, so a generator HX
        # injecting mass into an intermediate consumer junction has no path
        # to drain — outright infeasible.  Skip them in node-based mode.
        if not node_based_heat_loads:
            p_gen_mw = _node_power_gen_mw(power_net, node)
            if p_gen_mw > 0 and heat_gen_share > 0:
                q_mw = max(min_gen_mw, p_gen_mw * heat_gen_share)
                mx.create_heat_exchanger(
                    target_net,
                    from_node_id=return_junction,
                    to_node_id=supply_id,
                    q_mw=-q_mw,
                )

    slack_supply_junction = bus_index_to_supply_junction[slack_root]

    mx.create_water_pipe(
        target_net,
        from_node_id=return_junction,
        to_node_id=slack_supply_junction,
        diameter_m=return_diameter_m
        if return_diameter_m is not None
        else default_diameter_m * 1.5,
        length_m=return_length_m
        if return_length_m is not None
        else default_length * length_scale,
        temperature_ext_k=296.15,
        roughness=0.001,
        grid=heat_grid,
    )

    mx.create_ext_hydr_grid(
        target_net,
        node_id=slack_supply_junction,
        t_k=REF_TEMP,
        grid_key=mm.WATER_KEY,
        name="Grid Connection Heat",
    )
    return bus_index_to_supply_junction, return_junction


def create_coupling_points_for_mes(
    mes_net: mm.Network,
    bus_to_gas_junc,
    bus_to_heat_supply_junc,
    heat_return_junc,
    density=0.2,
    centralized=False,
    central_node_id=None,
    couplings=("chp", "p2g", "p2h"),
    chp_efficiency_power=0.4,
    chp_efficiency_heat=0.45,
    chp_diameter_m=0.05,
    p2g_efficiency=0.7,
    p2h_efficiency=0.95,
    p2h_diameter_m=0.01,
    regulation=1.0,
    use_hg_variants=False,
    seed=None,
):
    """Add multi-energy coupling points (CHP / P2G / P2H) to an MES network.

    Density and placement
    ---------------------
    * ``density`` (in [0, 1]) controls how many of the eligible power nodes
      receive a coupling unit.  In **decentralized** mode a per-node
      Bernoulli draw is performed; in **centralized** mode the count is
      ``ceil(density * |nodes|)`` and every drawn unit is attached to the
      same hub node (``central_node_id`` if given, else the slack/first
      node of the power graph).

    Capacity sizing
    ---------------
    Each unit is sized from the local power node so the coupling capacities
    stay coherent with the power, gas and heat grids built by
    :func:`create_gas_tree_net_for_power` and
    :func:`create_heat_supply_return_net_for_power`:

    * **CHP** — fuel mass flow scaled to cover ~50% of the bus's electrical
      generation (or its load, if the bus is a pure consumer).
    * **P2G** — electrical input scaled from the bus's load, output gas mass
      flow derived via the gas HHV.
    * **P2H** — heat output scaled from the bus's load.

    Returns
    -------
    list[dict]
        One entry per created unit: ``{"type", "node", "id"}``.
    """
    if not 0 <= regulation <= 1:
        raise ValueError("regulation must be in [0, 1]")
    if not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1]")

    rng = random.Random(seed) if seed is not None else random

    coupling_set = {c.lower() for c in couplings}
    valid = {"chp", "p2g", "p2h"}
    unknown = coupling_set - valid
    if unknown:
        raise ValueError(f"unknown coupling types: {sorted(unknown)}")

    candidate_node_ids = [
        nid for nid in bus_to_gas_junc.keys() if nid in bus_to_heat_supply_junc
    ]

    if centralized:
        hub = (
            central_node_id if central_node_id is not None else min(candidate_node_ids)
        )
        if hub not in candidate_node_ids:
            raise ValueError(
                f"central_node_id={hub} is not in the gas/heat coupled set"
            )
        n_units = int(round(density * len(candidate_node_ids)))
        target_nodes = [hub] * n_units
    else:
        target_nodes = [nid for nid in candidate_node_ids if rng.random() < density]

    created = []
    for power_node_id in target_nodes:
        gas_junc = bus_to_gas_junc[power_node_id]
        heat_supply_junc = bus_to_heat_supply_junc[power_node_id]

        # Sizing handle: prefer local generation magnitude, fall back to local
        # load magnitude.  A bus with no power activity at all (transit /
        # bare slack) provides no basis for sizing — skip it instead of
        # using a system-scale fallback that would oversize the unit by
        # orders of magnitude on LV grids.
        node = mes_net.node_by_id(power_node_id)
        p_ref_mw = _node_power_gen_mw(mes_net, node) or _node_power_load_mw(
            mes_net, node
        )
        if p_ref_mw <= 0:
            continue

        unit_type = rng.choice(sorted(coupling_set))

        if unit_type == "chp":
            # Fuel mass flow needed to supply ~half of p_ref_mw electrically
            # at the configured electrical efficiency (rough sizing).
            mass_flow = (
                0.5 * p_ref_mw / max(chp_efficiency_power, 1e-3) / GAS_HHV_MJ_PER_KG
            )
            if use_hg_variants:
                # HeatGenerator-based CHP: heat injection via q_w_heat at the
                # supply junction, no return-side branch.  Required for the
                # McCormick-DHS formulation, which only sees node-level heat
                # injection (q_w_heat) and pipe enthalpy (H_in_w/H_out_w).
                uid = mx.create_chp_hg(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    gas_node_id=gas_junc,
                    mass_flow_setpoint=round(mass_flow, 6),
                    efficiency_power=chp_efficiency_power,
                    efficiency_heat=chp_efficiency_heat,
                    regulation=regulation,
                )
            else:
                uid = mx.create_chp(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    heat_return_node_id=heat_return_junc,
                    gas_node_id=gas_junc,
                    mass_flow_setpoint=round(mass_flow, 6),
                    diameter_m=chp_diameter_m,
                    efficiency_power=chp_efficiency_power,
                    efficiency_heat=chp_efficiency_heat,
                    regulation=regulation,
                )
            created.append({"type": "chp", "node": power_node_id, "id": uid})

        elif unit_type == "p2g":
            mass_flow = p2g_efficiency * p_ref_mw / GAS_HHV_MJ_PER_KG
            bid = mx.create_p2g(
                mes_net,
                from_node_id=power_node_id,
                to_node_id=gas_junc,
                efficiency=p2g_efficiency,
                mass_flow_setpoint=round(mass_flow, 6),
                regulation=regulation,
            )
            created.append({"type": "p2g", "node": power_node_id, "id": bid})

        elif unit_type == "p2h":
            heat_w = p_ref_mw * 1e6 * p2h_efficiency
            if use_hg_variants:
                bid = mx.create_p2h_hg(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    heat_energy_w=round(heat_w, 1),
                    efficiency=p2h_efficiency,
                )
                created.append({"type": "p2h", "node": power_node_id, "id": bid})
            else:
                uid = mx.create_p2h(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    heat_return_node_id=heat_return_junc,
                    heat_energy_w=round(heat_w, 1),
                    diameter_m=p2h_diameter_m,
                    efficiency=p2h_efficiency,
                    regulation=regulation,
                )
                created.append({"type": "p2h", "node": power_node_id, "id": uid})

    return created


def generate_supply_return_mes_based_on_power_net(
    net_power: mm.Network,
    *,
    coupling_density=0.2,
    centralized=False,
    central_node_id=None,
    couplings=("chp", "p2g", "p2h"),
    heat_kwargs=None,
    gas_kwargs=None,
    coupling_kwargs=None,
):
    """High-level wrapper: build a tree-shaped gas grid, a supply/return DHS
    grid with one shared return junction, and a configurable set of coupling
    points (CHP/P2G/P2H) on top of an existing power network.

    Capacities for the gas/heat demands and generators are derived from the
    power network's ``PowerLoad`` / ``PowerGenerator`` magnitudes so the
    three grids remain dimensionally consistent.
    """
    new_mes_net = net_power.copy()
    bus_to_gas_junc = create_gas_tree_net_for_power(
        net_power, new_mes_net, **(gas_kwargs or {})
    )
    bus_to_heat_supply, heat_return = create_heat_supply_return_net_for_power(
        net_power, new_mes_net, **(heat_kwargs or {})
    )
    create_coupling_points_for_mes(
        new_mes_net,
        bus_to_gas_junc=bus_to_gas_junc,
        bus_to_heat_supply_junc=bus_to_heat_supply,
        heat_return_junc=heat_return,
        density=coupling_density,
        centralized=centralized,
        central_node_id=central_node_id,
        couplings=couplings,
        **(coupling_kwargs or {}),
    )
    return new_mes_net
