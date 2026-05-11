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
                    heat_energy_mw=0.015,
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
    gas_load_share=3.0,
    gas_gen_share=1.0,
    default_diameter_m=0.3,
    length_scale=1,
    default_length=DEFAULT_LENGTH,
    min_load_kgs=1.8e-4,
    min_source_kgs=1.8e-4,
    slack_node_id=None,
    extra_mesh_pipes=0,
    mesh_diameter_factor=0.5,
    mesh_length_factor=2.0,
    mesh_seed=None,
):
    """Create an acyclic gas grid mirroring the spanning tree of ``power_net``.

    Topology
    --------
    * One gas junction per power node, positioned at the corresponding bus.
    * Gas pipes mirror the spanning-tree branches of the power network, so the
      resulting gas grid is guaranteed to be non-cyclic.

    Resilience meshing
    ------------------
    With ``extra_mesh_pipes > 0`` an additional ``N`` cross-connection pipes
    are added between non-adjacent gas junctions to break the strict tree
    structure.  Without these, a single pipe failure isolates the entire
    downstream subtree, which is the dominant reason additive coupling
    points are statistically irrelevant on this layout: a CHP whose fuel
    path is severed contributes nothing regardless of how much rated
    capacity it carries.  Each tie pipe defaults to ``mesh_diameter_factor``
    × ``default_diameter_m`` and ``mesh_length_factor`` × ``default_length``
    (smaller and longer than primary pipes, modelling minor cross-feeders).

    Capacity sizing
    ---------------
    For each power node we sum its ``PowerLoad`` magnitudes (in MW) and treat
    ``gas_load_share`` × p_mw as the chemical thermal demand (in MW) to be
    covered by gas.  ``gas_load_share`` is a thermal-to-electrical multiplier:
    the default of 3.0 reflects the typical ~3× ratio of peak gas thermal
    demand to peak electric demand on a residential LV feeder (gas heating +
    DHW vs lights/appliances).  The equivalent mass flow is::

        kg/s = (MW * gas_load_share) / HHV[MJ/kg]

    Generation magnitudes (``PowerGenerator`` / ``GridFormingGenerator`` /
    ``ExtPowerGrid``) are mirrored as gas sources scaled by ``gas_gen_share``.
    The slack ``ExtHydrGrid`` is attached at ``slack_node_id`` (or the
    network's first node) so that any residual imbalance is absorbed.

    The per-bus floors ``min_load_kgs`` / ``min_source_kgs`` correspond to
    ~10 kW thermal at the gas HHV, matching the heat-grid floor; they are
    *not* a fraction-of-1.0 design parameter — interpreting them as such
    inflates per-bus gas demand by ~50× the realistic per-dwelling peak.

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

    if extra_mesh_pipes > 0:
        # Add resilience tie pipes: pick non-adjacent junction pairs uniformly
        # at random and connect them with a smaller-diameter / longer pipe.
        # The point is to give every junction more than one path to the
        # slack so single pipe failures don't isolate large subtrees — the
        # main reason additive CPs underperform on a tree layout.
        rng = random.Random(mesh_seed) if mesh_seed is not None else random
        junctions = list(bus_index_to_junction_index.values())
        existing_pairs = set()
        for branch in target_net.branches:
            if isinstance(branch.model, mm.GasPipe):
                a, b = branch.from_node_id, branch.to_node_id
                existing_pairs.add(frozenset((a, b)))
        candidates = [
            (j1, j2)
            for i, j1 in enumerate(junctions)
            for j2 in junctions[i + 1 :]
            if frozenset((j1, j2)) not in existing_pairs
        ]
        rng.shuffle(candidates)
        for j1, j2 in candidates[:extra_mesh_pipes]:
            mx.create_gas_pipe(
                target_net,
                from_node_id=j1,
                to_node_id=j2,
                diameter_m=default_diameter_m * mesh_diameter_factor,
                length_m=default_length * mesh_length_factor * length_scale,
                grid=gas_grid,
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
    heat_load_share=1.0,
    heat_gen_share=0.5,
    default_diameter_m=0.12,
    return_diameter_m=None,
    length_scale=1,
    default_length=DEFAULT_LENGTH,
    return_length_m=None,
    min_load_mw=0.005,
    min_gen_mw=0.005,
    slack_node_id=None,
    node_based_heat_loads=False,
    balance_gen_to_load=True,
    heat_plant_mode="closing_pipe",
    return_t_k=REF_TEMP - 30,
    return_pressure_pu=0.95,
    return_pin_temperature=False,
    node_heat_gen_share=1.0,
    supply_slack_t_k=REF_TEMP,
):
    """Create a supply/return DHS grid mirroring the spanning tree of ``power_net``.

    Topology
    --------
    * One **supply junction** per power node, mirroring its bus position.
    * Supply pipes mirror the spanning-tree branches, forming an acyclic
      supply network.
    * Exactly **one shared return junction** that collects every consumer
      return flow and feeds every generator suction line.
    * Linear heat exchangers (default ``HeatExchanger`` formulation) act as
      heat loads (supply -> return) and heat generators (return -> supply).

    Heat-plant abstraction (``heat_plant_mode``)
    --------------------------------------------
    * ``"closing_pipe"`` (default): a single supply slack plus a closing
      return pipe that loops the return junction back to the slack supply
      junction.  Hydraulically closed.  Caveat: the t-pin at the slack
      partially collapses the supply/return ΔT (water arriving via the
      closing pipe is forced toward ``REF_TEMP`` at the slack), so HX-Loads
      typically deliver only a fraction of design under EF.  This is the
      most robust option with the default ``LinearHeatExchanger``
      formulation.
    * ``"two_port"`` (experimental, physical aspiration): drops the closing
      pipe and adds a second ``ExtHydrGrid`` at the return junction (cold
      port).  In principle preserves the supply/return ΔT, but with the
      ``LinearHeatExchanger`` formulation the resulting nodal heat balance
      at the return junction is either over-constrained (when
      ``return_pin_temperature=True``) or introduces a bilinear
      ``m_ext · T_n`` term Gurobi struggles to converge on (when
      ``return_pin_temperature=False``).  Currently most useful as a
      starting point for studies that switch to the McCormick-DHS
      formulation (``node_based_heat_loads=True``), which has explicit
      nodal energy balances and handles two-port plant boundaries cleanly.
    * ``"screening"`` (open-loop relaxation): closing pipe plus a water
      ``Sink`` at the return junction sized to ~5× the consumer aggregate.
      The slack pumps fresh water; faster to solve because the LP-corner
      ambiguity in mass flow is removed by capping the drain — but the mass
      flow is no longer physically meaningful.  Use only for high-level
      screening studies where aggregate energy is what matters.

    Capacity sizing
    ---------------
    Heat capacities are derived from the power network: every power load
    contributes ``heat_load_share`` × ``p_mw`` as thermal demand, every
    power generator contributes ``heat_gen_share`` × rated power as thermal
    generation.  ``heat_load_share`` is a thermal-to-electrical multiplier
    (default 1.0): on a residential distribution feeder peak space-heating
    demand is roughly comparable to peak electric demand, so a 1:1 ratio is
    a reasonable starting point.  ``balance_gen_to_load`` rescales the
    generator side so total HX-Gen capacity matches total HX-Load demand.

    Node-based heat generators (``node_heat_gen_share``)
    ----------------------------------------------------
    In ``node_based_heat_loads`` mode, distributes node-based
    :class:`~monee.model.HeatGenerator` children across every bus with a
    ``PowerGenerator``, sized as ``node_heat_gen_share × p_gen_mw`` (clamped
    by ``min_gen_mw``).  These enter the McCormick-DHS nodal balance via
    ``q_mw_heat``, exactly like ``HeatLoad`` and the HG-variant CPs — they
    are the heat-side analogue of the gas ``Source`` injection at generator
    buses, and the correct primary fleet for replacement-mode studies to
    drain.

    Default is 1.0 to mirror ``gas_gen_share`` — without a distributed heat
    fleet every kJ flows through the unbounded supply slack, which makes
    heat-side resilience studies meaningless: bounding the slack's
    ``max_import_kgs`` is a hydraulic, not energetic, constraint (mass flow
    is set by sinks, so the bound either has no effect or makes the system
    infeasible without a smooth scarcity transition); lowering
    ``supply_slack_t_k`` only forces consumers to draw *more* mass to
    deliver the same demand.  Set to 0.0 to recover the legacy "slack does
    everything" behaviour explicitly.

    Slack temperature (``supply_slack_t_k``)
    -----------------------------------------
    Defaults to ``REF_TEMP`` (356 K, the network reference).  Exposed so
    studies can lower it to simulate degraded primary heat sources.  Note:
    in node-based mode lowering this *increases* the slack's mass-flow load
    (consumers compensate for lower enthalpy per kg with more mass), so the
    physical effect is to raise pipe velocities, not to cap heat power —
    pair it with distributed ``HeatGenerator`` capacity for a meaningful
    scarcity setup.

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
        # solve.  Pick a 2 m/s design velocity (typical for DHS).
        #
        # τ-envelope: ``[0.75, 1.15]`` ≈ ``[267 K, 409 K]``.
        # The envelope FORCES τ_node ∈ [τ_L, τ_U] via the McCormick partition
        # equations (paper eq. 18c-g), so it must cover the actual physical
        # τ range of every node — otherwise the formulation is provably
        # infeasible.  Anchoring τ_L at ambient (≈ 0.832) is too tight: the
        # LP relaxation tightens monotonically with ``num_partitions``, and
        # at ``S ≥ 7`` deeply-loaded consumer junctions whose physical τ
        # equilibrium falls just below the design return temperature
        # (cumulative pipe losses + 30 K demand drop) push τ < 0.832 and
        # render the formulation infeasible.  Bisection on a simbench
        # LV-rural3 net at ``S = 20`` showed the threshold sits at
        # ``τ_L ≈ 0.76``; ``0.75`` gives a small margin.  τ_U = 1.15
        # accommodates HeatGenerator-injecting CPs (CHP_HG, P2H_HG) that
        # locally raise τ above 1.0 by up to ~5 %.
        heat_grid.t_pu_min_env = 0.75
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

    # Cache the (from, to) supply-tree edges and the resulting WaterPipe ids
    # so we can decorate each pipe with a tighter ``m_U_design`` later in the
    # function once consumer demands are known.
    supply_pipe_for_edge = {}
    for p, c in bfs_edges:
        if p not in keep_nodes or c not in keep_nodes:
            continue
        branch = branch_lookup[(p, c)]
        from_id = bus_index_to_supply_junction[p]
        to_id = bus_index_to_supply_junction[c]
        pipe_id = mx.create_water_pipe(
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
        supply_pipe_for_edge[(p, c)] = pipe_id

    # Two-pass HX construction so HX-Generator capacities can be scaled to
    # match the total HX-Load demand.  Without this scaling the network is
    # systematically heat-deficient on grids where load_share > gen_share or
    # where the per-bus floors push loads above gens (e.g. simbench LV-rural3
    # has Σload_p ≈ 0.37 MW, Σgen_p ≈ 0.21 MW).  In the closed supply/return
    # loop with slack mass = 0 this caps q_delivered at ≈ Σ HX-Gen, so the
    # consumers under-deliver even at regulation = 1.  With the scaling, the
    # generator side has enough heat-injection capacity that an active
    # economic / shedding objective can drive each consumer to its design.
    load_specs = []  # list of (supply_id, q_mw)
    gen_specs = []  # list of (supply_id, q_mw_raw)
    for node in power_net_as_st.nodes:
        if node.id not in bus_index_to_supply_junction:
            continue
        supply_id = bus_index_to_supply_junction[node.id]

        p_load_mw = _node_power_load_mw(power_net, node)
        if p_load_mw > 0 and heat_load_share > 0:
            load_specs.append(
                (supply_id, max(min_load_mw, p_load_mw * heat_load_share))
            )

        if not node_based_heat_loads:
            p_gen_mw = _node_power_gen_mw(power_net, node)
            if p_gen_mw > 0 and heat_gen_share > 0:
                gen_specs.append(
                    (supply_id, max(min_gen_mw, p_gen_mw * heat_gen_share))
                )

    total_load_q = sum(q for _, q in load_specs)
    total_gen_q_raw = sum(q for _, q in gen_specs)
    if balance_gen_to_load and total_gen_q_raw > 0 and total_load_q > total_gen_q_raw:
        gen_scale = total_load_q / total_gen_q_raw
    else:
        gen_scale = 1.0

    # Create the HX-Loads (or node-based HeatLoad children).
    for supply_id, q_mw in load_specs:
        if node_based_heat_loads:
            # Node-based HeatLoad child (uses ``q_mw_heat`` on the junction).
            # Compatible with the McCormick-DHS formulation, which only
            # accounts for branch enthalpy via WaterPipe ``H_in_mw/H_out_mw``
            # and node-level ``q_mw_heat``.  Pair every HeatLoad with a
            # water Sink that drains the consumer's design mass flow —
            # without it the consumer junction is a hydraulic dead-end
            # (incoming pipe pinned to zero flow by McCormick-DHS), which
            # makes the heat-load demand structurally infeasible.
            mx.create_heat_load(target_net, supply_id, q_mw=q_mw)
            # Design flow same as the LinearHX would compute internally:
            # m = q / (c · ΔT) with c = 4180 J/(kg·K), ΔT = 30 K default.
            m_design = q_mw * 1e6 / (4180.0 * 30.0)
            mx.create_water_sink(target_net, supply_id, mass_flow=round(m_design, 6))
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

    # Create the HX-Generators (return → supply) in non-node-based mode,
    # scaled by ``gen_scale`` so total HX-Gen capacity matches HX-Load
    # demand.  Per-bus heat-generator HXs are only meaningful under
    # formulations that allow flow reversal (e.g. plain LinearHeatExchanger);
    # under McCormick-DHS the rooted-outward supply tree pins every pipe's
    # flow direction, so a generator HX injecting mass into an intermediate
    # consumer junction has no path to drain.  Skip them in node-based mode.
    if not node_based_heat_loads:
        for supply_id, q_mw_raw in gen_specs:
            q_mw = q_mw_raw * gen_scale
            mx.create_heat_exchanger(
                target_net,
                from_node_id=return_junction,
                to_node_id=supply_id,
                q_mw=-q_mw,
            )

    # Node-based HeatGenerators at PowerGenerator buses.  This is the
    # McCormick-DHS-compatible counterpart to the gas-side ``Source``
    # injection at generator buses: a distributed primary heat fleet whose
    # rated output participates in the nodal heat balance and gives
    # downstream resilience studies a finite, drainable pool (instead of
    # routing every kJ through the unbounded supply slack).
    if node_based_heat_loads and node_heat_gen_share > 0:
        for node in power_net_as_st.nodes:
            if node.id not in bus_index_to_supply_junction:
                continue
            p_gen_mw = _node_power_gen_mw(power_net, node)
            if p_gen_mw <= 0:
                continue
            q_mw = max(min_gen_mw, p_gen_mw * node_heat_gen_share)
            mx.create_heat_generator(
                target_net,
                node_id=bus_index_to_supply_junction[node.id],
                q_mw=round(q_mw, 6),
            )

    slack_supply_junction = bus_index_to_supply_junction[slack_root]

    if heat_plant_mode not in ("two_port", "closing_pipe", "screening"):
        raise ValueError(
            f"Unknown heat_plant_mode {heat_plant_mode!r}; expected one of "
            f"'two_port', 'closing_pipe', 'screening'."
        )

    # Hot port (supply slack): pins T = supply_slack_t_k, p = 1.0 at the
    # slack junction.  Always present.
    mx.create_ext_hydr_grid(
        target_net,
        node_id=slack_supply_junction,
        t_k=supply_slack_t_k,
        grid_key=mm.WATER_KEY,
        name="Grid Connection Heat Supply",
    )

    if heat_plant_mode == "two_port":
        # Cold port (return slack): anchors the return-junction pressure
        # so circulation can be driven, but does *not* pin the return
        # temperature.  The return T emerges from the upstream heat
        # balance (mass-weighted average of HX-Load outlet temperatures)
        # — pinning it to a fixed setpoint over-constrains the balance
        # whenever pipe losses or partial HX regulation make individual
        # HX outlets diverge from the setpoint.  The two slacks together
        # represent the heat plant as a two-port boundary: water exits
        # at the return slack, is reheated externally, and re-enters at
        # the supply slack.  Mass conservation forces
        # ``m_supply_slack ≈ -m_return_slack`` (steady-state plant pump
        # rate); the supply/return ΔT is preserved because the manifolds
        # are no longer hydraulically shorted by a closing pipe.
        mx.create_ext_hydr_grid(
            target_net,
            node_id=return_junction,
            t_k=return_t_k,
            pressure_pu=return_pressure_pu,
            pin_temperature=return_pin_temperature,
            grid_key=mm.WATER_KEY,
            name="Grid Connection Heat Return",
        )
    else:
        # Legacy closed-loop topology: a single supply slack plus a
        # closing return pipe back to it.  Hydraulically closed, but
        # collapses the supply/return ΔT under the t-pin at the slack —
        # see the docstring.  Optional open-loop relaxation via a Sink
        # at the return junction (``screening``) trades mass-flow
        # realism for solve speed.
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
        if not node_based_heat_loads and heat_plant_mode == "screening" and load_specs:
            sink_capacity = total_load_q * 1e6 / (4180.0 * 30.0) * 5.0
            mx.create_water_sink(
                target_net,
                node_id=return_junction,
                mass_flow=round(sink_capacity, 6),
                name="Grid Connection Heat Return Sink",
            )

    # ---- Per-pipe ``m_U_design`` (Mccormick #5) -----------------------------
    # For the McCormick-DHS formulation each pipe's mass-flow upper bound
    # determines the McCormick envelope width; the velocity-cap default is
    # tens of kg/s, while the actual downstream design demand on an LV-rural
    # tree is typically below 1 kg/s.  A forward sweep on the supply tree
    # gives every pipe its true demand; the formulation's ``_branch_m_U``
    # then prefers it over the velocity cap.  Only meaningful under
    # ``node_based_heat_loads`` (branch-HX trees route flow through HXs and
    # closing pipes, where the simple downstream sum is misleading).
    if node_based_heat_loads:
        bus_demand_kgs = {}
        for supply_id, q_mw in load_specs:
            m_design = q_mw * 1e6 / (4180.0 * 30.0)
            bus_demand_kgs[supply_id] = bus_demand_kgs.get(supply_id, 0.0) + m_design

        children_of = {}
        for p, c in bfs_edges:
            if p in keep_nodes and c in keep_nodes:
                children_of.setdefault(p, []).append(c)

        # Post-order accumulation: each pipe's design mass-flow cap equals
        # the sum of its downstream subtree's consumer demands.  Walk the
        # supply tree once with an iterative DFS — children are popped onto
        # the stack first and finalized before their parent, so when we
        # process a parent every ``children_of[parent]`` is already in
        # ``subtree``.
        subtree = {}
        stack = [(slack_root, False)]
        while stack:
            bus, visited = stack.pop()
            if visited:
                supply_id = bus_index_to_supply_junction[bus]
                total = bus_demand_kgs.get(supply_id, 0.0)
                for child_bus in children_of.get(bus, []):
                    total += subtree[child_bus]
                subtree[bus] = total
            else:
                stack.append((bus, True))
                for child_bus in children_of.get(bus, []):
                    stack.append((child_bus, False))

        # Decorate each pipe with a per-branch m_U_design.  The base estimate
        # ``m = q / (c · ΔT_design)`` assumes the rated 30 K supply/return
        # ΔT, but under tight network bounds (vm/p/t/loading checks) the
        # operating ΔT is smaller and the realised mass-flow is larger.  A
        # 5× safety margin covers ΔT down to ~6 K without making the
        # McCormick envelope infeasible at high partition counts (1.5× was
        # observed to push S=20 over the edge on simbench LV-rural3).  The
        # McCormick formulation still clips by the per-pipe velocity cap,
        # so this is always a tightening relative to the velocity ceiling,
        # never a loosening.
        SAFETY = 5.0
        for (p, c), pipe_id in supply_pipe_for_edge.items():
            m_design = subtree.get(c, 0.0)
            if m_design > 0:
                pipe = target_net.branch_by_id(pipe_id)
                pipe.model.m_U_design = m_design * SAFETY

    return bus_index_to_supply_junction, return_junction


def _drain_power_gen_capacity(net: mm.Network, total_mw: float) -> float:
    """Subtract ``total_mw`` of rated capacity from ``PowerGenerator`` /
    ``GridFormingGenerator`` children, walking them in iteration order.

    Generators whose rated drops to zero are removed.  Returns the unabsorbed
    remainder (positive if the network does not contain enough primary power
    generation to absorb the request).  ``ExtPowerGrid`` is intentionally not
    touched: its ``p_mw`` is a free slack variable, not a rated capacity.
    """
    remaining = float(total_mw)
    if remaining <= 0:
        return 0.0
    for child in list(net.childs):
        if remaining <= 1e-12:
            break
        if isinstance(child.model, mm.PowerGenerator):
            current = abs(float(child.model.p_mw))
            if current <= 0:
                continue
            absorb = min(current, remaining)
            new_mag = current - absorb
            if new_mag <= 1e-12:
                net.remove_child(child.id)
            else:
                child.model.p_mw = -new_mag
            remaining -= absorb
    return remaining


def _drain_gas_source_capacity(net: mm.Network, total_kgs: float) -> float:
    """Subtract ``total_kgs`` of rated capacity from gas-side ``Source``
    children.  Sources at water junctions are skipped (they belong to the heat
    grid).  Returns the unabsorbed remainder.
    """
    remaining = float(total_kgs)
    if remaining <= 0:
        return 0.0
    for child in list(net.childs):
        if remaining <= 1e-12:
            break
        if not isinstance(child.model, mm.Source):
            continue
        if not isinstance(child.grid, mm.GasGrid):
            continue
        current = abs(float(child.model.mass_flow))
        if current <= 0:
            continue
        absorb = min(current, remaining)
        new_mag = current - absorb
        if new_mag <= 1e-12:
            net.remove_child(child.id)
        else:
            child.model.mass_flow = -new_mag
        remaining -= absorb
    return remaining


def _drain_heat_gen_capacity(net: mm.Network, total_mw: float) -> float:
    """Subtract ``total_mw`` of rated heat-injection capacity from the
    primary heat-generation fleet.  Two pools are drained, in order:

    1. Node-based ``HeatGenerator`` children — the McCormick-DHS-compatible
       primary fleet (created by ``create_heat_supply_return_net_for_power``
       when ``node_heat_gen_share > 0``).  ``q_mw_heat`` is negated under
       the load convention, so the rated magnitude is ``abs(q_mw_heat)``.
    2. ``HeatExchanger`` branches with ``q_mw_set > 0`` — the non-node-based
       HX-Generator fleet (return → supply injection branches).

    Returns the unabsorbed remainder.

    When the network is in ``node_based_heat_loads`` mode with
    ``node_heat_gen_share = 0`` no primary fleet exists, and the request
    passes through unabsorbed — that is the correct outcome.  Bounding
    ``ExtHydrGrid.max_import_kgs`` is *not* attempted as a fallback: in
    node-based DHS the slack's mass flow is hydraulically determined by the
    consumers (sinks), so a mass-flow cap is not a smooth heat-power
    scarcity dial — it either has no effect (demand ≤ cap) or makes the
    model hard-infeasible (demand > cap), with no graceful in-between.  The
    principled fix is to give the network a distributed primary heat fleet
    upstream via ``node_heat_gen_share > 0``; lowering
    ``supply_slack_t_k`` is the other available knob but only affects pipe
    velocities, not heat power.
    """
    remaining = float(total_mw)
    if remaining <= 0:
        return 0.0

    for child in list(net.childs):
        if remaining <= 1e-12:
            break
        if not isinstance(child.model, mm.HeatGenerator):
            continue
        current = abs(float(child.model.q_mw_heat))
        if current <= 0:
            continue
        absorb = min(current, remaining)
        new_mag = current - absorb
        if new_mag <= 1e-12:
            net.remove_child(child.id)
        else:
            child.model.q_mw_heat = -new_mag
        remaining -= absorb

    for branch in list(net.branches):
        if remaining <= 1e-12:
            break
        if not isinstance(branch.model, mm.HeatExchanger):
            continue
        q_set = float(getattr(branch.model, "q_mw_set", 0.0) or 0.0)
        if q_set <= 0:
            continue
        absorb = min(q_set, remaining)
        new_q = q_set - absorb
        if new_q <= 1e-12:
            net.remove_branch_between(
                branch.from_node_id, branch.to_node_id, branch.id[2]
            )
        else:
            branch.model.q_mw_set = new_q
        remaining -= absorb

    return remaining


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
    chp_p_share=0.5,
    p2g_efficiency=0.7,
    p2g_p_share=1.0,
    p2h_efficiency=0.95,
    p2h_p_share=1.0,
    p2h_diameter_m=0.01,
    cp_size_multiplier=1.0,
    regulation=1.0,
    use_hg_variants=False,
    seed=None,
    replace_primary_generation=False,
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

    * **CHP** — fuel mass flow chosen so the *electrical* output covers
      ``chp_p_share`` × ``p_ref_mw`` (default 0.5, i.e. ~50 % of the bus's
      load / generation magnitude).
    * **P2G** — electrical input is ``p2g_p_share`` × ``p_ref_mw`` (default
      1.0); output gas mass flow follows from ``p2g_efficiency`` and HHV.
    * **P2H** — electrical input is ``p2h_p_share`` × ``p_ref_mw`` (default
      1.0); heat output follows via ``p2h_efficiency``.

    The global ``cp_size_multiplier`` (default 1.0) scales every unit's
    rated output uniformly on top of the per-type shares above.  Use it as
    the headline knob for resilience studies that ask "how big do CPs need
    to be before their contribution rises above noise?" — pair with
    ``extra_mesh_pipes`` on the gas grid to give the larger CPs a fuel
    path that survives single failures.

    Replacement mode
    ----------------
    Default (``replace_primary_generation=False``) is **purely additive**:
    coupling units stack on top of the existing primary generators, which
    makes coupling points look unconditionally beneficial for resilience —
    losing one is always recoverable by the unchanged primary fleet.

    With ``replace_primary_generation=True`` each unit's *output* capacity is
    absorbed from the matching pool of primary generation, keeping the
    network's total rated production per carrier invariant:

    * **CHP** (gas → power + heat): subtract its rated electrical output from
      ``PowerGenerator`` capacity and its rated thermal output from the
      heat-generator (HX-Gen) pool.
    * **G2P** (gas → power): subtract its rated electrical output from
      ``PowerGenerator`` capacity.
    * **P2G** (power → gas): subtract its rated gas mass flow from the
      gas-side ``Source`` pool.
    * **P2H** (power → heat): subtract its rated thermal output from the
      heat-generator pool.

    This flips the framing of the network from *redundancy* to *cross-carrier
    dependence*: losing a carrier now disables both the coupling unit *and*
    the primary generation it displaced.  In ``node_based_heat_loads`` mode
    the heat pool drains node-based ``HeatGenerator`` children created via
    ``heat_kwargs={"node_heat_gen_share": ...}`` upstream.  If no such
    fleet exists the heat-side reduction is reported as unabsorbed and *no*
    slack-bound fallback is attempted — bounding ``ExtHydrGrid.max_import_kgs``
    is a hydraulic, not energetic, constraint (mass flow is set by sinks),
    so it is not a smooth scarcity dial.  Provision a ``HeatGenerator``
    fleet upstream if you want CP heat output to displace something.

    Returns
    -------
    list[dict]
        One entry per created unit: ``{"type", "node", "id"}``.
    """
    if not 0 <= regulation <= 1:
        raise ValueError("regulation must be in [0, 1]")
    if not 0 <= density <= 1:
        raise ValueError("density must be in [0, 1]")
    if cp_size_multiplier <= 0:
        raise ValueError("cp_size_multiplier must be > 0")
    for share_name, share in (
        ("chp_p_share", chp_p_share),
        ("p2g_p_share", p2g_p_share),
        ("p2h_p_share", p2h_p_share),
    ):
        if share < 0:
            raise ValueError(f"{share_name} must be >= 0")

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
    # Tracks the cumulative rated *output* capacity of every CP we add, so we
    # can absorb the same amount from primary generation when
    # ``replace_primary_generation`` is set (see end of function).
    cp_power_out_mw = 0.0
    cp_gas_out_kgs = 0.0
    cp_heat_out_mw = 0.0
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
            # Fuel mass flow chosen so the electrical output covers
            # ``chp_p_share · cp_size_multiplier · p_ref_mw`` at the configured
            # electrical efficiency.
            chp_p_target_mw = chp_p_share * cp_size_multiplier * p_ref_mw
            mass_flow = round(
                chp_p_target_mw / max(chp_efficiency_power, 1e-3) / GAS_HHV_MJ_PER_KG,
                6,
            )
            cp_power_out_mw += mass_flow * GAS_HHV_MJ_PER_KG * chp_efficiency_power
            cp_heat_out_mw += mass_flow * GAS_HHV_MJ_PER_KG * chp_efficiency_heat
            if use_hg_variants:
                # HeatGenerator-based CHP: heat injection via q_mw_heat at the
                # supply junction, no return-side branch.  Required for the
                # McCormick-DHS formulation, which only sees node-level heat
                # injection (q_mw_heat) and pipe enthalpy (H_in_mw/H_out_mw).
                uid = mx.create_chp_hg(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    gas_node_id=gas_junc,
                    mass_flow_setpoint=mass_flow,
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
                    mass_flow_setpoint=mass_flow,
                    diameter_m=chp_diameter_m,
                    efficiency_power=chp_efficiency_power,
                    efficiency_heat=chp_efficiency_heat,
                    regulation=regulation,
                )
            created.append({"type": "chp", "node": power_node_id, "id": uid})

        elif unit_type == "p2g":
            p2g_p_in_mw = p2g_p_share * cp_size_multiplier * p_ref_mw
            mass_flow = round(p2g_efficiency * p2g_p_in_mw / GAS_HHV_MJ_PER_KG, 6)
            cp_gas_out_kgs += mass_flow
            bid = mx.create_p2g(
                mes_net,
                from_node_id=power_node_id,
                to_node_id=gas_junc,
                efficiency=p2g_efficiency,
                mass_flow_setpoint=mass_flow,
                regulation=regulation,
            )
            created.append({"type": "p2g", "node": power_node_id, "id": bid})

        elif unit_type == "p2h":
            p2h_p_in_mw = p2h_p_share * cp_size_multiplier * p_ref_mw
            heat_mw = round(p2h_p_in_mw * p2h_efficiency, 6)
            cp_heat_out_mw += heat_mw
            if use_hg_variants:
                bid = mx.create_p2h_hg(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    heat_energy_mw=heat_mw,
                    efficiency=p2h_efficiency,
                )
                created.append({"type": "p2h", "node": power_node_id, "id": bid})
            else:
                uid = mx.create_p2h(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    heat_return_node_id=heat_return_junc,
                    heat_energy_mw=heat_mw,
                    diameter_m=p2h_diameter_m,
                    efficiency=p2h_efficiency,
                    regulation=regulation,
                )
                created.append({"type": "p2h", "node": power_node_id, "id": uid})

    if replace_primary_generation:
        # Drain primary generation to keep total rated capacity per carrier
        # invariant.  The heat pool now covers both node-based
        # ``HeatGenerator`` children (created when
        # ``node_heat_gen_share > 0``) and non-node-based HX-Generator
        # branches.  No slack-bound fallback: in node-based DHS the slack's
        # mass flow is hydraulically determined by consumer sinks, so a
        # ``max_import_kgs`` cap is not a smooth scarcity dial — the
        # principled fix is to give the network a distributed
        # ``HeatGenerator`` fleet via ``heat_kwargs={"node_heat_gen_share":
        # ...}`` upstream.
        unabsorbed_p = _drain_power_gen_capacity(mes_net, cp_power_out_mw)
        unabsorbed_g = _drain_gas_source_capacity(mes_net, cp_gas_out_kgs)
        unabsorbed_h = _drain_heat_gen_capacity(mes_net, cp_heat_out_mw)
        for label, asked, left in (
            ("PowerGenerator", cp_power_out_mw, unabsorbed_p),
            ("gas Source", cp_gas_out_kgs, unabsorbed_g),
            ("heat (HeatGenerator + HX-Gen)", cp_heat_out_mw, unabsorbed_h),
        ):
            if left > 1e-9 and asked > 0:
                print(
                    f"[create_coupling_points_for_mes] replace_primary_generation: "
                    f"{label} pool absorbed {asked - left:g} of {asked:g} requested; "
                    f"{left:g} unabsorbed (likely no remaining primary capacity)."
                )

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

    Useful ``coupling_kwargs`` flags forwarded to
    :func:`create_coupling_points_for_mes`:

    * ``seed`` — RNG seed for reproducible CP placement.
    * ``use_hg_variants`` — switch CHP / P2H to their ``HeatGenerator``-style
      compounds (required by the McCormick-DHS formulation).
    * ``cp_size_multiplier`` — global scaling on every CP's rated capacity
      (default 1.0).  Headline knob for resilience studies that ask how
      large CPs need to be before their additive contribution rises above
      noise; per-type overrides via ``chp_p_share`` / ``p2g_p_share`` /
      ``p2h_p_share`` (defaults 0.5 / 1.0 / 1.0).
    * ``replace_primary_generation`` — make every CP's rated *output* absorb
      capacity from the matching primary pool (PowerGenerator / gas Source /
      HX-Generator), keeping total rated production per carrier invariant.
      Default is purely additive.  Use the replacement mode to study the
      *cross-carrier dependence cost* of coupling points (the "danger" framing,
      complementing the additive "redundancy" framing).

    Useful ``gas_kwargs`` flags forwarded to
    :func:`create_gas_tree_net_for_power`:

    * ``extra_mesh_pipes`` — add ``N`` random cross-connection pipes to the
      otherwise-strict gas tree.  Recommended for resilience studies in
      additive CP mode: without meshing, a single gas-pipe failure isolates
      the entire downstream subtree, which is the dominant reason additive
      coupling points are statistically irrelevant under random-failure
      sampling on this layout (a CHP whose fuel path is severed contributes
      nothing regardless of how much rated capacity it carries).
    * ``mesh_seed`` — RNG seed for reproducible tie-pipe placement.

    Useful ``heat_kwargs`` flags forwarded to
    :func:`create_heat_supply_return_net_for_power`:

    * ``node_based_heat_loads`` — node-based ``HeatLoad`` consumers
      (required by McCormick-DHS); default ``False``.
    * ``node_heat_gen_share`` — in node-based mode, distribute
      ``HeatGenerator`` children at every ``PowerGenerator`` bus with rated
      output ``node_heat_gen_share × p_gen_mw``.  Default is 1.0, mirroring
      ``gas_gen_share``; the heat sector gets a finite, drainable primary
      fleet rather than routing every kJ through the unbounded supply
      slack.  Set to 0.0 to recover the legacy slack-only behaviour
      explicitly.
    * ``supply_slack_t_k`` — supply slack pinned temperature (default
      ``REF_TEMP``).  Lowering it does *not* cap slack heat power directly
      (mass flow is hydraulically determined by sinks); it forces
      consumers to pull more mass to meet the same demand and is mostly
      useful as a "degraded primary" scenario when combined with finite
      ``node_heat_gen_share``.
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
