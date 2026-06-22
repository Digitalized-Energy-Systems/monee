import random

import networkx as nx
from geopy import distance

import monee.express as mx
import monee.model as mm
import monee.model.phys.core.hydraulics as hyd
from monee.model.grid import DEFAULT_GAS_HHV_MJ_PER_KG
from monee.model.phys.nonlinear.gf import reference_gas_density

REF_PA = 1000000
REF_TEMP = 356
DEFAULT_LENGTH = 100

# Gas HHV [MJ/kg] for sizing gas demands from power magnitudes. Sourced from the
# lgas grid definition so the sizing uses the SAME heating value as the coupling
# physics (see monee.model.grid.DEFAULT_GAS_HHV_MJ_PER_KG).
GAS_HHV_MJ_PER_KG = DEFAULT_GAS_HHV_MJ_PER_KG


def _node_power_load_mw(power_net: mm.Network, node):
    p_mw = 0.0
    for child in power_net.childs_by_ids(node.child_ids):
        if isinstance(child.model, mm.PowerLoad):
            p_mw += float(getattr(child.model, "p_mw", 0.0) or 0.0)
    return p_mw


def _node_power_gen_mw(power_net: mm.Network, node):
    """|generation| at *node* from PowerGenerator + GridFormingGenerator.
    Excludes ExtPowerGrid (its p_mw is a free slack Var pre-solve)."""
    p_mw = 0.0
    for child in power_net.childs_by_ids(node.child_ids):
        model = child.model
        if isinstance(model, mm.PowerGenerator):
            p_mw += abs(float(getattr(model, "p_mw", 0.0) or 0.0))
        elif isinstance(model, mm.GridFormingGenerator):
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


def _length_weight(branch, node_from, node_to):
    """Edge weight = pipe length [m]: ``length_m`` if set, else geodesic
    distance between the endpoint positions, else the default length."""
    length = getattr(branch.model, "length_m", None)
    if length is not None:
        return float(length)
    if node_from.position is not None and node_to.position is not None:
        return distance.distance(node_from.position, node_to.position).m
    return float(DEFAULT_LENGTH)


def _resolve_backbone_weight(weight):
    return _length_weight if weight == "length" else weight


def _demand_supply_terminals(power_net, slack_node_id):
    """Buses a Steiner backbone must connect: the slack plus every bus that
    carries electrical load or generation."""
    terminals = {slack_node_id}
    for node in power_net.nodes:
        if (
            _node_power_load_mw(power_net, node) > 0
            or _node_power_gen_mw(power_net, node) > 0
        ):
            terminals.add(node.id)
    return terminals


def _make_backbone(power_net, slack_node_id, method, weight):
    """Spanning-tree (default) or Steiner backbone of ``power_net`` used as the
    shared gas/heat layout skeleton."""
    weight_fn = _resolve_backbone_weight(weight)
    if method == "steiner":
        return mm.to_backbone(
            power_net,
            method="steiner",
            weight=weight_fn,
            terminals=_demand_supply_terminals(power_net, slack_node_id),
        )
    return mm.to_backbone(power_net, method="span", weight=weight_fn)


def create_heat_net_for_power(
    power_net,
    target_net,
    heat_deployment_rate,
    mass_flow_rate_kgs=0.075,
    default_diameter_m=0.12,
    length_scale=1,
    default_length=DEFAULT_LENGTH,
    power_scale=1,
    design_flow_headroom=1.5,
    seed=None,
):
    rng = random.Random(seed) if seed is not None else random
    heat_grid = mm.create_water_grid("water", t_ref_k=REF_TEMP, pressure_ref_pa=REF_PA)
    target_net.set_default_grid("water", heat_grid)

    power_net_as_st = mm.to_spanning_tree(power_net)
    bus_index_to_junction_index = {}
    bus_index_to_end_junction_index = {}
    node_demand_kgs = {}

    for node in power_net_as_st.nodes:
        junc_id = mx.create_junction(target_net, position=node.position, grid=heat_grid)
        sink_mass_flow = mass_flow_rate_kgs + rng.random() * mass_flow_rate_kgs / 10
        node_demand_kgs[node.id] = sink_mass_flow
        mx.create_sink(
            target_net,
            junc_id,
            mass_flow_kgs=sink_mass_flow,
        )
        bus_index_to_junction_index[node.id] = junc_id
        bus_index_to_end_junction_index[node.id] = junc_id
        deployment_c_value = rng.random()
        if deployment_c_value < heat_deployment_rate:
            bus_index_to_end_junction_index[node.id] = mx.create_junction(
                target_net, position=node.position, grid=heat_grid
            )
            # Passive: an in-line device on the distribution run - the tree
            # flow is already fixed by the downstream sinks, so a design-flow
            # prescribing HeatExchanger would over-determine the hydraulics.
            mx.create_passive_heat_exchanger(
                target_net,
                from_node_id=bus_index_to_junction_index[node.id],
                to_node_id=bus_index_to_end_junction_index[node.id],
                q_mw=(-1 if rng.random() > 0.8 else 1)
                * -0.1
                * rng.random()
                * power_scale,
                diameter_m=default_diameter_m,
            )
            end_sink_mass_flow = (
                mass_flow_rate_kgs + rng.random() * mass_flow_rate_kgs / 10
            )
            node_demand_kgs[node.id] += end_sink_mass_flow
            mx.create_sink(
                target_net,
                bus_index_to_end_junction_index[node.id],
                mass_flow_kgs=end_sink_mass_flow,
            )

    # Capacity-based trunk sizing (same design as the auto_diameter mode of
    # create_heat_supply_return_net_for_power): every tree pipe must carry the
    # cumulative downstream sink demand, so size its diameter for that flow at
    # the grid's design velocity_mps - floored at default_diameter_m so leaf pipes
    # are never thinned. Without this, a flat default diameter (or the grid
    # max_mass_flow_kgs) caps the slack trunk below total demand and the net is infeasible
    # by construction.
    root = power_net_as_st.first_node()
    parent = {root: None}
    undirected = nx.Graph(power_net_as_st.graph)
    for p, c in nx.bfs_edges(undirected, source=root):
        parent[c] = p
    subtree_kgs = dict(node_demand_kgs)
    for c in reversed(list(nx.topological_sort(nx.bfs_tree(undirected, root)))):
        if parent.get(c) is not None:
            subtree_kgs[parent[c]] += subtree_kgs[c]

    def _trunk_diameter(downstream_kgs):
        d = hyd.calc_min_diameter_for_mass_flow(
            downstream_kgs * design_flow_headroom,
            heat_grid.fluid_density_kg_per_m3,
            heat_grid.v_max_mps,
        )
        return max(d, default_diameter_m)

    for branch in power_net_as_st.branches:
        # The endpoint whose tree-parent is the other endpoint sits downstream.
        if parent.get(branch.to_node_id) == branch.from_node_id:
            downstream = subtree_kgs[branch.to_node_id]
        else:
            downstream = subtree_kgs[branch.from_node_id]
        from_node_id = bus_index_to_end_junction_index[branch.from_node_id]
        to_node_id = bus_index_to_junction_index[branch.to_node_id]
        mx.create_water_pipe(
            target_net,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            diameter_m=_trunk_diameter(downstream),
            length_m=get_length(
                target_net,
                branch,
                from_node_id,
                to_node_id,
                default_length=default_length,
            )
            * length_scale,
            temperature_ext_k=296.15,
            roughness_m=0.001,
            grid=heat_grid,
        )
    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[root],
        t_k=REF_TEMP,
        name="Grid Connection Heat",
    )
    # max_mass_flow_kgs is the grid-level per-branch cap: it must at least admit the slack
    # trunk's design flow or the sized diameters are moot. Leaf pipes stay
    # tightly capped through their velocity_mps-based per-pipe bound.
    heat_grid.max_mass_flow_kgs = max(
        heat_grid.max_mass_flow_kgs, design_flow_headroom * subtree_kgs[root]
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
    seed=None,
    gas_type="lgas",
):
    rng = random.Random(seed) if seed is not None else random
    gas_grid = mm.create_gas_grid(
        "gas", type=gas_type, t_ref_k=REF_TEMP, pressure_ref_pa=REF_PA
    )

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
        deployment_c_value = rng.random()
        if deployment_c_value <= gas_deployment_rate:
            mx.create_sink(
                target_net,
                bus_index_to_junction_index[node.id],
                mass_flow_kgs=round(0.1 + 0.5 * rng.random() * scaling, 2),
            )
    mx.create_source(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()],
        mass_flow_kgs=10 * scaling * source_scaling,
    )
    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[power_net_as_st.first_node()],
        t_k=REF_TEMP,
        name="Grid Connection Gas",
    )
    return bus_index_to_junction_index


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
    bus_index_to_junction_index, _ = create_heat_net_for_power(
        pn, new_mes, 1, mass_flow_rate_kgs=1, default_diameter_m=0.16
    )
    new_water_junc = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc,
        mass_flow_kgs=0.05,
    )
    new_water_junc_2 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_2,
        mass_flow_kgs=0.05,
    )
    mx.create_passive_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc,
        to_node_id=new_water_junc_2,
        q_mw=0.03,
        diameter_m=0.16,
    )
    new_water_junc_3 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_3,
        mass_flow_kgs=0.06,
    )
    mx.create_passive_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc_2,
        to_node_id=new_water_junc_3,
        q_mw=0.03,
        diameter_m=0.16,
    )
    mx.create_p2g(
        new_mes,
        from_node_id=node_4,
        to_node_id=bus_to_gas_junc[node_4],
        efficiency=0.7,
        mass_flow_setpoint_kgs=1,
        regulation=0,
    )
    mx.create_chp(
        new_mes,
        power_node_id=node_1,
        heat_node_id=bus_index_to_junction_index[node_0],
        heat_return_node_id=new_water_junc,
        gas_node_id=bus_to_gas_junc[node_3],
        mass_flow_setpoint_kgs=0.005,
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
        # This reference grid was tuned for the pre-2026 methane-like gas; keep
        # it on that gas so the bench net stays feasible under realistic lgas.
        gas_type="methane",
    )
    create_heat_net_for_power(
        monee_net,
        new_mes,
        0.5,
        mass_flow_rate_kgs=25,
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
        mass_flow_setpoint_kgs=0.5,
        regulation=0.1,
    )
    mx.create_chp(
        new_mes,
        power_node_id=2,
        heat_node_id=43,
        heat_return_node_id=44,
        gas_node_id=25,
        mass_flow_setpoint_kgs=0.5,
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


def _add_gas_mesh_pipes(
    target_net,
    bus_index_to_junction_index,
    gas_grid,
    extra_mesh_pipes,
    mesh_diameter_factor,
    mesh_length_factor,
    default_diameter_m,
    default_length,
    length_scale,
    mesh_seed,
):
    # Add resilience tie pipes: pick non-adjacent junction pairs uniformly
    # at random and connect them with a smaller-diameter / longer pipe.
    # The point is to give every junction more than one path to the
    # slack so single pipe failures don't isolate large subtrees - the
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


def create_gas_tree_net_for_power(  # NOSONAR
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
    auto_diameter=False,
    auto_diameter_v_mps=None,
    auto_diameter_headroom=1.5,
    auto_min_diameter_m=0.02,
    pressure_ref_pa=REF_PA,
    pressure_ambient_pa=0.0,
    diameter_tiers=None,
    backbone_method="span",
    backbone_weight=None,
):
    """Build a gas grid on the spanning tree of ``power_net``.

    ``pressure_ref_pa`` / ``pressure_ambient_pa`` set the gas grid's reference
    (nominal) pressure and the absolute/gauge convention (see
    :func:`monee.model.grid.create_gas_grid`); the defaults reproduce the
    historical 10 bar absolute grid.

    ``diameter_tiers`` (default ``None``) selects a realistic trunk->branch->
    service diameter taper instead of flow-based ``auto_diameter`` or the flat
    ``default_diameter_m``. It is a list of ``(min_downstream_consumers,
    diameter_m)`` pairs, sorted by descending threshold; each tree pipe is sized
    by the number of demand-bearing junctions in the subtree it feeds, so mains
    near the slack come out wide (many consumers) and leaf service pipes narrow.
    At LV gas demands (grams/s) flow never sets the pipe size, so a structural
    tier schedule is the realistic sizing rule rather than hydraulics.

    Loads/generators are sized from electrical magnitudes via ``gas_load_share`` /
    ``gas_gen_share`` and the HHV; ``extra_mesh_pipes`` add tie-lines so a single
    failure doesn't isolate a subtree. ``min_load_kgs`` / ``min_source_kgs``
    floor per-bus values at ~10 kW thermal.

    ``auto_diameter`` (default off) sizes each tree pipe to the cumulative
    downstream sink demand it must carry, instead of the flat
    ``default_diameter_m``.  The required diameter is the smallest keeping the
    gas design velocity at ``auto_diameter_v_mps`` (defaults to the grid's
    ``v_max_mps``, i.e. the same 20 m/s cap the gas formulation enforces) after a
    ``auto_diameter_headroom`` margin on the flow, floored at
    ``auto_min_diameter_m`` (default 0.02 m, a DN20 service pipe, so leaf pipes
    stay realistic rather than collapsing to zero).  Without it, the flat 0.3 m
    default over LV-scale demands (~0.03 kg/s) makes the Weymouth pressure drop
    (Δp² ∝ L·f·m²/D⁵) vanish - the gas hydraulics never move regardless of load.
    Mirrors the ``auto_diameter`` mode of
    :func:`create_heat_supply_return_net_for_power`.

    Returns ``{power_node_id → gas_junction_id}``.
    """
    gas_grid = mm.create_gas_grid(
        "gas",
        type="lgas",
        t_ref_k=REF_TEMP,
        pressure_ref_pa=pressure_ref_pa,
        pressure_ambient_pa=pressure_ambient_pa,
    )
    target_net.set_default_grid("gas", gas_grid)
    # HHV [MJ/kg] of this grid's gas fluid (sizing matches the gas physics).
    gas_hhv_mj = gas_grid.higher_heating_value_kwh_per_kg * 3.6

    slack_node = slack_node_id if slack_node_id is not None else power_net.first_node()
    power_net_as_st = _make_backbone(
        power_net, slack_node, backbone_method, backbone_weight
    )

    bus_index_to_junction_index = {}
    for node in power_net_as_st.nodes:
        bus_index_to_junction_index[node.id] = mx.create_junction(
            target_net, position=node.position, grid=gas_grid
        )

    # Per-bus gas sink demand [kg/s]. Computed up front (not inline at sink
    # creation) so capacity-based pipe sizing can sum it over each subtree.
    bus_sink_kgs: dict = {}
    for node in power_net_as_st.nodes:
        p_load_mw = _node_power_load_mw(power_net, node)
        if p_load_mw > 0 and gas_load_share > 0:
            bus_sink_kgs[node.id] = round(
                max(min_load_kgs, p_load_mw * gas_load_share / gas_hhv_mj), 4
            )

    # Capacity-based diameter: size every tree pipe for its cumulative downstream
    # sink demand at the gas design velocity, so the Weymouth drop (∝ L·f·m²/D⁵)
    # stays physical instead of vanishing under a flat oversized diameter. The
    # subtree totals mirror the heat net's auto_diameter accumulation.
    parent: dict = {}
    subtree_kgs: dict = {}
    subtree_consumers: dict = {}
    if auto_diameter or diameter_tiers:
        undirected = nx.Graph(power_net_as_st.graph)
        parent = {slack_node: None}
        for p, c in nx.bfs_edges(undirected, source=slack_node):
            parent[c] = p
        order = list(
            reversed(list(nx.topological_sort(nx.bfs_tree(undirected, slack_node))))
        )
        subtree_kgs = {nid: bus_sink_kgs.get(nid, 0.0) for nid in parent}
        subtree_consumers = {nid: (1 if nid in bus_sink_kgs else 0) for nid in parent}
        for c in order:
            if parent.get(c) is not None:
                subtree_kgs[parent[c]] += subtree_kgs[c]
                subtree_consumers[parent[c]] += subtree_consumers[c]

    if diameter_tiers:
        # Descending threshold order so the first match is the widest tier.
        _tiers = sorted(diameter_tiers, key=lambda t: t[0], reverse=True)

        def _tier_diameter(n_consumers):
            for thresh, dia in _tiers:
                if n_consumers >= thresh:
                    return dia
            return _tiers[-1][1]

    if auto_diameter:
        design_v = (
            auto_diameter_v_mps
            if auto_diameter_v_mps is not None
            else getattr(gas_grid, "v_max_mps", 20.0)
        )
        gas_density = reference_gas_density(gas_grid)

        def _gas_diameter(downstream_kgs):
            """Smallest gas-pipe diameter [m] carrying ``downstream_kgs`` at the
            design velocity (with headroom), floored at ``auto_min_diameter_m``."""
            d = hyd.calc_min_diameter_for_mass_flow(
                downstream_kgs * auto_diameter_headroom, gas_density, design_v
            )
            return max(d, auto_min_diameter_m)

    for branch in power_net_as_st.branches:
        from_id = bus_index_to_junction_index[branch.from_node_id]
        to_id = bus_index_to_junction_index[branch.to_node_id]
        # Downstream endpoint = the one whose tree-parent is the other.
        if parent.get(branch.to_node_id) == branch.from_node_id:
            downstream_node = branch.to_node_id
        else:
            downstream_node = branch.from_node_id
        if diameter_tiers:
            diameter_m = _tier_diameter(subtree_consumers[downstream_node])
        elif auto_diameter:
            diameter_m = _gas_diameter(subtree_kgs[downstream_node])
        else:
            diameter_m = default_diameter_m
        mx.create_gas_pipe(
            target_net,
            from_node_id=from_id,
            to_node_id=to_id,
            diameter_m=diameter_m,
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
        if node.id in bus_sink_kgs:
            mx.create_sink(
                target_net,
                bus_index_to_junction_index[node.id],
                mass_flow_kgs=bus_sink_kgs[node.id],
            )

        p_gen_mw = _node_power_gen_mw(power_net, node)
        if p_gen_mw > 0 and gas_gen_share > 0:
            mass_flow_kgs = max(min_source_kgs, p_gen_mw * gas_gen_share / gas_hhv_mj)
            mx.create_source(
                target_net,
                node_id=bus_index_to_junction_index[node.id],
                mass_flow_kgs=round(mass_flow_kgs, 4),
            )

    if extra_mesh_pipes > 0:
        _add_gas_mesh_pipes(
            target_net,
            bus_index_to_junction_index,
            gas_grid,
            extra_mesh_pipes=extra_mesh_pipes,
            mesh_diameter_factor=mesh_diameter_factor,
            mesh_length_factor=mesh_length_factor,
            default_diameter_m=default_diameter_m,
            default_length=default_length,
            length_scale=length_scale,
            mesh_seed=mesh_seed,
        )

    mx.create_ext_hydr_grid(
        target_net,
        node_id=bus_index_to_junction_index[slack_node],
        t_k=REF_TEMP,
        name="Grid Connection Gas",
    )
    return bus_index_to_junction_index


def create_heat_supply_return_net_for_power(  # NOSONAR
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
    auto_diameter=False,
    auto_diameter_v_mps=None,
    auto_diameter_headroom=1.5,
    auto_min_diameter_m=None,
    backbone_method="span",
    backbone_weight=None,
):
    """Build a supply/return DHS grid on the spanning tree of ``power_net``.

    Supply junctions mirror the buses; one shared return junction collects all
    consumer returns and feeds the generators. HX-Load and HX-Gen branches
    bridge supply and return.

    ``heat_plant_mode``:
      * ``"closing_pipe"`` (default) - single supply slack + return-to-supply
        closing pipe. Robust under the LinearHeatExchanger formulation but the
        slack t-pin collapses some supply/return ΔT.
      * ``"two_port"`` - second slack at the return junction. Cleaner physics
        but needs McCormick-DHS (``node_based_heat_loads=True``).
      * ``"screening"`` - closing pipe plus an oversized return Sink; faster
        but mass flow is no longer physical.

    Capacities scale from electrical magnitudes via ``heat_load_share`` /
    ``heat_gen_share``; ``balance_gen_to_load`` rescales generators to match.
    In node-based mode ``node_heat_gen_share`` distributes :class:`HeatGenerator`
    children at every PowerGenerator bus (set to 0.0 for slack-only).

    ``auto_diameter`` (default off) sizes each supply pipe - and the return /
    closing pipe - to the mass flow it must carry instead of the flat
    ``default_diameter_m``.  The required flow is the cumulative downstream
    consumer demand (the same per-pipe ``subtree`` total used for the
    McCormick-DHS envelopes), so trunk pipes near the slack come out wider than
    leaf pipes.  Each diameter is the smallest that keeps the design velocity_mps at
    ``auto_diameter_v_mps`` (defaults to the grid's ``v_max_mps``) after a
    ``auto_diameter_headroom`` margin on the flow, floored at
    ``auto_min_diameter_m`` (defaults to ``default_diameter_m`` so leaf pipes are
    never thinned below the flat default).  Without it, a large radial DHS (e.g.
    simbench MV-urban: ~256 kg/s through 0.12 m pipes) is physically infeasible -
    the trunk flow exceeds the velocity_mps cap and the Darcy pressure drop blows up.

    Returns ``({power_node_id → supply_junction_id}, return_junction_id)``.
    """
    heat_grid = mm.create_water_grid("water", t_ref_k=REF_TEMP, pressure_ref_pa=REF_PA)
    if node_based_heat_loads:
        # Tighten McCormick-DHS envelopes
        heat_grid.t_pu_min_env = 0.75
        heat_grid.t_pu_max_env = 1.15
        heat_grid.v_max_mps = 2.0
    target_net.set_default_grid("water", heat_grid)

    auto_v = (
        auto_diameter_v_mps if auto_diameter_v_mps is not None else heat_grid.v_max_mps
    )
    auto_floor_m = (
        auto_min_diameter_m if auto_min_diameter_m is not None else default_diameter_m
    )

    def _auto_diameter(mass_flow_kgs):
        """Capacity-sized diameter [m] for ``mass_flow_kgs``, floored."""
        d = hyd.calc_min_diameter_for_mass_flow(
            mass_flow_kgs * auto_diameter_headroom,
            heat_grid.fluid_density_kg_per_m3,
            auto_v,
        )
        return max(d, auto_floor_m)

    # Orient supply pipes outward from the slack via BFS so every supply
    # junction has an incoming pipe (McCormick-DHS pins direction).
    slack_root = slack_node_id if slack_node_id is not None else power_net.first_node()
    power_net_as_st = _make_backbone(
        power_net, slack_root, backbone_method, backbone_weight
    )
    undirected = nx.Graph(power_net_as_st.graph)
    branch_lookup = {}
    for branch in power_net_as_st.branches:
        branch_lookup[(branch.from_node_id, branch.to_node_id)] = branch
        branch_lookup[(branch.to_node_id, branch.from_node_id)] = branch

    bfs_edges = list(nx.bfs_edges(undirected, source=slack_root))

    if node_based_heat_loads:
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
            roughness_m=0.001,
            grid=heat_grid,
        )
        supply_pipe_for_edge[(p, c)] = pipe_id

    load_specs = []  # list of (supply_id, heat_mw)
    gen_specs = []  # list of (supply_id, heat_mw_raw)
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
    for supply_id, heat_mw in load_specs:
        if node_based_heat_loads:
            mx.create_heat_load(target_net, supply_id, q_mw=heat_mw)
            # Design flow same as the LinearHX would compute internally:
            # m = q / (c · ΔT) with c = 4180 J/(kg·K), ΔT = 30 K default.
            m_design = heat_mw * 1e6 / (4180.0 * 30.0)
            mx.create_water_sink(
                target_net, supply_id, mass_flow_kgs=round(m_design, 6)
            )
        else:
            mx.create_heat_exchanger(
                target_net,
                from_node_id=supply_id,
                to_node_id=return_junction,
                q_mw=heat_mw,
            )

    if not node_based_heat_loads:
        for supply_id, heat_mw_raw in gen_specs:
            heat_mw = heat_mw_raw * gen_scale
            mx.create_heat_exchanger(
                target_net,
                from_node_id=return_junction,
                to_node_id=supply_id,
                q_mw=-heat_mw,
            )

    bus_demand_kgs: dict[int, float] = {}
    subtree: dict = {}
    # The per-pipe cumulative downstream demand feeds both the McCormick-DHS
    # envelopes (node-based mode) and capacity-based pipe sizing (auto_diameter),
    # so compute it whenever either is requested.
    if node_based_heat_loads or auto_diameter:
        for supply_id, heat_mw in load_specs:
            m_design = heat_mw * 1e6 / (4180.0 * 30.0)
            bus_demand_kgs[supply_id] = bus_demand_kgs.get(supply_id, 0.0) + m_design

        children_of: dict = {}
        for p, c in bfs_edges:
            if p in keep_nodes and c in keep_nodes:
                children_of.setdefault(p, []).append(c)

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

    if node_based_heat_loads and node_heat_gen_share > 0:
        for node in power_net_as_st.nodes:
            if node.id not in bus_index_to_supply_junction:
                continue
            p_gen_mw = _node_power_gen_mw(power_net, node)
            if p_gen_mw <= 0:
                continue
            heat_mw = max(min_gen_mw, p_gen_mw * node_heat_gen_share)
            m_downstream = subtree.get(node.id, 0.0)
            q_cap_mw = 4180.0 * m_downstream * 30.0 / 1e6
            if q_cap_mw > 0:
                heat_mw = min(heat_mw, q_cap_mw)
            elif q_cap_mw == 0:
                # No downstream consumers - no heat can be transported.
                continue
            mx.create_heat_generator(
                target_net,
                node_id=bus_index_to_supply_junction[node.id],
                q_mw=round(heat_mw, 6),
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
        if return_diameter_m is not None:
            closing_diameter_m = return_diameter_m
        elif auto_diameter:
            # The closing pipe returns the whole network's flow to the slack.
            closing_diameter_m = _auto_diameter(subtree.get(slack_root, 0.0))
        else:
            closing_diameter_m = default_diameter_m * 1.5
        mx.create_water_pipe(
            target_net,
            from_node_id=return_junction,
            to_node_id=slack_supply_junction,
            diameter_m=closing_diameter_m,
            length_m=return_length_m
            if return_length_m is not None
            else default_length * length_scale,
            temperature_ext_k=296.15,
            roughness_m=0.001,
            grid=heat_grid,
        )
        if not node_based_heat_loads and heat_plant_mode == "screening" and load_specs:
            sink_capacity = total_load_q * 1e6 / (4180.0 * 30.0) * 5.0
            mx.create_water_sink(
                target_net,
                node_id=return_junction,
                mass_flow_kgs=round(sink_capacity, 6),
                name="Grid Connection Heat Return Sink",
            )

    if node_based_heat_loads:
        SAFETY = 5.0
        for (p, c), pipe_id in supply_pipe_for_edge.items():
            m_design = subtree.get(c, 0.0)
            if m_design > 0:
                pipe = target_net.branch_by_id(pipe_id)
                pipe.model.m_U_design = m_design * SAFETY
                pipe.model.mass_flow_nominal_kgs = m_design

    if auto_diameter:
        # Widen each supply pipe to carry its cumulative downstream demand at
        # the design velocity_mps; trunk pipes near the slack end up widest.
        for (p, c), pipe_id in supply_pipe_for_edge.items():
            pipe = target_net.branch_by_id(pipe_id)
            pipe.model.diameter_m = _auto_diameter(subtree.get(c, 0.0))

    return bus_index_to_supply_junction, return_junction


def _drain_proportionally(items, get_mag, set_mag, remove, total) -> float:
    """Drain ``total`` from *items* by scaling every item with one common
    factor ``(pool − absorbed) / pool``.

    Proportional draining preserves the spatial distribution of the
    remaining capacity — which generators carry the replaced output is then
    independent of child insertion order, so CP-replacement variants differ
    from their no-CP baseline only by the CP routing, not by an arbitrary
    set of deleted generators. Items scaled to ~0 are removed entirely.

    Returns the unabsorbed remainder (> 0 iff ``total`` exceeds the pool).
    """
    remaining = float(total)
    if remaining <= 0:
        return 0.0
    items = [i for i in items if get_mag(i) > 0]
    pool = sum(get_mag(i) for i in items)
    if pool <= 0:
        return remaining
    absorb = min(pool, remaining)
    factor = (pool - absorb) / pool
    for item in items:
        new_mag = get_mag(item) * factor
        if new_mag <= 1e-12:
            remove(item)
        else:
            set_mag(item, new_mag)
    return remaining - absorb


def _drain_power_gen_capacity(net: mm.Network, total_mw: float) -> float:
    """Drain ``total_mw`` proportionally from PowerGenerator children
    (skips ExtPowerGrid - slack). Returns the unabsorbed remainder."""
    return _drain_proportionally(
        [c for c in net.childs if isinstance(c.model, mm.PowerGenerator)],
        get_mag=lambda c: abs(float(c.model.p_mw)),
        set_mag=lambda c, v: setattr(c.model, "p_mw", -v),
        remove=lambda c: net.remove_child(c.id),
        total=total_mw,
    )


def _drain_gas_source_capacity(net: mm.Network, total_kgs: float) -> float:
    """Drain ``total_kgs`` proportionally from gas-side Sources only.
    Returns the unabsorbed remainder."""
    return _drain_proportionally(
        [
            c
            for c in net.childs
            if isinstance(c.model, mm.Source) and isinstance(c.grid, mm.GasGrid)
        ],
        get_mag=lambda c: abs(float(c.model.mass_flow_kgs)),
        set_mag=lambda c, v: setattr(c.model, "mass_flow_kgs", -v),
        remove=lambda c: net.remove_child(c.id),
        total=total_kgs,
    )


def _drain_heat_gen_capacity(net: mm.Network, total_mw: float) -> float:
    """Drain ``total_mw`` proportionally from HeatGenerator children, then
    (only if their pool is exhausted) proportionally from HX-Gen branches.
    Returns the unabsorbed remainder; no slack fallback by design."""
    remaining = _drain_proportionally(
        [c for c in net.childs if isinstance(c.model, mm.HeatGenerator)],
        get_mag=lambda c: abs(float(c.model.q_mw_heat)),
        set_mag=lambda c, v: setattr(c.model, "q_mw_heat", -v),
        remove=lambda c: net.remove_child(c.id),
        total=total_mw,
    )
    if remaining <= 1e-12:
        return remaining

    return _drain_proportionally(
        [
            b
            for b in net.branches
            if isinstance(b.model, mm.HeatExchanger)
            and float(getattr(b.model, "q_mw_set", 0.0) or 0.0) > 0
        ],
        get_mag=lambda b: float(getattr(b.model, "q_mw_set", 0.0) or 0.0),
        set_mag=lambda b, v: setattr(b.model, "q_mw_set", v),
        remove=lambda b: net.remove_branch_between(
            b.from_node_id, b.to_node_id, b.id[2]
        ),
        total=remaining,
    )


def create_coupling_points_for_mes(  # NOSONAR
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
    """Add CHP / P2G / P2H coupling points to an MES network.

    ``density`` ∈ [0,1] is per-node Bernoulli in decentralised mode and
    ``ceil(density·N)`` units on one hub in centralised mode. Capacities scale
    from each bus's local p_ref via the per-type ``*_p_share`` and a global
    ``cp_size_multiplier``.

    ``replace_primary_generation=True`` (default False is additive) drains the
    matching primary fleet to keep total rated production per carrier invariant.
    In node-based heat mode, heat-side replacement drains :class:`HeatGenerator`
    children (no slack fallback).

    Returns ``list[{"type", "node", "id"}]``.
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
    if not candidate_node_ids:
        raise ValueError("no nodes are coupled to both gas and heat supply")

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

    # HHV [MJ/kg] of the gas fluid actually registered on the net, so coupling-
    # point sizing uses the same heating value as the gas physics regardless of
    # which gas type the grid was built with.
    gas_hhv_mj = (
        mes_net.node_by_id(
            next(iter(bus_to_gas_junc.values()))
        ).grid.higher_heating_value_kwh_per_kg
        * 3.6
    )

    cp_power_out_mw = 0.0
    cp_gas_out_kgs = 0.0
    cp_heat_out_mw = 0.0
    for power_node_id in target_nodes:
        gas_junc = bus_to_gas_junc[power_node_id]
        heat_supply_junc = bus_to_heat_supply_junc[power_node_id]

        node = mes_net.node_by_id(power_node_id)
        p_ref_mw = _node_power_gen_mw(mes_net, node) or _node_power_load_mw(
            mes_net, node
        )
        if p_ref_mw <= 0:
            continue

        unit_type = rng.choice(sorted(coupling_set))

        if unit_type == "chp":
            chp_p_target_mw = chp_p_share * cp_size_multiplier * p_ref_mw
            mass_flow_kgs = round(
                chp_p_target_mw / max(chp_efficiency_power, 1e-3) / gas_hhv_mj,
                6,
            )
            cp_power_out_mw += mass_flow_kgs * gas_hhv_mj * chp_efficiency_power
            cp_heat_out_mw += mass_flow_kgs * gas_hhv_mj * chp_efficiency_heat
            if use_hg_variants:
                uid = mx.create_chp_hg(
                    mes_net,
                    power_node_id=power_node_id,
                    heat_node_id=heat_supply_junc,
                    gas_node_id=gas_junc,
                    mass_flow_setpoint_kgs=mass_flow_kgs,
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
                    mass_flow_setpoint_kgs=mass_flow_kgs,
                    diameter_m=chp_diameter_m,
                    efficiency_power=chp_efficiency_power,
                    efficiency_heat=chp_efficiency_heat,
                    regulation=regulation,
                )
            created.append({"type": "chp", "node": power_node_id, "id": uid})

        elif unit_type == "p2g":
            p2g_p_in_mw = p2g_p_share * cp_size_multiplier * p_ref_mw
            mass_flow_kgs = round(p2g_efficiency * p2g_p_in_mw / gas_hhv_mj, 6)
            cp_gas_out_kgs += mass_flow_kgs
            bid = mx.create_p2g(
                mes_net,
                from_node_id=power_node_id,
                to_node_id=gas_junc,
                efficiency=p2g_efficiency,
                mass_flow_setpoint_kgs=mass_flow_kgs,
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
        # Drain primary generation so total rated capacity per carrier stays invariant.
        # Measure the gas primary pool BEFORE draining: a gas distribution grid may
        # carry no distributed primary gas generation (gas_gen_share=0), in which
        # case there is nothing for P2G to displace and its gas output is additive
        # green gas, not a replacement — so an empty pool is NOT an invariance
        # violation and the gas check is skipped (electricity/heat still enforced).
        gas_primary_pool_kgs = sum(
            abs(float(getattr(c.model, "mass_flow_kgs", 0.0) or 0.0))
            for c in mes_net.childs
            if isinstance(c.model, mm.Source) and isinstance(c.grid, mm.GasGrid)
        )
        unabsorbed_p = _drain_power_gen_capacity(mes_net, cp_power_out_mw)
        unabsorbed_g = _drain_gas_source_capacity(mes_net, cp_gas_out_kgs)
        unabsorbed_h = _drain_heat_gen_capacity(mes_net, cp_heat_out_mw)
        checks = [
            ("PowerGenerator", cp_power_out_mw, unabsorbed_p),
            ("heat (HeatGenerator + HX-Gen)", cp_heat_out_mw, unabsorbed_h),
        ]
        if gas_primary_pool_kgs > 1e-12:
            checks.append(("gas Source", cp_gas_out_kgs, unabsorbed_g))
        for label, asked, left in checks:
            if left > 1e-9 and asked > 0:
                # Hard error by design: an unabsorbed remainder means the CP
                # output exceeds the primary pool, so total rated capacity is
                # NOT invariant any more and every density-sweep comparison
                # built on that invariance is silently biased.
                raise ValueError(
                    f"replace_primary_generation: {label} pool absorbed "
                    f"{asked - left:g} of {asked:g} requested; {left:g} "
                    f"unabsorbed. CP output exceeds the primary pool — "
                    f"capacity invariance would break. Reduce CP density / "
                    f"size (or raise the primary generation share)."
                )

    return created


def generate_supply_return_mes_based_on_power_net(
    net_power: mm.Network,
    *,
    coupling_density=0.2,
    centralized=False,
    central_node_id=None,
    couplings=("chp", "p2g", "p2h"),
    backbone_method="span",
    backbone_weight=None,
    heat_kwargs=None,
    gas_kwargs=None,
    coupling_kwargs=None,
):
    """Wrap :func:`create_gas_tree_net_for_power` + :func:`create_heat_supply_return_net_for_power`
    + :func:`create_coupling_points_for_mes` into one call. ``gas_kwargs`` /
    ``heat_kwargs`` / ``coupling_kwargs`` forward to those builders.

    ``backbone_method`` (``"span"`` default, or ``"steiner"``) and
    ``backbone_weight`` (``None`` for unit weights, ``"length"``, or a callable)
    set the shared gas/heat layout skeleton; both are forwarded to the layer
    builders unless already overridden in ``gas_kwargs`` / ``heat_kwargs``."""
    gas_kwargs = dict(gas_kwargs or {})
    heat_kwargs = dict(heat_kwargs or {})
    for kw in (gas_kwargs, heat_kwargs):
        kw.setdefault("backbone_method", backbone_method)
        kw.setdefault("backbone_weight", backbone_weight)
    new_mes_net = net_power.copy()
    bus_to_gas_junc = create_gas_tree_net_for_power(
        net_power, new_mes_net, **gas_kwargs
    )
    bus_to_heat_supply, heat_return = create_heat_supply_return_net_for_power(
        net_power, new_mes_net, **heat_kwargs
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
