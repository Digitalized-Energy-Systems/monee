import copy
import json
import os
import tempfile
import uuid

from pandapower.converter.matpower import to_mpc

from monee.model.child import PowerGenerator, PowerLoad

from .matpower import read_matpower_case


def aggregated_pp_load_name(pp_loads_at_bus) -> str:
    return "+".join(pp_loads_at_bus["name"].astype(str).tolist())


def _coerce_positive_int(value, default=1):
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv <= 0:  # NOSONAR NaN or non-positive
        return default
    return max(1, int(fv))


def _coerce_positive_float(value, default=1.0):
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv <= 0:  # NOSONAR
        return default
    return fv


def _pp_branch_max_i_ka_overrides(net):
    overrides = {}
    seen = {}

    bus_index = net.bus.index if hasattr(net, "bus") else None

    if bus_index is None:
        return overrides

    def mpc_bus(pp_bus):
        return int(bus_index.get_loc(int(pp_bus))) + 1

    if hasattr(net, "line") and len(net.line):
        for row in net.line.itertuples():
            f = mpc_bus(row.from_bus)
            t = mpc_bus(row.to_bus)
            key = seen.get((f, t), 0)
            seen[(f, t)] = key + 1
            max_i_ka = float(row.max_i_ka)
            parallel = _coerce_positive_int(getattr(row, "parallel", 1))
            df = _coerce_positive_float(getattr(row, "df", 1.0))
            overrides[(f, t, key)] = max_i_ka * parallel * df

    if hasattr(net, "trafo") and len(net.trafo):
        for row in net.trafo.itertuples():
            f = mpc_bus(row.hv_bus)
            t = mpc_bus(row.lv_bus)
            seen[(f, t)] = seen.get((f, t), 0) + 1

    return overrides


def _extract_sgens(net):
    if not hasattr(net, "sgen") or not len(net.sgen):
        return net, []
    sgens = []
    for row in net.sgen.itertuples():
        if not getattr(row, "in_service", True):
            continue

        scaling = _coerce_positive_float(getattr(row, "scaling", 1.0))
        name = getattr(row, "name", None)
        sgens.append(
            (
                int(row.bus),
                float(row.p_mw) * scaling,
                float(row.q_mvar) * scaling,
                None if name is None else str(name),
            )
        )
    net = copy.deepcopy(net)
    net.sgen = net.sgen.drop(net.sgen.index)
    return net, sgens


def _pp_bus_to_node_id(net):
    lookups = getattr(net, "_pd2ppc_lookups", None)
    bus_lookup = None if lookups is None else lookups.get("bus")
    mapping = {}
    for raw_bus in net.bus.index:
        bus = int(raw_bus)

        if bus_lookup is not None and 0 <= bus < len(bus_lookup):
            ppc_idx = int(bus_lookup[bus])
            if ppc_idx >= 0:
                mapping[bus] = ppc_idx + 1
                continue
        mapping[bus] = int(net.bus.index.get_loc(bus)) + 1

    return mapping


def _attach_sgens(monee_net, sgens, bus_to_node):
    if not sgens:
        return
    nodes_by_id = {n.id: n for n in monee_net.nodes}
    for bus, p_mw, q_mvar, name in sgens:
        node = nodes_by_id.get(bus_to_node.get(bus))

        if node is None:
            continue

        monee_net.child_to(
            PowerGenerator(p_mw=p_mw, q_mvar=q_mvar),
            node_id=node.id,
            name=name,
        )


def _strip_transformer_vector_group_shifts(monee_net):
    for branch in monee_net.branches:
        shift = getattr(branch.model, "shift", None)
        if shift is None:
            continue
        try:
            shift_val = float(shift)
        except (TypeError, ValueError):
            continue
        if abs(shift_val) > 1e-9:
            branch.model.shift = 0.0


def _bus_position(net, pp_id):
    if "geo" in net.bus.columns:
        geo = net.bus["geo"].iloc[pp_id]
        if not isinstance(geo, str):  # None / NaN -> no coordinates
            return None
        try:
            coords = json.loads(geo)["coordinates"]
        except (ValueError, KeyError, TypeError):
            return None
        return (coords[0], coords[1])
    if hasattr(net, "bus_geodata") and len(net.bus_geodata) > pp_id:
        return (
            net.bus_geodata["x"].iloc[pp_id],
            net.bus_geodata["y"].iloc[pp_id],
        )
    return None


def from_pandapower_net(net):  # NOSONAR

    net, sgens = _extract_sgens(net)

    with tempfile.TemporaryDirectory() as tmp_dir:
        name_file = os.path.join(tmp_dir, f"{uuid.uuid4()}.mat")
        to_mpc(net, init="flat", filename=name_file)
        monee_net = read_matpower_case(name_file)

    for node in monee_net.nodes:
        pp_id = node.id - 1
        if len(net.bus) > pp_id:
            node.name = net.bus["name"].iloc[pp_id]
            position = _bus_position(net, pp_id)
            if position is not None:
                node.position = position

    overrides = _pp_branch_max_i_ka_overrides(net)
    if overrides:
        for branch in monee_net.branches:
            if not hasattr(branch.model, "max_i_ka"):
                continue
            bid = (branch.from_node_id, branch.to_node_id, branch.id[2])
            if bid in overrides:
                branch.model.max_i_ka = overrides[bid]

    _strip_transformer_vector_group_shifts(monee_net)

    # pandapower bus -> monee node id, accounting for bus fusion in to_mpc.
    bus_to_node = _pp_bus_to_node_id(net)

    if hasattr(net, "load") and len(net.load):
        nodes_by_id = {n.id: n for n in monee_net.nodes}
        for pp_bus, group in net.load.groupby("bus", sort=False):
            monee_node = nodes_by_id.get(bus_to_node.get(int(pp_bus)))

            if monee_node is None:
                continue

            agg_name = aggregated_pp_load_name(group)

            for child in monee_net.childs_by_ids(monee_node.child_ids):
                if isinstance(child.model, PowerLoad):
                    child.name = agg_name

    _attach_sgens(monee_net, sgens, bus_to_node)

    return monee_net
