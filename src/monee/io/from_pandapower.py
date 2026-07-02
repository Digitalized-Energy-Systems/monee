import copy
import json
import math
import os
import tempfile
import uuid
import warnings

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


def _coerce_positive_float(value, default=1.0, allow_zero=False):
    """None/NaN/negative -> *default*; 0.0 -> default unless *allow_zero*."""
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv < 0 or (fv == 0 and not allow_zero):  # NOSONAR
        return default
    return fv


def _pp_branch_max_i_ka_overrides(net, bus_to_node, monee_node_ids):
    overrides = {}
    seen = {}

    if not hasattr(net, "bus"):
        return overrides

    if "in_service" in net.bus.columns:
        live_buses = {int(b) for b in net.bus.index[net.bus["in_service"].astype(bool)]}
    else:
        live_buses = {int(b) for b in net.bus.index}

    def resolve_pair(from_bus, to_bus):
        # to_mpc renumbers out-of-service buses with a stale lookup, so their
        # mapped node id is unreliable (it can even collide with another bus).
        if int(from_bus) not in live_buses or int(to_bus) not in live_buses:
            return None
        f = bus_to_node.get(int(from_bus))
        t = bus_to_node.get(int(to_bus))
        if f not in monee_node_ids or t not in monee_node_ids:
            return None
        return f, t

    def take_parallel_key(f, t):
        # The monee branch key is the nx.MultiGraph edge key, which counts
        # parallels per *undirected* node pair.
        pair = (f, t) if f <= t else (t, f)
        key = seen.get(pair, 0)
        seen[pair] = key + 1
        return key

    skipped_lines = []
    if hasattr(net, "line") and len(net.line):
        for row in net.line.itertuples():
            # to_mpc drops out-of-service lines, so they consume no branch slot.
            if not getattr(row, "in_service", True):
                continue
            pair = resolve_pair(row.from_bus, row.to_bus)
            if pair is None:
                skipped_lines.append(row.Index)
                continue
            f, t = pair
            key = take_parallel_key(f, t)
            max_i_ka = float(row.max_i_ka)
            parallel = _coerce_positive_int(getattr(row, "parallel", 1))
            df = _coerce_positive_float(getattr(row, "df", 1.0))
            overrides[(f, t, key)] = max_i_ka * parallel * df

    if hasattr(net, "trafo") and len(net.trafo):
        for row in net.trafo.itertuples():
            if not getattr(row, "in_service", True):
                continue
            pair = resolve_pair(row.hv_bus, row.lv_bus)
            if pair is not None:
                take_parallel_key(*pair)

    if skipped_lines:
        warnings.warn(
            "from_pandapower: could not attach max_i_ka rating of line(s) "
            f"{skipped_lines}: an endpoint bus is out of service or was not "
            "mapped by to_mpc; the affected branches keep the MATPOWER-derived "
            "rating.",
            stacklevel=3,
        )

    return overrides


def _extract_sgens(net):
    # Always work on a copy: to_mpc mutates the net it converts.
    net = copy.deepcopy(net)
    if not hasattr(net, "sgen") or not len(net.sgen):
        return net, []
    sgens = []
    for row in net.sgen.itertuples():
        if not getattr(row, "in_service", True):
            continue

        # scaling=0 is a legitimate switched-off sgen, not missing data.
        scaling = _coerce_positive_float(getattr(row, "scaling", 1.0), allow_zero=True)
        name = getattr(row, "name", None)
        sgens.append(
            (
                int(row.bus),
                float(row.p_mw) * scaling,
                float(row.q_mvar) * scaling,
                None if name is None else str(name),
            )
        )
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

        if p_mw < 0:
            # sgen convention: positive p_mw = injection. A negative-p sgen is
            # a consumer; PowerLoad stores consumption as positive.
            model = PowerLoad(p_mw=-p_mw, q_mvar=-q_mvar)
        else:
            model = PowerGenerator(p_mw=p_mw, q_mvar=q_mvar)
        monee_net.child_to(model, node_id=node.id, name=name)


def _strip_transformer_vector_group_shifts(monee_net):
    # Vector-group shifts (multiples of 30 deg) collapse the CasADi flat start,
    # and monee has no per-unit rebasing use for them; genuine phase-shifter
    # angles (non-multiples of 30 deg) are kept.
    stripped = []
    for branch in monee_net.branches:
        shift = getattr(branch.model, "shift", None)
        if shift is None:
            continue
        try:
            shift_val = float(shift)
        except (TypeError, ValueError):
            continue
        if abs(shift_val) <= 1e-9:
            continue
        shift_deg = math.degrees(shift_val)
        remainder = abs(shift_deg) % 30.0
        if min(remainder, 30.0 - remainder) > 1e-6:
            continue
        branch.model.shift = 0.0
        stripped.append((branch.id, round(shift_deg, 3)))
    if stripped:
        warnings.warn(
            "from_pandapower: stripped transformer vector-group phase shifts "
            "(multiples of 30 deg) to keep the flat start solvable: "
            + ", ".join(f"branch {bid} ({deg} deg)" for bid, deg in stripped),
            stacklevel=3,
        )


def _bus_position(net, pp_id):
    if "geo" in net.bus.columns:
        geo = net.bus["geo"].iloc[pp_id]
        if not isinstance(geo, str):  # None / NaN -> no coordinates
            return None
        try:
            coords = json.loads(geo)["coordinates"]
            return (coords[0], coords[1])
        except (ValueError, KeyError, TypeError, IndexError):
            return None
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

    # pandapower bus -> monee node id, accounting for bus fusion in to_mpc.
    bus_to_node = _pp_bus_to_node_id(net)

    nodes_by_id = {n.id: n for n in monee_net.nodes}
    named_node_ids = set()
    for raw_bus in net.bus.index:
        pp_bus = int(raw_bus)
        node = nodes_by_id.get(bus_to_node.get(pp_bus))
        # Fused buses share one node; the first pandapower bus wins.
        if node is None or node.id in named_node_ids:
            continue
        named_node_ids.add(node.id)
        pp_pos = int(net.bus.index.get_loc(raw_bus))
        node.name = net.bus["name"].iloc[pp_pos]
        position = _bus_position(net, pp_pos)
        if position is not None:
            node.position = position

    overrides = _pp_branch_max_i_ka_overrides(net, bus_to_node, set(nodes_by_id))
    if overrides:
        for branch in monee_net.branches:
            if not hasattr(branch.model, "max_i_ka"):
                continue
            bid = (branch.from_node_id, branch.to_node_id, branch.id[2])
            if bid in overrides:
                branch.model.max_i_ka = overrides[bid]

    _strip_transformer_vector_group_shifts(monee_net)

    if hasattr(net, "load") and len(net.load):
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
