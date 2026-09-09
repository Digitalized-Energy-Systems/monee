import copy
import json
import math
import os
import tempfile
import uuid
import warnings

from pandapower.converter.matpower import to_mpc

from monee.model.child import PowerGenerator, PowerLoad, VoltageControlledGenerator

from .matpower import read_matpower_case, read_matpower_opf_case


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


def _clean_pp_name(value):
    if value is None or value != value:  # NOSONAR NaN
        return None
    return str(value)


def _pp_branch_max_i_ka_overrides(net, bus_to_node, monee_node_ids):
    overrides = {}
    kinds = {}
    meta = {}
    seen = {}

    if not hasattr(net, "bus"):
        return overrides, kinds, meta

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
            kinds[(f, t, key)] = "line"
            meta[(f, t, key)] = (
                "line",
                row.Index,
                _clean_pp_name(getattr(row, "name", None)),
            )

    if hasattr(net, "trafo") and len(net.trafo):
        for row in net.trafo.itertuples():
            if not getattr(row, "in_service", True):
                continue
            pair = resolve_pair(row.hv_bus, row.lv_bus)
            if pair is not None:
                f, t = pair
                key = take_parallel_key(f, t)
                kinds[(f, t, key)] = "trafo"
                meta[(f, t, key)] = (
                    "trafo",
                    row.Index,
                    _clean_pp_name(getattr(row, "name", None)),
                )

    # ppc branch order is line, trafo, trafo3w, impedance; trafo3w arms end on
    # a unique star bus and never collide with these pairs, so counting
    # impedances here keeps the parallel keys aligned with to_mpc.
    if hasattr(net, "impedance") and len(net.impedance):
        for row in net.impedance.itertuples():
            if not getattr(row, "in_service", True):
                continue
            pair = resolve_pair(row.from_bus, row.to_bus)
            if pair is not None:
                f, t = pair
                key = take_parallel_key(f, t)
                kinds[(f, t, key)] = "impedance"
                meta[(f, t, key)] = (
                    "impedance",
                    row.Index,
                    _clean_pp_name(getattr(row, "name", None)),
                )

    if skipped_lines:
        warnings.warn(
            "from_pandapower: could not attach max_i_ka rating of line(s) "
            f"{skipped_lines}: an endpoint bus is out of service or was not "
            "mapped by to_mpc; the affected branches keep the MATPOWER-derived "
            "rating.",
            stacklevel=3,
        )

    return overrides, kinds, meta


def _trafo3w_terminal_node_ids(net, bus_to_node):
    terminals = set()
    if hasattr(net, "trafo3w") and len(net.trafo3w):
        for row in net.trafo3w.itertuples():
            if not getattr(row, "in_service", True):
                continue
            for bus in (row.hv_bus, row.mv_bus, row.lv_bus):
                node = bus_to_node.get(int(bus))
                if node is not None:
                    terminals.add(node)
    return terminals


def _apply_branch_kinds(monee_net, kinds, real_node_ids, trafo3w_terminals):
    for branch in monee_net.branches:
        if not hasattr(branch.model, "max_i_ka"):
            continue
        kind = kinds.get((branch.from_node_id, branch.to_node_id, branch.id[2]))
        if kind is None:
            from_real = branch.from_node_id in real_node_ids
            to_real = branch.to_node_id in real_node_ids
            if from_real != to_real:
                real_end = branch.from_node_id if from_real else branch.to_node_id
                kind = "trafo3w" if real_end in trafo3w_terminals else "switch-stub"
            else:
                kind = "unknown"
        branch.model.kind = kind


def _apply_max_i_ka_overrides(monee_net, overrides, real_node_ids):
    """Write the recovered ratings and return ``{requested bid: branch}``."""
    resolved = {}
    if not overrides:
        return resolved
    rated = {
        (branch.from_node_id, branch.to_node_id, branch.id[2]): branch
        for branch in monee_net.branches
        if hasattr(branch.model, "max_i_ka")
    }
    unmatched = {}
    for bid, max_i_ka in overrides.items():
        branch = rated.get(bid)
        if branch is None:
            unmatched[bid] = max_i_ka
        else:
            branch.model.max_i_ka = max_i_ka
            resolved[bid] = branch

    if not unmatched:
        return resolved

    # to_mpc materialises an open line switch as an auxiliary bus, so such a
    # line ends on a stub node that the (from_bus, to_bus) key cannot name.
    stubs = [
        branch
        for bid, branch in rated.items()
        if (bid[0] in real_node_ids) != (bid[1] in real_node_ids)
    ]
    still_unmatched = []
    for (from_node, to_node, key), max_i_ka in unmatched.items():
        candidates = [
            branch
            for branch in stubs
            if branch.from_node_id in (from_node, to_node)
            or branch.to_node_id in (from_node, to_node)
        ]
        if len(candidates) == 1:
            candidates[0].model.max_i_ka = max_i_ka
            resolved[(from_node, to_node, key)] = candidates[0]
            stubs.remove(candidates[0])
        else:
            still_unmatched.append((from_node, to_node))

    if still_unmatched:
        warnings.warn(
            "from_pandapower: no imported branch matches the node pair(s) "
            f"{still_unmatched}; the corresponding pandapower max_i_ka rating "
            "was dropped and those branches keep the MATPOWER-derived rating.",
            stacklevel=3,
        )
    return resolved


def _apply_branch_identity(monee_net, meta, resolved):
    branch_by_bid = {
        (branch.from_node_id, branch.to_node_id, branch.id[2]): branch
        for branch in monee_net.branches
    }
    line_to_branch = {}
    trafo_to_branch = {}
    for bid, (table, pp_index, name) in meta.items():
        branch = branch_by_bid.get(bid)
        if branch is None:
            branch = resolved.get(bid)
        if branch is None:
            continue
        if name is not None and branch.name is None:
            branch.name = name
        if table == "line":
            line_to_branch[pp_index] = branch.id
        elif table == "trafo":
            trafo_to_branch[pp_index] = branch.id
    return line_to_branch, trafo_to_branch


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


def _name_generators(monee_net, net, bus_to_node, nodes_by_id):
    if not hasattr(net, "gen") or not len(net.gen) or "name" not in net.gen.columns:
        return
    names_by_node = {}
    for row in net.gen.itertuples():
        if not getattr(row, "in_service", True):
            continue
        node = nodes_by_id.get(bus_to_node.get(int(row.bus)))
        if node is None:
            continue
        names_by_node.setdefault(node.id, []).append(getattr(row, "name", None))

    for node_id, names in names_by_node.items():
        children = [
            child
            for child in monee_net.childs_by_ids(nodes_by_id[node_id].child_ids)
            if child.name is None
            and isinstance(child.model, (PowerGenerator, VoltageControlledGenerator))
        ]
        # A gen at the slack bus becomes an ExtPowerGrid, so a count mismatch
        # means the row-to-child order is not trustworthy.
        if len(children) != len(names):
            continue
        for child, name in zip(children, names):
            if name is not None and name == name:  # NOSONAR NaN
                child.name = str(name)


def strip_transformer_vector_group_shifts(monee_net):
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


_strip_transformer_vector_group_shifts = strip_transformer_vector_group_shifts


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


def _has_costs(net) -> bool:
    return any(len(getattr(net, table, ())) for table in ("poly_cost", "pwl_cost"))


def from_pandapower_net(net, opf=False, max_loading=1.0, limit_basis="mva"):  # NOSONAR
    """Convert a pandapower net into a monee :class:`~monee.model.network.Network`.

    The conversion goes through pandapower's MATPOWER export, so the monee node
    id is the position of the bus in the exported case, not the pandapower bus
    index; the returned network carries the real lookups as ``pp_bus_to_node``
    (pandapower bus index to monee node id), ``pp_line_to_branch`` and
    ``pp_trafo_to_branch`` (pandapower line/trafo index to monee branch id,
    including lines split at an open switch's auxiliary bus). Line, trafo and
    impedance names are copied onto the created branches, so named branches
    show a ``name`` column in the branch result frame. Every imported
    electrical branch model additionally gets a ``kind`` attribute naming its
    pandapower origin (``"line"``, ``"trafo"``, ``"trafo3w"``, ``"impedance"``,
    ``"switch-stub"`` for a line split at an open switch's auxiliary bus, or
    ``"unknown"``); it shows up as a column in the branch result frame.

    With ``opf=False`` (the default) generators keep their pandapower setpoint
    (a gen at a PV bus becomes a
    :class:`~monee.model.child.VoltageControlledGenerator`) and any
    ``net.poly_cost`` / ``net.pwl_cost`` is dropped, which is warned about.

    With ``opf=True`` the case is read through
    :func:`~monee.io.matpower.read_matpower_opf_case` and ``(network, problem)``
    is returned: every in-service generator becomes dispatchable within
    PMIN/PMAX/QMIN/QMAX and carries its polynomial cost, the ext_grid cost maps
    onto the slack, and *max_loading* / *limit_basis* cap the branch loading.
    Static generators are stripped before the MATPOWER export and re-attached
    afterwards, so they stay non-dispatchable fixed injections. The slack bus
    voltage is optimised within [VMIN, VMAX], so results track pandapower
    ``runopp`` closely but are not bit-identical to it.
    """

    if opf and not _has_costs(net):
        raise ValueError(
            "from_pandapower_net(opf=True) needs generator costs: the net has "
            "neither 'poly_cost' nor 'pwl_cost' entries, so the MATPOWER export "
            "carries no objective. Add costs with pandapower's create_poly_cost, "
            "or convert with opf=False for a plain power flow."
        )
    if not opf and _has_costs(net):
        warnings.warn(
            "from_pandapower: the net carries poly_cost/pwl_cost entries and "
            "generator limits that a power-flow import drops. Use "
            "from_pandapower_net(net, opf=True) to keep them and get an "
            "optimisation problem alongside the network.",
            stacklevel=2,
        )

    net, sgens = _extract_sgens(net)

    problem = None
    with tempfile.TemporaryDirectory() as tmp_dir:
        name_file = os.path.join(tmp_dir, f"{uuid.uuid4()}.mat")
        to_mpc(net, init="flat", filename=name_file)
        if opf:
            monee_net, problem = read_matpower_opf_case(
                name_file, max_loading=max_loading, limit_basis=limit_basis
            )
        else:
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

    overrides, kinds, branch_meta = _pp_branch_max_i_ka_overrides(
        net, bus_to_node, set(nodes_by_id)
    )
    real_node_ids = set(bus_to_node.values())
    resolved = _apply_max_i_ka_overrides(monee_net, overrides, real_node_ids)
    _apply_branch_kinds(
        monee_net, kinds, real_node_ids, _trafo3w_terminal_node_ids(net, bus_to_node)
    )
    line_to_branch, trafo_to_branch = _apply_branch_identity(
        monee_net, branch_meta, resolved
    )

    strip_transformer_vector_group_shifts(monee_net)

    if hasattr(net, "load") and len(net.load):
        for pp_bus, group in net.load.groupby("bus", sort=False):
            monee_node = nodes_by_id.get(bus_to_node.get(int(pp_bus)))

            if monee_node is None:
                continue

            agg_name = aggregated_pp_load_name(group)

            for child in monee_net.childs_by_ids(monee_node.child_ids):
                if isinstance(child.model, PowerLoad):
                    child.name = agg_name

    _name_generators(monee_net, net, bus_to_node, nodes_by_id)

    _attach_sgens(monee_net, sgens, bus_to_node)

    monee_net.pp_bus_to_node = dict(bus_to_node)
    monee_net.pp_line_to_branch = line_to_branch
    monee_net.pp_trafo_to_branch = trafo_to_branch

    if opf:
        return monee_net, problem
    return monee_net
