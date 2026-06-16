import copy
import json
import math
import os
import tempfile
import uuid

from pandapower.converter.matpower import to_mpc

from monee.model.child import PowerGenerator, PowerLoad

from .matpower import read_matpower_case

_SQRT3 = math.sqrt(3.0)


def aggregated_pp_load_name(pp_loads_at_bus) -> str:
    """``"+"``-joined names of pandapower loads aggregated onto one monee PowerLoad."""
    return "+".join(pp_loads_at_bus["name"].astype(str).tolist())


def _coerce_positive_int(value, default=1):
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv <= 0:  # NaN or non-positive
        return default
    return max(1, int(fv))


def _coerce_positive_float(value, default=1.0):
    if value is None:
        return default
    try:
        fv = float(value)
    except (TypeError, ValueError):
        return default
    if fv != fv or fv <= 0:
        return default
    return fv


def _pp_branch_max_i_ka_overrides(net):
    """``{monee_branch_id → max_i_ka}`` rebuilt from ``net.line``/``net.trafo``.

    The matpower roundtrip drops current limits and our writer pins a 0.319 kA
    placeholder, so we re-derive line ratings as ``max_i_ka·parallel·df`` here.
    Trafos are *not* overridden: monee's single ``max_i_ka`` cannot represent
    both HV and LV sides of a Y-Δ trafo simultaneously, so any tight bound
    would make one side infeasible. Switch-aux branches stay on the placeholder.
    """
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

    # Trafos: bump the parallel-key counter so later lines on the same pair
    # get the right key, but don't override (see docstring).
    if hasattr(net, "trafo") and len(net.trafo):
        for row in net.trafo.itertuples():
            f = mpc_bus(row.hv_bus)
            t = mpc_bus(row.lv_bus)
            seen[(f, t)] = seen.get((f, t), 0) + 1

    return overrides


def _extract_sgens(net):
    """Pull static generators out of *net* so they import as distinct PowerGenerators.

    pandapower's ``to_mpc`` folds non-controllable sgens into each bus's net
    ``(Pd, Qd)`` injection rather than emitting them as ``gen`` rows. A bus that
    carries both a (reactive-consuming) load and an (active-injecting) sgen then
    collapses to a single child typed only by the net *active* sign - yielding a
    PowerGenerator that wrongly carries the load's positive reactive power.

    Removing the sgens here keeps the bus injection load-only; the caller re-adds
    each sgen as its own PowerGenerator via :func:`_attach_sgens`. The caller's
    net is never mutated - a copy is returned.

    Returns ``(net_without_sgens, sgens)`` where each sgen is a
    ``(bus, p_mw, q_mvar, name)`` tuple with ``scaling`` already applied (matching
    to_mpc's snapshot) and out-of-service rows dropped (to_mpc ignores them too).
    """
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
    """Map each pandapower bus index to its monee node id.

    monee node ids equal the matpower bus numbers, which are pandapower's
    internal ppc bus indices (0-based) + 1. ``to_mpc`` fuses buses joined by
    closed bus-bus switches (e.g. simbench HV grids collapse 306 buses to 64),
    so several pandapower buses can share one node. The authoritative mapping is
    pandapower's own ``_pd2ppc_lookups['bus']``, populated by the preceding
    ``to_mpc`` call; positional order (``get_loc + 1``) is only correct when no
    fusion happens, so it is used solely as a fallback.
    """
    lookups = getattr(net, "_pd2ppc_lookups", None)
    bus_lookup = None if lookups is None else lookups.get("bus")
    mapping = {}
    for bus in net.bus.index:
        bus = int(bus)
        if bus_lookup is not None and 0 <= bus < len(bus_lookup):
            ppc_idx = int(bus_lookup[bus])
            if ppc_idx >= 0:
                mapping[bus] = ppc_idx + 1
                continue
        mapping[bus] = int(net.bus.index.get_loc(bus)) + 1
    return mapping


def _attach_sgens(monee_net, sgens, bus_to_node):
    """Add each extracted sgen as its own PowerGenerator on its bus's monee node.

    PowerGenerator takes positive magnitudes and stores them in load convention
    (negative = injection); pandapower sgen power is already a positive
    generation magnitude (positive q_mvar = reactive injection), so the values
    pass straight through.
    """
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
    """Zero the phase ``shift`` on imported transformer branches.

    pandapower models a transformer's winding vector group as a phase shift (a
    multiple of 30°, e.g. Dyn5 = 150°). pandapower **3.x** applies that shift in
    its power-flow model and ``to_mpc`` exports it; pandapower **2.x** dropped it
    (exported ``shift = 0``). The two are otherwise byte-identical - same r/x/b,
    same nodal injections (verified against each version's own solved ``_ppc``;
    ``to_mpc`` is faithful in both, so this is a pandapower transformer-model
    change, not an export bug).

    For monee's single-reference PQ/slack distribution power flow a vector-group
    shift is a pure *reference rotation*: it offsets every downstream bus angle
    by the same constant and leaves all reported magnitudes (voltages, branch
    flows, |currents|) unchanged. But a lone 150° jump across one branch, with
    the rest of the grid at 0°, wrecks the conditioning of the flat-start NLP -
    the IPOPT/CasADi solve collapses onto the spurious low-voltage root (GEKKO's
    square Newton solve tolerates it; the default in-process CasADi backend does
    not). Lines never carry a shift, so zeroing every non-zero branch shift
    normalises exactly the transformer vector groups - reproducing monee's
    pre-pandapower-3 behaviour while keeping every magnitude identical to
    pandapower's own solution.
    """
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
    """``(x, y)`` for the pandapower bus at positional index ``pp_id``, or ``None``.

    pandapower 3.x stores bus coordinates as a GeoJSON ``Point`` string in
    ``net.bus['geo']`` (``{"coordinates": [x, y], "type": "Point"}``); 2.x kept
    them in a separate ``net.bus_geodata`` frame. Support both so the bridge is
    not pinned to one pandapower major.
    """
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


def from_pandapower_net(net):
    # Import sgens as distinct PowerGenerators instead of letting to_mpc fold
    # them into each bus's net injection (see _extract_sgens). Works on the
    # returned copy so the caller's net is left untouched.
    net, sgens = _extract_sgens(net)
    # Write the intermediate matpower case into a private temp dir so it is
    # always cleaned up (even if the read raises) and never collides with or
    # pollutes the caller's working directory.
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

    # Recover max_i_ka from pandapower (dropped by the matpower roundtrip).
    overrides = _pp_branch_max_i_ka_overrides(net)
    if overrides:
        for branch in monee_net.branches:
            if not hasattr(branch.model, "max_i_ka"):
                continue
            bid = (branch.from_node_id, branch.to_node_id, branch.id[2])
            if bid in overrides:
                branch.model.max_i_ka = overrides[bid]

    # Normalise transformer vector-group phase shifts (pandapower 3.x exports
    # them where 2.x didn't); a lone 150° shift across one branch otherwise
    # collapses the flat-start AC NLP onto the spurious low-voltage root.
    _strip_transformer_vector_group_shifts(monee_net)

    # pandapower bus -> monee node id, accounting for bus fusion in to_mpc.
    bus_to_node = _pp_bus_to_node_id(net)

    # Tag aggregated PowerLoad with a deterministic name so simbench
    # timeseries can be matched back by name.
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
