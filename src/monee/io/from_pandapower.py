import math
import os
import uuid

import pandapower.converter as pc

from monee.model.child import PowerLoad

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


def from_pandapower_net(net):
    id_file = uuid.uuid4()
    name_file = f"{id_file}.mat"
    pc.to_mpc(net, init="flat", filename=name_file)
    monee_net = read_matpower_case(name_file)
    os.remove(name_file)
    for node in monee_net.nodes:
        pp_id = node.id - 1
        if len(net.bus) > pp_id:
            node.name = net.bus["name"].iloc[pp_id]
            if hasattr(net, "bus_geodata"):
                node.position = (
                    net.bus_geodata["x"].iloc[pp_id],
                    net.bus_geodata["y"].iloc[pp_id],
                )

    # Recover max_i_ka from pandapower (dropped by the matpower roundtrip).
    overrides = _pp_branch_max_i_ka_overrides(net)
    if overrides:
        for branch in monee_net.branches:
            if not hasattr(branch.model, "max_i_ka"):
                continue
            bid = (branch.from_node_id, branch.to_node_id, branch.id[2])
            if bid in overrides:
                branch.model.max_i_ka = overrides[bid]

    # Tag aggregated PowerLoad with a deterministic name so simbench
    # timeseries can be matched back by name.
    if hasattr(net, "load") and len(net.load):
        nodes_by_id = {n.id: n for n in monee_net.nodes}
        for pp_bus, group in net.load.groupby("bus", sort=False):
            monee_node = nodes_by_id.get(int(pp_bus) + 1)
            if monee_node is None:
                continue
            agg_name = aggregated_pp_load_name(group)
            for child in monee_net.childs_by_ids(monee_node.child_ids):
                if isinstance(child.model, PowerLoad):
                    child.name = agg_name

    return monee_net
