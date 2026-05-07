import os
import uuid

import pandapower.converter as pc

from monee.model.child import PowerLoad

from .matpower import read_matpower_case


def aggregated_pp_load_name(pp_loads_at_bus) -> str:
    """Stable joined name for the pandapower loads sharing a single bus.

    The matpower converter aggregates every pandapower load on a bus into a
    single :class:`~monee.model.child.PowerLoad`.  We mirror that grouping
    by joining the contributing pandapower load names with ``"+"`` so the
    monee child has a deterministic name that
    :func:`monee.io.from_simbench.obtain_simbench_profile_by_pp_net` can use
    to bind the corresponding (summed) timeseries.
    """
    return "+".join(pp_loads_at_bus["name"].astype(str).tolist())


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

    # Tag aggregated PowerLoad children with a deterministic name derived
    # from the contributing pandapower loads, so simbench timeseries can be
    # matched back to them by name.  Buses with no load are silently
    # skipped; buses where matpower produced no PowerLoad child (zero
    # aggregate p/q) are likewise ignored.
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
