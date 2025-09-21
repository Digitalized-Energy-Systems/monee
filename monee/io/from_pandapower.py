import os
import uuid

import pandapower.converter as pc

from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad

from .matpower import read_matpower_case


def from_pandapower_net(net):
    id_file = uuid.uuid4()
    name_file = f"{id_file}.mat"
    pc.to_mpc(net, init="flat", filename=name_file)
    monee_net = read_matpower_case(name_file)
    os.remove(name_file)
    monee_net.clear_childs()

    for _, row in net.load.iterrows():
        monee_net.child_to(
            PowerLoad(row["p_mw"], row["q_mvar"]), row["bus"] + 1, name=row["name"]
        )
    for _, row in net.sgen.iterrows():
        monee_net.child_to(
            PowerGenerator(row["p_mw"], row["q_mvar"], name=row["name"]),
            row["bus"] + 1,
            name=row["name"],
        )
    for _, row in net.ext_grid.iterrows():
        monee_net.child_to(
            ExtPowerGrid(1, 1, vm_pu=row["vm_pu"], va_degree=row["va_degree"]),
            row["bus"] + 1,
            name=row["name"],
        )
    for node in monee_net.nodes:
        # per convention the index of the node minus 1 is equal to the id of the bus in pandapower
        pp_id = node.id - 1
        if len(net.bus) > pp_id:
            node.name = net.bus["name"].iloc[pp_id]
            if hasattr(net, "bus_geodata"):
                node.position = (
                    net.bus_geodata["x"].iloc[pp_id],
                    net.bus_geodata["y"].iloc[pp_id],
                )

    return monee_net
