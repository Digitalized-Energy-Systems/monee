import os
import uuid

import pandapower.converter as pc

from .matpower import read_matpower_case


def from_pandapower_net(net):
    """
    No docstring provided.
    """
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
    return monee_net
