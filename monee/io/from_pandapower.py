import os
import pandapower.converter as pc
from .matpower import read_matpower_case
from monee.model.child import PowerLoad, PowerGenerator, ExtPowerGrid


def from_pandapower_net(net):
    pc.to_mpc(net, init="flat", filename="temp.mat")
    monee_net = read_matpower_case("temp.mat")
    os.remove("temp.mat")
    monee_net.clear_childs()

    for _, row in net.load.iterrows():
        monee_net.child_to(
            PowerLoad(row["p_mw"], row["q_mvar"], name=row["name"]), row["bus"] + 1
        )
    for _, row in net.sgen.iterrows():
        monee_net.child_to(
            PowerGenerator(row["p_mw"], row["q_mvar"], name=row["name"]), row["bus"] + 1
        )
    for _, row in net.ext_grid.iterrows():
        monee_net.child_to(
            ExtPowerGrid(
                1, 1, vm_pu=row["vm_pu"], va_degree=row["va_degree"], name=row["name"]
            ),
            row["bus"] + 1,
        )
    return monee_net
