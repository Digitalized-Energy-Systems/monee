import os

from monee.model.core import Network
from monee.model.grid import PowerGrid
from monee.model.branch import PowerLine
from monee.model.node import Bus
from monee.model.child import PowerGenerator, ExtPowerGrid

from monee.io.native import write_omef_network, load_to_network


def test_load():

    pn = Network(PowerGrid(name="power", sn_mva=1))

    node_0 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerGenerator(p_mw=100, q_mvar=0))],
    )

    node_1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=10, q_mvar=0, vm_pu=10, va_degree=1))],
    )

    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.0001, x_ohm_per_m=0.0005, parallel=1),
        node_0,
        node_1,
    )

    write_omef_network("test.nt", pn)
    network = load_to_network("test.nt")
    assert network is not None
    os.remove("test.nt")    
