import os

import pytest

import monee.model as mm
from monee.io.native import PersistenceException, load_to_network, write_omef_network


def create_compound_test_network():
    pn = mm.Network(mm.create_power_grid("power"))

    # POWER
    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0)),
        ],
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
    )

    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow=0.1))], grid=gas_grid
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=1000, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=1500, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_2,
    )

    # HEAT
    heating_grid = mm.create_water_grid("heat")
    h_node_0 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow=0.1))], grid=heating_grid
    )
    h_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=heating_grid
    )
    h_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=heating_grid
    )

    pn.branch(
        mm.WaterPipe(diameter_m=0.35, length_m=1000),
        h_node_0,
        h_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.35, length_m=1500),
        h_node_0,
        h_node_2,
    )

    # MULTI
    pn.compound(
        mm.CHP(0.1, 0.9, 0.1),
        gas_node=g_node_0,
        heat_node=h_node_0,
        heat_return_node=h_node_1,
        power_node=el_node_0,
    )
    return pn


def test_write_load_with_compound():
    compound_test_network = create_compound_test_network()
    write_omef_network("test.nt", compound_test_network)
    network = load_to_network("test.nt")
    assert network is not None
    assert len(network.compounds) == 1
    assert type(network.compounds[0].model) is mm.CHP
    assert len(network.compounds[0].connected_to) == 4
    os.remove("test.nt")


def test_load():
    pn = mm.Network(mm.create_power_grid("power"))

    node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerGenerator(p_mw=100, q_mvar=0))],
    )

    node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=10, q_mvar=0, vm_pu=10, va_degree=1))],
    )

    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=0.0001, x_ohm_per_m=0.0005, parallel=1),
        node_0,
        node_1,
    )

    write_omef_network("test.nt", pn)
    network = load_to_network("test.nt")
    assert network is not None
    os.remove("test.nt")


def test_multi_grid_error():
    pn = mm.Network(mm.create_power_grid("power"))
    other_power_grid = mm.create_power_grid("power", sn_mva=2)
    node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerGenerator(p_mw=100, q_mvar=0))],
        grid=other_power_grid,
    )

    node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=10, q_mvar=0, vm_pu=10, va_degree=1))],
    )

    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=0.0001, x_ohm_per_m=0.0005, parallel=1),
        node_0,
        node_1,
    )

    with pytest.raises(PersistenceException):
        write_omef_network("test.nt", pn)


def test_model_unknown():
    class BusUnknown(mm.Bus):
        pass

    pn = mm.Network(mm.create_power_grid("power"))
    node_0 = pn.node(
        BusUnknown(base_kv=1),
        child_ids=[pn.child(mm.PowerGenerator(p_mw=100, q_mvar=0))],
    )
    node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=10, q_mvar=0, vm_pu=10, va_degree=1))],
    )
    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=0.0001, x_ohm_per_m=0.0005, parallel=1),
        node_0,
        node_1,
    )

    with pytest.raises(PersistenceException):
        write_omef_network("test_error.nt", pn)
        load_to_network("test_error.nt")
    os.remove("test_error.nt")
