import os

import pytest

import monee.model as mm
from monee.io.native import (
    FORMAT_VERSION,
    PersistenceException,
    load_to_network,
    native_dict_to_network,
    network_to_native_dict,
    write_omef_network,
)


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
        mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=0.1))], grid=gas_grid
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=1000, temperature_ext_k=300, roughness_m=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=1500, temperature_ext_k=300, roughness_m=0.01
        ),
        g_node_0,
        g_node_2,
    )

    # HEAT
    heating_grid = mm.create_water_grid("heat")
    h_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow_kgs=0.1))],
        grid=heating_grid,
    )
    h_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=heating_grid
    )
    h_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))], grid=heating_grid
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
        mm.CHP(0.1, 0.9, 0.1, 1),
        gas_node_id=g_node_0,
        heat_node_id=h_node_0,
        heat_return_node_id=h_node_1,
        power_node_id=el_node_0,
    )
    return pn


def test_write_load_with_compound():
    # GIVEN
    compound_test_network = create_compound_test_network()

    # WHEN
    write_omef_network("test.nt", compound_test_network)
    network = load_to_network("test.nt")

    # THEN
    assert network is not None
    assert len(network.compounds) == 1
    assert type(network.compounds[0].model) is mm.CHP
    assert len(network.compounds[0].connected_to) == 4

    os.remove("test.nt")


def test_load():
    # GIVEN
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

    # WHEN
    write_omef_network("test.nt", pn)
    network = load_to_network("test.nt")

    # THEN
    assert network is not None

    os.remove("test.nt")


def test_multi_grid_error():
    # GIVEN
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

    # WHEN / THEN
    with pytest.raises(PersistenceException):
        write_omef_network("test.nt", pn)


def test_model_unknown():
    # GIVEN
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

    # WHEN / THEN
    with pytest.raises(PersistenceException):
        write_omef_network("test_error.nt", pn)
        load_to_network("test_error.nt")

    os.remove("test_error.nt")


def _simple_power_network():
    pn = mm.Network(mm.create_power_grid("power", sn_mva=42))
    node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerGenerator(p_mw=100, q_mvar=0))],
    )
    node_1 = pn.node(
        mm.Bus(base_kv=2),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=10, q_mvar=0, vm_pu=10, va_degree=1))],
    )
    pn.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=0.0001, x_ohm_per_m=0.0005, parallel=1),
        node_0,
        node_1,
    )
    return pn, node_0, node_1


def _roundtrip(network):
    """Serialize to the native dict and back, in memory."""
    return native_dict_to_network(network_to_native_dict(network))


def test_native_dict_has_version_and_sections():
    # GIVEN
    pn, _, _ = _simple_power_network()

    # WHEN
    struct = network_to_native_dict(pn)

    # THEN
    assert struct["version"] == FORMAT_VERSION
    assert set(struct) >= {"grids", "nodes", "childs", "branches", "compounds"}


def test_roundtrip_preserves_structure():
    # GIVEN
    pn = create_compound_test_network()

    # WHEN
    out = _roundtrip(pn)

    # THEN
    assert len(out.nodes) == len(pn.nodes)
    assert len(out.branches) == len(pn.branches)
    assert len(out.childs) == len(pn.childs)
    assert len(out.compounds) == len(pn.compounds)


def test_var_value_bounds_integer_name_preserved():
    # GIVEN
    pn, node_0, _ = _simple_power_network()
    bus = pn.node_by_id(node_0).model
    bus.vm_pu = mm.Var(1.23, max=9.0, min=-3.0, integer=True, name="custom_vm")

    # WHEN
    out = _roundtrip(pn)

    # THEN
    restored = out.node_by_id(node_0).model.vm_pu
    assert isinstance(restored, mm.Var)
    assert restored.value == 1.23
    assert restored.max == 9.0
    assert restored.min == -3.0
    assert restored.integer is True
    assert restored.name == "custom_vm"


def test_integer_var_flag_preserved_on_branch():
    # GIVEN
    pn = mm.Network(mm.create_water_grid("water"))
    n0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Source(mass_flow_kgs=0.1))])
    n1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    pn.branch(mm.WaterPipe(diameter_m=0.3, length_m=100), n0, n1)

    # WHEN
    out = _roundtrip(pn)

    # THEN
    # WaterPipe.direction is an integer Var; losing the flag would make it continuous
    direction = out.branches[0].model.direction
    assert isinstance(direction, mm.Var)
    assert direction.integer is True


def test_intermediate_value_preserved():
    # GIVEN
    pn, node_0, _ = _simple_power_network()
    bus = pn.node_by_id(node_0).model
    assert isinstance(bus.p_mw, mm.Intermediate)
    bus.p_mw.value = 7.5

    # WHEN
    out = _roundtrip(pn)

    # THEN
    restored = out.node_by_id(node_0).model.p_mw
    assert isinstance(restored, mm.Intermediate)
    assert restored.value == 7.5


def test_const_preserved():
    # GIVEN
    pn, node_0, _ = _simple_power_network()
    pn.node_by_id(node_0).model.vm_pu = mm.Const(0.97)

    # WHEN
    out = _roundtrip(pn)

    # THEN
    restored = out.node_by_id(node_0).model.vm_pu
    assert isinstance(restored, mm.Const)
    assert restored.value == 0.97


def test_node_position_preserved():
    # GIVEN
    pn = mm.Network(mm.create_power_grid("power"))
    n0 = pn.node(mm.Bus(base_kv=1), position=(1.5, -2.0))
    pn.node(mm.Bus(base_kv=1))
    pn.branch(
        mm.PowerLine(length_m=10, r_ohm_per_m=1e-4, x_ohm_per_m=5e-4, parallel=1),
        n0,
        1,
    )

    # WHEN
    out = _roundtrip(pn)

    # THEN
    assert tuple(out.node_by_id(n0).position) == (1.5, -2.0)


def test_active_flag_preserved():
    # GIVEN
    pn, node_0, node_1 = _simple_power_network()
    pn.node_by_id(node_1).active = False

    # WHEN
    out = _roundtrip(pn)

    # THEN
    assert out.node_by_id(node_0).active is True
    assert out.node_by_id(node_1).active is False


def test_component_name_preserved():
    # GIVEN
    pn = mm.Network(mm.create_power_grid("power"))
    n0 = pn.node(mm.Bus(base_kv=1), name="slack_bus")
    pn.node(mm.Bus(base_kv=1))
    pn.branch(
        mm.PowerLine(length_m=10, r_ohm_per_m=1e-4, x_ohm_per_m=5e-4, parallel=1),
        n0,
        1,
        name="line_a",
    )

    # WHEN
    out = _roundtrip(pn)

    # THEN
    assert out.node_by_id(n0).name == "slack_bus"
    assert out.branches[0].name == "line_a"


def test_grid_attributes_preserved():
    # GIVEN
    pn, _, _ = _simple_power_network()

    # WHEN
    out = _roundtrip(pn)

    # THEN
    grid = out.nodes[0].grid
    assert isinstance(grid, mm.PowerGrid)
    assert grid.sn_mva == 42


def test_native_dict_to_network_does_not_mutate_input():
    # GIVEN
    pn = create_compound_test_network()
    struct = network_to_native_dict(pn)
    import copy

    snapshot = copy.deepcopy(struct)

    # WHEN
    native_dict_to_network(struct)
    native_dict_to_network(struct)

    # THEN
    # a second load must succeed and the input must be untouched
    assert struct == snapshot


def test_gen_load_values_roundtrip():
    # GIVEN
    pn, node_0, node_1 = _simple_power_network()

    # WHEN
    out = _roundtrip(pn)

    # THEN
    # PowerGenerator stores the negated magnitude; round-trip must preserve it
    gen = out.childs_by_type(mm.PowerGenerator)[0].model
    assert gen.p_mw == -100


def test_file_roundtrip(tmp_path):
    # GIVEN
    pn = create_compound_test_network()
    path = tmp_path / "net.json"

    # WHEN
    write_omef_network(str(path), pn)
    out = load_to_network(str(path))

    # THEN
    assert len(out.nodes) == len(pn.nodes)
    assert len(out.compounds) == 1


def test_reads_legacy_untagged_format():
    # GIVEN
    # legacy on-disk / matpower format: untagged Var dicts, no "version"/"compounds"
    legacy = {
        "grids": {"power": {"model_type": "PowerGrid", "values": {"name": "power"}}},
        "nodes": [
            {
                "id": 0,
                "grid_id": "power",
                "child_ids": [0],
                "values": {
                    "base_kv": 1,
                    "vm_pu": {"value": 1.0, "max": 1.5, "min": 0.0},
                },
                "model_type": "Bus",
            },
            {
                "id": 1,
                "grid_id": "power",
                "child_ids": [],
                "values": {"base_kv": 1},
                "model_type": "Bus",
            },
        ],
        "childs": [
            {
                "id": 0,
                "values": {"p_mw": -5, "q_mvar": 0},
                "model_type": "PowerGenerator",
            }
        ],
        "branches": [
            {
                "id": [0, 1, 0],
                "from_node": 0,
                "to_node": 1,
                "grid_id": "power",
                "values": {
                    "length_m": 100,
                    "r_ohm_per_m": 1e-4,
                    "x_ohm_per_m": 5e-4,
                    "parallel": 1,
                },
                "model_type": "PowerLine",
            }
        ],
    }

    # WHEN
    network = native_dict_to_network(legacy)

    # THEN
    assert len(network.nodes) == 2
    assert len(network.branches) == 1

    vm_pu = network.node_by_id(0).model.vm_pu
    assert isinstance(vm_pu, mm.Var)
    assert vm_pu.max == 1.5


def test_storage_private_state_preserved():
    # GIVEN
    pn = mm.Network(mm.create_power_grid("power"))
    n0 = pn.node(
        mm.Bus(base_kv=1), child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0))]
    )
    n1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(
                mm.ElectricStorage(
                    e_mwh_initial=2.0,
                    e_mwh_max=5.0,
                    p_max_mw=3.0,
                    efficiency_charge=0.9,
                    efficiency_discharge=0.8,
                )
            )
        ],
    )
    pn.branch(
        mm.PowerLine(length_m=10, r_ohm_per_m=1e-4, x_ohm_per_m=5e-4, parallel=1),
        n0,
        n1,
    )

    # WHEN
    out = _roundtrip(pn)

    # THEN
    # private power bounds and lossy flag must round-trip
    storage = out.childs_by_type(mm.ElectricStorage)[0].model
    assert storage._e_mwh_initial == 2.0
    assert storage._p_max == 3.0
    assert storage._p_min == -3.0
    assert storage._lossy is True

    # behaviour driven by that private state must be intact
    storage.make_controllable()
    assert storage.p_mw.max == 3.0
    assert storage.p_mw.min == -3.0
    assert hasattr(storage, "p_charge_mw")


def test_heat_exchanger_nondefault_private_param_preserved():
    # GIVEN
    pn = mm.Network(mm.create_water_grid("water"))
    n0 = pn.node(mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())])
    n1 = pn.node(mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=0.5))])
    pn.branch(mm.HeatExchanger(q_mw=2.0, T_delta_design_K=45), n0, n1)

    # WHEN
    out = _roundtrip(pn)

    # THEN
    hx = out.branches_by_type(mm.HeatExchanger)[0].model
    assert hx._T_delta_design_K == 45


def test_compound_id_and_active_preserved():
    # GIVEN
    pn = create_compound_test_network()
    compound = pn.compounds[0]
    original_id = compound.id
    pn.deactivate_by_id(type(compound), original_id)

    # WHEN
    out = _roundtrip(pn)

    # THEN
    assert len(out.compounds) == 1
    restored = out.compounds[0]
    assert restored.id == original_id
    assert restored.active is False


def test_unserializable_public_attribute_raises():
    # GIVEN
    class Weird(mm.Bus):
        def __init__(self):
            super().__init__(base_kv=1)
            self.gadget = object()  # public, non-encodable

    pn = mm.Network(mm.create_power_grid("power"))
    pn.node(Weird())
    pn.node(mm.Bus(base_kv=1))
    pn.branch(
        mm.PowerLine(length_m=10, r_ohm_per_m=1e-4, x_ohm_per_m=5e-4, parallel=1),
        0,
        1,
    )

    # WHEN / THEN
    with pytest.raises(PersistenceException):
        network_to_native_dict(pn)
