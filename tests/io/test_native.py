import os

import pytest

import monee.express as mx
import monee.model as mm
from monee.io.native import (
    FORMAT_VERSION,
    PersistenceException,
    _decode_values,
    load_to_network,
    native_dict_to_network,
    network_to_native_dict,
    preprocess_dict,
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


def test_var_scale_roundtrip():
    # GIVEN
    pn, node_0, _ = _simple_power_network()
    bus = pn.node_by_id(node_0).model
    bus.vm_pu = mm.Var(1.0, name="scaled_vm")
    bus.vm_pu.scale = 100.0

    # WHEN
    out = _roundtrip(pn)

    # THEN
    restored = out.node_by_id(node_0).model.vm_pu
    assert isinstance(restored, mm.Var)
    assert restored.scale == 100.0


def test_load_edit_via_default_grid_save_roundtrip():
    # GIVEN a loaded network whose file carries a grid named like a default
    pn, node_0, _ = _simple_power_network()
    out = _roundtrip(pn)

    # WHEN adding a component through the default-grid path
    new_node = out.node(mm.Bus(base_kv=1), grid=mm.EL_KEY)
    out.branch(
        mm.PowerLine(length_m=10, r_ohm_per_m=1e-4, x_ohm_per_m=5e-4, parallel=1),
        node_0,
        new_node,
    )

    # THEN the loaded grid replaced the default: one grid instance, save works
    assert len({id(n.grid) for n in out.nodes}) == 1
    struct = network_to_native_dict(out)
    assert set(struct["grids"]) == {"power"}
    assert struct["grids"]["power"]["values"]["sn_mva"] == 42


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


def test_preprocess_dict_delegates_to_decode():
    sample = {"a": 1, "b": "x"}
    assert preprocess_dict(sample) == _decode_values(sample)


def _create_all_couplers_network():
    """One network containing every registered compound and coupler type."""
    net = mm.Network(mm.create_power_grid("power"))
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    water_grid = mm.create_water_grid("heat")

    el = [
        net.node(
            mm.Bus(base_kv=1),
            child_ids=[
                net.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))
            ],
        )
    ]
    el += [net.node(mm.Bus(base_kv=1)) for _ in range(4)]
    for node in el[1:]:
        net.branch(
            mm.PowerLine(length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
            el[0],
            node,
        )

    gas = [
        net.node(mm.Junction(), child_ids=[net.child(mm.ExtHydrGrid())], grid=gas_grid)
    ]
    gas += [net.node(mm.Junction(), grid=gas_grid) for _ in range(4)]
    for node in gas[1:]:
        net.branch(
            mm.GasPipe(
                diameter_m=0.3, length_m=100, temperature_ext_k=300, roughness_m=0.01
            ),
            gas[0],
            node,
        )

    heat = [
        net.node(
            mm.Junction(), child_ids=[net.child(mm.ExtHydrGrid())], grid=water_grid
        )
    ]
    heat += [net.node(mm.Junction(), grid=water_grid) for _ in range(3)]
    for node in heat[1:]:
        net.branch(mm.WaterPipe(diameter_m=0.2, length_m=100), heat[0], node)

    net.compound(
        mm.PowerToHeat(
            heat_energy_mw=0.1, diameter_m=0.1, temperature_ext_k=293, efficiency=0.9
        ),
        power_node_id=el[1],
        heat_node_id=heat[1],
        heat_return_node_id=heat[2],
    )
    net.compound(
        mm.CHP(
            diameter_m=0.1,
            efficiency_power=0.4,
            efficiency_heat=0.45,
            mass_flow_setpoint_kgs=0.01,
        ),
        power_node_id=el[2],
        heat_node_id=heat[1],
        heat_return_node_id=heat[2],
        gas_node_id=gas[1],
    )
    net.compound(
        mm.GasToHeat(
            heat_energy_mw=0.2, diameter_m=0.1, temperature_ext_k=293, efficiency=0.85
        ),
        gas_node_id=gas[2],
        heat_node_id=heat[1],
        heat_return_node_id=heat[2],
    )
    net.compound(
        mm.CHPHG(
            efficiency_power=0.35, efficiency_heat=0.5, mass_flow_setpoint_kgs=0.02
        ),
        power_node_id=el[3],
        heat_node_id=heat[3],
        gas_node_id=gas[3],
    )
    net.branch(
        mm.PowerToGas(efficiency=0.7, mass_flow_setpoint_kgs=0.05), el[4], gas[4]
    )
    net.branch(mm.GasToPower(efficiency=0.6, p_mw_setpoint=0.3), gas[4], el[4])
    net.branch(mm.PowerToHeatHG(heat_energy_mw=0.15, efficiency=0.95), el[4], heat[3])
    net.branch(mm.GasToHeatHG(heat_energy_mw=0.25, efficiency=0.8), gas[4], heat[3])
    return net


def _model_of(components, model_cls):
    matching = [c.model for c in components if type(c.model) is model_cls]
    assert len(matching) == 1
    return matching[0]


def test_roundtrip_every_coupler_type(tmp_path):
    # GIVEN
    pn = _create_all_couplers_network()
    path = tmp_path / "couplers.json"

    # WHEN
    write_omef_network(str(path), pn)
    out = load_to_network(str(path))

    # THEN structure survives
    assert len(out.nodes) == len(pn.nodes)
    assert len(out.branches) == len(pn.branches)
    assert len(out.childs) == len(pn.childs)
    assert len(out.compounds) == len(pn.compounds)

    # THEN key attributes survive
    p2h = _model_of(out.compounds, mm.PowerToHeat)
    assert p2h.heat_energy_mw == 0.1
    assert p2h.efficiency == 0.9
    assert p2h.load_p_mw == pytest.approx(0.1 / 0.9)

    chp = _model_of(out.compounds, mm.CHP)
    assert chp.efficiency_power == 0.4
    assert chp.efficiency_heat == 0.45
    assert chp.mass_flow_setpoint_kgs == 0.01

    g2h = _model_of(out.compounds, mm.GasToHeat)
    assert g2h.heat_energy_mw == -0.2
    assert g2h.efficiency == 0.85

    chp_hg = _model_of(out.compounds, mm.CHPHG)
    assert chp_hg.efficiency_power == 0.35
    assert chp_hg.efficiency_heat == 0.5

    p2g = _model_of(out.branches, mm.PowerToGas)
    assert p2g.efficiency == 0.7
    assert p2g.gas_mass_flow_kgs == -0.05

    g2p = _model_of(out.branches, mm.GasToPower)
    assert g2p.efficiency == 0.6
    assert g2p.el_mw == -0.3

    p2h_hg = _model_of(out.branches, mm.PowerToHeatHG)
    assert p2h_hg.heat_energy_mw == 0.15
    assert p2h_hg.efficiency == 0.95
    assert p2h_hg.load_p_mw == pytest.approx(0.15 / 0.95)

    g2h_hg = _model_of(out.branches, mm.GasToHeatHG)
    assert g2h_hg.heat_energy_mw == -0.25
    assert g2h_hg.efficiency == 0.8


def _docs_quick_look_network():
    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    j_supply = mx.create_water_junction(net)
    j_mid = mx.create_water_junction(net)
    j_return = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, j_supply)
    mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.12, length_m=100)
    mx.create_sink(net, j_return, mass_flow_kgs=1)

    mx.create_p2h(
        net, bus_1, j_mid, j_return, heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9
    )
    return net


def test_roundtrip_docs_quick_look_network(tmp_path):
    # GIVEN
    pn = _docs_quick_look_network()
    path = tmp_path / "quick_look.json"

    # WHEN
    write_omef_network(str(path), pn)
    out = load_to_network(str(path))

    # THEN
    assert len(out.nodes) == len(pn.nodes)
    assert len(out.branches) == len(pn.branches)
    assert len(out.childs) == len(pn.childs)
    p2h = _model_of(out.compounds, mm.PowerToHeat)
    assert p2h.heat_energy_mw == 0.1
    assert p2h.load_p_mw == pytest.approx(0.1 / 0.9)


def test_roundtrip_preserves_adjacency_and_branch_order():
    # GIVEN a network whose compounds interleave branches with plain ones
    from monee.network import create_urban_district_net

    pn = create_urban_district_net()

    # WHEN
    out = _roundtrip(pn)

    # THEN global branch order and per-node adjacency order survive
    assert [b.id for b in out.branches] == [b.id for b in pn.branches]
    for orig_node in pn.nodes:
        loaded = out.node_by_id(orig_node.id)
        assert loaded.from_branch_ids == orig_node.from_branch_ids
        assert loaded.to_branch_ids == orig_node.to_branch_ids


def test_serialization_independent_of_solve_state():
    # GIVEN
    import monee

    pn = _stub_bus_network()
    before = network_to_native_dict(pn)

    # WHEN solving in place
    monee.run_energy_flow(pn, formulation="smooth_nlp")

    # THEN the default encoding is unchanged by the solve
    assert network_to_native_dict(pn) == before

    # THEN the opt-in encoding carries the solved operating point
    checkpoint = network_to_native_dict(pn, include_solve_state=True)
    assert checkpoint != before
    live_vm = pn.node_by_id(1).model.vm_pu.value
    encoded_vm = next(n for n in checkpoint["nodes"] if n["id"] == 1)["values"][
        "vm_pu"
    ]["value"]
    assert encoded_vm == live_vm


def test_loaded_solved_network_starts_from_declared_values(tmp_path):
    # GIVEN a solved network saved with the default (solve-independent) encoding
    import monee

    pn = _stub_bus_network()
    declared_vm = pn.node_by_id(1).model.vm_pu.value
    monee.run_energy_flow(pn, formulation="smooth_nlp")
    assert pn.node_by_id(1).model.vm_pu.value != declared_vm
    path = tmp_path / "solved.json"
    write_omef_network(str(path), pn)

    # WHEN
    out = load_to_network(str(path))

    # THEN the loaded network re-solves from the declared start, not the solution
    assert out.node_by_id(1).model.vm_pu.value == declared_vm


def _stub_bus_network():
    import monee

    net = monee.Network(mm.create_power_grid("power"))
    b0 = net.node(
        mm.Bus(base_kv=20), child_ids=[net.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0))]
    )
    b1 = net.node(
        mm.Bus(base_kv=20), child_ids=[net.child(mm.PowerLoad(p_mw=0.1, q_mvar=0))]
    )
    b2 = net.node(mm.Bus(base_kv=20))
    line = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net.branch(mm.PowerLine(**line), b0, b1)
    net.branch(mm.PowerLine(**line), b1, b2)
    return net


def test_postprocess_nan_survives_roundtrip():
    # GIVEN a network with a dead-end stub whose bus is pruned to NaN
    import math

    import monee

    net = _stub_bus_network()
    r1 = monee.run_energy_flow(net, formulation="smooth_nlp")

    # WHEN
    out = _roundtrip(net)
    r2 = monee.run_energy_flow(out, formulation="smooth_nlp")

    # THEN the pruned bus reports NaN in both, and live values match
    va1 = list(r1.dataframes["Bus"]["va_degree"])
    va2 = list(r2.dataframes["Bus"]["va_degree"])
    assert math.isnan(va1[2]) and math.isnan(va2[2])
    assert va1[:2] == pytest.approx(va2[:2], abs=1e-10)


def test_solve_save_load_solve_roundtrip_end_to_end(tmp_path):
    # GIVEN
    import monee

    net = _docs_quick_look_network()

    # WHEN build -> solve -> save -> load -> solve -> save
    r1 = monee.run_energy_flow(net, formulation="smooth_nlp")
    path_1 = tmp_path / "first.json"
    write_omef_network(str(path_1), net)

    out = load_to_network(str(path_1))
    r2 = monee.run_energy_flow(out, formulation="smooth_nlp")
    path_2 = tmp_path / "second.json"
    write_omef_network(str(path_2), out)

    # THEN the two serializations are byte-identical
    assert path_1.read_bytes() == path_2.read_bytes()

    # THEN the two solves agree
    for frame_name, frame in r1.dataframes.items():
        other = r2.dataframes[frame_name]
        left = frame.select_dtypes("number").fillna(0.0)
        right = other.select_dtypes("number").fillna(0.0)
        assert ((left - right).abs().max(axis=None) or 0.0) == pytest.approx(
            0.0, abs=1e-8
        ), frame_name


def test_unknown_type_error_names_both_causes():
    from monee.io.native import _resolve_model_type

    with pytest.raises(PersistenceException) as excinfo:
        _resolve_model_type("NotRegisteredBranch")

    message = str(excinfo.value)
    assert "never imported in this process" in message
    assert "not decorated with @model" in message
    assert "concepts/data_model" in message
