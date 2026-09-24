"""Deactivate/activate round-trips for coupling compounds: a repeated
deactivate must not clobber the saved regulation, so a later activate
restores the original setpoints instead of a permanently dead unit."""

from monee import mm, mx

REGULATION = 0.7
MASS_FLOW_KGS = 0.15


def _nodes():
    net = mm.Network()
    bus = mx.create_bus(net)
    j_heat = mx.create_water_junction(net)
    j_heat_return = mx.create_water_junction(net)
    j_gas = mx.create_gas_junction(net)
    return net, bus, j_heat, j_heat_return, j_gas


def _assert_roundtrip(net, compound_id):
    compound = net.compound_by_id(compound_id)
    control = compound.model._control_node
    original = control.regulation

    net.deactivate(compound)
    net.deactivate(compound)
    assert control.regulation == 0

    net.activate(compound)
    assert control.regulation == original
    return control


def test_chp_double_deactivate_then_activate_restores_regulation():
    net, bus, j_heat, j_heat_return, j_gas = _nodes()
    cid = mx.create_chp(
        net,
        power_node_id=bus,
        heat_node_id=j_heat,
        heat_return_node_id=j_heat_return,
        gas_node_id=j_gas,
        diameter_m=0.1,
        efficiency_power=0.4,
        efficiency_heat=0.4,
        mass_flow_setpoint_kgs=MASS_FLOW_KGS,
        regulation=REGULATION,
    )
    control = _assert_roundtrip(net, cid)
    assert control.regulation == REGULATION
    assert control.gas_mass_flow_kgs == MASS_FLOW_KGS


def test_g2h_double_deactivate_then_activate_restores_regulation():
    net, _, j_heat, j_heat_return, j_gas = _nodes()
    cid = mx.create_g2h(
        net,
        gas_node_id=j_gas,
        heat_node_id=j_heat,
        heat_return_node_id=j_heat_return,
        heat_energy_mw=0.2,
        diameter_m=0.1,
        efficiency=0.9,
        regulation=REGULATION,
    )
    control = _assert_roundtrip(net, cid)
    assert control.regulation == REGULATION


def test_p2h_double_deactivate_then_activate_restores_regulation():
    net, bus, j_heat, j_heat_return, _ = _nodes()
    cid = mx.create_p2h(
        net,
        power_node_id=bus,
        heat_node_id=j_heat,
        heat_return_node_id=j_heat_return,
        heat_energy_mw=0.2,
        diameter_m=0.1,
        efficiency=0.9,
        regulation=REGULATION,
    )
    control = _assert_roundtrip(net, cid)
    assert control.regulation == REGULATION


def test_chp_single_deactivate_activate_roundtrip():
    net, bus, j_heat, j_heat_return, j_gas = _nodes()
    cid = mx.create_chp(
        net,
        power_node_id=bus,
        heat_node_id=j_heat,
        heat_return_node_id=j_heat_return,
        gas_node_id=j_gas,
        diameter_m=0.1,
        efficiency_power=0.4,
        efficiency_heat=0.4,
        mass_flow_setpoint_kgs=MASS_FLOW_KGS,
        regulation=REGULATION,
    )
    compound = net.compound_by_id(cid)
    control = compound.model._control_node

    net.deactivate(compound)
    assert control.regulation == 0
    assert control.gas_mass_flow_kgs == 0

    net.activate(compound)
    assert control.regulation == REGULATION
    assert control.gas_mass_flow_kgs == MASS_FLOW_KGS
