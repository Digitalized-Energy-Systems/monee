import pytest

import monee
import monee.express as mx
import monee.model as mm


def _dead_end_heat_load_net():
    net = mm.Network(mm.create_water_grid("water"))
    j0 = mx.create_water_junction(net)
    j1 = mx.create_water_junction(net)
    j2 = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, j0)
    mx.create_water_pipe(net, j0, j1, diameter_m=0.2, length_m=100)
    mx.create_water_pipe(net, j1, j2, diameter_m=0.2, length_m=100)
    mx.create_sink(net, j1, mass_flow_kgs=1)
    mx.create_heat_load(net, j2, q_mw=0.1, name="my heat load")
    return net


def test_dead_end_heat_load_pruning_names_load_and_rule():
    with pytest.warns(UserWarning, match="my heat load"):
        result = monee.run_energy_flow(_dead_end_heat_load_net())

    assert result.success
    pruned = [w for w in result.warnings if w.category == "pruned"]
    assert len(pruned) == 1
    message = pruned[0].message
    assert "HeatLoad" in message
    assert "my heat load" in message
    assert "heat-only" in message
    assert "diagnose_infeasibility" in message


def test_islanded_component_pruning_names_rule():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.0)
    b2 = mx.create_bus(net)
    b3 = mx.create_bus(net)
    mx.create_line(net, b2, b3, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b3, p_mw=0.5, q_mvar=0.0, name="islanded load")

    with pytest.warns(UserWarning, match="grid-forming"):
        result = monee.run_energy_flow(net)

    assert result.success
    pruned = next(w for w in result.warnings if w.category == "pruned")
    assert "islanded load" in pruned.message
    assert "slack or grid-forming" in pruned.message


def test_unpruned_net_emits_no_pruned_entry():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.0)

    result = monee.run_energy_flow(net)

    assert result.success
    assert [w for w in result.warnings if w.category == "pruned"] == []
