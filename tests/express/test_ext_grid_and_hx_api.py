import pytest

import monee
import monee.express as mx
import monee.model as mm
from monee.solver.casadi import CasADiSolveError


def small_gas_net(**ext_kwargs):
    net = mm.Network(el_model=mm.PowerGrid("pg"))
    j0 = mx.create_gas_junction(net)
    j1 = mx.create_gas_junction(net)
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.3, length_m=500)
    eid = mx.create_gas_ext_grid(net, j0, **ext_kwargs)
    mx.create_gas_sink(net, j1, mass_flow_kgs=0.1)
    return net, eid


def small_heat_ring(**hx_kwargs):
    net = mm.Network(el_model=mm.PowerGrid("pg"))
    s0 = mx.create_water_junction(net)
    s1 = mx.create_water_junction(net)
    r1 = mx.create_water_junction(net)
    r0 = mx.create_water_junction(net)
    mx.create_water_pipe(net, s0, s1, diameter_m=0.2, length_m=200)
    mx.create_water_pipe(net, r1, r0, diameter_m=0.2, length_m=200)
    hx = mx.create_heat_exchanger(net, s1, r1, q_mw=0.5, **hx_kwargs)
    ext = mx.create_water_ext_grid(net, s0)
    mx.create_consume_hydr_grid(net, r0)
    return net, hx, ext


def test_gas_ext_grid_default_t_k_resolves_to_gas_grid_t_k():
    net, eid = small_gas_net()
    assert net.child_by_id(eid).model.t_k is None
    result = monee.run_energy_flow(net)
    assert result.dataframes["Junction"].iloc[0]["t_k"] == pytest.approx(300)


def test_gas_ext_grid_explicit_t_k_still_pins():
    net, eid = small_gas_net(t_k=356)
    result = monee.run_energy_flow(net)
    assert result.dataframes["Junction"].iloc[0]["t_k"] == pytest.approx(356)


def test_water_ext_grid_default_t_k_resolves_to_t_ref_k():
    net, _, ext = small_heat_ring()
    assert net.child_by_id(ext).model.t_k is None
    result = monee.run_energy_flow(net)
    t_pinned = result.dataframes["Junction"].iloc[0]["t_k"]
    assert t_pinned == pytest.approx(356)


def test_max_import_kgs_mutation_takes_effect_on_next_solve():
    net, eid = small_gas_net(max_import_kgs=0.2)
    first = monee.run_energy_flow(net)
    assert first.dataframes["ExtHydrGrid"].iloc[0]["mass_flow_kgs"] == pytest.approx(
        -0.1, abs=1e-4
    )
    model = net.child_by_id(eid).model
    model.max_import_kgs = 0.05
    # Batch solver-ergonomics item S1: reported on the result; strict=True
    # raises. Neither may leave the caller's network unable to solve again.
    assert monee.run_energy_flow(net).success is False
    with pytest.raises(CasADiSolveError):
        monee.run_energy_flow(net, strict=True)
    model.max_import_kgs = 0.15
    relaxed = monee.run_energy_flow(net)
    assert relaxed.dataframes["ExtHydrGrid"].iloc[0]["mass_flow_kgs"] == pytest.approx(
        -0.1, abs=1e-4
    )


def test_heat_exchanger_mass_flow_design_kgs_parameter_drives_flow():
    net, hx, _ = small_heat_ring(mass_flow_design_kgs=2.0)
    result = monee.run_energy_flow(net)
    row = result.dataframes["HeatExchangerLoad"].iloc[0]
    assert row["mass_flow_kgs"] == pytest.approx(-2.0, abs=1e-4)
    assert row["q_mw"] == pytest.approx(0.5, abs=1e-4)


def test_heat_creator_docstrings_state_part_load_semantics():
    for creator in (
        mx.create_heat_exchanger,
        mx.create_passive_heat_exchanger,
        mx.create_heat_load,
        mx.create_heat_generator,
    ):
        assert "Part load in one line" in creator.__doc__
    assert "q_mw_set" in mx.create_heat_exchanger.__doc__
    assert "regulation" in mx.create_heat_exchanger.__doc__


def test_gas_ext_grid_docstring_names_model_class():
    assert "ExtHydrGrid" in mx.create_gas_ext_grid.__doc__


def test_water_ext_grid_docstring_states_backup_plant_semantics():
    assert "backup heat plant" in mx.create_water_ext_grid.__doc__
    assert "backup" in mm.ExtHydrGrid.__doc__
    assert "bounds_ext_heat" in mm.ExtHydrGrid.__doc__
