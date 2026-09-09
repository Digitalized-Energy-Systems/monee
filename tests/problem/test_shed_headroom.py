"""Unit tests for the load-shedding local-optimality headroom check."""

import monee.express as mx
from monee.model.core import Var
from monee.problem.min_load_shedding import (
    _attach_shed_headroom_validator,
    _shed_headroom_validator,
    create_min_load_shedding_problem,
)
from monee.solver.core import SolverResult, validate_result


def _el_net(regulation, ext_p_mw=0.0):
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    mx.create_line(net, b0, b1, length_m=1000, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    ext_id = mx.create_ext_power_grid(net, b0)
    load_id = mx.create_power_load(net, b1, p_mw=0.4, q_mvar=0.0)
    net.child_by_id(load_id).model.regulation = regulation
    net.child_by_id(ext_id).model.p_mw = ext_p_mw
    return net, net.child_by_id(ext_id), net.child_by_id(load_id)


def test_shed_with_unbounded_slack_warns():
    net, _, _ = _el_net(regulation=0.5)
    entries = _shed_headroom_validator(net, None)
    assert len(entries) == 1
    w = entries[0]
    assert w.category == "shed_headroom"
    assert w.value == 0.2
    assert "PowerLoad" in w.message
    assert "unbounded" in w.message
    assert "solver='scip'" in w.message
    assert "load_shedding" in w.message


def test_shed_with_slack_at_import_bound_stays_silent():
    net, ext, _ = _el_net(regulation=0.5)
    ext.model.p_mw = Var(-1.0, max=1.0, min=-1.0, name="p_mw")
    assert _shed_headroom_validator(net, None) == []


def test_shed_with_slack_strictly_inside_bounds_warns_with_headroom():
    net, ext, _ = _el_net(regulation=0.5)
    ext.model.p_mw = Var(-0.4, max=1.0, min=-1.0, name="p_mw")
    entries = _shed_headroom_validator(net, None)
    assert len(entries) == 1
    assert "0.6 MW" in entries[0].message


def test_fully_served_load_stays_silent():
    net, _, _ = _el_net(regulation=1.0)
    assert _shed_headroom_validator(net, None) == []


def test_ignored_load_is_structural_and_stays_silent():
    net, _, load = _el_net(regulation=0.0)
    load.ignored = True
    assert _shed_headroom_validator(net, None) == []


def test_problem_attaches_validator_and_validate_result_runs_it():
    net, _, _ = _el_net(regulation=0.5)
    problem = create_min_load_shedding_problem()
    assert _attach_shed_headroom_validator in problem._controllable_appliables
    _attach_shed_headroom_validator(net)
    _attach_shed_headroom_validator(net)
    assert net._result_validators == [_shed_headroom_validator]
    result = SolverResult(net, {}, 0.0, True)
    validate_result(net, result)
    assert any(w.category == "shed_headroom" for w in result.warnings)


def test_gas_sink_shed_converts_to_mw_and_warns():
    net = mx.create_multi_energy_network()
    j0 = mx.create_gas_junction(net)
    j1 = mx.create_gas_junction(net)
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.3, length_m=100)
    mx.create_ext_hydr_grid(net, j0)
    sink_id = mx.create_sink(net, j1, mass_flow_kgs=0.1)
    net.child_by_id(sink_id).model.regulation = 0.0
    entries = _shed_headroom_validator(net, None)
    assert len(entries) == 1
    assert entries[0].category == "shed_headroom"
    assert "gas demand" in entries[0].message
    assert entries[0].value > 0.5
