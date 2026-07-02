"""
Tests for :class:`monee.simulation.stepper.Stepper`.

Covers the externally-paced co-simulation API: variable ``dt_h``,
``data_overrides``, ``ts_index``, ``initial_state`` seeding, error
handling, parity with :func:`run_timeseries`, and the StepState extension.
"""

from __future__ import annotations

import dataclasses
import math

import pytest

import monee.express as mx
import monee.model as mm
from monee.simulation import (
    NetworkChange,
    Stepper,
    StepState,
    TimeseriesData,
    run_timeseries,
)
from tests.util import child_id_by_type


def _power_net(p_load: float = 1.0):
    net = mm.Network()
    b1 = mx.create_bus(net, base_kv=20.0)
    b2 = mx.create_bus(net, base_kv=20.0)
    mx.create_ext_power_grid(net, b1, p_mw=0, q_mvar=0)
    mx.create_power_load(net, b2, p_mw=p_load, q_mvar=0.1)
    mx.create_line(
        net,
        b1,
        b2,
        length_m=500,
        r_ohm_per_m=2e-4,
        x_ohm_per_m=4e-4,
        parallel=1,
    )
    return net


def test_Stepper_default_constructs_and_steps():
    stepper = Stepper(_power_net())
    assert stepper.step_count == 0
    assert stepper.t_h == 0.0

    sr = stepper.step(dt_h=1.0)

    assert sr.failed is False
    assert sr.result.success is True
    assert stepper.step_count == 1
    assert stepper.t_h == 1.0
    assert len(stepper.history) == 1


def test_Stepper_accumulates_t_h_across_variable_dt():
    stepper = Stepper(_power_net())
    stepper.step(dt_h=1.0 / 3600)  # 1 s
    stepper.step(dt_h=4.0 / 3600)  # 4 s
    stepper.step(dt_h=10.0 / 3600)  # 10 s
    stepper.step(dt_h=1.0 / 60)  # 60 s

    assert math.isclose(stepper.t_h, (1 + 4 + 10 + 60) / 3600, rel_tol=1e-12)
    assert stepper.step_count == 4


def test_Stepper_rejects_non_positive_dt():
    stepper = Stepper(_power_net())
    with pytest.raises(ValueError, match="dt_h must be > 0"):
        stepper.step(dt_h=0)
    with pytest.raises(ValueError, match="dt_h must be > 0"):
        stepper.step(dt_h=-1.0)


def test_Stepper_base_network_not_mutated():
    """Per-step deep-copy contract: the user's net is never touched."""
    net = _power_net()
    original_repr = repr(net.as_result_dataframe_dict())

    stepper = Stepper(net)
    stepper.step(dt_h=1.0)

    assert repr(net.as_result_dataframe_dict()) == original_repr


def test_Stepper_data_overrides_apply_per_step():
    """Override changes drive the solve; the base net is untouched."""
    net = _power_net(p_load=1.0)
    stepper = Stepper(net)

    load_id = child_id_by_type(net, mm.PowerLoad)

    sr_low = stepper.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 0.2})
    sr_high = stepper.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 2.0})

    p_from_low = sr_low.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
    p_from_high = sr_high.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
    assert p_from_high > p_from_low > 0
    assert math.isclose(p_from_low, 0.2, rel_tol=5e-2)
    assert math.isclose(p_from_high, 2.0, rel_tol=5e-2)


def test_Stepper_overrides_unknown_id_raises():
    stepper = Stepper(_power_net())
    with pytest.raises(KeyError, match="not found in network"):
        stepper.step(dt_h=1.0, data_overrides={(99999, "p_mw"): 1.0})


def test_Stepper_overrides_unknown_attr_raises():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)
    with pytest.raises(AttributeError, match="attribute"):
        stepper.step(dt_h=1.0, data_overrides={(load_id, "bogus_attr"): 0.5})


def test_Stepper_ts_index_applies_profile():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.2, 1.0, 2.0])

    stepper = Stepper(net, timeseries_data=td)
    p_values = []
    for t in range(3):
        sr = stepper.step(dt_h=1.0, ts_index=t)
        p_values.append(sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0])

    assert p_values[0] < p_values[1] < p_values[2]
    for actual, target in zip(p_values, [0.2, 1.0, 2.0]):
        assert math.isclose(actual, target, rel_tol=5e-2)


def test_Stepper_overrides_win_over_ts_data():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.2])

    stepper = Stepper(net, timeseries_data=td)
    sr = stepper.step(
        dt_h=1.0,
        ts_index=0,
        data_overrides={(load_id, "p_mw"): 1.5},
    )

    p = sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
    assert math.isclose(p, 1.5, rel_tol=5e-2), f"override should win, got p={p}"


def test_step_state_returns_initial_when_no_prior_solve():
    """The extended ``StepState`` falls back to initial_state on the first step."""
    state = StepState(initial_state={(42, "soc_mwh"): 7.5})
    assert state.get(42, "soc_mwh") == 7.5
    assert state.get(42, "missing_attr") is None
    assert state.get(99, "soc_mwh") is None  # unknown id


def test_step_state_prior_solve_wins_over_initial():
    """Once a network has been pushed, its values win over initial_state."""
    state = StepState(initial_state={(42, "vm_pu"): 1.5})
    assert state.get(42, "vm_pu") == 1.5

    # Push a fake solved net carrying vm_pu = 1.02 on node 42.
    net = mm.Network()
    bid = mx.create_bus(net, base_kv=20.0)
    mx.create_ext_power_grid(net, bid, p_mw=0, q_mvar=0)
    bus = net.node_by_id(bid)
    # The default Bus carries a Var(1) for vm_pu - pin it to 1.02 by hand.
    bus.model.vm_pu.value = 1.02

    # The state.get lookup keys on node.id (which equals bid here).
    state.push(net)
    assert math.isclose(state.get(bid, "vm_pu"), 1.02, rel_tol=1e-9)


def test_Stepper_on_step_error_skip_records_failure():
    """``on_step_error='skip'`` swallows solver exceptions and records them."""
    stepper = Stepper(_power_net(), on_step_error="skip")

    def _exploding_solve(*a, **kw):
        raise RuntimeError("simulated solver failure")

    stepper._solver.solve = _exploding_solve

    sr = stepper.step(dt_h=1.0)
    assert sr.failed is True
    assert isinstance(sr.error, RuntimeError)
    assert sr.result is None
    assert stepper.step_count == 1
    assert stepper.t_h == 1.0  # advances even on failure


def test_Stepper_on_step_error_raise_propagates():
    stepper = Stepper(_power_net(), on_step_error="raise")

    def _exploding_solve(*a, **kw):
        raise RuntimeError("boom")

    stepper._solver.solve = _exploding_solve

    with pytest.raises(RuntimeError, match="boom"):
        stepper.step(dt_h=1.0)
    assert stepper.step_count == 0  # not recorded


def test_Stepper_invalid_on_step_error_value():
    with pytest.raises(ValueError, match="on_step_error must be"):
        Stepper(_power_net(), on_step_error="silent")


def test_Stepper_reset_clears_history_and_state():
    stepper = Stepper(_power_net())
    stepper.step(dt_h=1.0)
    stepper.step(dt_h=2.0)
    assert stepper.step_count == 2
    assert stepper.t_h == 3.0
    assert len(stepper.state) == 2

    stepper.reset()

    assert stepper.step_count == 0
    assert stepper.t_h == 0.0
    assert len(stepper.state) == 0


def test_Stepper_reset_can_replace_initial_state():
    stepper = Stepper(_power_net(), initial_state={(0, "x"): 1.0})
    assert stepper.state.get(0, "x") == 1.0
    stepper.reset(initial_state={(0, "x"): 99.0})
    assert stepper.state.get(0, "x") == 99.0


def test_Stepper_context_manager():
    with Stepper(_power_net()) as stepper:
        stepper.step(dt_h=1.0)
        assert stepper.step_count == 1
    # Should exit cleanly; no resource leak assertion possible without
    # introspection of the solver backend.


def test_Stepper_matches_run_timeseries_on_fixed_grid():
    """Same network + same TD + same dt_h â†’ identical objectives across steps."""
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.5, 1.0, 1.5])

    ts_result = run_timeseries(net, td, steps=3)
    ts_pvals = [
        sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
        for sr in ts_result.step_results
    ]

    stepper = Stepper(net, timeseries_data=td)
    cond_pvals = []
    for t in range(3):
        sr = stepper.step(dt_h=1.0, ts_index=t)
        cond_pvals.append(sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0])

    for a, b in zip(ts_pvals, cond_pvals):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_Stepper_to_timeseries_result():
    stepper = Stepper(_power_net())
    for _ in range(3):
        stepper.step(dt_h=1.0)
    res = stepper.to_timeseries_result()
    assert len(res.step_results) == 3
    # Should expose the same get_result_for API as the timeseries runner.
    p_series = res.get_result_for(mm.PowerLine, "p_from_mw")
    assert len(p_series) == 3


def test_Stepper_repr_shows_progress():
    stepper = Stepper(_power_net())
    assert "steps=0" in repr(stepper)
    stepper.step(dt_h=0.5)
    assert "steps=1" in repr(stepper)
    assert "t_h=0.5" in repr(stepper)


def test_stepper_max_history_caps_retention():
    stepper = Stepper(_power_net(), max_history=2)
    for _ in range(5):
        stepper.step(dt_h=1.0)

    assert stepper.step_count == 5  # logical count keeps running
    assert len(stepper.history) == 2  # only the last two StepResults retained
    assert [sr.step for sr in stepper.history] == [3, 4]
    assert len(stepper.state) == 5  # logical step count in the state too


def test_stepper_max_history_invalid():
    with pytest.raises(ValueError, match="max_history must be"):
        Stepper(_power_net(), max_history=0)


def test_step_state_max_steps_absolute_index_fallback():
    """Absolute indices keep their meaning under retention; dropped steps
    fall back to initial_state."""
    state = StepState(initial_state={(0, "x"): 7.0}, max_steps=2)

    nets = []
    for vm in (1.01, 1.02, 1.03):
        net = mm.Network()
        bid = mx.create_bus(net, base_kv=20.0)
        net.node_by_id(bid).model.vm_pu.value = vm
        nets.append((bid, net))
        state.push(net)

    bid = nets[0][0]
    assert len(state) == 3
    # step 0 was dropped -> falls back (no initial entry for vm_pu -> None)
    assert state.get(bid, "vm_pu", step=0) is None
    assert math.isclose(state.get(bid, "vm_pu", step=1), 1.02)
    assert math.isclose(state.get(bid, "vm_pu", step=2), 1.03)
    assert math.isclose(state.get(bid, "vm_pu"), 1.03)  # relative latest
    # dropped step with an initial_state entry returns the seed
    assert state.get(0, "x", step=0) == 7.0


def test_stepper_get_returns_latest_solved_value():
    net = _power_net(p_load=1.0)
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    assert stepper.get(load_id, "p_mw") is None  # nothing solved yet

    stepper.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 0.4})
    assert math.isclose(stepper.get(load_id, "p_mw"), 0.4, rel_tol=1e-9)

    stepper.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 1.2})
    assert math.isclose(stepper.get(load_id, "p_mw"), 1.2, rel_tol=1e-9)
    # absolute lookback to the first step
    assert math.isclose(stepper.get(load_id, "p_mw", step=0), 0.4, rel_tol=1e-9)


# --- network change recording -------------------------------------------------


def _line_id(net):
    return net.branches[0].id


def _kinds(changes, kind):
    return [c for c in changes if c.kind == kind]


def test_Stepper_deactivate_records_and_persists():
    """deactivate() records one 'deactivated' change (source mutation) and the
    component stays off across later steps without re-recording."""
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    stepper.step(dt_h=1.0)

    deact = _kinds(stepper.changes, "deactivated")
    assert len(deact) == 1
    change = deact[0]
    assert change.step == 0
    assert change.component_type == "Child"
    assert change.component_id == load_id
    assert change.source == "mutation"
    # base net untouched: the user's load is still active.
    assert net.child_by_id(load_id).active is True


def test_Stepper_ambiguous_bare_id_raises():
    """A bare id shared by a node and a child must be disambiguated."""
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)
    with pytest.raises(ValueError, match="ambiguous"):
        stepper.deactivate(load_id)


def test_Stepper_activate_flips_back():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    stepper.activate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)

    assert len(_kinds(stepper.changes, "deactivated")) == 1
    react = _kinds(stepper.changes, "activated")
    assert len(react) == 1
    assert react[0].step == 1


def test_Stepper_fail_and_restore_labels():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.fail(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    stepper.restore(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)

    mutated = [c for c in stepper.changes if c.source == "mutation"]
    assert [c.kind for c in mutated] == ["failed", "restored"]
    assert all(c.component_id == load_id for c in mutated)


def test_Stepper_remove_branch_records_and_drops_component():
    net = _power_net()
    line_id = _line_id(net)
    stepper = Stepper(net, on_step_error="skip")

    stepper.step(dt_h=1.0)  # line present
    stepper.remove_branch(line_id)
    sr = stepper.step(dt_h=1.0)

    removed = _kinds(stepper.changes, "removed")
    assert [c.component_id for c in removed] == [line_id]
    assert removed[0].component_type == "Branch"
    assert removed[0].source == "mutation"
    if not sr.failed:
        assert sr.result.get(mm.PowerLine).empty


def test_Stepper_remove_node_cascade_is_detected():
    """Removing the load bus cascades: the bus is a mutation, the incident line
    and the orphaned load are detected removals."""
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    line_id = _line_id(net)
    load_bus_id = net.child_by_id(load_id).node_id
    stepper = Stepper(net, on_step_error="skip")

    stepper.remove_node(load_bus_id)
    stepper.step(dt_h=1.0)

    # Node and child ids collide, so key removals by (type, id).
    removed = {
        (c.component_type, c.component_id): c
        for c in _kinds(stepper.changes, "removed")
    }
    assert removed[("Node", load_bus_id)].source == "mutation"
    assert removed[("Branch", line_id)].source == "detected"
    assert removed[("Child", load_id)].source == "detected"


def test_Stepper_islanding_recorded_as_solver_source():
    """Deactivating the only line islands the load bus; the solver-side drop is
    recorded with source 'solver'."""
    net = _power_net()
    line_id = _line_id(net)
    stepper = Stepper(net)

    stepper.deactivate(line_id)
    stepper.step(dt_h=1.0)

    islanded = _kinds(stepper.changes, "islanded")
    assert islanded, "expected at least one islanded component"
    assert all(c.source == "solver" for c in islanded)
    assert all(c.step == 0 for c in islanded)


def test_Stepper_record_changes_disabled():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, record_changes=False)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)

    assert stepper.changes == []
    assert stepper.changes_df().empty


def test_Stepper_changes_df_columns():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)
    stepper.fail(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)

    df = stepper.changes_df()
    assert list(df.columns) == [
        "step",
        "t_h",
        "kind",
        "component_type",
        "component_id",
        "name",
        "source",
    ]
    assert (df["kind"] == "failed").any()


def test_Stepper_reset_clears_changes_and_reverts_mutations():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    assert stepper.changes

    stepper.reset()
    assert stepper.changes == []

    # Mutation reverted: stepping again records no deactivation.
    stepper.step(dt_h=1.0)
    assert _kinds(stepper.changes, "deactivated") == []


def test_Stepper_changes_not_capped_by_max_history():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, max_history=1)

    # Toggle the load every step; each flip is recorded at the following step.
    for i in range(6):
        if i % 2 == 0:
            stepper.deactivate(load_id, mm.PowerLoad)
        else:
            stepper.activate(load_id, mm.PowerLoad)
        stepper.step(dt_h=1.0)

    assert len(stepper.history) == 1  # results capped
    # Every flip retained even though history is capped at 1.
    flips = _kinds(stepper.changes, "deactivated") + _kinds(
        stepper.changes, "activated"
    )
    assert len(flips) == 6


def test_Stepper_unknown_mutation_id_raises():
    stepper = Stepper(_power_net())
    with pytest.raises(KeyError, match="not found in network"):
        stepper.deactivate(99999)


def test_NetworkChange_is_importable_and_frozen():
    change = NetworkChange(0, 0.0, "failed", "Child", 3, None, "mutation")
    assert change.kind == "failed"
    with pytest.raises(dataclasses.FrozenInstanceError):
        change.kind = "x"


def _raising_solve(*_a, **_kw):
    raise RuntimeError("boom")


class _UnsuccessfulResult:
    success = False
    network = None
    solver_status = "aborted"
    termination_condition = "infeasible"


def _p2h_net():
    """Two-grid (power + heat/water) network with one P2H compound."""
    net = mm.Network(mm.create_water_grid("heat"))
    w0 = net.node(mm.Junction(), child_ids=[net.child(mm.ConsumeHydrGrid(0.1))])
    w1 = net.node(mm.Junction())
    w2 = net.node(mm.Junction())
    w3 = net.node(mm.Junction(), child_ids=[net.child(mm.ExtHydrGrid(t_k=356))])
    net.branch(mm.WaterPipe(diameter_m=0.15, length_m=100), w1, w0)
    net.branch(mm.WaterPipe(diameter_m=0.15, length_m=200), w2, w3)

    power_grid = mm.create_power_grid("power")
    p0 = net.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[
            net.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))
        ],
    )
    p1 = net.node(
        mm.Bus(base_kv=1),
        grid=power_grid,
        child_ids=[net.child(mm.PowerLoad(p_mw=0.5, q_mvar=0))],
    )
    net.branch(
        mm.PowerLine(length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        p0,
        p1,
    )
    compound_id = net.compound(
        mm.PowerToHeat(0.02, 0.15, 300, 1.0),
        power_node_id=p1,
        heat_node_id=w2,
        heat_return_node_id=w1,
    )
    return net, compound_id


def test_Stepper_remove_auto_dispatches_by_type():
    net = _power_net()
    line_id = _line_id(net)
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, on_step_error="skip")

    stepper.remove(line_id)  # branch ids are unambiguous
    stepper.remove(load_id, mm.PowerLoad)  # typed dispatch to remove_child
    stepper.step(dt_h=1.0)

    removed = {
        (c.component_type, c.component_id): c
        for c in _kinds(stepper.changes, "removed")
    }
    assert removed[("Branch", line_id)].source == "mutation"
    assert removed[("Child", load_id)].source == "mutation"


def test_Stepper_remove_child_records_and_drops_component():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, on_step_error="skip")

    stepper.remove_child(load_id)
    sr = stepper.step(dt_h=1.0)

    removed = _kinds(stepper.changes, "removed")
    assert [(c.component_type, c.component_id) for c in removed] == [("Child", load_id)]
    assert removed[0].source == "mutation"
    if not sr.failed:
        assert sr.result.get(mm.PowerLoad).empty


def test_Stepper_remove_compound_records_mutation_and_cascade():
    net, compound_id = _p2h_net()
    stepper = Stepper(net, on_step_error="skip")

    stepper.remove_compound(compound_id)
    stepper.step(dt_h=1.0)

    removed = {
        (c.component_type, c.component_id): c
        for c in _kinds(stepper.changes, "removed")
    }
    assert removed[("Compound", compound_id)].source == "mutation"
    # internal subcomponents disappear as detected cascade removals
    cascade = [c for key, c in removed.items() if key != ("Compound", compound_id)]
    assert cascade
    assert all(c.source == "detected" for c in cascade)


def test_Stepper_rejoined_and_no_echo_for_explicit_toggle():
    """Reactivating a previously islanded line yields 'rejoined' events for the
    dependents; the explicitly toggled line itself is not double-reported as a
    solver islanded/rejoined echo."""
    net = _power_net()
    line_id = _line_id(net)
    stepper = Stepper(net)

    stepper.deactivate(line_id)
    stepper.step(dt_h=1.0)
    stepper.activate(line_id)
    stepper.step(dt_h=1.0)

    islanded_keys = {
        (c.component_type, c.component_id) for c in _kinds(stepper.changes, "islanded")
    }
    rejoined = _kinds(stepper.changes, "rejoined")
    rejoined_keys = {(c.component_type, c.component_id) for c in rejoined}

    assert islanded_keys, "dependents should be reported islanded"
    assert rejoined_keys == islanded_keys
    assert ("Branch", line_id) not in islanded_keys
    assert ("Branch", line_id) not in rejoined_keys
    assert all(c.source == "solver" and c.step == 1 for c in rejoined)


def test_Stepper_added_component_detected():
    net = _power_net()
    stepper = Stepper(net)
    stepper.step(dt_h=1.0)

    bus2 = stepper._work_net.nodes[1].id
    new_load = mx.create_power_load(stepper._work_net, bus2, p_mw=0.1, q_mvar=0.0)
    stepper.step(dt_h=1.0)

    added = _kinds(stepper.changes, "added")
    assert [(c.component_type, c.component_id) for c in added] == [("Child", new_load)]
    assert added[0].step == 1
    assert added[0].source == "detected"


def test_Stepper_added_inactive_component_not_double_reported():
    """A component added already-inactive records one 'added' event; the
    solver ignoring the inactive child is an echo, not 'islanded'."""
    net = _power_net()
    stepper = Stepper(net)
    stepper.step(dt_h=1.0)

    bus2 = stepper._work_net.nodes[1].id
    new_load = mx.create_power_load(stepper._work_net, bus2, p_mw=0.1, q_mvar=0.0)
    stepper._work_net.deactivate_by_id(mm.Child, new_load)
    stepper.step(dt_h=1.0)

    added = _kinds(stepper.changes, "added")
    assert [(c.component_type, c.component_id) for c in added] == [("Child", new_load)]
    assert _kinds(stepper.changes, "islanded") == []


def test_Stepper_added_disconnected_component_still_reports_islanded():
    """An active but disconnected addition is a genuine solver decision and
    must keep its 'islanded' event alongside 'added'."""
    net = _power_net()
    stepper = Stepper(net)
    stepper.step(dt_h=1.0)

    new_bus = mx.create_bus(stepper._work_net, base_kv=20.0)
    stepper.step(dt_h=1.0)

    added_keys = {
        (c.component_type, c.component_id) for c in _kinds(stepper.changes, "added")
    }
    islanded_keys = {
        (c.component_type, c.component_id) for c in _kinds(stepper.changes, "islanded")
    }
    assert ("Node", new_bus) in added_keys
    assert ("Node", new_bus) in islanded_keys


def test_Stepper_typed_removers_missing_id_friendly_error():
    stepper = Stepper(_power_net())
    for remover in (
        stepper.remove_branch,
        stepper.remove_node,
        stepper.remove_child,
        stepper.remove_compound,
    ):
        with pytest.raises(KeyError, match="not found in network"):
            remover(99999)


def test_Stepper_rollback_truncates_after_max_changes_trim():
    """Rollback of a raising step must undo the entries it appended even when
    a max_changes trim dropped older entries from the front during that step."""
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    # second load keeps the bus live, so a toggle records exactly one change
    mx.create_power_load(net, net.nodes[1].id, p_mw=0.1, q_mvar=0.0)
    stepper = Stepper(net, on_step_error="raise", max_changes=2)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    stepper.activate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    assert [c.kind for c in stepper.changes] == ["deactivated", "activated"]

    real_solve = stepper._solver.solve
    stepper._solver.solve = _raising_solve
    stepper.deactivate(load_id, mm.PowerLoad)
    with pytest.raises(RuntimeError, match="boom"):
        stepper.step(dt_h=1.0)

    # the trim during the failed step dropped the oldest entry for good; the
    # entry appended by the failed step itself is rolled back
    assert [c.kind for c in stepper.changes] == ["activated"]

    stepper._solver.solve = real_solve
    stepper.step(dt_h=1.0)
    assert [c.kind for c in stepper.changes] == ["activated", "deactivated"]
    assert stepper.changes[-1].step == 2
    assert stepper.changes[-1].source == "mutation"


def test_Stepper_changes_attributed_to_skipped_failed_step():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, on_step_error="skip")
    real_solve = stepper._solver.solve
    stepper._solver.solve = _raising_solve

    stepper.deactivate(load_id, mm.PowerLoad)
    sr = stepper.step(dt_h=1.0)
    assert sr.failed

    deact = _kinds(stepper.changes, "deactivated")
    assert len(deact) == 1
    assert deact[0].step == 0
    assert deact[0].source == "mutation"

    stepper._solver.solve = real_solve
    stepper.step(dt_h=1.0)
    # not re-recorded at the next (successful) step
    assert len(_kinds(stepper.changes, "deactivated")) == 1


def test_Stepper_changes_rolled_back_when_step_raises():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, on_step_error="raise")
    real_solve = stepper._solver.solve
    stepper._solver.solve = _raising_solve

    stepper.deactivate(load_id, mm.PowerLoad)
    with pytest.raises(RuntimeError, match="boom"):
        stepper.step(dt_h=1.0)

    # the step never happened: no change events, mutation stays pending
    assert stepper.changes == []
    assert stepper.step_count == 0

    stepper._solver.solve = real_solve
    stepper.step(dt_h=1.0)
    deact = _kinds(stepper.changes, "deactivated")
    assert len(deact) == 1
    assert deact[0].step == 0
    assert deact[0].source == "mutation"


def test_Stepper_noop_mutation_leaves_no_stale_annotation():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.step(dt_h=1.0)
    # no-op: already inactive relative to the recorded snapshot
    stepper.deactivate(load_id, mm.PowerLoad)
    assert stepper._pending_annotations == {}

    # a later non-mutation reactivation must be detected, not mislabeled by a
    # stale 'deactivated'/'mutation' annotation
    stepper._work_net.activate_by_id(mm.Child, load_id)
    stepper.step(dt_h=1.0)
    react = _kinds(stepper.changes, "activated")
    assert len(react) == 1
    assert react[0].source == "detected"


def test_Stepper_mutation_pair_cancels_cleanly():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.deactivate(load_id, mm.PowerLoad)
    stepper.activate(load_id, mm.PowerLoad)
    assert stepper._pending_annotations == {}
    stepper.step(dt_h=1.0)

    assert [c for c in stepper.changes if c.source == "mutation"] == []


def test_Stepper_success_false_result_is_step_failure_skip():
    stepper = Stepper(_power_net(), on_step_error="skip")
    stepper._solver.solve = lambda *a, **kw: _UnsuccessfulResult()

    sr = stepper.step(dt_h=1.0)

    assert sr.failed
    assert sr.result is None
    assert "success=False" in str(sr.error)
    assert "solver_status=aborted" in str(sr.error)
    assert len(stepper.state) == 0  # nothing pushed


def test_Stepper_success_false_result_is_step_failure_raise():
    stepper = Stepper(_power_net(), on_step_error="raise")
    stepper._solver.solve = lambda *a, **kw: _UnsuccessfulResult()

    with pytest.raises(RuntimeError, match="success=False"):
        stepper.step(dt_h=1.0)
    assert stepper.step_count == 0


def test_Stepper_skipped_failure_dt_carried_into_next_solve():
    stepper = Stepper(_power_net(), on_step_error="skip")
    stepper.step(dt_h=1.0)

    real_solve = stepper._solver.solve
    stepper._solver.solve = _raising_solve
    stepper.step(dt_h=2.0)  # fails: the 2 h interval must not vanish
    stepper._solver.solve = real_solve

    stepper.step(dt_h=1.0)
    assert math.isclose(stepper.state.dt_h, 3.0)  # 2 h backlog + 1 h
    stepper.step(dt_h=1.0)
    assert math.isclose(stepper.state.dt_h, 1.0)  # backlog consumed
    assert math.isclose(stepper.t_h, 5.0)  # wall clock unaffected


def test_Stepper_get_absolute_step_aligns_after_skipped_failure():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, on_step_error="skip")

    stepper.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 0.4})
    real_solve = stepper._solver.solve
    stepper._solver.solve = _raising_solve
    stepper.step(dt_h=1.0)  # step 1 fails
    stepper._solver.solve = real_solve
    sr = stepper.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 1.2})

    assert sr.step == 2
    assert math.isclose(stepper.get(load_id, "p_mw", step=2), 1.2, rel_tol=1e-9)
    assert math.isclose(stepper.get(load_id, "p_mw", step=0), 0.4, rel_tol=1e-9)
    assert stepper.get(load_id, "p_mw", step=1) is None  # failed step: no entry


def test_Stepper_max_changes_caps_recorded_changes():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net, max_changes=3)

    for i in range(6):
        if i % 2 == 0:
            stepper.deactivate(load_id, mm.PowerLoad)
        else:
            stepper.activate(load_id, mm.PowerLoad)
        stepper.step(dt_h=1.0)

    assert len(stepper.changes) == 3
    assert stepper.changes[-1].step == 5  # oldest dropped first


def test_Stepper_max_changes_invalid():
    with pytest.raises(ValueError, match="max_changes must be"):
        Stepper(_power_net(), max_changes=0)


def test_Stepper_typed_tuple_id_disambiguates_mutation():
    net = _power_net()
    load_id = child_id_by_type(net, mm.PowerLoad)
    stepper = Stepper(net)

    stepper.deactivate((mm.PowerLoad, load_id))
    stepper.step(dt_h=1.0)

    deact = _kinds(stepper.changes, "deactivated")
    assert [c.component_id for c in deact] == [load_id]
    assert deact[0].component_type == "Child"


def test_Stepper_overrides_ambiguous_raises_and_typed_key_resolves():
    net = _power_net()
    ext_id = child_id_by_type(net, mm.ExtPowerGrid)
    stepper = Stepper(net)

    # bare ext_id collides with a bus id; both carry a settable vm_pu
    with pytest.raises(ValueError, match="ambiguous"):
        stepper.step(dt_h=1.0, data_overrides={(ext_id, "vm_pu"): 1.05})

    sr = stepper.step(
        dt_h=1.0,
        data_overrides={((mm.ExtPowerGrid, ext_id), "vm_pu"): 1.05},
    )
    vm_slack = sr.result.get(mm.Bus)["vm_pu"].iloc[0]
    assert math.isclose(vm_slack, 1.05, rel_tol=1e-3)
