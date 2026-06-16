"""
Tests for :class:`monee.simulation.stepper.Stepper`.

Covers the externally-paced co-simulation API: variable ``dt_h``,
``data_overrides``, ``ts_index``, ``initial_state`` seeding, error
handling, parity with :func:`run_timeseries`, and the StepState extension.
"""

from __future__ import annotations

import math

import pytest

import monee.express as mx
import monee.model as mm
from monee.simulation import Stepper, StepState, TimeseriesData, run_timeseries
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
