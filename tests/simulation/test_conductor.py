"""
Tests for :class:`monee.simulation.conductor.Conductor`.

Covers the externally-paced co-simulation API: variable ``dt_h``,
``data_overrides``, ``ts_index``, ``initial_state`` seeding, error
handling, parity with :func:`run_timeseries`, and the StepState extension.
"""

from __future__ import annotations

import math

import pytest

import monee.express as mx
import monee.model as mm
from monee.simulation import Conductor, StepState, TimeseriesData, run_timeseries

# ── fixtures: a minimal power network and a battery+gas net for storage ──


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


# ── basic stepping ──────────────────────────────────────────────────────


def test_conductor_default_constructs_and_steps():
    cond = Conductor(_power_net())
    assert cond.step_count == 0
    assert cond.t_h == 0.0

    sr = cond.step(dt_h=1.0)

    assert sr.failed is False
    assert sr.result.success is True
    assert cond.step_count == 1
    assert cond.t_h == 1.0
    assert len(cond.history) == 1


def test_conductor_accumulates_t_h_across_variable_dt():
    cond = Conductor(_power_net())
    cond.step(dt_h=1.0 / 3600)  # 1 s
    cond.step(dt_h=4.0 / 3600)  # 4 s
    cond.step(dt_h=10.0 / 3600)  # 10 s
    cond.step(dt_h=1.0 / 60)  # 60 s

    assert math.isclose(cond.t_h, (1 + 4 + 10 + 60) / 3600, rel_tol=1e-12)
    assert cond.step_count == 4


def test_conductor_rejects_non_positive_dt():
    cond = Conductor(_power_net())
    with pytest.raises(ValueError, match="dt_h must be > 0"):
        cond.step(dt_h=0)
    with pytest.raises(ValueError, match="dt_h must be > 0"):
        cond.step(dt_h=-1.0)


def test_conductor_base_network_not_mutated():
    """Per-step deep-copy contract: the user's net is never touched."""
    net = _power_net()
    original_repr = repr(net.as_result_dataframe_dict())

    cond = Conductor(net)
    cond.step(dt_h=1.0)

    assert repr(net.as_result_dataframe_dict()) == original_repr


# ── data_overrides ──────────────────────────────────────────────────────


def test_conductor_data_overrides_apply_per_step():
    """Override changes drive the solve; the base net is untouched."""
    net = _power_net(p_load=1.0)
    cond = Conductor(net)

    load_id = next(c.id for c in net.childs if isinstance(c.model, mm.PowerLoad))

    sr_low = cond.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 0.2})
    sr_high = cond.step(dt_h=1.0, data_overrides={(load_id, "p_mw"): 2.0})

    p_from_low = sr_low.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
    p_from_high = sr_high.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
    assert p_from_high > p_from_low > 0
    assert math.isclose(p_from_low, 0.2, rel_tol=5e-2)
    assert math.isclose(p_from_high, 2.0, rel_tol=5e-2)


def test_conductor_overrides_unknown_id_raises():
    cond = Conductor(_power_net())
    with pytest.raises(KeyError, match="not found in network"):
        cond.step(dt_h=1.0, data_overrides={(99999, "p_mw"): 1.0})


def test_conductor_overrides_unknown_attr_raises():
    net = _power_net()
    load_id = next(c.id for c in net.childs if isinstance(c.model, mm.PowerLoad))
    cond = Conductor(net)
    with pytest.raises(AttributeError, match="attribute"):
        cond.step(dt_h=1.0, data_overrides={(load_id, "bogus_attr"): 0.5})


# ── TimeseriesData integration ──────────────────────────────────────────


def test_conductor_ts_index_applies_profile():
    net = _power_net()
    load_id = next(c.id for c in net.childs if isinstance(c.model, mm.PowerLoad))

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.2, 1.0, 2.0])

    cond = Conductor(net, timeseries_data=td)
    p_values = []
    for t in range(3):
        sr = cond.step(dt_h=1.0, ts_index=t)
        p_values.append(sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0])

    assert p_values[0] < p_values[1] < p_values[2]
    for actual, target in zip(p_values, [0.2, 1.0, 2.0]):
        assert math.isclose(actual, target, rel_tol=5e-2)


def test_conductor_overrides_win_over_ts_data():
    net = _power_net()
    load_id = next(c.id for c in net.childs if isinstance(c.model, mm.PowerLoad))

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.2])

    cond = Conductor(net, timeseries_data=td)
    sr = cond.step(
        dt_h=1.0,
        ts_index=0,
        data_overrides={(load_id, "p_mw"): 1.5},
    )

    p = sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
    assert math.isclose(p, 1.5, rel_tol=5e-2), f"override should win, got p={p}"


# ── initial_state seeding ───────────────────────────────────────────────


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


# ── error handling ──────────────────────────────────────────────────────


def test_conductor_on_step_error_skip_records_failure():
    """``on_step_error='skip'`` swallows solver exceptions and records them."""
    cond = Conductor(_power_net(), on_step_error="skip")

    def _exploding_solve(*a, **kw):
        raise RuntimeError("simulated solver failure")

    cond._solver.solve = _exploding_solve

    sr = cond.step(dt_h=1.0)
    assert sr.failed is True
    assert isinstance(sr.error, RuntimeError)
    assert sr.result is None
    assert cond.step_count == 1
    assert cond.t_h == 1.0  # advances even on failure


def test_conductor_on_step_error_raise_propagates():
    cond = Conductor(_power_net(), on_step_error="raise")

    def _exploding_solve(*a, **kw):
        raise RuntimeError("boom")

    cond._solver.solve = _exploding_solve

    with pytest.raises(RuntimeError, match="boom"):
        cond.step(dt_h=1.0)
    assert cond.step_count == 0  # not recorded


def test_conductor_invalid_on_step_error_value():
    with pytest.raises(ValueError, match="on_step_error must be"):
        Conductor(_power_net(), on_step_error="silent")


# ── reset & context manager ─────────────────────────────────────────────


def test_conductor_reset_clears_history_and_state():
    cond = Conductor(_power_net())
    cond.step(dt_h=1.0)
    cond.step(dt_h=2.0)
    assert cond.step_count == 2
    assert cond.t_h == 3.0
    assert len(cond.state) == 2

    cond.reset()

    assert cond.step_count == 0
    assert cond.t_h == 0.0
    assert len(cond.state) == 0


def test_conductor_reset_can_replace_initial_state():
    cond = Conductor(_power_net(), initial_state={(0, "x"): 1.0})
    assert cond.state.get(0, "x") == 1.0
    cond.reset(initial_state={(0, "x"): 99.0})
    assert cond.state.get(0, "x") == 99.0


def test_conductor_context_manager():
    with Conductor(_power_net()) as cond:
        cond.step(dt_h=1.0)
        assert cond.step_count == 1
    # Should exit cleanly; no resource leak assertion possible without
    # introspection of the solver backend.


# ── parity with run_timeseries ──────────────────────────────────────────


def test_conductor_matches_run_timeseries_on_fixed_grid():
    """Same network + same TD + same dt_h → identical objectives across steps."""
    net = _power_net()
    load_id = next(c.id for c in net.childs if isinstance(c.model, mm.PowerLoad))
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.5, 1.0, 1.5])

    ts_result = run_timeseries(net, td, steps=3)
    ts_pvals = [
        sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0]
        for sr in ts_result.step_results
    ]

    cond = Conductor(net, timeseries_data=td)
    cond_pvals = []
    for t in range(3):
        sr = cond.step(dt_h=1.0, ts_index=t)
        cond_pvals.append(sr.result.get(mm.PowerLine)["p_from_mw"].iloc[0])

    for a, b in zip(ts_pvals, cond_pvals):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)


def test_conductor_to_timeseries_result():
    cond = Conductor(_power_net())
    for _ in range(3):
        cond.step(dt_h=1.0)
    res = cond.to_timeseries_result()
    assert len(res.step_results) == 3
    # Should expose the same get_result_for API as the timeseries runner.
    p_series = res.get_result_for(mm.PowerLine, "p_from_mw")
    assert len(p_series) == 3


# ── repr ────────────────────────────────────────────────────────────────


def test_conductor_repr_shows_progress():
    cond = Conductor(_power_net())
    assert "steps=0" in repr(cond)
    cond.step(dt_h=0.5)
    assert "steps=1" in repr(cond)
    assert "t_h=0.5" in repr(cond)
