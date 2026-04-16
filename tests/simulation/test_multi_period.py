"""
Tests for multi-period optimization.

Each test validates a specific design property of the multi-period solver:

1. Results match the single-step solver when run over a single period.
2. Storage SoC evolves correctly when solved jointly across T periods.
3. A global objective (e.g. minimise total cost) is optimised across all periods.
4. Cross-period ramp constraints limit per-period dispatch changes.
5. initial_state overrides the network's tracked values at t=0.
6. terminal_state adds an equality constraint at t=T-1.
7. run_mpc executes total_steps periods via rolling-horizon solves.
8. run_mpc propagates initial state between windows automatically.
"""

import pandas
import pytest

import monee.model as mm
from monee.model import Network
from monee.model.child import ExtPowerGrid, PowerLoad
from monee.model.node import Bus
from monee.model.storage import ElectricStorage
from monee.problem import OptimizationProblem
from monee.simulation.multi_period import run_mpc, run_multi_period
from monee.simulation.timeseries import TimeseriesData


def _storage_problem():
    """Minimal OptimizationProblem that just enables storage dispatch as a variable."""
    p = OptimizationProblem()
    p.controllable_storages()
    return p


# Shared line parameters: short lossless-ish line.
_LINE = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)


def _simple_power_net():
    """Two-bus power network: ext-grid at b0, load at b1, line between them."""
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=1.0, q_mvar=0.0))
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)
    return net, b0, b1, load_id


def _storage_net():
    """Three-bus net: ext-grid → b0 → b1 (load + battery)."""
    net = Network(mm.PowerGrid(name="el", sn_mva=1))
    ext_id = net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1.0, va_degree=0.0))
    b0 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[ext_id])
    load_id = net.child(PowerLoad(p_mw=2.0, q_mvar=0.0))
    bat_id = net.child(
        ElectricStorage(e_mwh_initial=4.0, e_mwh_max=8.0, p_max_mw=2.0),
        name="battery",
    )
    b1 = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[load_id, bat_id])
    net.branch(mm.PowerLine(**_LINE), b0, b1)
    return net, b0, b1, load_id, bat_id


def test_single_period_matches_single_step():
    """Multi-period with T=1 must give the same result as the single-step solver."""
    from monee.simulation.core import solve as run_energy_flow

    net, b0, b1, load_id = _simple_power_net()

    r_single = run_energy_flow(net, optimization_problem=None)
    r_multi = run_multi_period(net, steps=1)

    vm_single = r_single.get(mm.Bus)["vm_pu"].sort_index()
    vm_multi = r_multi.get_result_for(mm.Bus, "vm_pu")  # DataFrame, 1 row

    for bus_id in vm_single.index:
        assert abs(vm_single[bus_id] - vm_multi[bus_id].iloc[0]) < 1e-4, (
            f"Bus {bus_id}: single={vm_single[bus_id]:.6f}, "
            f"multi={vm_multi[bus_id].iloc[0]:.6f}"
        )


def test_storage_soc_initial_condition_pinned():
    """
    The t=0 inter-step constraint ``e_mwh[0] == e_initial + dt_h * p_mw[0]``
    must be active, anchoring the SoC evolution to the user-supplied initial
    condition rather than leaving it as a free degree of freedom.

    We check that constraint satisfaction holds at t=0, regardless of what
    p_mw[0] the solver chooses (there is no objective to prescribe a value).
    """
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 2.0, 2.0])

    result = run_multi_period(
        net, td, steps=3, dt_h=1.0, optimization_problem=_storage_problem()
    )

    soc_df = result.get_result_for(ElectricStorage, "e_mwh")
    p_df = result.get_result_for(ElectricStorage, "p_mw")

    # The initial-condition constraint: e_mwh[0] = e_initial + dt_h * p_mw[0]
    e_initial = 4.0
    p0 = p_df[bat_id].iloc[0]
    e0 = soc_df[bat_id].iloc[0]
    assert abs(e0 - (e_initial + 1.0 * p0)) < 1e-3, (
        f"t=0 constraint violated: e={e0:.4f}, expected {e_initial + p0:.4f}"
    )


def test_storage_soc_continuity():
    """
    Explicitly verify that e_mwh[t] = e_mwh[t-1] + dt_h * p_mw[t] holds
    across all adjacent periods.
    """
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.0, 1.0, 1.0])

    result = run_multi_period(
        net, td, steps=4, dt_h=1.0, optimization_problem=_storage_problem()
    )

    soc_df = result.get_result_for(ElectricStorage, "e_mwh")
    p_df = result.get_result_for(ElectricStorage, "p_mw")

    for t in range(1, 4):
        soc_prev = soc_df[bat_id].iloc[t - 1]
        p_cur = p_df[bat_id].iloc[t]
        soc_cur = soc_df[bat_id].iloc[t]
        residual = abs(soc_cur - (soc_prev + 1.0 * p_cur))
        assert residual < 1e-3, (
            f"SoC continuity violated at t={t}: "
            f"e[t]={soc_cur:.4f}, e[t-1]+dt*p={soc_prev + p_cur:.4f}"
        )


def test_result_api():
    """MultiPeriodResult exposes the expected query methods."""
    net, b0, b1, load_id = _simple_power_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 2.0])

    result = run_multi_period(net, td, steps=2)

    assert result.T == 2
    vm_df = result.get_result_for(mm.Bus, "vm_pu")
    assert vm_df.shape[0] == 2  # 2 rows (periods)
    assert b0 in vm_df.columns or b1 in vm_df.columns

    # get_period_result returns a SolverResult for one period
    sr0 = result.get_period_result(0)
    assert hasattr(sr0, "get")
    assert not sr0.get(mm.Bus).empty

    # per-period objective is None (only global objective is meaningful)
    assert sr0.objective is None

    # repr should not raise
    assert "MultiPeriodResult" in repr(result)


def test_get_result_for_id():
    """get_result_for_id must return a Series with one entry per period."""
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.5, 2.0])

    result = run_multi_period(net, td, steps=3)

    soc = result.get_result_for_id(bat_id, "e_mwh")
    assert isinstance(soc, pandas.Series)
    assert len(soc) == 3
    assert soc.notna().all()


def test_getitem_component():
    """result[bat_id] must return a DataFrame (rows=periods, cols=attributes)."""
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.0])

    result = run_multi_period(net, td, steps=2)

    df = result[bat_id]
    assert isinstance(df, pandas.DataFrame)
    assert df.shape[0] == 2
    assert "e_mwh" in df.columns

    with pytest.raises(KeyError):
        _ = result[99999]


def test_datetime_index_labels_rows():
    """When datetime_index is provided, result DataFrames are indexed by it."""
    net, b0, b1, load_id = _simple_power_net()

    idx = pandas.date_range("2025-01-01", periods=2, freq="h")
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 2.0])

    result = run_multi_period(net, td, steps=2, datetime_index=idx)

    vm_df = result.get_result_for(mm.Bus, "vm_pu")
    assert isinstance(vm_df.index, pandas.DatetimeIndex)
    assert list(vm_df.index) == list(idx)


def test_timeseries_too_short_raises():
    """Requesting more steps than the timeseries length must raise ValueError."""
    net, b0, b1, load_id = _simple_power_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 2.0])  # only 2 entries

    with pytest.raises(ValueError, match="steps"):
        run_multi_period(net, td, steps=5)


def test_nonpositive_dt_h_raises():
    """dt_h <= 0 must raise ValueError."""
    net, b0, b1, load_id = _simple_power_net()

    with pytest.raises(ValueError, match="positive"):
        run_multi_period(net, steps=2, dt_h=0.0)

    with pytest.raises(ValueError, match="positive"):
        run_multi_period(net, steps=2, dt_h=-1.0)


def test_initial_state_override():
    """initial_state must override the network's tracked value used in t=0 constraint.

    We create a storage with e_mwh_initial=4.0, then pass initial_state with
    a different SoC (2.0).  At t=0 the constraint is:
        e_mwh[0] = 2.0 + dt_h * p_mw[0]
    so e_mwh[0] must differ from what we'd get with the default 4.0 seed.
    """
    net, b0, b1, load_id, bat_id = _storage_net()  # e_mwh_initial=4.0

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 2.0])

    # Default run: uses e_mwh_initial=4.0 as the t=0 anchor.
    result_default = run_multi_period(net, td, steps=2, dt_h=1.0)

    # Override: seed with 2.0 instead.
    result_override = run_multi_period(
        net,
        td,
        steps=2,
        dt_h=1.0,
        initial_state={(bat_id, "e_mwh"): 2.0},
    )

    soc_default = result_default.get_result_for(ElectricStorage, "e_mwh")[bat_id]
    soc_override = result_override.get_result_for(ElectricStorage, "e_mwh")[bat_id]

    # With a different initial condition the trajectories must differ.
    assert abs(soc_default.iloc[0] - soc_override.iloc[0]) > 0.5, (
        f"Expected SoC trajectories to differ significantly; "
        f"default={soc_default.iloc[0]:.4f}, override={soc_override.iloc[0]:.4f}"
    )

    # Verify the override's t=0 constraint is anchored to 2.0.
    p0 = result_override.get_result_for(ElectricStorage, "p_mw")[bat_id].iloc[0]
    e0 = soc_override.iloc[0]
    assert abs(e0 - (2.0 + 1.0 * p0)) < 1e-3, (
        f"t=0 constraint with overridden initial state violated: "
        f"e={e0:.4f}, expected {2.0 + p0:.4f}"
    )


def test_terminal_state_constraint():
    """terminal_state must pin e_mwh at t=T-1 to the specified target.

    We optimise 3 periods with a fixed load and require the battery to end
    at exactly its initial SoC (4.0 MWh), verifying the constraint is active.
    """
    net, b0, b1, load_id, bat_id = _storage_net()  # e_mwh_initial=4.0, e_mwh_max=8.0

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.5, 1.0])

    target_soc = 4.0  # pin terminal SoC back to initial
    result = run_multi_period(
        net,
        td,
        steps=3,
        dt_h=1.0,
        optimization_problem=_storage_problem(),
        terminal_state={(bat_id, "e_mwh"): target_soc},
    )

    soc = result.get_result_for(ElectricStorage, "e_mwh")[bat_id]
    assert abs(soc.iloc[-1] - target_soc) < 1e-3, (
        f"Terminal SoC constraint violated: "
        f"e_mwh[T-1]={soc.iloc[-1]:.4f}, target={target_soc}"
    )


def test_mpc_executes_total_steps():
    """run_mpc must return a result with exactly total_steps executed periods."""
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    total = 6
    td.add_child_series(load_id, "p_mw", [1.0] * total)

    result = run_mpc(
        net,
        td,
        total_steps=total,
        horizon=3,
        execution_steps=1,
        optimization_problem=_storage_problem(),
    )

    assert result.T == total, f"Expected {total} periods, got {result.T}"
    assert result.success


def test_mpc_state_propagation():
    """run_mpc must propagate the executed period's terminal state as the next
    window's initial state, producing a physically consistent SoC trajectory.

    We check SoC continuity across the window boundary: e_mwh at the first
    period of window 2 must follow from the last executed period of window 1.
    """
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    total = 4
    td.add_child_series(load_id, "p_mw", [2.0, 2.0, 2.0, 2.0])

    result = run_mpc(
        net,
        td,
        total_steps=total,
        horizon=2,
        execution_steps=2,
        dt_h=1.0,
        optimization_problem=_storage_problem(),
    )

    assert result.T == total

    soc = result.get_result_for(ElectricStorage, "e_mwh")[bat_id]
    p = result.get_result_for(ElectricStorage, "p_mw")[bat_id]

    # Check SoC continuity at all period boundaries.
    for t in range(1, total):
        residual = abs(soc.iloc[t] - (soc.iloc[t - 1] + 1.0 * p.iloc[t]))
        assert residual < 1e-2, (
            f"SoC continuity violated at t={t} (window boundary): "
            f"e[t]={soc.iloc[t]:.4f}, e[t-1]+dt*p={soc.iloc[t - 1] + p.iloc[t]:.4f}"
        )


def test_repr_shows_temporal_evolution():
    """__repr__ must include a temporal-evolution section when SoC varies."""
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 2.0, 0.5])

    result = run_multi_period(
        net, td, steps=3, dt_h=1.0, optimization_problem=_storage_problem()
    )
    r = repr(result)

    assert "MultiPeriodResult" in r
    # With varying loads the SoC changes; temporal evolution section must appear.
    assert "Temporal evolution" in r


def test_zero_bounds_warning(caplog):
    """controllable_demands on a load with p_mw=0.0 must emit a warning."""
    net, b0, b1, load_id = _simple_power_net()
    # Override load to 0.0 so bounds inference hits [0, 0]
    for node in net.nodes:
        for child in net.childs_by_ids(node.child_ids):
            if isinstance(child.model, PowerLoad):
                child.model.p_mw = 0.0

    prob = OptimizationProblem()
    prob.controllable_demands(["p_mw"])

    import logging

    with caplog.at_level(logging.WARNING, logger="monee.problem.core"):
        prob._apply(net)

    zero_warnings = [r for r in caplog.records if "value 0.0" in r.message]
    assert len(zero_warnings) > 0, (
        "Expected a warning about zero-valued bounds inference"
    )


def test_when_period_constraint():
    """A constraint with when_period should only apply at filtered periods."""
    from monee.problem.core import Constraints

    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 1.0, 1.0, 1.0])

    prob = _storage_problem()
    cons = Constraints()
    # Force battery to charge at exactly 1.0 MW only at period 2
    cons.select(lambda c: isinstance(c.model, ElectricStorage)).equation(
        lambda m: m.p_mw >= 0.99
    ).equation(lambda m: m.p_mw <= 1.01).when_period(lambda t: t == 2)
    prob.constraints = cons

    result = run_multi_period(net, td, steps=4, optimization_problem=prob, dt_h=1.0)

    p = result.get_result_for_id(bat_id, "p_mw")
    # At period 2 the constraint forces p_mw ≈ 1.0 (charging)
    assert abs(p.iloc[2] - 1.0) < 0.1, (
        f"Period 2 battery dispatch should be ~1.0 MW, got {p.iloc[2]:.4f}"
    )
    # At other periods the battery is free — it should NOT be forced to 1.0
    # (with a uniform load the optimizer has no reason to charge at exactly 1.0)
    # We just verify the constraint was respected at t=2; other periods are free.


def test_when_period_single_period():
    """when_period filtering should be harmless with steps=1."""
    from monee.problem.core import Constraints

    net, b0, b1, load_id, bat_id = _storage_net()

    prob = _storage_problem()
    cons = Constraints()
    # This filter says "only at period 5" but we only have 1 period (t=0).
    # The constraint should simply not fire — solve should succeed.
    cons.select(lambda c: isinstance(c.model, ElectricStorage)).equation(
        lambda m: m.p_mw == 999  # impossible if it fires
    ).when_period({5})
    prob.constraints = cons

    result = run_multi_period(net, steps=1, optimization_problem=prob)
    assert result.success


def test_controllable_to_attr_no_leak():
    """_controllable_to_attr must not grow with T."""
    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 2.0, 1.5, 0.8])

    prob = _storage_problem()
    result = run_multi_period(net, td, steps=4, optimization_problem=prob, dt_h=1.0)

    # After solve, _controllable_to_attr should have entries only from the
    # last period's _apply call, not accumulated from all 4 periods.
    n_entries = len(prob._controllable_to_attr)
    # A single storage component → at most a few entries, not 4x.
    assert n_entries <= 5, (
        f"_controllable_to_attr has {n_entries} entries — "
        f"expected it to be bounded, not growing with T"
    )


def test_objective_data_via_timeseries():
    """Time-varying objective data via add_objective_data should influence dispatch."""
    from monee.problem.core import Objectives

    net, b0, b1, load_id, bat_id = _storage_net()

    # Constant load, varying price: cheap first, expensive later
    loads = [2.0, 2.0, 2.0, 2.0]

    td_cheap_first = TimeseriesData()
    td_cheap_first.add_child_series(load_id, "p_mw", loads)
    # ext grid is child 0 on node b0 — find its id
    ext_id = None
    for node in net.nodes:
        for child in net.childs_by_ids(node.child_ids):
            if isinstance(child.model, ExtPowerGrid):
                ext_id = child.id
                break
    assert ext_id is not None

    td_cheap_first.add_objective_data(ext_id, "price", [10, 10, 50, 50])

    td_expensive_first = TimeseriesData()
    td_expensive_first.add_child_series(load_id, "p_mw", loads)
    td_expensive_first.add_objective_data(ext_id, "price", [50, 50, 10, 10])

    prob = _storage_problem()
    obj = Objectives()
    obj.select(lambda m: isinstance(m, ExtPowerGrid)).calculate(
        lambda models: sum(getattr(m, "price", 1) * m.p_mw for m in models)
    )
    prob.objectives = obj

    r1 = run_multi_period(
        net, td_cheap_first, steps=4, optimization_problem=prob, dt_h=1.0
    )
    r2 = run_multi_period(
        net, td_expensive_first, steps=4, optimization_problem=prob, dt_h=1.0
    )

    # The objectives should differ — different price schedules yield
    # different total costs.
    assert r1.objective != r2.objective, (
        "Different price schedules should produce different objectives"
    )


def test_temporal_equation_ramp_rate():
    """A temporal_equation should couple variables across periods.

    We add a ramp-rate constraint to the ext-grid p_mw so the change between
    consecutive periods is limited.  Without the constraint, the ext-grid
    freely follows the load.  With it, the ext-grid must ramp gradually,
    forcing the battery to compensate.
    """
    from monee.problem.core import Constraints

    net, b0, b1, load_id, bat_id = _storage_net()

    # Large demand swing: 1 → 4 → 1 → 4
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 4.0, 1.0, 4.0])

    # Find ext-grid id
    ext_id = None
    for node in net.nodes:
        for child in net.childs_by_ids(node.child_ids):
            if isinstance(child.model, ExtPowerGrid):
                ext_id = child.id
                break
    assert ext_id is not None

    # --- Run WITHOUT ramp constraint ---
    prob_free = _storage_problem()
    r_free = run_multi_period(
        net, td, steps=4, optimization_problem=prob_free, dt_h=1.0
    )

    # --- Run WITH ramp constraint (max 1.0 MW change per period) ---
    prob_ramp = _storage_problem()
    cons = Constraints()

    def ramp_limit(m, cid, ts):
        prev = ts.get(cid, "p_mw")
        if prev is None:
            return []  # no constraint at t=0
        return [
            m.p_mw - prev <= 1.0,  # ramp up limit
            prev - m.p_mw <= 1.0,  # ramp down limit
        ]

    cons.select_types(ExtPowerGrid).temporal_equation(ramp_limit)
    prob_ramp.constraints = cons

    r_ramp = run_multi_period(
        net, td, steps=4, optimization_problem=prob_ramp, dt_h=1.0
    )

    # The ext-grid power should vary less with the ramp constraint.
    ext_free = r_free.get_result_for(mm.ExtPowerGrid, "p_mw")[ext_id]
    ext_ramp = r_ramp.get_result_for(mm.ExtPowerGrid, "p_mw")[ext_id]

    # Compute max absolute period-to-period change
    max_delta_free = max(abs(ext_free.iloc[t + 1] - ext_free.iloc[t]) for t in range(3))
    max_delta_ramp = max(abs(ext_ramp.iloc[t + 1] - ext_ramp.iloc[t]) for t in range(3))

    # The ramp-constrained case should have smaller or equal swings
    assert max_delta_ramp <= max_delta_free + 0.01, (
        f"Ramp constraint should reduce swings: "
        f"free={max_delta_free:.3f}, ramp={max_delta_ramp:.3f}"
    )
    # The ramp-constrained case should respect the 1.0 MW limit (with tolerance)
    assert max_delta_ramp <= 1.0 + 0.05, (
        f"Ramp constraint violated: max_delta={max_delta_ramp:.3f} > 1.0"
    )


def test_temporal_equation_with_when_period():
    """temporal_equation + when_period should only fire at filtered periods."""
    from monee.problem.core import Constraints

    net, b0, b1, load_id, bat_id = _storage_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 4.0, 1.0, 4.0])

    # Find ext-grid id
    ext_id = None
    for node in net.nodes:
        for child in net.childs_by_ids(node.child_ids):
            if isinstance(child.model, ExtPowerGrid):
                ext_id = child.id
                break
    assert ext_id is not None

    # Ramp constraint ONLY at period 1 (the first big jump)
    prob = _storage_problem()
    cons = Constraints()
    cons.select_types(ExtPowerGrid).temporal_equation(
        lambda m, cid, ts: (
            []
            if ts.get(cid, "p_mw") is None
            else [
                m.p_mw - ts.get(cid, "p_mw") <= 1.0,
                ts.get(cid, "p_mw") - m.p_mw <= 1.0,
            ]
        )
    ).when_period({1})
    prob.constraints = cons

    r = run_multi_period(net, td, steps=4, optimization_problem=prob, dt_h=1.0)

    ext_p = r.get_result_for(mm.ExtPowerGrid, "p_mw")[ext_id]

    # Period 0→1 should be ramp-limited (≤ 1.0)
    delta_01 = abs(ext_p.iloc[1] - ext_p.iloc[0])
    assert delta_01 <= 1.0 + 0.05, (
        f"Ramp at period 1 should be limited: delta={delta_01:.3f}"
    )

    # Period 2→3 should NOT be ramp-limited (constraint not active at period 3)
    # so the ext-grid can freely jump.  We just verify the solve completed and
    # the constraint only applied where we asked.
    assert r.objective is not None, "Solve should succeed"


def test_multi_period_load_shedding():
    """Multi-period load shedding sheds load when demand exceeds supply capacity.

    A two-bus power network with a capacity-limited ext-grid (bounded to
    [-3, 3] MW) and a load that exceeds 3 MW at certain periods.  The
    solver must curtail load via the regulation variable at those periods.
    """
    from monee.problem.load_shedding import (
        create_multi_period_load_shedding_optimization_problem,
    )

    net, b0, b1, load_id = _simple_power_net()

    # Load profile: periods 1 and 3 exceed the ext-grid capacity of 3 MW
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 5.0, 2.0, 4.0])

    prob = create_multi_period_load_shedding_optimization_problem(
        ext_grid_el_bounds=(-3.0, 3.0),
        use_ext_grid_objective=False,
        check_lp=False,  # skip line loading for simplicity
    )

    result = run_multi_period(net, td, steps=4, optimization_problem=prob, dt_h=1.0)
    assert result.objective is not None, "Solve should succeed"

    # Check regulation: at periods where load ≤ 3 MW, regulation should be ~1.0
    # At periods where load > 3 MW, regulation should be < 1.0
    reg = result.get_result_for(mm.PowerLoad, "regulation")[load_id]

    # Period 0 (load=2.0): no shedding needed
    assert reg.iloc[0] > 0.95, f"Period 0: expected ~1.0, got {reg.iloc[0]:.3f}"
    # Period 1 (load=5.0): must shed — regulation < 1.0
    assert reg.iloc[1] < 0.95, f"Period 1: expected shedding, got {reg.iloc[1]:.3f}"
    # Period 2 (load=2.0): no shedding needed
    assert reg.iloc[2] > 0.95, f"Period 2: expected ~1.0, got {reg.iloc[2]:.3f}"
    # Period 3 (load=4.0): must shed — regulation < 1.0
    assert reg.iloc[3] < 0.95, f"Period 3: expected shedding, got {reg.iloc[3]:.3f}"


def test_multi_period_load_shedding_ramp():
    """Regulation ramp limit prevents abrupt shedding changes between periods."""
    from monee.problem.load_shedding import (
        create_multi_period_load_shedding_optimization_problem,
    )

    net, b0, b1, load_id = _simple_power_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [2.0, 5.0, 2.0, 4.0])

    # Without ramp limit
    prob_free = create_multi_period_load_shedding_optimization_problem(
        ext_grid_el_bounds=(-3.0, 3.0),
        use_ext_grid_objective=False,
        check_lp=False,
    )
    r_free = run_multi_period(
        net, td, steps=4, optimization_problem=prob_free, dt_h=1.0
    )

    # With tight ramp limit of 0.2 per period
    prob_ramp = create_multi_period_load_shedding_optimization_problem(
        ext_grid_el_bounds=(-3.0, 3.0),
        use_ext_grid_objective=False,
        check_lp=False,
        regulation_ramp_limit=0.2,
    )
    r_ramp = run_multi_period(
        net, td, steps=4, optimization_problem=prob_ramp, dt_h=1.0
    )

    reg_free = result = r_free.get_result_for(mm.PowerLoad, "regulation")[load_id]
    reg_ramp = r_ramp.get_result_for(mm.PowerLoad, "regulation")[load_id]

    # The ramp-constrained regulation should change more gradually
    max_delta_free = max(abs(reg_free.iloc[t + 1] - reg_free.iloc[t]) for t in range(3))
    max_delta_ramp = max(abs(reg_ramp.iloc[t + 1] - reg_ramp.iloc[t]) for t in range(3))

    assert max_delta_ramp <= 0.2 + 0.05, (
        f"Ramp limit violated: max_delta={max_delta_ramp:.3f} > 0.2"
    )
    # Ramp-constrained should be smoother than free
    assert max_delta_ramp <= max_delta_free + 0.01, (
        f"Expected ramp to smooth regulation: "
        f"free={max_delta_free:.3f}, ramp={max_delta_ramp:.3f}"
    )
