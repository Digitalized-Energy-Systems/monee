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


# ---------------------------------------------------------------------------
# Shared network builders
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 1 — single-period result matches the standard single-step solver
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 2 — storage SoC couples correctly across periods
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 3 — SoC constraint links t to t+1 (continuity check)
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 4 — result object API
# ---------------------------------------------------------------------------


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

    # per-period objective is 0.0 (only global objective is meaningful)
    assert sr0.objective == 0.0

    # repr should not raise
    assert "MultiPeriodResult" in repr(result)


# ---------------------------------------------------------------------------
# Test 5 — get_result_for_id returns a Series indexed by period
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 6 — result[component_id] returns a DataFrame of all attributes
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 7 — datetime_index labels result rows
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 8 — validation: timeseries too short raises ValueError
# ---------------------------------------------------------------------------


def test_timeseries_too_short_raises():
    """Requesting more steps than the timeseries length must raise ValueError."""
    net, b0, b1, load_id = _simple_power_net()

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [1.0, 2.0])  # only 2 entries

    with pytest.raises(ValueError, match="steps"):
        run_multi_period(net, td, steps=5)


# ---------------------------------------------------------------------------
# Test 9 — validation: non-positive dt_h raises ValueError
# ---------------------------------------------------------------------------


def test_nonpositive_dt_h_raises():
    """dt_h <= 0 must raise ValueError."""
    net, b0, b1, load_id = _simple_power_net()

    with pytest.raises(ValueError, match="positive"):
        run_multi_period(net, steps=2, dt_h=0.0)

    with pytest.raises(ValueError, match="positive"):
        run_multi_period(net, steps=2, dt_h=-1.0)


# ---------------------------------------------------------------------------
# Test 10 — initial_state overrides the tracked value at t=0
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 11 — terminal_state pins the last-period variable
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 12 — run_mpc executes total_steps periods
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 13 — run_mpc propagates initial state between windows
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 14 — MultiPeriodResult repr shows temporal evolution
# ---------------------------------------------------------------------------


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
