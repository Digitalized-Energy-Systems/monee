"""Tests for monee.simulation.timeseries."""

import logging
import math
import warnings

import pandas
import pytest

import monee.express as mx
import monee.model as mm
from monee import run_energy_flow
from monee.model import Network, Var
from monee.model.branch import PowerLine
from monee.model.child import ExtHydrGrid, ExtPowerGrid, PowerLoad, Sink
from monee.model.grid import PowerGrid
from monee.model.node import Bus, Junction
from monee.simulation.timeseries import StepHook, StepResult, TimeseriesData, run
from monee.solver import GEKKOSolver


def _el_net(load_p_mw=1.0, load_name="load"):
    """Simple 2-bus radial power network."""
    net = Network(PowerGrid(name="power", sn_mva=1))
    n0 = net.node(
        Bus(base_kv=1),
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=mm.EL,
    )
    n1 = net.node(
        Bus(base_kv=1),
        child_ids=[net.child(PowerLoad(p_mw=load_p_mw, q_mvar=0), name=load_name)],
        grid=mm.EL,
    )
    net.branch(
        PowerLine(length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, parallel=1),
        n0,
        n1,
    )
    return net


def _gas_net():
    """Simple 3-junction gas network: ext grid → pipe → sink."""
    net = Network()
    net.activate_grid(grid=mm.GAS)
    n0 = net.node(Junction(), mm.GAS, child_ids=[net.child(ExtHydrGrid())])
    n1 = net.node(Junction(), mm.GAS)
    n2 = net.node(Junction(), mm.GAS, child_ids=[net.child(Sink(mass_flow_kgs=0.1))])
    net.branch(mm.GasPipe(diameter_m=0.5, length_m=500, temperature_ext_k=300), n0, n1)
    net.branch(mm.GasPipe(diameter_m=0.5, length_m=500, temperature_ext_k=300), n1, n2)
    return net


def _make_failing_solve(fail_on_call, message):
    """Return a solve replacement raising RuntimeError on the given call number."""
    call_count = [0]

    def failing_solve(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == fail_on_call:
            raise RuntimeError(message)
        from monee.simulation.core import solve as real_solve

        return real_solve(*args, **kwargs)

    return failing_solve


def test_series_length_validation_mismatch_raises():
    # GIVEN
    td = TimeseriesData()
    td.add_child_series(1, "p_mw", [1.0, 2.0, 3.0])

    # WHEN / THEN
    with pytest.raises(ValueError, match="length"):
        td.add_child_series(2, "p_mw", [1.0, 2.0])


def test_series_length_inferred_after_first_add():
    # GIVEN
    td = TimeseriesData()
    assert td.length is None

    # WHEN
    td.add_child_series(1, "p_mw", [1.0, 2.0])

    # THEN
    assert td.length == 2


def test_series_length_accepts_pandas_series():
    # GIVEN
    td = TimeseriesData()
    s = pandas.Series([1.0, 2.0, 3.0])

    # WHEN
    td.add_child_series(1, "p_mw", s)

    # THEN
    assert td.length == 3


def test_add_node_series_registered():
    # GIVEN
    td = TimeseriesData()

    # WHEN
    td.add_node_series(42, "some_attr", [0.1, 0.2])

    # THEN
    assert 42 in td._node_id_to_series
    assert "some_attr" in td._node_id_to_series[42]


def test_add_branch_series_by_name_registered():
    # GIVEN
    td = TimeseriesData()

    # WHEN
    td.add_branch_series_by_name("pipe_1", "on_off", [1, 0, 1])

    # THEN
    assert "pipe_1" in td._branch_name_to_series


def test_add_compound_series_by_name_registered():
    # GIVEN
    td = TimeseriesData()

    # WHEN
    td.add_compound_series_by_name("chp_1", "regulation", [0.8, 0.9, 1.0])

    # THEN
    assert "chp_1" in td._compound_name_to_series


def test_extend_adds_new_component():
    # GIVEN
    td1 = TimeseriesData()
    td1.add_child_series(1, "p_mw", [1.0, 2.0])
    td2 = TimeseriesData()
    td2.add_child_series(2, "p_mw", [3.0, 4.0])

    # WHEN
    td1.extend(td2)

    # THEN
    assert 1 in td1._child_id_to_series
    assert 2 in td1._child_id_to_series


def test_extend_self_wins_on_attribute_conflict():
    # GIVEN
    td1 = TimeseriesData()
    td1.add_child_series(1, "p_mw", [10.0, 20.0])
    td2 = TimeseriesData()
    td2.add_child_series(1, "p_mw", [99.0, 99.0])

    # WHEN
    td1.extend(td2)

    # THEN
    assert td1._child_id_to_series[1]["p_mw"] == [10.0, 20.0]


def test_extend_merges_disjoint_attributes():
    # GIVEN
    td1 = TimeseriesData()
    td1.add_child_series(1, "p_mw", [1.0, 2.0])
    td2 = TimeseriesData()
    td2.add_child_series(1, "q_mvar", [0.1, 0.2])

    # WHEN
    td1.extend(td2)

    # THEN
    assert "p_mw" in td1._child_id_to_series[1]
    assert "q_mvar" in td1._child_id_to_series[1]


def test_add_combines_two_timeseries_data():
    # GIVEN
    td1 = TimeseriesData()
    td1.add_child_series(1, "p_mw", [1.0])
    td2 = TimeseriesData()
    td2.add_child_series(2, "p_mw", [2.0])

    # WHEN
    combined = td1 + td2

    # THEN
    assert 1 in combined._child_id_to_series
    assert 2 in combined._child_id_to_series


def test_extend_unequal_lengths_truncates_to_shorter_and_warns():
    # GIVEN
    td_long = TimeseriesData()
    td_long.add_child_series(1, "p_mw", [1.0, 2.0, 3.0, 4.0])
    td_short = TimeseriesData()
    td_short.add_child_series(2, "p_mw", [5.0, 6.0])

    # WHEN
    with pytest.warns(UserWarning, match=r"unequal lengths \(4 and 2\).*head\(2\)"):
        td_long.extend(td_short)

    # THEN  (both sides truncated, incoming object untouched)
    assert td_long.length == 2
    assert td_long._child_id_to_series[1]["p_mw"] == [1.0, 2.0]
    assert td_long._child_id_to_series[2]["p_mw"] == [5.0, 6.0]
    assert td_short.length == 2


def test_add_unequal_lengths_truncates_and_keeps_operands():
    # GIVEN
    td_short = TimeseriesData()
    td_short.add_child_series(1, "p_mw", [1.0])
    td_long = TimeseriesData()
    td_long.add_child_series(2, "p_mw", [2.0, 3.0, 4.0])

    # WHEN
    with pytest.warns(UserWarning, match="unequal lengths"):
        combined = td_short + td_long

    # THEN
    assert combined.length == 1
    assert combined._child_id_to_series[2]["p_mw"] == [2.0]
    assert td_long.length == 3


@mm.model
class _FreeVarLoad(mm.ChildModel):
    def __init__(self, p_mw=1.0, **kwargs):
        super().__init__(**kwargs)
        self.p_mw = Var(p_mw, min=0.5, max=3.0, name="p_mw")
        self.q_mvar = 0.0

    def equations(self, grid, node_model, **kwargs):
        return []


def _free_var_net():
    net = _el_net()
    net.child_to(_FreeVarLoad(), node_id=net.nodes[1].id, name="process")
    return net


def test_series_on_var_attribute_warns_up_front_in_simulation_run():
    # GIVEN
    net = _free_var_net()
    td = TimeseriesData()
    td.add_child_series_by_name("process", "p_mw", [0.75, 0.8])

    # WHEN  (one warning at run start, naming attribute and settable inputs)
    with pytest.warns(UserWarning, match=r"'p_mw' is a solved variable.*q_mvar"):
        result = run(net, td, steps=2)

    # THEN  (values are still bound-pinned to the series)
    assert not result.failed_steps
    p = result.get_result_for(_FreeVarLoad, "p_mw")
    assert math.isclose(p.iloc[0, 0], 0.75, rel_tol=1e-9)
    assert math.isclose(p.iloc[1, 0], 0.8, rel_tol=1e-9)


def test_series_on_var_attribute_silent_without_solve():
    # GIVEN
    net = _free_var_net()
    td = TimeseriesData()
    td.add_child_series_by_name("process", "p_mw", [0.75, 0.8])

    # WHEN / THEN  (dry runs do not warn)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = run(net, td, steps=2, solve_flag=False)
    assert all(sr.skipped for sr in result.step_results)


def test_from_dataframe_child_by_id():
    # GIVEN
    df = pandas.DataFrame({"p_mw": [1.0, 2.0, 3.0], "q_mvar": [0.1, 0.2, 0.3]})

    # WHEN
    td = TimeseriesData.from_dataframe(df, "child", component_id=42)

    # THEN
    assert td.length == 3
    assert 42 in td._child_id_to_series
    assert list(td._child_id_to_series[42]["p_mw"]) == [1.0, 2.0, 3.0]
    assert list(td._child_id_to_series[42]["q_mvar"]) == [0.1, 0.2, 0.3]


def test_from_dataframe_child_by_name():
    # GIVEN
    df = pandas.DataFrame({"p_mw": [5.0, 6.0]})

    # WHEN
    td = TimeseriesData.from_dataframe(df, "child", component_name="my_load")

    # THEN
    assert "my_load" in td._child_name_to_series


def test_from_dataframe_requires_id_or_name():
    # GIVEN
    df = pandas.DataFrame({"p_mw": [1.0]})

    # WHEN / THEN
    with pytest.raises(ValueError):
        TimeseriesData.from_dataframe(df, "child")


def test_from_dataframe_unknown_type_raises():
    # GIVEN
    df = pandas.DataFrame({"x": [1.0]})

    # WHEN / THEN
    with pytest.raises(ValueError, match="Unknown"):
        TimeseriesData.from_dataframe(df, "unknown_type", component_id=1)


def test_from_dataframe_node_by_name_raises():
    # GIVEN
    df = pandas.DataFrame({"x": [1.0]})

    # WHEN / THEN
    with pytest.raises(ValueError):
        TimeseriesData.from_dataframe(df, "node", component_name="n")


def test_basic_timeseries_run_returns_correct_length():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    td.add_child_series(load.id, "p_mw", [0.5, 0.8, 1.2])

    # WHEN
    result = run(net, td, steps=3)

    # THEN
    assert not result.failed_steps
    assert len(result.raw) == 3


def test_steps_inferred_from_series_length():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    td.add_child_series(load.id, "p_mw", [0.5, 0.8])

    # WHEN
    result = run(net, td)

    # THEN
    assert not result.failed_steps
    assert len(result.raw) == 2


def test_steps_not_provided_without_series_raises():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()

    # WHEN / THEN
    with pytest.raises(ValueError, match="step count"):
        run(net, td)


def test_get_result_for_returns_correct_shape():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    td.add_child_series(load.id, "p_mw", [0.4, 0.9])
    result = run(net, td)

    # WHEN
    df = result.get_result_for(PowerLoad, "p_mw")

    # THEN
    assert not result.failed_steps

    # one row per step
    assert len(df) == 2
    assert isinstance(df, pandas.DataFrame)


def test_get_result_for_is_cached():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    td.add_child_series(load.id, "p_mw", [0.5])
    result = run(net, td)

    # WHEN
    df1 = result.get_result_for(PowerLoad, "p_mw")
    df2 = result.get_result_for(PowerLoad, "p_mw")

    # THEN
    assert not result.failed_steps

    assert df1 is df2


def test_get_result_for_id_returns_series():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    td.add_child_series(load.id, "p_mw", [0.4, 0.9])
    result = run(net, td)

    # WHEN
    s = result.get_result_for_id(load.id, "p_mw", PowerLoad)

    # THEN
    assert not result.failed_steps

    assert isinstance(s, pandas.Series)
    assert len(s) == 2


def test_get_result_for_id_values_match_input():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    p_series = [0.3, 0.7, 1.1]
    td.add_child_series(load.id, "p_mw", p_series)
    result = run(net, td)

    # WHEN
    s = result.get_result_for_id(load.id, "p_mw", PowerLoad)

    # THEN
    assert not result.failed_steps

    for step, expected in enumerate(p_series):
        assert math.isclose(s.iloc[step], expected, rel_tol=1e-3), (
            f"step {step}: got {s.iloc[step]}, expected {expected}"
        )


def test_child_series_by_name_applied():
    # GIVEN
    net = _el_net(load_name="demand")
    td = TimeseriesData()
    td.add_child_series_by_name("demand", "p_mw", [0.3, 0.6])

    # WHEN
    result = run(net, td)

    # THEN
    assert not result.failed_steps

    load_id = net.childs[1].id
    s = result.get_result_for_id(load_id, "p_mw", PowerLoad)
    assert math.isclose(s.iloc[0], 0.3, rel_tol=1e-3)
    assert math.isclose(s.iloc[1], 0.6, rel_tol=1e-3)


def test_datetime_index_propagates_to_result_dataframe():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8])
    idx = pandas.date_range("2024-01-01", periods=2, freq="h")

    # WHEN
    result = run(net, td, datetime_index=idx)

    # THEN
    assert not result.failed_steps

    df = result.get_result_for(PowerLoad, "p_mw")
    assert list(df.index) == list(idx)


def test_datetime_index_on_get_result_for_id():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    load = net.childs[1]
    td.add_child_series(load.id, "p_mw", [0.5, 0.8])
    idx = pandas.date_range("2024-01-01", periods=2, freq="h")

    # WHEN
    result = run(net, td, datetime_index=idx)

    # THEN
    assert not result.failed_steps

    s = result.get_result_for_id(load.id, "p_mw", PowerLoad)
    assert list(s.index) == list(idx)


def test_on_step_error_raise_is_default(monkeypatch):
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8])
    monkeypatch.setattr(
        "monee.simulation.timeseries.solve",
        _make_failing_solve(2, "solver exploded"),
    )

    # WHEN / THEN  (pin a non-CasADi solver so the per-step loop - which calls
    # the monkeypatched ``solve`` - runs, rather than the CasADi build-once fast
    # path, which bypasses ``solve``; the error-handling contract under test is
    # backend-independent).
    with pytest.raises(RuntimeError, match="solver exploded"):
        run(net, td, solver=GEKKOSolver())


def test_on_step_error_skip_records_failure(monkeypatch):
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8, 1.0])
    monkeypatch.setattr(
        "monee.simulation.timeseries.solve",
        _make_failing_solve(2, "step 1 failed"),
    )

    # WHEN  (per-step loop, see test_on_step_error_raise_is_default)
    result = run(net, td, on_step_error="skip", solver=GEKKOSolver())

    # THEN
    assert result.failed_steps == [1]

    # raw contains only the successful steps
    assert len(result.raw) == 2
    assert len(result.step_results) == 3


def test_on_step_error_skip_step_result_has_error(monkeypatch):
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5])
    monkeypatch.setattr(
        "monee.simulation.timeseries.solve",
        lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("bad")),
    )

    # WHEN  (per-step loop, see test_on_step_error_raise_is_default)
    result = run(net, td, on_step_error="skip", solver=GEKKOSolver())

    # THEN
    sr = result.step_results[0]
    assert sr.failed
    assert isinstance(sr.error, RuntimeError)
    assert sr.result is None


def test_on_step_error_invalid_value_raises():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5])

    # WHEN / THEN
    with pytest.raises(ValueError, match="on_step_error"):
        run(net, td, on_step_error="ignore")


def test_progress_callback_called_each_step():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.4, 0.7, 1.0])
    calls = []

    # WHEN
    result = run(
        net, td, progress_callback=lambda step, total: calls.append((step, total))
    )

    # THEN
    assert not result.failed_steps

    assert calls == [(0, 3), (1, 3), (2, 3)]


class _RecordingHook(StepHook):
    def __init__(self):
        self.pre_calls = []
        self.post_calls = []

    def pre_run(self, net, step, step_state):
        self.pre_calls.append((step, type(step_state).__name__))

    def post_run(self, net, step, step_state, step_result, base_net):
        self.post_calls.append((step, step_result.failed))


def test_step_hook_pre_and_post_called_each_step():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8])
    hook = _RecordingHook()

    # WHEN
    result = run(net, td, step_hooks=[hook])

    # THEN
    assert not result.failed_steps

    assert hook.pre_calls == [(0, "StepState"), (1, "StepState")]
    assert hook.post_calls == [(0, False), (1, False)]


def test_callable_hook_called_post_step():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8])
    post_args = []

    # WHEN
    result = run(net, td, step_hooks=[lambda nc, s, *_: post_args.append(s)])

    # THEN
    assert not result.failed_steps

    assert post_args == [0, 1]


def test_step_hook_post_receives_step_result():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5])
    received = []

    class _ResultHook(StepHook):
        def post_run(self, net, step, step_state, step_result, base_net):
            received.append(step_result)

    # WHEN
    result = run(net, td, step_hooks=[_ResultHook()])

    # THEN
    assert not result.failed_steps

    assert len(received) == 1
    assert isinstance(received[0], StepResult)
    assert not received[0].failed


def test_step_results_contains_all_steps():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8, 1.0])

    # WHEN
    result = run(net, td)

    # THEN
    assert not result.failed_steps

    assert len(result.step_results) == 3
    for i, sr in enumerate(result.step_results):
        assert sr.step == i
        assert not sr.failed
        assert sr.result is not None


def test_failed_steps_empty_when_all_succeed():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5])

    # WHEN
    result = run(net, td)

    # THEN
    assert result.failed_steps == []


def test_base_network_not_mutated():
    # GIVEN
    net = _el_net()
    original_load_p = net.childs[1].model.p_mw
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.3, 0.7])

    # WHEN
    result = run(net, td)

    # THEN
    assert not result.failed_steps

    assert net.childs[1].model.p_mw == original_load_p


def test_gas_timeseries_varying_sink():
    # GIVEN
    net = _gas_net()
    sink = net.childs[1]
    td = TimeseriesData()
    td.add_child_series(sink.id, "mass_flow_kgs", [0.05, 0.10, 0.15])

    # WHEN
    result = run(net, td)

    # THEN
    assert not result.failed_steps
    assert len(result.raw) == 3


def test_tracked_var_extracted_into_step_state():
    # GIVEN
    net = _el_net()
    load = net.childs[1]
    # tracked Var replaces the plain float p_mw; series values overwrite it pre-solve
    load.model.p_mw = Var(0.5, min=0.0, max=10.0)
    td = TimeseriesData()
    td.add_child_series(load.id, "p_mw", [0.4, 0.7])
    state_snapshots = []

    class _StateCapture(StepHook):
        def post_run(self, net, step, step_state, step_result, base_net):
            val = step_state.get(load.id, "p_mw")
            state_snapshots.append(val)

    # WHEN
    result = run(net, td, step_hooks=[_StateCapture()])

    # THEN
    assert not result.failed_steps

    # StepState holds the solved p_mw of each step
    assert state_snapshots[0] is not None
    assert math.isclose(state_snapshots[0], 0.4, rel_tol=1e-2)
    assert math.isclose(state_snapshots[1], 0.7, rel_tol=1e-2)


@pytest.mark.pptest
def test_timeseries_with_simbench():
    from monee.io.from_simbench import obtain_simbench_net_with_td

    # GIVEN
    steps = 3
    net, td = obtain_simbench_net_with_td("1-LV-rural3--1-no_sw")
    print(net.as_dataframe_dict_str())
    assert run_energy_flow(net).success

    # WHEN
    result = run(net, td, steps)

    # THEN
    assert not result.failed_steps
    assert len(result.raw) == steps
    assert len(result.step_results) == steps
    assert len(result.get_result_for(PowerLoad, "p_mw")) == steps


class _UnsuccessfulResult:
    success = False
    network = None
    solver_status = "aborted"
    termination_condition = "infeasible"


def _make_unsuccessful_solve(fail_on_call):
    """Return a solve replacement reporting success=False on the given call."""
    call_count = [0]

    def fake_solve(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == fail_on_call:
            return _UnsuccessfulResult()
        from monee.simulation.core import solve as real_solve

        return real_solve(*args, **kwargs)

    return fake_solve


def test_success_false_result_raises_by_default(monkeypatch):
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8])
    monkeypatch.setattr(
        "monee.simulation.timeseries.solve", _make_unsuccessful_solve(2)
    )

    # WHEN / THEN  (per-step loop, see test_on_step_error_raise_is_default)
    with pytest.raises(RuntimeError, match="success=False"):
        run(net, td, solver=GEKKOSolver())


def test_success_false_result_skip_records_failure_without_push(monkeypatch):
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8, 1.0])
    monkeypatch.setattr(
        "monee.simulation.timeseries.solve", _make_unsuccessful_solve(2)
    )

    # WHEN  (per-step loop, see test_on_step_error_raise_is_default)
    result = run(net, td, on_step_error="skip", solver=GEKKOSolver())

    # THEN: failed like a raising solver; bad network not treated as a result
    assert result.failed_steps == [1]
    assert len(result.raw) == 2
    sr = result.step_results[1]
    assert sr.failed
    assert sr.result is None
    assert "solver_status=aborted" in str(sr.error)


def test_result_index_uses_step_numbers_after_skipped_failure(monkeypatch):
    # GIVEN
    net = _el_net()
    load_id = net.childs[1].id
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.5, 0.8, 1.0])
    monkeypatch.setattr(
        "monee.simulation.timeseries.solve", _make_failing_solve(2, "boom")
    )

    # WHEN  (per-step loop, see test_on_step_error_raise_is_default)
    result = run(net, td, on_step_error="skip", solver=GEKKOSolver())

    # THEN: rows labelled by step number, not renumbered positionally
    df = result.get_result_for(PowerLoad, "p_mw")
    assert list(df.index) == [0, 2]
    s = result.get_result_for_id(load_id, "p_mw", PowerLoad)
    assert list(s.index) == [0, 2]


def test_datetime_index_shorter_than_steps_raises():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8, 1.0])
    idx = pandas.date_range("2025-01-01", periods=2, freq="h")

    # WHEN / THEN: clear error up front instead of a mid-run IndexError
    with pytest.raises(ValueError, match="datetime_index length"):
        run(net, td, datetime_index=idx, solver=GEKKOSolver())


def test_dt_h_threaded_to_step_state():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5, 0.8])
    seen = []

    class _DtCapture(StepHook):
        def pre_run(self, base_net, step, step_state):
            seen.append(step_state.dt_h)

    # WHEN
    result = run(net, td, dt_h=0.25, step_hooks=[_DtCapture()], solver=GEKKOSolver())

    # THEN
    assert not result.failed_steps
    assert seen == [0.25, 0.25]


def test_dt_h_invalid_raises():
    # GIVEN
    net = _el_net()
    td = TimeseriesData()
    td.add_child_series(net.childs[1].id, "p_mw", [0.5])

    # WHEN / THEN
    with pytest.raises(ValueError, match="dt_h must be positive"):
        run(net, td, dt_h=0.0)


def test_add_objective_data_by_name():
    td = TimeseriesData()
    td.add_objective_data_by_name("load1", "p_mw", [1, 2, 3])
    assert td.child_name_data["load1"]["p_mw"] == [1, 2, 3]


def test_id_data_properties():
    td = TimeseriesData()
    td.add_child_series("c1", "p_mw", [1, 2])
    td.add_branch_series("b1", "on_off", [1, 0])
    td.add_compound_series("cmp1", "x", [3, 4])
    assert td.child_id_data["c1"]["p_mw"] == [1, 2]
    assert td.branch_id_data["b1"]["on_off"] == [1, 0]
    assert td.compound_id_data["cmp1"]["x"] == [3, 4]


def _chp_net():
    """Multi-energy net with a single CHP compound; returns (net, compound_id)."""
    net = mx.create_multi_energy_network()
    gas_junction = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas_junction)
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    mx.create_power_load(net, bus, p_mw=1.0, q_mvar=0.0)
    supply = mx.create_water_junction(net)
    return_junction = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, supply)
    mx.create_sink(net, return_junction, mass_flow_kgs=15)
    chp_id = mx.create_chp(
        net,
        bus,
        supply,
        return_junction,
        gas_junction,
        diameter_m=0.15,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.1,
    )
    return net, chp_id


def _chp_control_node(net):
    return next(
        node.model for node in net.nodes if isinstance(node.model, mm.CHPControlNode)
    )


def test_compound_series_reaches_control_node():
    # GIVEN
    net, chp_id = _chp_net()
    td = TimeseriesData()
    td.add_compound_series(chp_id, "regulation", [1.0, 0.4, 0.0])

    # WHEN
    td.apply_to_network(net, 1)

    # THEN
    assert _chp_control_node(net).regulation == 0.4


def test_compound_series_does_not_reactivate_deactivated_compound():
    # GIVEN
    net, chp_id = _chp_net()
    net.compound_by_id(chp_id).model.set_active(False)
    td = TimeseriesData()
    td.add_compound_series(chp_id, "regulation", [1.0, 0.4, 0.0])

    # WHEN
    td.apply_to_network(net, 0)

    # THEN
    assert _chp_control_node(net).regulation == 0


def test_compound_series_does_not_clobber_controllable_var():
    # GIVEN
    net, chp_id = _chp_net()
    control_node = _chp_control_node(net)
    control_node.regulation = Var(1, min=0, max=1, name="regulation")
    td = TimeseriesData()
    td.add_compound_series(chp_id, "regulation", [1.0, 0.4, 0.0])

    # WHEN
    td.apply_to_network(net, 1)

    # THEN
    assert control_node.regulation is not None
    assert isinstance(control_node.regulation, Var)


def test_unmatched_series_key_is_reported_before_the_first_step():
    # GIVEN
    net = _el_net(load_name="load")
    td = TimeseriesData()
    td.add_child_series_by_name("typo_load", "p_mw", [1.0, 0.5])
    td.add_child_series(9999, "p_mw", [1.0, 0.5])
    td.add_node_series(4242, "vm_pu", [1.0, 1.0])

    # WHEN
    with pytest.warns(UserWarning) as caught:
        run(net, td, solve_flag=False)

    # THEN
    message = " ".join(str(w.message) for w in caught)
    assert "typo_load" in message
    assert "9999" in message
    assert "4242" in message

    # ... and validate_bound itself is strict
    with pytest.raises(KeyError, match="typo_load"):
        td.validate_bound(net)


def test_matched_series_keys_pass_validation():
    # GIVEN
    net = _el_net(load_name="load")
    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", [1.0, 0.5])
    td.add_node_series(0, "vm_pu", [1.0, 1.0])

    # WHEN
    unbound = td.validate_bound(net)

    # THEN
    assert unbound == []


def test_series_longer_than_explicit_steps_does_not_warn_but_logs(caplog):
    # GIVEN  (an explicit steps= is a deliberate truncation)
    net = _el_net(load_name="load")
    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", [0.5, 0.6, 0.7, 0.8, 0.9])

    # WHEN / THEN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.INFO, logger="monee.simulation.timeseries"):
            result = run(net, td, steps=2, solve_flag=False)
    assert len(result.step_results) == 2
    notes = [r.getMessage() for r in caplog.records if "length 5" in r.getMessage()]
    assert notes and "head(2)" in notes[0]


def test_series_matching_the_run_length_does_not_warn():
    # GIVEN
    net = _el_net(load_name="load")
    td = TimeseriesData()
    td.add_child_series_by_name("load", "p_mw", [0.5, 0.6])

    # WHEN / THEN
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        run(net, td, steps=2, solve_flag=False)


def test_slice_cuts_every_series_positionally():
    # GIVEN
    td = TimeseriesData()
    td.add_child_series(1, "p_mw", pandas.Series([0.1, 0.2, 0.3, 0.4]))
    td.add_child_series_by_name("load", "q_mvar", [1.0, 2.0, 3.0, 4.0])

    # WHEN
    window = td.slice(1, 3)

    # THEN
    assert window.length == 2
    assert window._child_id_to_series[1]["p_mw"] == [0.2, 0.3]
    assert window._child_name_to_series["load"]["q_mvar"] == [2.0, 3.0]
    # the source object is untouched
    assert td.length == 4


def test_head_truncates_to_the_horizon():
    # GIVEN
    td = TimeseriesData()
    td.add_child_series(1, "p_mw", [0.1, 0.2, 0.3])

    # WHEN
    head = td.head(2)

    # THEN
    assert head.length == 2
    assert head._child_id_to_series[1]["p_mw"] == [0.1, 0.2]


def test_slice_out_of_range_raises():
    # GIVEN
    td = TimeseriesData()
    td.add_child_series(1, "p_mw", [0.1, 0.2, 0.3])

    # WHEN / THEN
    with pytest.raises(ValueError, match="out of range"):
        td.slice(0, 4)
    with pytest.raises(ValueError, match="out of range"):
        td.slice(2, 2)
    with pytest.raises(ValueError, match="no series registered"):
        TimeseriesData().slice(0, 1)


def _shadowing_net():
    """Bus ids and child ids both start at 0, so every child id shadows a bus."""
    net = _el_net(load_p_mw=0.6)
    return net, net.childs[1]


def test_get_result_for_id_refuses_an_id_shared_by_two_categories():
    # GIVEN
    net, load = _shadowing_net()
    result = run(net, steps=1)

    # WHEN / THEN
    with pytest.raises(ValueError, match="Bus, PowerLoad"):
        result.get_result_for_id(load.id, "p_mw")
    with pytest.raises(ValueError, match="Bus, PowerLoad"):
        result[load.id]
    with pytest.raises(ValueError, match="Bus, PowerLoad"):
        result.step_results[0].result[load.id]


def test_model_type_selects_the_component_behind_a_shared_id():
    # GIVEN
    net, load = _shadowing_net()
    result = run(net, steps=1)

    # WHEN
    series = result.get_result_for_id(load.id, "p_mw", model_type=PowerLoad)
    row = result[load.id, PowerLoad]
    solver_row = result.step_results[0].result[load.id, PowerLoad]

    # THEN
    assert math.isclose(series.iloc[0], 0.6)
    assert "base_kv" not in row.columns
    assert math.isclose(row["p_mw"].iloc[0], 0.6)
    assert math.isclose(solver_row["p_mw"], 0.6)


def test_ambiguous_id_error_shows_the_model_type_form_and_the_docs_page():
    # GIVEN
    net, load = _shadowing_net()
    result = run(net, steps=1)

    # WHEN
    with pytest.raises(ValueError) as attr_exc:
        result.get_result_for_id(load.id, "p_mw")
    with pytest.raises(ValueError) as row_exc:
        result[load.id]

    # THEN
    attr_msg = str(attr_exc.value)
    assert "model_type=" in attr_msg
    assert f"get_result_for_id({load.id!r}, 'p_mw', mm.PowerLoad)" in attr_msg
    assert "concepts/data_model" in attr_msg
    assert f"result[{load.id!r}, mm.PowerLoad]" in str(row_exc.value)


def test_ambiguous_gas_ext_grid_id_names_ext_hydr_grid():
    # GIVEN a gas ext grid child sharing id 0 with the first junction
    net = _gas_net()
    result = run(net, steps=1)

    # WHEN
    with pytest.raises(ValueError) as exc:
        result.get_result_for_id(0, "mass_flow_kgs")

    # THEN
    msg = str(exc.value)
    assert "Junction" in msg
    assert "mm.ExtHydrGrid" in msg
    series = result.get_result_for_id(0, "mass_flow_kgs", mm.ExtHydrGrid)
    assert math.isclose(abs(series.iloc[0]), 0.1, abs_tol=1e-6)


def test_single_step_matches_run_energy_flow():
    # GIVEN
    from monee.network import create_urban_district_net

    # WHEN
    flow = run_energy_flow(create_urban_district_net(), formulation="smooth_nlp")
    ts = run(create_urban_district_net(), steps=1, formulation="smooth_nlp")

    # THEN
    expected = flow.dataframes["Junction"].set_index("id")["t_k"]
    actual = ts.get_result_for(Junction, "t_k").iloc[0]
    for node_id, t_k in expected.items():
        assert math.isclose(float(actual[node_id]), float(t_k), rel_tol=1e-4), (
            f"Junction {node_id}: energy flow {t_k}, timeseries {actual[node_id]}"
        )


def test_named_compound_series_reaches_control_node():
    # GIVEN a CHP created with name=; register the series by that name
    net = mx.create_multi_energy_network()
    gas_junction = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas_junction)
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    supply = mx.create_water_junction(net)
    return_junction = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, supply)
    mx.create_sink(net, return_junction, mass_flow_kgs=15)
    mx.create_chp(
        net,
        bus,
        supply,
        return_junction,
        gas_junction,
        diameter_m=0.15,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.1,
        name="chp_a",
    )
    td = TimeseriesData()
    td.add_compound_series_by_name("chp_a", "regulation", [1.0, 0.4, 0.0])

    # WHEN
    assert td.unbound_keys(net) == []
    td.apply_to_network(net, 1)

    # THEN
    assert _chp_control_node(net).regulation == 0.4


@mm.model
class _RampLimitedProcess(mm.ChildModel):
    """Custom child whose own inter-temporal constraint limits |dp| to 1 MW."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p_mw = 0.5
        self.q_mvar = 0.0

    def equations(self, grid, node_model, **kwargs):
        return []

    def inter_temporal_equations(self, temporal_state, component_id, **kwargs):
        previous = temporal_state.get(component_id, "p_mw")
        if previous is None:
            return []
        return [self.p_mw - previous <= 1.0, previous - self.p_mw <= 1.0]


def _ramp_net():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_line(net, b0, b1, length_m=800, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, b0)
    process_id = mx.create_el_child(net, _RampLimitedProcess(), node_id=b1, name="proc")
    return net, process_id


def test_prescribed_replay_violating_inter_temporal_equation_warns():
    # GIVEN a profile whose jumps exceed the model's own 1 MW/h ramp limit
    net, process_id = _ramp_net()
    td = TimeseriesData()
    td.add_child_series_by_name("proc", "p_mw", [0.5, 3.0, 0.5, 3.0])

    # WHEN
    with pytest.warns(UserWarning, match="violate inter-temporal constraints"):
        result = run(net, td, dt_h=1.0)

    # THEN the violation is reported per step, naming model and step
    entries = [
        warning
        for step_result in result.step_results
        for warning in step_result.result.warnings
        if warning.category == "temporal"
    ]
    assert [entry.component for entry in entries] == [str(process_id)] * 3
    assert all("_RampLimitedProcess" in entry.message for entry in entries)
    assert {"step 1", "step 2", "step 3"} == {
        entry.message.split(":")[0] for entry in entries
    }


def test_prescribed_replay_respecting_inter_temporal_equation_is_clean():
    # GIVEN a profile within the ramp limit
    net, _ = _ramp_net()
    td = TimeseriesData()
    td.add_child_series_by_name("proc", "p_mw", [0.5, 1.0, 1.5, 1.0])

    # WHEN
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = run(net, td, dt_h=1.0)

    # THEN
    assert result.failed_steps == []
    assert not [
        warning
        for step_result in result.step_results
        for warning in step_result.result.warnings
        if warning.category == "temporal"
    ]


def test_prescribed_replay_violation_raises_under_strict():
    # GIVEN the violating profile and strict=True
    from monee.solver.core import ValidationError

    net, _ = _ramp_net()
    td = TimeseriesData()
    td.add_child_series_by_name("proc", "p_mw", [0.5, 3.0])

    # WHEN / THEN
    with pytest.raises(ValidationError) as excinfo:
        run(net, td, dt_h=1.0, strict=True)
    assert any(w.category == "temporal" for w in excinfo.value.warnings)
