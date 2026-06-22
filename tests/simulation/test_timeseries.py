"""Tests for monee.simulation.timeseries."""

import math

import pandas
import pytest

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
    s = result.get_result_for_id(load.id, "p_mw")

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
    s = result.get_result_for_id(load.id, "p_mw")

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
    s = result.get_result_for_id(load_id, "p_mw")
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

    s = result.get_result_for_id(load.id, "p_mw")
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
