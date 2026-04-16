import monee.model as mm
from monee.model import Network
from monee.model.child import ExtHydrGrid, ExtPowerGrid, PowerGenerator
from monee.model.node import Bus, Junction
from monee.model.storage import ElectricStorage, GasStorage
from monee.simulation.timeseries import TimeseriesData, run

# helpers


def _el_net_with_storage(p_max_mw=2.0, e_initial=5.0, e_max=10.0):
    """
    3-bus power network:
      ext-grid (slack) ──line── bus0 ──line── bus1 (storage)
    """
    net = Network(mm.PowerGrid(name="power", sn_mva=1))
    n_slack = net.node(
        Bus(base_kv=1),
        child_ids=[net.child(ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=mm.EL,
    )
    gen_id = net.child(PowerGenerator(p_mw=0.5, q_mvar=0))
    n_gen = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[gen_id])
    storage_id = net.child(
        ElectricStorage(
            e_mwh_initial=e_initial,
            e_mwh_max=e_max,
            p_max_mw=p_max_mw,
        ),
        name="storage",
    )
    n_storage = net.node(Bus(base_kv=1), grid=mm.EL, child_ids=[storage_id])
    line_kw = dict(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1)
    net.branch(mm.PowerLine(**line_kw), n_slack, n_gen)
    net.branch(mm.PowerLine(**line_kw), n_gen, n_storage)
    return net, storage_id


def _gas_net_with_storage(flow_max=0.2, m_initial=100.0, m_max=500.0):
    """
    3-junction gas network:
      ext-grid (slack) ──pipe── jct0 ──pipe── jct1 (storage)
    """
    from monee.model.child import Source

    net = Network()
    net.activate_grid(grid=mm.GAS)
    n0 = net.node(Junction(), mm.GAS, child_ids=[net.child(ExtHydrGrid())])
    src_id = net.child(Source(mass_flow=0.05))
    n_src = net.node(Junction(), mm.GAS, child_ids=[src_id])
    storage_id = net.child(
        GasStorage(
            m_stored_kg_initial=m_initial,
            m_stored_kg_max=m_max,
            flow_max_kgs=flow_max,
        ),
        name="gas_storage",
    )
    n_storage = net.node(Junction(), mm.GAS, child_ids=[storage_id])
    pipe_kw = dict(diameter_m=0.3, length_m=200, temperature_ext_k=300)
    net.branch(mm.GasPipe(**pipe_kw), n0, n_src)
    net.branch(mm.GasPipe(**pipe_kw), n_src, n_storage)
    return net, storage_id


# Plain energy flow — fixed dispatch via TimeseriesData


def test_electric_storage_inter_step_constraint_holds():
    """
    In plain energy flow, p_mw is a fixed dispatch (set via timeseries).
    The inter-step invariant must hold for steps 2+:
        e_mwh[t] == e_mwh[t-1] + dt_h * p_mw[t]   (dt_h = 1.0)
    """
    net, storage_id = _el_net_with_storage(p_max_mw=2.0, e_initial=5.0, e_max=10.0)

    # Prescribe a non-trivial dispatch schedule so SoC actually changes.
    td = TimeseriesData()
    td.add_child_series(storage_id, "p_mw", [1.0, -0.5, 0.8])
    ts_result = run(net, td, steps=3)

    e_series = ts_result.get_result_for_id(storage_id, "e_mwh")
    p_series = ts_result.get_result_for_id(storage_id, "p_mw")
    assert e_series is not None and p_series is not None
    assert len(e_series) == 3

    dt_h = 1.0
    for t in range(1, 3):
        expected = e_series.iloc[t - 1] + dt_h * p_series.iloc[t]
        assert abs(e_series.iloc[t] - expected) < 1e-3, (
            f"Step {t}: e_mwh={e_series.iloc[t]:.4f}, "
            f"expected={expected:.4f} (prev={e_series.iloc[t - 1]:.4f}, "
            f"p_mw={p_series.iloc[t]:.4f})"
        )


def test_electric_storage_fixed_zero_dispatch_no_soc_change():
    """With p_mw=0 (default), SoC must stay constant across all steps."""
    net, storage_id = _el_net_with_storage(p_max_mw=2.0, e_initial=5.0, e_max=10.0)

    ts_result = run(net, steps=3)

    e_series = ts_result.get_result_for_id(storage_id, "e_mwh")
    assert len(e_series) == 3
    for t in range(3):
        assert abs(e_series.iloc[t] - 5.0) < 1e-3, (
            f"Step {t}: expected e_mwh=5.0, got {e_series.iloc[t]:.4f}"
        )


def test_electric_storage_soc_recorded_first_step():
    """First step records e_mwh correctly with zero default dispatch."""
    net, storage_id = _el_net_with_storage(p_max_mw=1.0, e_initial=5.0, e_max=10.0)

    ts_result = run(net, steps=1)

    e_series = ts_result.get_result_for_id(storage_id, "e_mwh")
    assert e_series is not None
    assert len(e_series) == 1
    # With p_mw=0 and initial SoC=5.0, t=0 SoC must be 5.0.
    assert abs(e_series.iloc[0] - 5.0) < 1e-3


def test_electric_storage_soc_within_bounds():
    """e_mwh must stay within [0, e_max] even with prescribed dispatch."""
    e_max = 8.0
    net, storage_id = _el_net_with_storage(p_max_mw=2.0, e_initial=4.0, e_max=e_max)

    # Prescribe a dispatch that stays within capacity.
    td = TimeseriesData()
    td.add_child_series(storage_id, "p_mw", [1.0, -1.0, 0.5, -0.5])
    ts_result = run(net, td, steps=4)

    e_series = ts_result.get_result_for_id(storage_id, "e_mwh")
    assert e_series is not None
    assert (e_series >= -1e-6).all(), "e_mwh went negative"
    assert (e_series <= e_max + 1e-6).all(), "e_mwh exceeded e_max"


def test_gas_storage_inter_step_constraint_holds():
    """
    In plain energy flow, mass_flow is a fixed dispatch (set via timeseries).
    The inter-step invariant must hold for steps 2+:
        m_stored_kg[t] == m_stored_kg[t-1] + dt_s * mass_flow[t]
    """
    net, storage_id = _gas_net_with_storage(flow_max=0.2, m_initial=100.0, m_max=500.0)

    td = TimeseriesData()
    td.add_child_series(storage_id, "mass_flow", [0.02, 0.01, -0.01])
    ts_result = run(net, td, steps=3)

    m_series = ts_result.get_result_for_id(storage_id, "m_stored_kg")
    f_series = ts_result.get_result_for_id(storage_id, "mass_flow")
    assert m_series is not None and f_series is not None
    assert len(m_series) == 3

    dt_s = 1.0 * 3600.0  # dt_h=1.0 default
    for t in range(1, 3):
        expected = m_series.iloc[t - 1] + dt_s * f_series.iloc[t]
        assert abs(m_series.iloc[t] - expected) < 1e-2, (
            f"Step {t}: m={m_series.iloc[t]:.4f}, expected={expected:.4f} "
            f"(prev={m_series.iloc[t - 1]:.4f}, mf={f_series.iloc[t]:.6f})"
        )


def test_gas_storage_soc_within_bounds():
    """m_stored_kg must stay within [0, m_max] with prescribed dispatch."""
    m_max = 300.0
    net, storage_id = _gas_net_with_storage(flow_max=0.15, m_initial=150.0, m_max=m_max)

    td = TimeseriesData()
    td.add_child_series(storage_id, "mass_flow", [0.02, -0.01, 0.01, -0.01])
    ts_result = run(net, td, steps=4)

    m_series = ts_result.get_result_for_id(storage_id, "m_stored_kg")
    assert m_series is not None
    assert (m_series >= -1e-3).all(), "m_stored_kg went negative"
    assert (m_series <= m_max + 1e-3).all(), "m_stored_kg exceeded m_max"


# Optimisation mode — make_controllable / controllable_storages


def test_make_controllable_converts_p_mw_to_var():
    """make_controllable() must convert p_mw from float to Var."""
    from monee.model.core import Var

    model = ElectricStorage(e_mwh_initial=5.0, e_mwh_max=10.0, p_max_mw=2.0)
    assert isinstance(model.p_mw, (int, float)), (
        "p_mw should be float before make_controllable"
    )

    model.make_controllable()
    assert isinstance(model.p_mw, Var), "p_mw should be Var after make_controllable"
    assert model.p_mw.min == -2.0
    assert model.p_mw.max == 2.0


def test_controllable_storages_via_problem():
    """OptimizationProblem.controllable_storages() makes storage dispatch a Var."""
    from monee.problem import OptimizationProblem
    from monee.simulation.multi_period import run_multi_period

    net, storage_id = _el_net_with_storage(p_max_mw=2.0, e_initial=4.0, e_max=8.0)

    td = TimeseriesData()
    td.add_child_series(
        net.child_by_id(storage_id).id if hasattr(net, "child_by_id") else storage_id,
        "p_mw",
        [0.0, 0.0],
    ) if False else None  # no external dispatch; let solver optimise

    problem = OptimizationProblem()
    problem.controllable_storages()

    # Run multi-period with the problem — p_mw must now be optimised (non-zero
    # SoC change is possible).
    td2 = TimeseriesData()
    result = run_multi_period(net, td2, steps=2, optimization_problem=problem)
    assert result.success

    p_df = result.get_result_for(ElectricStorage, "p_mw")
    # The result p_mw columns should be present (storage participated in solve).
    assert storage_id in p_df.columns
