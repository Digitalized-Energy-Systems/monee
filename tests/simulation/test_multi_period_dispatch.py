"""Economic dispatch over a horizon: per-carrier ramps, storage arbitrage and
per-period prices on gas and heat units."""

import numpy as np
import pytest

import monee.express as mx
import monee.model as mm
from monee import TimeseriesData, run_multi_period
from monee.problem import create_economic_dispatch_problem
from monee.problem.utils import gas_mw_per_kgs
from monee.simulation.multi_period import run_mpc
from tests.problem._ed_networks import (
    gas_merit_net,
    gas_storage_net,
    heat_dhs_net,
    heat_loop_net,
    ramp_collision_net,
)


def _column(result, model_type, attr):
    return result.get_result_for(model_type, attr).iloc[:, 0].tolist()


def test_power_ramp_binds_generator_not_bus():
    net, generator_id = ramp_collision_net()
    td = TimeseriesData()
    td.add_objective_data(generator_id, "cost", [100, 1, 100, 1])

    result = run_multi_period(
        net,
        td,
        steps=4,
        optimization_problem=create_economic_dispatch_problem(
            ext_grid_cost_default=50, ramp_limit=0.5
        ),
    )

    assert result.success
    p_mw = _column(result, mm.PowerGenerator, "p_mw")
    assert np.all(np.abs(np.diff(p_mw)) <= 0.5 + 1e-6)
    assert p_mw == pytest.approx([-1.259, -1.759, -1.2597, -1.7597], abs=1e-3)


def test_power_ramp_limits_the_served_power():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    generator_id = mx.create_power_generator(
        net, bus, p_mw=2.0, q_mvar=0.0, cost=10, regulation=0.5
    )
    mx.create_power_load(net, bus, p_mw=0.1, q_mvar=0.0)
    td = TimeseriesData()
    td.add_objective_data(generator_id, "cost", [100, 1, 100, 1])

    result = run_multi_period(
        net,
        td,
        steps=4,
        optimization_problem=create_economic_dispatch_problem(
            ext_grid_cost_default=50, ramp_limit=0.25
        ),
        initial_state={(generator_id, "p_mw"): 0.0},
    )

    assert result.success
    assert _column(result, mm.PowerGenerator, "p_mw") == pytest.approx(
        [0.0, -0.5, 0.0, -0.5], abs=1e-4
    )


def test_power_ramp_follows_a_changing_regulation():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    generator_id = mx.create_power_generator(net, bus, p_mw=2.0, q_mvar=0.0, cost=10)
    mx.create_power_load(net, bus, p_mw=3.0, q_mvar=0.0)
    td = TimeseriesData()
    td.add_child_series(generator_id, "regulation", [0.5, 1.0])

    result = run_multi_period(
        net,
        td,
        steps=2,
        optimization_problem=create_economic_dispatch_problem(
            ext_grid_cost_default=50, ramp_limit=0.25
        ),
    )

    assert result.success
    p_mw = _column(result, mm.PowerGenerator, "p_mw")
    served = [-p * r for p, r in zip(p_mw, [0.5, 1.0])]
    assert served == pytest.approx([1.0, 1.25], abs=1e-4)


def _chp_sharing_the_source_id():
    net = mx.create_multi_energy_network()
    g0, g1, g2 = (mx.create_gas_junction(net) for _ in range(3))
    source_id = mx.create_gas_source(net, g2, mass_flow_kgs=0.1, cost=10.0)
    mx.create_gas_pipe(net, g0, g1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(net, g1, g2, diameter_m=0.15, length_m=400)
    mx.create_gas_ext_grid(net, g0, max_export_kgs=0)
    mx.create_gas_sink(net, g1, mass_flow_kgs=0.15)
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    mx.create_power_load(net, bus, p_mw=0.5, q_mvar=0.0)
    w0, w1 = mx.create_water_junction(net), mx.create_water_junction(net)
    mx.create_water_ext_grid(net, w0)
    mx.create_water_pipe(net, w0, w1, diameter_m=0.1, length_m=100)
    mx.create_water_sink(net, w1, mass_flow_kgs=3.0)
    mx.create_heat_load(net, w1, q_mw=0.3)
    chp_id = mx.create_chp_hg(
        net,
        bus,
        w1,
        g1,
        efficiency_power=0.4,
        efficiency_heat=0.4,
        mass_flow_setpoint_kgs=0.01,
    )
    assert chp_id == source_id
    return net, source_id


def test_run_mpc_seeds_the_gas_ramp_from_the_source_not_the_compound():
    net, source_id = _chp_sharing_the_source_id()
    td = TimeseriesData()
    td.add_objective_data(source_id, "cost", [50, 5, 50, 5])
    mw_per_kgs = gas_mw_per_kgs(net.child_by_id(source_id).grid)

    result = run_mpc(
        net,
        td,
        total_steps=4,
        horizon=1,
        execution_steps=1,
        optimization_problem=create_economic_dispatch_problem(
            gas_cost_default=30, ramp_limit={"gas": 0.02 * mw_per_kgs}
        ),
        initial_state={(source_id, "mass_flow_kgs"): 0.0},
    )

    assert result.success
    assert _column(result, mm.Source, "mass_flow_kgs") == pytest.approx(
        [0.0, -0.02, 0.0, -0.02], abs=1e-4
    )


def test_gas_ramp_limit_is_in_mw():
    net, source_id, _ = gas_merit_net(10.0)
    td = TimeseriesData()
    td.add_objective_data(source_id, "cost", [50, 5, 50, 5])
    mw_per_kgs = gas_mw_per_kgs(net.child_by_id(source_id).grid)

    result = run_multi_period(
        net,
        td,
        steps=4,
        optimization_problem=create_economic_dispatch_problem(
            gas_cost_default=30, ramp_limit={"gas": 0.02 * mw_per_kgs}
        ),
        initial_state={(source_id, "mass_flow_kgs"): 0.0},
    )

    assert result.success
    assert _column(result, mm.Source, "mass_flow_kgs") == pytest.approx(
        [-0.02, -0.04, -0.06, -0.08], abs=1e-4
    )
    assert result.user_objective == pytest.approx(704.577, abs=1e-2)


def test_float_ramp_limit_covers_every_dispatched_carrier():
    net, source_id, _ = gas_merit_net(10.0)
    td = TimeseriesData()
    td.add_objective_data(source_id, "cost", [50, 5, 50, 5])
    mw_per_kgs = gas_mw_per_kgs(net.child_by_id(source_id).grid)

    result = run_multi_period(
        net,
        td,
        steps=4,
        optimization_problem=create_economic_dispatch_problem(
            gas_cost_default=30,
            heat_cost_default=30,
            ramp_limit=0.02 * mw_per_kgs,
        ),
        initial_state={(source_id, "mass_flow_kgs"): 0.0},
    )

    assert result.success
    assert _column(result, mm.Source, "mass_flow_kgs") == pytest.approx(
        [-0.02, -0.04, -0.06, -0.08], abs=1e-4
    )


def _hx_ramp_case():
    net, hx_id, _ = heat_loop_net(30.0)
    td = TimeseriesData()
    td.add_branch_series(hx_id, "cost", [10.0, 100.0, 10.0])
    return net, td, {(hx_id, "q_mw_set"): -0.3}, mm.HeatExchangerGenerator, "q_mw_set"


def _heat_generator_ramp_case():
    net, generator_id, _, _ = heat_dhs_net(30.0)
    td = TimeseriesData()
    td.add_objective_data(generator_id, "cost", [10.0, 100.0, 10.0])
    return net, td, {(generator_id, "q_mw_heat"): 0.0}, mm.HeatGenerator, "q_mw_heat"


@pytest.mark.parametrize(
    "make_case, limit, expected",
    [
        (_hx_ramp_case, 0.1, [-0.4, -0.3, -0.4]),
        (_heat_generator_ramp_case, 0.05, [-0.05, 0.0, -0.05]),
    ],
    ids=["heat_exchanger", "heat_generator"],
)
def test_heat_ramp_limit(make_case, limit, expected):
    net, td, initial_state, model_type, attr = make_case()

    result = run_multi_period(
        net,
        td,
        steps=3,
        optimization_problem=create_economic_dispatch_problem(
            heat_cost_default=50, ramp_limit={"heat": limit}
        ),
        initial_state=initial_state,
    )

    assert result.success
    assert _column(result, model_type, attr) == pytest.approx(expected, abs=1e-3)


def test_gas_storage_arbitrage():
    net, storage_id, ext_id = gas_storage_net()
    td = TimeseriesData()
    td.add_objective_data(ext_id, "cost", [10, 50, 10, 50])

    result = run_multi_period(
        net,
        td,
        steps=4,
        optimization_problem=create_economic_dispatch_problem(
            gas_cost_default=10, include_storages=True
        ),
        terminal_state={(storage_id, "m_stored_kg"): 0.0},
    )

    assert result.success
    assert _column(result, mm.GasStorage, "mass_flow_kgs") == pytest.approx(
        [0.1, -0.1, 0.1, -0.1], abs=1e-4
    )
    assert _column(result, mm.GasStorage, "m_stored_kg") == pytest.approx(
        [360, 0, 360, 0], abs=1e-2
    )
    assert _column(result, mm.ExtHydrGrid, "mass_flow_kgs") == pytest.approx(
        [-0.2, 0, -0.2, 0], abs=1e-4
    )
    assert result.user_objective == pytest.approx(169.7776, abs=1e-3)


def test_electric_storage_arbitrage():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    ext_id = mx.create_ext_power_grid(net, bus)
    mx.create_power_load(net, bus, p_mw=0.5, q_mvar=0.0)
    storage_id = mx.create_el_child(
        net, mm.ElectricStorage(e_mwh_initial=0, e_mwh_max=1, p_max_mw=0.5), bus
    )
    td = TimeseriesData()
    td.add_objective_data(ext_id, "cost", [10, 50, 10, 50])

    result = run_multi_period(
        net,
        td,
        steps=4,
        optimization_problem=create_economic_dispatch_problem(
            ext_grid_cost_default=10, include_storages=True
        ),
        terminal_state={(storage_id, "e_mwh"): 0.0},
    )

    assert result.success
    assert _column(result, mm.ElectricStorage, "p_mw") == pytest.approx(
        [0.5, -0.5, 0.5, -0.5], abs=1e-4
    )


def test_run_mpc_dispatches_storages():
    net, storage_id, ext_id = gas_storage_net()
    td = TimeseriesData()
    td.add_objective_data(ext_id, "cost", [10, 50, 10, 50])

    result = run_mpc(
        net,
        td,
        total_steps=2,
        horizon=2,
        execution_steps=2,
        optimization_problem=create_economic_dispatch_problem(
            gas_cost_default=10, include_storages=True
        ),
        terminal_state={(storage_id, "m_stored_kg"): 0.0},
    )

    assert result.success
    assert _column(result, mm.GasStorage, "mass_flow_kgs") == pytest.approx(
        [0.1, -0.1], abs=1e-4
    )


@pytest.mark.parametrize(
    "bounds, duties",
    [("default", [-0.5007, 0.0]), (None, [-0.6, 0.0])],
    ids=["supply_only_default", "surplus_credited"],
)
def test_hx_cost_series_via_branch_series(bounds, duties):
    net, hx_id, _ = heat_loop_net(30.0)
    td = TimeseriesData()
    td.add_branch_series(hx_id, "cost", [10.0, 100.0])
    extra = {} if bounds == "default" else {"bounds_ext_heat_mw": bounds}

    result = run_multi_period(
        net,
        td,
        steps=2,
        optimization_problem=create_economic_dispatch_problem(
            heat_cost_default=50, **extra
        ),
    )

    assert result.success
    assert _column(result, mm.HeatExchangerGenerator, "q_mw_set") == pytest.approx(
        duties, abs=1e-3
    )
