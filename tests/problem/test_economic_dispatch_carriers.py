"""Economic dispatch over gas and heat: merit orders per carrier, coupling
points, heat pricing at the water slack, and the guards around them."""

import warnings

import pytest

import monee.express as mx
import monee.model as mm
from monee import run_energy_flow_optimization
from monee.model.core import Var
from monee.model.grid import GasGrid
from monee.model.multi import SubHE
from monee.problem import create_economic_dispatch_problem
from monee.problem.utils import gas_mw_per_kgs
from monee.solver.core import prepare_solve_network
from monee.solver.pyo import _pyscipopt_available
from tests.problem._ed_networks import (
    chp_hg_net,
    docs_el_net,
    g2p_net,
    gas_merit_net,
    gas_pressure_net,
    gas_storage_net,
    heat_dhs_net,
    heat_loop_net,
    hg_couplers_net,
)
from tests.problem.test_problem_core import _gas_only_net
from tests.solver.test_chp import _chp_behind_power_line_network

needs_scip = pytest.mark.skipif(not _pyscipopt_available(), reason="pyscipopt missing")
SOLVERS = [None, pytest.param("scip", marks=needs_scip)]


def _solve(net, problem, solver=None):
    result = run_energy_flow_optimization(net, problem, solver=solver)
    assert result.success
    return result


def _zero_match_warnings(caught):
    return [w for w in caught if "matched no component" in str(w.message)]


def _control_node(result, table):
    return float(result.dataframes[table]["regulation"].iloc[0])


def test_el_only_problem_unchanged_by_disabled_or_enabled_carriers():
    base = _solve(
        docs_el_net(),
        create_economic_dispatch_problem(
            ext_grid_cost_default=25, ext_grid_bounds=(-0.5, 0.0)
        ),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        widened = _solve(
            docs_el_net(),
            create_economic_dispatch_problem(
                ext_grid_cost_default=25,
                ext_grid_bounds=(-0.5, 0.0),
                gas_cost_default=5.0,
                heat_cost_default=5.0,
            ),
        )

    assert base.user_objective == pytest.approx(11.357, abs=1e-3)
    assert widened.user_objective == pytest.approx(base.user_objective, rel=1e-9)
    assert list(widened.get(mm.PowerGenerator)["p_mw"]) == pytest.approx(
        list(base.get(mm.PowerGenerator)["p_mw"]), abs=1e-9
    )
    assert _zero_match_warnings(caught) == []


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "gas_type, source_cost, ext_cost, source_kgs, ext_kgs, objective",
    [
        ("lgas", 10.0, 30.0, -0.1, -0.05, 106.111),
        ("lgas", 30.0, 10.0, 0.0, -0.15, 63.667),
        ("methane", 10.0, 30.0, -0.1, -0.05, 137.70),
        ("methane", 30.0, 10.0, 0.0, -0.15, 82.62),
    ],
)
def test_gas_merit_order(
    solver, gas_type, source_cost, ext_cost, source_kgs, ext_kgs, objective
):
    net, source_id, ext_id = gas_merit_net(source_cost, gas_type)

    result = _solve(
        net, create_economic_dispatch_problem(gas_cost_default=ext_cost), solver
    )

    source = result.get(mm.Source)["mass_flow_kgs"].iloc[0]
    ext = result.get(mm.ExtHydrGrid)["mass_flow_kgs"].iloc[0]
    assert source == pytest.approx(source_kgs, abs=1e-4)
    assert ext == pytest.approx(ext_kgs, abs=1e-4)
    mw_per_kgs = gas_mw_per_kgs(net.child_by_id(source_id).grid)
    by_hand = mw_per_kgs * (source_cost * -source + ext_cost * -ext)
    assert result.user_objective == pytest.approx(by_hand, rel=1e-6)
    assert result.user_objective == pytest.approx(objective, abs=1e-2)


def test_gas_pressure_band_limits_the_source():
    free = _solve(
        gas_pressure_net()[0], create_economic_dispatch_problem(gas_cost_default=10)
    )
    net, _, far_junction = gas_pressure_net()
    problem = create_economic_dispatch_problem(gas_cost_default=10)
    problem.bounds(
        (0.81, 1.1025),
        lambda m, g: type(m) is mm.Junction and type(g) is GasGrid,
        ["pressure_squared_pu"],
        optional=True,
    )
    banded = _solve(net, problem)

    assert free.get(mm.Source)["mass_flow_kgs"].iloc[0] == pytest.approx(
        -0.1024, abs=1e-3
    )
    assert free.user_objective == pytest.approx(45.789, abs=1e-2)
    assert banded.get(mm.Source)["mass_flow_kgs"].iloc[0] == pytest.approx(
        -0.0599, abs=1e-3
    )
    assert banded.get(mm.Junction, index="id").loc[
        far_junction, "pressure_pu"
    ] == pytest.approx(1.05, abs=1e-3)
    assert banded.user_objective == pytest.approx(62.011, abs=1e-2)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "plant_cost, generator_mw, objective", [(50.0, -0.2, 13.6128), (20.0, 0.0, 7.0451)]
)
def test_heat_dhs_merit_order(solver, plant_cost, generator_mw, objective):
    net, _, plant_id, return_junction = heat_dhs_net(30.0)

    result = _solve(
        net, create_economic_dispatch_problem(heat_cost_default=plant_cost), solver
    )

    assert result.get(mm.HeatGenerator)["q_mw_heat"].iloc[0] == pytest.approx(
        generator_mw, abs=1e-4
    )
    assert result.user_objective == pytest.approx(objective, abs=1e-3)
    loads = result.get(mm.HeatExchangerLoad)
    assert list(loads["q_mw_delivered"]) == pytest.approx(
        list(loads["q_mw_set"]), abs=1e-5
    )
    assert not any("shortfall" in key for key in result.violations)
    m_plant = abs(result.get(mm.ExtHydrGrid, index="id").loc[plant_id, "mass_flow_kgs"])
    t_return = result.get(mm.Junction, index="id").loc[return_junction, "t_k"]
    plant_mw = 4184 * m_plant * (358.0 - t_return) / 1e6
    assert (result.user_objective - 30.0 * -generator_mw) / plant_cost == pytest.approx(
        plant_mw, rel=2e-3
    )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "slack_cost, bounds, duty, objective",
    [
        (50.0, None, -0.6, 13.0386),
        (20.0, None, 0.0, 10.0082),
        (50.0, "default", -0.5007, 15.0213),
    ],
    ids=["surplus_credited", "slack_cheaper", "supply_only_default"],
)
def test_heat_loop_merit_order(solver, slack_cost, bounds, duty, objective):
    net, hx_id, _ = heat_loop_net(30.0)
    extra = {} if bounds == "default" else {"bounds_ext_heat_mw": bounds}

    result = _solve(
        net,
        create_economic_dispatch_problem(heat_cost_default=slack_cost, **extra),
        solver,
    )

    hx = result.get(mm.HeatExchangerGenerator, index="id").loc[hx_id]
    assert hx["q_mw_set"] == pytest.approx(duty, abs=1e-3)
    assert hx["q_mw_delivered"] == pytest.approx(hx["q_mw_set"], abs=1e-6)
    assert result.user_objective == pytest.approx(objective, abs=1e-3)


def test_heat_mode_removes_duty_reward():
    net, _, _ = heat_loop_net(30.0)

    result = _solve(net, create_economic_dispatch_problem(heat_cost_default=50.0))

    assert result.aux_objective == pytest.approx(0.0, abs=1e-6)


def test_supply_cap_on_the_heat_slack():
    net, hx_id, _ = heat_loop_net(30.0)

    result = _solve(
        net,
        create_economic_dispatch_problem(
            heat_cost_default=20.0, bounds_ext_heat_mw=(-0.1, 0.0)
        ),
    )

    assert (
        result.get(mm.HeatExchangerGenerator, index="id").loc[hx_id, "q_mw_set"] < -0.39
    )


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "el, gas, heat, regulation, objective",
    [
        (100.0, 20.0, 30.0, 1.0, 95.9235),
        (10.0, 50.0, 30.0, 0.0, 19.0885),
        (30.0, 40.0, 100.0, 1.0, 55.1115),
        (30.0, 40.0, 10.0, 0.0, 33.2188),
    ],
)
def test_chp_hg_merit_order(solver, el, gas, heat, regulation, objective):
    result = _solve(
        chp_hg_net(),
        create_economic_dispatch_problem(
            ext_grid_cost_default=el, gas_cost_default=gas, heat_cost_default=heat
        ),
        solver,
    )

    assert _control_node(result, "CHPHGControlNode") == pytest.approx(
        regulation, abs=1e-4
    )
    assert result.user_objective == pytest.approx(objective, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "bounds, el, gas, regulation, subhe_mw, objective",
    [
        (None, 100.0, 20.0, 1.0, -1.91, -20.5039),
        (None, 10.0, 50.0, 0.0, 0.0, 10.2085),
        ("default", 100.0, 20.0, 0.0, 0.0, 102.0851),
    ],
    ids=["surplus_credited", "gas_expensive", "supply_only_default"],
)
def test_chp_he_exact_subhe_duty(
    solver, bounds, el, gas, regulation, subhe_mw, objective
):
    extra = {} if bounds == "default" else {"bounds_ext_heat_mw": bounds}

    result = _solve(
        _chp_behind_power_line_network(),
        create_economic_dispatch_problem(
            ext_grid_cost_default=el,
            gas_cost_default=gas,
            heat_cost_default=30.0,
            **extra,
        ),
        solver,
    )

    subhe = result.dataframes["SubHE"].iloc[0]
    assert _control_node(result, "CHPControlNode") == pytest.approx(
        regulation, abs=1e-4
    )
    assert subhe["q_mw"] == pytest.approx(subhe_mw, abs=1e-3)
    assert subhe["q_mw_delivered"] == pytest.approx(subhe["q_mw"], abs=1e-6)
    assert result.user_objective == pytest.approx(objective, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "el, gas, p2h, g2h, objective",
    [(20.0, 100.0, 1.0, 0.0, 9.2397), (100.0, 20.0, 0.0, 1.0, 9.4736)],
)
def test_p2h_vs_g2h_merit_order(solver, el, gas, p2h, g2h, objective):
    result = _solve(
        hg_couplers_net(),
        create_economic_dispatch_problem(
            ext_grid_cost_default=el, gas_cost_default=gas, heat_cost_default=50.0
        ),
        solver,
    )

    assert _control_node(result, "PowerToHeatHG") == pytest.approx(p2h, abs=1e-4)
    assert _control_node(result, "GasToHeatHG") == pytest.approx(g2h, abs=1e-4)
    assert result.user_objective == pytest.approx(objective, abs=1e-3)


@pytest.mark.parametrize("solver", SOLVERS)
@pytest.mark.parametrize(
    "gas, regulation, objective", [(20.0, 1.0, 50.5029), (30.0, 0.0, 63.0883)]
)
def test_g2p_merit_order(solver, gas, regulation, objective):
    result = _solve(
        g2p_net(),
        create_economic_dispatch_problem(
            ext_grid_cost_default=50.0, gas_cost_default=gas
        ),
        solver,
    )

    assert _control_node(result, "GasToPower") == pytest.approx(regulation, abs=1e-4)
    assert result.user_objective == pytest.approx(objective, abs=1e-3)


def test_g2p_stays_at_setpoint_without_coupler_dispatch():
    result = _solve(
        g2p_net(),
        create_economic_dispatch_problem(
            ext_grid_cost_default=50.0,
            gas_cost_default=30.0,
            dispatch_coupling_points=False,
        ),
    )

    assert _control_node(result, "GasToPower") == pytest.approx(1.0)


def _prepared(net, problem):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        network, _, _ = prepare_solve_network(net, optimization_problem=problem)
    return network


def _regulation(network, model_type):
    return next(
        c.model.regulation
        for c in network.all_components()
        if type(c.model) is model_type
    )


def test_coupler_gating_follows_the_dispatched_carriers():
    gas_only = _prepared(
        chp_hg_net(), create_economic_dispatch_problem(gas_cost_default=1.0)
    )
    heat_only = _prepared(
        g2p_net(), create_economic_dispatch_problem(heat_cost_default=1.0)
    )
    both = _prepared(
        chp_hg_net(),
        create_economic_dispatch_problem(gas_cost_default=1.0, heat_cost_default=1.0),
    )
    switched_off = _prepared(
        chp_hg_net(),
        create_economic_dispatch_problem(
            gas_cost_default=1.0, heat_cost_default=1.0, dispatch_coupling_points=False
        ),
    )

    assert not isinstance(_regulation(gas_only, mm.CHPHGControlNode), Var)
    assert not isinstance(_regulation(heat_only, mm.GasToPower), Var)
    promoted = _regulation(both, mm.CHPHGControlNode)
    assert isinstance(promoted, Var)
    assert (promoted.min, promoted.max) == (0, 1)
    assert not isinstance(_regulation(switched_off, mm.CHPHGControlNode), Var)


def test_derated_coupler_keeps_its_regulation_as_upper_bound():
    net = g2p_net()
    next(
        c for c in net.branches if isinstance(c.model, mm.GasToPower)
    ).model.regulation = 0.4

    promoted = _regulation(
        _prepared(net, create_economic_dispatch_problem(gas_cost_default=1.0)),
        mm.GasToPower,
    )

    assert (promoted.min, promoted.max) == (0, 0.4)


def test_hx_generator_promoted_and_duties_exact():
    net, hx_id, _ = heat_loop_net(30.0)

    heat = _prepared(net, create_economic_dispatch_problem(heat_cost_default=1.0))
    power = _prepared(net, create_economic_dispatch_problem())

    hx = heat.branch_by_id(hx_id).model
    assert isinstance(hx.q_mw_set, Var)
    assert (hx.q_mw_set.min, hx.q_mw_set.max) == (-0.6, 0)
    assert hx._he_duty_exact
    assert hx._ed_heat_generator
    assert not getattr(power.branch_by_id(hx_id).model, "_he_duty_exact", False)


def test_consumer_promoted_by_another_rule_is_not_a_heat_unit():
    net, _, _, _ = heat_dhs_net(30.0)
    problem = create_economic_dispatch_problem(heat_cost_default=50.0)
    problem.controllable_demands(["q_mw_set"])

    network = _prepared(net, problem)

    loads = [
        b.model for b in network.branches if isinstance(b.model, mm.HeatExchangerLoad)
    ]
    assert loads
    assert all(isinstance(m.q_mw_set, Var) for m in loads)
    assert not any(getattr(m, "_ed_heat_generator", False) for m in loads)


def test_subhe_duty_marked_exact():
    network = _prepared(
        _chp_behind_power_line_network(),
        create_economic_dispatch_problem(gas_cost_default=1.0, heat_cost_default=1.0),
    )

    subhe = next(c.model for c in network.branches if isinstance(c.model, SubHE))
    assert subhe._he_duty_exact


def test_water_temperature_band_applies_to_water_components_only():
    net = chp_hg_net()
    network = _prepared(
        net,
        create_economic_dispatch_problem(gas_cost_default=1.0, heat_cost_default=1.0),
    )

    water = [
        c.model
        for c in network.nodes
        if type(c.model) is mm.Junction and c.grid.name == mm.WATER_KEY
    ]
    gas = [
        c.model
        for c in network.nodes
        if type(c.model) is mm.Junction and isinstance(c.grid, GasGrid)
    ]
    for junction in water:
        if isinstance(junction.t_pu, Var):
            assert (junction.t_pu.min, junction.t_pu.max) == pytest.approx(
                (278.15 / 356, 423.15 / 356)
            )
    assert all((j.t_pu.min, j.t_pu.max) == (0.3, 2) for j in gas)


@pytest.mark.parametrize(
    "make_net, kwargs",
    [
        (docs_el_net, dict(gas_cost_default=1.0, heat_cost_default=1.0)),
        (_gas_only_net, dict(gas_cost_default=1.0, check_vm=False)),
        (lambda: heat_loop_net(30.0)[0], dict(heat_cost_default=50.0)),
        (
            chp_hg_net,
            dict(
                ext_grid_cost_default=30.0,
                gas_cost_default=40.0,
                heat_cost_default=50.0,
            ),
        ),
    ],
    ids=["el_only", "gas_only", "heat_only", "all_carriers"],
)
def test_no_zero_match_warnings(make_net, kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(make_net(), create_economic_dispatch_problem(**kwargs))

    assert _zero_match_warnings(caught) == []


def test_free_power_for_couplers_warns():
    with pytest.warns(UserWarning, match="ext_grid_cost_default"):
        _solve(
            hg_couplers_net(),
            create_economic_dispatch_problem(
                gas_cost_default=20.0, heat_cost_default=50.0
            ),
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _solve(
            hg_couplers_net(),
            create_economic_dispatch_problem(
                ext_grid_cost_default=20.0,
                gas_cost_default=20.0,
                heat_cost_default=50.0,
            ),
        )
    assert not [w for w in caught if "ext_grid_cost_default is None" in str(w.message)]


def test_gas_to_heat_only_does_not_warn_about_free_power():
    net = hg_couplers_net()
    p2h = next(c for c in net.branches if isinstance(c.model, mm.PowerToHeatHG))
    p2h.active = False

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        create_economic_dispatch_problem(
            gas_cost_default=20.0, heat_cost_default=50.0
        )._apply(prepare_solve_network(net)[0])

    assert not [w for w in caught if "ext_grid_cost_default is None" in str(w.message)]


def test_gas_cost_on_disabled_carrier_warns():
    net, _, _ = gas_merit_net(10.0)

    with pytest.warns(UserWarning, match="gas_cost_default"):
        result = _solve(net, create_economic_dispatch_problem(check_vm=False))

    assert result.get(mm.Source)["mass_flow_kgs"].iloc[0] == pytest.approx(-0.1)


def test_heat_cost_on_disabled_carrier_warns():
    net, _, _ = heat_loop_net(30.0)

    with pytest.warns(UserWarning, match="heat_cost_default"):
        create_economic_dispatch_problem()._apply(prepare_solve_network(net)[0])


def test_cost_on_water_source_warns_in_heat_mode():
    net, _, _ = heat_loop_net(30.0)
    junction = next(n.id for n in net.nodes)
    mx.create_water_source(net, junction, mass_flow_kgs=0.1, cost=5.0)

    with pytest.warns(UserWarning, match="water Source"):
        create_economic_dispatch_problem(heat_cost_default=10.0)._apply(
            prepare_solve_network(net)[0]
        )


def test_generator_cost_counts_regulation():
    net = mx.create_multi_energy_network()
    bus = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus)
    mx.create_power_generator(net, bus, p_mw=1.0, q_mvar=0.0, cost=10, regulation=0.5)
    mx.create_power_load(net, bus, p_mw=0.3, q_mvar=0.0)

    result = _solve(net, create_economic_dispatch_problem(ext_grid_cost_default=50))

    p_gen = result.get(mm.PowerGenerator)["p_mw"].iloc[0]
    p_ext = result.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
    assert p_gen == pytest.approx(-1.0, abs=1e-4)
    assert result.user_objective == pytest.approx(
        10 * 0.5 * -p_gen + 50 * -p_ext, rel=1e-6
    )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(bounds_ext_heat_mw=(None, 0.0)), "bounds_ext_heat_mw"),
        (dict(bounds_t_k=(300, 400)), "bounds_t_k"),
        (dict(ramp_limit={"steam": 1}), "ramp_limit"),
        (dict(ramp_limit={"gas": 1}), "gas_cost_default"),
        (dict(heat_cost_default=1.0, bounds_t_k=(400, 300)), "bounds_t_k"),
        (
            dict(heat_cost_default=1.0, bounds_ext_heat_mw=(0.0, -1.0)),
            "bounds_ext_heat_mw",
        ),
    ],
)
def test_factory_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        create_economic_dispatch_problem(**kwargs)


def test_two_water_slacks_in_one_island_raise():
    net, _, _ = heat_loop_net(30.0)
    junction = [n.id for n in net.nodes][1]
    mx.create_water_ext_grid(net, junction)

    with pytest.raises(ValueError, match="ExtHydrGrid"):
        run_energy_flow_optimization(
            net, create_economic_dispatch_problem(heat_cost_default=10.0)
        )


@needs_scip
def test_mccormick_heat_pricing_raises():
    import monee.solver as ms

    net = mm.Network()
    juncs = [mx.create_water_junction(net) for _ in range(3)]
    for a, b in zip(juncs, juncs[1:]):
        mx.create_water_pipe(net, a, b, diameter_m=0.15, length_m=100)
    mx.create_water_ext_grid(net, juncs[0], t_k=360.0)
    mx.create_water_sink(net, juncs[-1], mass_flow_kgs=1.0)
    mx.create_heat_load(net, juncs[1], q_mw=0.01)
    net.apply_formulation(mm.make_heat_convex_milp_formulation())

    with pytest.raises(ValueError, match="heat_convex_milp"):
        ms.PyomoSolver().solve(
            net,
            optimization_problem=create_economic_dispatch_problem(
                heat_cost_default=10.0
            ),
        )


def test_storages_fixed_outside_multi_period():
    net, storage_id, _ = gas_storage_net()

    with pytest.warns(UserWarning, match="include_storages"):
        result = _solve(
            net,
            create_economic_dispatch_problem(
                gas_cost_default=10.0, include_storages=True
            ),
        )

    assert result.get(mm.GasStorage, index="id").loc[
        storage_id, "mass_flow_kgs"
    ] == pytest.approx(0.0)
