"""Unit tests for controllable selection, bounds validation and the vm-bounds
hook of the problem layer."""

import logging
import warnings

import pytest

import monee.express as mx
import monee.model as mm
from monee import run_energy_flow_optimization
from monee.model.core import Intermediate, Var
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.model.formulation.registry import attach_formulations
from monee.problem.core import REGULATION_ATTR, Constraints, OptimizationProblem
from monee.problem.economic_dispatch import create_economic_dispatch_problem
from monee.problem.min_load_shedding import create_min_load_shedding_problem


def _hx_pair_net():
    """Water net with one consuming and one generating bare HeatExchanger."""
    net = mx.create_multi_energy_network()
    j1 = mx.create_water_junction(net)
    j2 = mx.create_water_junction(net)
    j3 = mx.create_water_junction(net)
    # HeatExchanger stores q_mw_set = -q_mw: q_mw=-0.1 -> q_mw_set=+0.1 (load),
    # q_mw=+0.1 -> q_mw_set=-0.1 (generator).
    load_id = net.branch(
        mm.HeatExchanger(q_mw=-0.1), j1, j2, grid=mm.WATER, name="hx_load"
    )
    gen_id = net.branch(
        mm.HeatExchanger(q_mw=0.1), j2, j3, grid=mm.WATER, name="hx_gen"
    )
    return net, net.branch_by_id(load_id).model, net.branch_by_id(gen_id).model


def test_controllable_demands_selects_consuming_hx_only():
    net, hx_load, hx_gen = _hx_pair_net()
    problem = OptimizationProblem()
    problem.controllable_demands(REGULATION_ATTR)
    problem._apply(net)

    selected = set(problem._controllable_to_attr.keys())
    assert hx_load in selected
    assert hx_gen not in selected


def test_controllable_generators_selects_generating_hx_only():
    net, hx_load, hx_gen = _hx_pair_net()
    problem = OptimizationProblem()
    problem.controllable_generators(REGULATION_ATTR)
    problem._apply(net)

    selected = set(problem._controllable_to_attr.keys())
    assert hx_gen in selected
    assert hx_load not in selected


def test_bounds_requires_attributes():
    problem = OptimizationProblem()
    with pytest.raises(ValueError, match="non-empty list of attribute names"):
        problem.bounds((0.9, 1.1))
    with pytest.raises(ValueError, match="non-empty list of attribute names"):
        problem.bounds((0.9, 1.1), attributes=[])


def _small_power_net():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    mx.create_ext_power_grid(net, b0, p_mw=0, q_mvar=0, vm_pu=1.0)
    mx.create_power_load(net, b1, p_mw=1, q_mvar=0.1)
    mx.create_line(net, b0, b1, length_m=100, r_ohm_per_m=3e-4, x_ohm_per_m=3e-4)
    return net


def _bus_models(network):
    return [n.model for n in network.nodes if type(n.model) is mm.Bus]


@pytest.mark.parametrize(
    "make_problem",
    [
        lambda: create_min_load_shedding_problem(bounds_vm=(0.9, 1.1)),
        lambda: create_economic_dispatch_problem(bounds_vm=(0.9, 1.1)),
    ],
    ids=["min_load_shedding", "economic_dispatch"],
)
def test_check_vm_bounds_squared_var_under_misocp(make_problem):
    net = _small_power_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    network = net.copy()
    attach_formulations(network, None)

    problem = make_problem()
    problem._apply(network)

    buses = _bus_models(network)
    assert buses
    for bus in buses:
        # MISOCP: vm_pu is only a reporting Intermediate; the decision variable
        # vm_pu_squared must carry the squared bounds.
        assert isinstance(bus.vm_pu, Intermediate)
        assert type(bus.vm_pu_squared) is Var
        assert bus.vm_pu_squared.min == pytest.approx(0.81)
        assert bus.vm_pu_squared.max == pytest.approx(1.21)


def test_check_vm_bounds_vm_pu_under_nlp():
    net = _small_power_net()
    network = net.copy()
    attach_formulations(network, None)

    problem = create_min_load_shedding_problem(bounds_vm=(0.9, 1.1))
    problem._apply(network)

    buses = _bus_models(network)
    assert buses
    for bus in buses:
        assert type(bus.vm_pu) is Var
        assert bus.vm_pu.min == pytest.approx(0.9)
        assert bus.vm_pu.max == pytest.approx(1.1)


def test_economic_dispatch_displaces_an_expensive_slack():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net)
    ba = mx.create_bus(net)
    bb = mx.create_bus(net)
    mx.create_line(net, b0, ba, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_line(net, b0, bb, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, ba, p_mw=0.1, q_mvar=0.0)
    gen_a = mx.create_power_generator(net, ba, p_mw=0.15, q_mvar=0.0)
    gen_b = mx.create_power_generator(net, bb, p_mw=0.15, q_mvar=0.0)
    net.child_by_id(gen_a).model.cost = 1.0
    net.child_by_id(gen_b).model.cost = 2.0

    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=100.0, ext_grid_bounds=(-1, 0)
    )
    result = run_energy_flow_optimization(net, problem)

    generation = -result.get(mm.PowerGenerator)["p_mw"].sum()
    slack_import = -result.get(mm.ExtPowerGrid)["p_mw"].sum()
    assert generation == pytest.approx(0.1, abs=1e-3)
    assert slack_import == pytest.approx(0.0, abs=1e-3)
    assert result.objective > 0


def test_economic_dispatch_rejects_nonzero_min_line_loading():
    with pytest.raises(ValueError, match="bounds_lp"):
        create_economic_dispatch_problem(bounds_lp=(0.1, 1.0))


def test_select_grids_returns_constraint():
    constraint = Constraints().select_grids((mm.PowerGrid,))
    assert constraint is not None


def test_controllable_all_returns_self():
    problem = OptimizationProblem()
    assert problem.controllable_all(["p_mw"]) is problem


def test_controllables_link_returns_callable():
    assert callable(OptimizationProblem().controllables_link())


def test_bounds_on_vm_pu_redirects_to_squared_var_under_misocp():
    net = _small_power_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    network = net.copy()
    attach_formulations(network, None)

    problem = OptimizationProblem()
    problem.bounds((0.95, 1.05), lambda m, _g: isinstance(m, mm.Bus), ["vm_pu"])
    problem._apply(network)

    buses = _bus_models(network)
    assert buses
    for bus in buses:
        assert isinstance(bus.vm_pu, Intermediate)
        assert bus.vm_pu_squared.min == pytest.approx(0.9025)
        assert bus.vm_pu_squared.max == pytest.approx(1.1025)


def test_bounds_on_a_non_var_attribute_warns(caplog):
    net = _small_power_net()
    network = net.copy()
    attach_formulations(network, None)

    problem = OptimizationProblem()
    problem.bounds((0.0, 1.0), lambda m, _g: isinstance(m, mm.Bus), ["base_kv"])
    with caplog.at_level(logging.WARNING, logger="monee.problem.core"):
        problem._apply(network)

    assert any("base_kv" in record.message for record in caplog.records)


def test_misocp_bus_keeps_the_grid_voltage_floor():
    net = _small_power_net()
    net.apply_formulation(EL_MISOCP_FORMULATION)
    network = net.copy()
    attach_formulations(network, None)

    for bus in _bus_models(network):
        assert bus.vm_pu_squared.min == pytest.approx(0.25)
        assert bus.vm_pu_squared.max == pytest.approx(2.25)


def _importing_net():
    net = mx.create_multi_energy_network()
    b0, b1, b2 = mx.create_bus(net), mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, 1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_line(net, b1, b2, 1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, b1, p_mw=0.6, q_mvar=0.0)
    mx.create_power_load(net, b2, p_mw=0.4, q_mvar=0.0)
    return net


def test_ext_power_grid_import_is_negative_p_mw():
    from monee import run_energy_flow

    result = run_energy_flow(_importing_net())

    p_mw = result.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
    assert p_mw < -1.0


def test_max_import_mw_becomes_the_lower_bound():
    ext = mm.ExtPowerGrid(p_mw=0, q_mvar=0, max_import_mw=0.5, max_export_mw=0.2)

    assert ext.p_mw.min == pytest.approx(-0.5)
    assert ext.p_mw.max == pytest.approx(0.2)


def test_import_only_bounds_ext_el_serve_the_load():
    from monee import solve_load_shedding_problem

    result = solve_load_shedding_problem(
        _importing_net(),
        bounds_vm=(0.9, 1.1),
        bounds_ext_el=(-9.8, 0.0),
        check_pressure=False,
        check_t=False,
    )

    assert result.get(mm.ExtPowerGrid)["p_mw"].iloc[0] < -0.9
    assert result.get(mm.PowerLoad)["regulation"].max() == pytest.approx(1.0, abs=1e-3)


def test_non_negative_bounds_ext_el_lower_bound_warns():
    with pytest.warns(UserWarning, match="forbids importing"):
        create_min_load_shedding_problem(bounds_ext_el=(0.0, 9.8))


def test_default_bounds_ext_el_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        create_min_load_shedding_problem()


def test_default_ext_bounds_do_not_shed_an_intact_4mw_feeder():
    from monee import solve_load_shedding_problem

    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    prev = b0
    for _ in range(3):
        b = mx.create_bus(net)
        mx.create_line(net, prev, b, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
        mx.create_power_load(net, b, p_mw=4.35 / 3, q_mvar=0.0)
        prev = b

    result = solve_load_shedding_problem(net, check_pressure=False, check_t=False)

    assert result.get(mm.PowerLoad)["regulation"].min() == pytest.approx(1.0, abs=1e-3)
    assert result.get(mm.ExtPowerGrid)["p_mw"].iloc[0] < -4.0


def test_constraint_selector_written_for_the_model_is_applied_with_warning():
    net = _small_power_net()
    constraints = Constraints()
    constraints.select(lambda m: isinstance(m, mm.ExtPowerGrid)).equation(
        lambda m: m.p_mw >= -0.5
    )

    with pytest.warns(UserWarning, match="applying the model interpretation"):
        equations = constraints.all(net)
    assert len(equations) == 1

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(constraints.all(net)) == 1


def test_controllable_condition_written_for_the_model_is_applied_with_warning():
    net = _small_power_net()
    problem = OptimizationProblem()
    problem.controllable(
        ["p_mw"], component_condition=lambda m: isinstance(m, mm.PowerLoad)
    )

    with pytest.warns(UserWarning, match="applying the model interpretation"):
        problem._apply(net)

    assert problem._controllable_to_attr
    assert all(type(m) is mm.PowerLoad for m in problem._controllable_to_attr)


def test_empty_typed_selector_warns_once():
    net = _small_power_net()
    constraints = Constraints()
    constraints.select_types(mm.HeatExchanger).equation(lambda m: m.q_mw >= 0)

    with pytest.warns(UserWarning, match="matched no component and has no effect"):
        assert constraints.all(net) == []

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert constraints.all(net) == []


def test_objective_selector_written_for_the_component_is_applied_with_warning():
    net = _small_power_net()
    from monee.problem.core import Objectives

    objectives = Objectives()
    objectives.select(
        lambda c: hasattr(c, "model") and isinstance(c.model, mm.PowerLoad)
    ).calculate(lambda models: sum(m.p_mw for m in models))

    with pytest.warns(UserWarning, match="applying the Component interpretation"):
        exprs = objectives.all(net)

    assert exprs == [1]


def test_two_arg_selectors_receive_model_and_component():
    net = _small_power_net()
    from monee.problem.core import Objectives

    constraints = Constraints()
    constraints.select(lambda m, c: isinstance(m, mm.ExtPowerGrid)).equation(
        lambda m: m.p_mw >= -0.5
    )
    objectives = Objectives()
    objectives.select(lambda m, c: isinstance(m, mm.PowerLoad) and c.active).calculate(
        lambda models: sum(m.p_mw for m in models)
    )
    problem = OptimizationProblem()
    problem.controllable(
        ["p_mw"], component_condition=lambda m, c: isinstance(m, mm.PowerLoad)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(constraints.all(net)) == 1
        assert objectives.all(net) == [1]
        problem._apply(net)

    assert [type(m) for m in problem._controllable_to_attr] == [mm.PowerLoad]


def test_one_arg_native_predicates_keep_working_without_warning():
    net = _small_power_net()
    from monee.problem.core import Objectives

    constraints = Constraints()
    constraints.select(lambda c: isinstance(c.model, mm.ExtPowerGrid)).equation(
        lambda m: m.p_mw >= -0.5
    )
    objectives = Objectives()
    objectives.select(lambda m: isinstance(m, mm.PowerLoad)).calculate(
        lambda models: sum(m.p_mw for m in models)
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(constraints.all(net)) == 1
        assert objectives.all(net) == [1]


def test_one_arg_bounds_condition_receives_the_model():
    net = _small_power_net()
    network = net.copy()
    attach_formulations(network, None)

    problem = OptimizationProblem()
    problem.bounds((0.92, 1.08), lambda m: isinstance(m, mm.Bus), ["vm_pu"])
    problem._apply(network)

    buses = _bus_models(network)
    assert buses
    for bus in buses:
        assert bus.vm_pu.min == pytest.approx(0.92)
        assert bus.vm_pu.max == pytest.approx(1.08)


def test_objective_selector_matching_nothing_warns():
    net = _small_power_net()
    from monee.problem.core import Objectives

    objectives = Objectives()
    objectives.select(lambda m: isinstance(m, mm.HeatExchanger))

    with pytest.warns(UserWarning, match="Objectives.select matched no component"):
        objectives.all(net)


def test_bounds_matching_nothing_warns_once():
    net = _small_power_net()
    problem = OptimizationProblem()
    problem.bounds(
        (0.0, 1.0),
        component_condition=lambda m, g: isinstance(m, mm.HeatExchanger),
        attributes=["q_mw"],
    )

    with pytest.warns(UserWarning, match=r"bounds\(\) on attributes \['q_mw'\]"):
        problem._apply(net)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problem._apply(net)


def _gas_only_net():
    net = mx.create_multi_energy_network()
    j0 = mx.create_gas_junction(net)
    j1 = mx.create_gas_junction(net)
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.3, length_m=100)
    mx.create_gas_ext_grid(net, j0)
    mx.create_sink(net, j1, mass_flow_kgs=0.1)
    return net


def _zero_match_warnings(caught):
    return [
        w
        for w in caught
        if issubclass(w.category, UserWarning)
        and "matched no component" in str(w.message)
    ]


@pytest.mark.parametrize(
    "make_net,kwargs",
    [
        (_small_power_net, dict(check_pressure=False, check_t=False)),
        (_small_power_net, dict()),
        (_gas_only_net, dict(check_vm=False, check_t=False)),
    ],
    ids=["el_only_no_checks", "el_only_default_checks", "gas_only"],
)
def test_factory_shedding_solve_has_no_zero_match_warnings(make_net, kwargs):
    from monee import solve_load_shedding_problem

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        solve_load_shedding_problem(make_net(), **kwargs)

    assert _zero_match_warnings(caught) == []


def test_user_zero_match_selector_warns_naming_the_types():
    net = _small_power_net()
    constraints = Constraints()
    constraints.select_types(mm.HeatExchanger).equation(lambda m: m.q_mw >= 0)

    with pytest.warns(UserWarning, match="It looked for HeatExchanger components"):
        assert constraints.all(net) == []


def test_optional_zero_match_selector_is_silent_and_logged(caplog):
    net = _small_power_net()
    constraints = Constraints()
    constraints.select_types(mm.HeatExchanger, optional=True).equation(
        lambda m: m.q_mw >= 0
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with caplog.at_level(logging.DEBUG, logger="monee.problem.core"):
            assert constraints.all(net) == []

    assert any(
        "matched no component" in record.message and "HeatExchanger" in record.message
        for record in caplog.records
    )


def test_user_controllable_zero_match_still_warns():
    net = _small_power_net()
    problem = OptimizationProblem()
    problem.controllable(
        ["q_mw"], component_condition=lambda m, c: isinstance(m, mm.HeatExchanger)
    )

    with pytest.warns(UserWarning, match=r"controllable\(\['q_mw'\]\) matched no"):
        problem._apply(net)


def test_optional_controllable_helper_zero_match_is_silent():
    net = _small_power_net()
    problem = OptimizationProblem()
    problem.controllable_cps(REGULATION_ATTR, optional=True)
    problem.controllable_backup_lines(optional=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problem._apply(net)


def test_optional_bounds_zero_match_is_silent():
    net = _small_power_net()
    problem = OptimizationProblem()
    problem.bounds(
        (0.9, 1.1),
        lambda m, g: isinstance(m, mm.Junction),
        ["t_pu"],
        optional=True,
        looked_for="Junction nodes on a WaterGrid",
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        problem._apply(net)
