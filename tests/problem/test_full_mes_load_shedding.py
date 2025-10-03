import math
import random

import monee.express as mx
import monee.model as mm
import monee.network.mes as mes
import monee.problem as mp
from monee import run_energy_flow, run_energy_flow_optimization
from monee.model import Bus, ExtPowerGrid, PowerGenerator, PowerLine, PowerLoad, Source


def create_four_line_example():
    random.seed(9002)
    pn = mm.Network()

    node_0 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerGenerator(p_mw=0.1, q_mvar=0.1, regulation=0.5))],
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0.1, vm_pu=1, va_degree=0))],
    )
    node_2 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerLoad(p_mw=0.1, q_mvar=0))],
    )
    node_3 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerLoad(p_mw=0.2, q_mvar=0))],
    )
    node_4 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerLoad(p_mw=0.2, q_mvar=0))],
    )
    node_5 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerGenerator(p_mw=0.3, q_mvar=0.1, regulation=0.5))],
    )
    node_6 = pn.node(
        Bus(base_kv=1),
        mm.EL,
        child_ids=[pn.child(PowerGenerator(p_mw=0.2, q_mvar=0.1, regulation=0.5))],
    )

    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_1,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_1,
        node_2,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_1,
        node_5,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_2,
        node_3,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_3,
        node_4,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_3,
        node_6,
    )

    new_mes = pn.copy()

    # gas
    bus_to_gas_junc = mes.create_gas_net_for_power(pn, new_mes, 1, scaling=1)
    new_mes.childs_by_type(Source)[0].model.regulation = 1

    # heat
    bus_index_to_junction_index, bus_index_to_end_junction_index = (
        mes.create_heat_net_for_power(pn, new_mes, 1)
    )
    new_water_junc = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc,
        mass_flow=0.075,
    )
    new_water_junc_2 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_2,
        mass_flow=0.075,
    )
    mx.create_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc,
        to_node_id=new_water_junc_2,
        diameter_m=0.20,
        q_mw=0.005,
    )
    new_water_junc_3 = mx.create_water_junction(new_mes)
    mx.create_sink(
        new_mes,
        new_water_junc_3,
        mass_flow=0.075,
    )
    mx.create_heat_exchanger(
        new_mes,
        from_node_id=new_water_junc_2,
        to_node_id=new_water_junc_3,
        diameter_m=0.20,
        q_mw=0.005,
    )
    mx.create_p2g(
        new_mes,
        from_node_id=node_4,
        to_node_id=bus_to_gas_junc[node_4],
        efficiency=0.7,
        mass_flow_setpoint=0.01,
        regulation=0,
    )
    mx.create_chp(
        new_mes,
        power_node_id=node_1,
        heat_node_id=bus_index_to_junction_index[node_0],
        heat_return_node_id=new_water_junc,
        gas_node_id=bus_to_gas_junc[node_3],
        mass_flow_setpoint=0.0005,
        diameter_m=0.3,
        efficiency_power=0.5,
        efficiency_heat=0.5,
        regulation=0,
    )
    mx.create_g2p(
        new_mes,
        from_node_id=bus_to_gas_junc[node_1],
        to_node_id=node_1,
        efficiency=0.9,
        p_mw_setpoint=0.3,
        regulation=0,
    )
    mx.create_g2p(
        new_mes,
        from_node_id=bus_to_gas_junc[node_6],
        to_node_id=node_6,
        efficiency=0.9,
        p_mw_setpoint=1.5,
        regulation=0,
    )
    new_mes.branch(
        PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            parallel=1,
            backup=True,
            on_off=0,
        ),
        node_4,
        node_0,
    )
    new_mes.branch(
        PowerLine(
            length_m=100,
            r_ohm_per_m=0.00007,
            x_ohm_per_m=0.00007,
            parallel=1,
            backup=True,
            on_off=0,
        ),
        node_5,
        node_2,
    )
    return new_mes


BOUND_EL = ("vm_pu", 1, 0.1)
BOUND_GAS = ("pressure_pu", 1, 0.1)
BOUND_HEAT = ("t_pu", 1, 0.1)

bounds_el = (
    BOUND_EL[1] * (1 - BOUND_EL[2]),
    BOUND_EL[1] * (1 + BOUND_EL[2]),
)
bounds_heat = (
    BOUND_HEAT[1] * (1 - BOUND_HEAT[2]),
    BOUND_HEAT[1] * (1 + BOUND_HEAT[2]),
)
bounds_gas = (
    BOUND_GAS[1] * (1 - BOUND_GAS[2]),
    BOUND_GAS[1] * (1 + BOUND_GAS[2]),
)


def test_load_shedding_multimicrogrid():
    net_multi = create_four_line_example()

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0.0, 0.1),
        ext_grid_gas_bounds=(0.0, 0.1),
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience == (0.0, 0, 0.0)
    assert result is not None


def test_load_shedding_multimicrogrid_chp_save():
    net_multi = create_four_line_example()
    net_multi.branch_by_id((3, 6, 0)).active = False

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0.0, 0.1),
        ext_grid_gas_bounds=(0.0, 0.1),
        use_ext_grid_bounds=True,
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience == (0.089030811558, 0, 0.0)
    assert result is not None


def test_load_shedding_multimicrogrid_gas_shedding():
    net_multi: mm.Network = create_four_line_example()
    net_multi.childs_by_type(Source)[0].model.mass_flow = -2

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0.0, 0),
        ext_grid_gas_bounds=(0.0, 0),
        use_ext_grid_bounds=True,
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert math.isclose(resilience[2], 27.472577276197804, abs_tol=0.01)
    assert result is not None


def test_load_shedding_multimicrogrid_heat_cooldown():
    net_multi: mm.Network = create_four_line_example()
    net_multi.branch_by_id((26, 27, 0)).model.q_w_set = 10000

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0.0, 0.1),
        ext_grid_gas_bounds=(0.0, 0.1),
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert math.isclose(resilience[1], 0.0, abs_tol=0.0001)
    assert result is not None


def test_scaled_example_gas_incident():
    net_multi: mm.Network = mes.create_monee_benchmark_net()
    net_multi.childs_by_type(Source)[0].model.mass_flow = -1.3

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0, 10),
        ext_grid_gas_bounds=(0, 10),
        use_ext_grid_bounds=False,
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience[0] == 0
    assert math.isclose(resilience[2], -28.643805241083786, abs_tol=0.01)
    assert result is not None


def test_scaled_load_shedding_def():
    net_multi = mes.create_monee_benchmark_net()

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0.0, 1),
        ext_grid_gas_bounds=(0.0, 1),
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience == (0, 0, 0.0)
    assert result is not None


def test_scaled_load_shedding_def_2_3():
    net_multi = mes.create_monee_benchmark_net()
    net_multi.branch_by_id((1, 2, 0)).active = False

    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0, 10),
        ext_grid_gas_bounds=(0, 10),
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience == (0, 0, 0.0)
    assert result is not None


def test_scaled_load_shedding_def_multi_line():
    net_multi = mes.create_monee_benchmark_net()
    net_multi.branch_by_id((1, 2, 0)).active = False
    net_multi.branch_by_id((10, 13, 0)).active = False

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=(0.0, 10),
        ext_grid_gas_bounds=(0.0, 10),
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience == (0, 0.0, 38.55489725826209)
    assert result is not None


def test_scaled_load_shedding_def_cigre():
    net_multi = mes.create_mv_multi_cigre()
    net_multi.branch_by_id((2, 3, 0)).active = False
    print(run_energy_flow(net_multi))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=(0, 2),
        bounds_heat=(0, 2),
        bounds_gas=(0, 2),
        ext_grid_el_bounds=(-0.0, 10),
        ext_grid_gas_bounds=(-0.0, 10),
        use_ext_grid_bounds=True,
        use_ext_grid_objective=True,
        debug=True,
    )
    result = run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience == (22.06759231082297, 0.0, 0)
    assert result is not None
