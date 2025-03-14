import math

from monee.model.branch import PowerLine, Trafo
from monee.model.child import ExtPowerGrid, PowerGenerator, PowerLoad
from monee.model.core import Network, Var
from monee.model.grid import PowerGrid
from monee.model.node import Bus
from monee.problem.load_shedding import create_load_shedding_optimization_problem
from monee.solver.gekko import GEKKOSolver


def create_two_line_example_with_vm(vm, controllable_gen=False):
    pn = Network(PowerGrid(name="power", sn_mva=1))

    node_0 = pn.node(
        Bus(base_kv=1),
        child_ids=[
            pn.child(PowerGenerator(p_mw=Var(1) if controllable_gen else 1, q_mvar=0))
        ],
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=vm, va_degree=0))],
    )
    node_2 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))],
    )

    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_1,
    )
    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_2,
    )
    return pn


def create_two_gen_network(power_gen=1):
    pn = Network(PowerGrid(name="power", sn_mva=1))

    node_0 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerGenerator(p_mw=power_gen, q_mvar=0))],
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    node_2 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))],
    )
    node_3 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerGenerator(p_mw=0.1, q_mvar=0))],
    )

    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_1,
    )
    pn.branch(
        PowerLine(length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_0,
        node_2,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_2,
        node_3,
    )
    return pn, node_1


def create_trafo_network():
    pn = Network(PowerGrid(name="power", sn_mva=1))

    node_0 = pn.node(
        Bus(base_kv=10),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))],
    )
    node_2 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))],
    )

    pn.branch(
        Trafo(),
        node_0,
        node_1,
    )
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_1,
        node_2,
    )
    return pn


def create_four_line_example_with_inactive():
    pn = Network(PowerGrid(name="power", sn_mva=1))

    node_0 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerGenerator(p_mw=1, q_mvar=0))],
    )
    node_1 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    node_2 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))],
    )
    node_3 = pn.node(
        Bus(base_kv=1),
    )
    node_4 = pn.node(
        Bus(base_kv=1),
        child_ids=[pn.child(PowerLoad(p_mw=1, q_mvar=0))],
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
    pn.branch_by_id(
        pn.branch(
            PowerLine(
                length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
            ),
            node_2,
            node_3,
        )
    ).active = False
    pn.branch(
        PowerLine(length_m=100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1),
        node_3,
        node_4,
    )
    return pn


def test_simple_trafo():
    pn = create_trafo_network()
    solver = GEKKOSolver()

    result = solver.solve(pn)

    assert len(pn.as_dataframe_dict()) == 5
    assert result.dataframes["Bus"]["vm_pu"][0] == 1
    assert math.isclose(result.dataframes["Bus"]["vm_pu"][1], 0.98425648993)


def test_two_lines_example():
    pn = create_two_line_example_with_vm(1)
    solver = GEKKOSolver()

    result = solver.solve(pn)

    assert len(pn.as_dataframe_dict()) == 5
    assert len(pn.node_by_id(1).model.vars) == 5
    assert len(result.dataframes) == 5
    assert (
        result.dataframes["ExtPowerGrid"]["p_mw"][0] > -0.09
        and result.dataframes["ExtPowerGrid"]["p_mw"][0] < -0.08
    )


def test_two_lines_example_big_vm():
    pn = create_two_line_example_with_vm(2)
    solver = GEKKOSolver()

    result = solver.solve(pn)

    assert len(pn.as_dataframe_dict()) == 5
    assert len(pn.node_by_id(1).model.vars) == 5
    assert len(result.dataframes) == 5
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -0.018169406301)


def test_two_gen_example():
    pn, node_1 = create_two_gen_network()
    solver = GEKKOSolver()
    result = solver.solve(pn)

    assert len(pn.as_dataframe_dict()) == 5
    assert len(pn.node_by_id(node_1).model.vars) == 5
    assert len(result.dataframes) == 5
    assert (
        result.dataframes["ExtPowerGrid"]["p_mw"][0] > 0.03
        and result.dataframes["ExtPowerGrid"]["p_mw"][0] < 0.04
    )
    assert (
        result.dataframes["PowerLine"]["p_from_mw"][0] > 0
        and result.dataframes["PowerLine"]["p_to_mw"][0] < 0
    )


def test_two_controllable_lines_example_simple_constraint():
    pn = create_two_line_example_with_vm(1, controllable_gen=True)
    pn.constraint(lambda net: net.childs[1].model.vars["p_mw"] == 1)
    solver = GEKKOSolver()

    result = solver.solve(pn)

    assert len(pn.as_dataframe_dict()) == 5
    assert len(pn.node_by_id(1).model.vars) == 5
    assert len(result.dataframes) == 5
    assert result.dataframes["ExtPowerGrid"]["p_mw"][0] == 1
    assert math.isclose(result.dataframes["PowerGenerator"]["p_mw"][0], -2.142071799)


def test_two_controllable_lines_example_simple_objective():
    pn = create_two_line_example_with_vm(1, controllable_gen=True)
    pn.objective(lambda net: net.childs[1].model.vars["p_mw"])
    solver = GEKKOSolver()

    result = solver.solve(pn)

    assert len(pn.as_dataframe_dict()) == 5
    assert len(pn.node_by_id(1).model.vars) == 5
    assert len(result.dataframes) == 5
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -4.1428573107)
    assert math.isclose(result.dataframes["PowerGenerator"]["p_mw"][0], 1.1685869004)


def test_load_shedding_network_regulate_gen():
    pn, _ = create_two_gen_network()
    load_shedding_problem = create_load_shedding_optimization_problem(
        ext_grid_el_bounds=(0, 0)
    )

    result = GEKKOSolver().solve(pn, optimization_problem=load_shedding_problem)

    assert len(result.dataframes) == 5
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], 0)
    assert math.isclose(result.dataframes["PowerGenerator"]["p_mw"][0], -0.90687803989)


def test_load_shedding_network_regulate_load():
    pn, _ = create_two_gen_network(power_gen=0.1)
    load_shedding_problem = create_load_shedding_optimization_problem(
        ext_grid_el_bounds=(0, 0)
    )

    result = GEKKOSolver().solve(pn, optimization_problem=load_shedding_problem)

    assert len(result.dataframes) == 5
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], 0)
    assert math.isclose(result.dataframes["PowerLoad"]["p_mw"][0], 0.19922893999)


def test_not_connected_due_to_deactivation():
    pn = create_four_line_example_with_inactive()

    result = GEKKOSolver().solve(pn)

    assert len(result.dataframes) == 5
    assert math.isclose(result.dataframes["ExtPowerGrid"]["p_mw"][0], -0.01400300199)
    assert math.isclose(result.dataframes["PowerLoad"]["p_mw"][0], 1)
    assert math.isnan(result.dataframes["Bus"]["vm_pu"][3])
