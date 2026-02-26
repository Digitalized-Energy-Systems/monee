import math

import monee
import monee.model as mm
import monee.problem as mp
import monee.solver as ms
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network.mes import create_monee_benchmark_net
from monee.solver import PyomoSolver

BOUND_EL = ("vm_pu", 1, 0.5)
BOUND_GAS = ("pressure_pu", 1, 0.5)
BOUND_HEAT = ("t_pu", 1, 0.5)

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

ext_grid_el_bounds = (0, 100)
ext_grid_gas_bounds = (0, 100)


def test_scaled_example_gas_incident_pyo():
    net_multi: mm.Network = create_monee_benchmark_net()
    net_multi.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # net_multi.childs_by_type(mm.Source)[0].model.mass_flow = -1.3

    print(monee.run_energy_flow(net_multi, solver=PyomoSolver()))

    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_el,
        bounds_heat=bounds_heat,
        bounds_gas=bounds_gas,
        ext_grid_el_bounds=ext_grid_el_bounds,
        ext_grid_gas_bounds=ext_grid_gas_bounds,
        use_ext_grid_bounds=False,
        debug=True,
    )
    result = monee.run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem, solver=PyomoSolver()
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)
    assert resilience[0] == 0
    assert math.isclose(resilience[2], 0, abs_tol=0.01)
    assert result is not None


def create_in_line_p2h():
    pn = mm.Network(mm.create_water_grid("heat"))

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Sink(mass_flow=0.1))],
    )
    w_node_1 = pn.node(mm.Junction())
    w_node_2 = pn.node(mm.Junction())
    w_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=356))],
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        w_node_1,
        w_node_0,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_2,
        w_node_3,
    )

    # POWER
    power_grid = mm.create_power_grid("power")
    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0)),
        ],
        grid=power_grid,
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=power_grid,
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
        grid=power_grid,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )

    # multi
    pn.compound(
        mm.PowerToHeat(0.02 * 1000000, 0.15, 300, 1),
        power_node_id=el_node_2,
        heat_node_id=w_node_2,
        heat_return_node_id=w_node_1,
    )
    return pn


def test_in_line_p2h():
    multi_energy_network = create_in_line_p2h()
    multi_energy_network.apply_formulation(MISOCP_NETWORK_FORMULATION)

    result = ms.PyomoSolver().solve(multi_energy_network)
    print(result)
    assert len(result.dataframes) == 12
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][0], 394.13290124571745, abs_tol=0.001
    )


def create_multi_chp():
    pn = mm.Network()

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.Sink(mass_flow=1))],
    )
    w_node_1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_3 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.35, length_m=100),
        w_node_0,
        w_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.35, length_m=200),
        w_node_3,
        w_node_2,
    )

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow=0.1))],
        grid=gas_grid,
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow=1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=100, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=150, temperature_ext_k=300, roughness=0.01
        ),
        g_node_0,
        g_node_2,
    )

    # POWER
    power_grid = mm.create_power_grid("power")
    el_node_0 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[
            pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0)),
        ],
        grid=power_grid,
    )
    el_node_1 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0.1, q_mvar=0, vm_pu=1, va_degree=0))],
        grid=power_grid,
    )
    el_node_2 = pn.node(
        mm.Bus(base_kv=1),
        child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))],
        grid=power_grid,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.000007, x_ohm_per_m=0.000007, parallel=1
        ),
        el_node_0,
        el_node_1,
    )
    pn.branch(
        mm.PowerLine(
            length_m=1000, r_ohm_per_m=0.000007, x_ohm_per_m=0.000007, parallel=1
        ),
        el_node_0,
        el_node_2,
    )

    # multi
    pn.compound(
        mm.CHP(0.5, 0.6, 0.4, 0.0, regulation=0.5),
        gas_node_id=g_node_2,
        heat_node_id=w_node_1,
        heat_return_node_id=w_node_2,
        power_node_id=el_node_2,
    )

    return pn


def test_simple_chp():
    multi_energy_network = create_multi_chp()
    multi_energy_network.apply_formulation(MISOCP_NETWORK_FORMULATION)

    result = ms.PyomoSolver().solve(multi_energy_network)
    print(result)

    assert len(result.dataframes) == 14
    assert math.isclose(
        result.dataframes["ExtPowerGrid"]["p_mw"][0],
        -0.006264089217161262,
        abs_tol=0.001,
    )
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow"][1],
        -0.9,
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][1], 357.924809287306, abs_tol=0.001
    )
