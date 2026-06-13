import math

import monee
import monee.model as mm
import monee.problem as mp
import monee.solver as ms
from monee.model.formulation import EL_MISOCP_FORMULATION
from monee.network.mes import create_monee_benchmark_net
from monee.solver import PyomoSolver

BOUND_EL = ("vm_pu", 1, 0.5)
BOUND_GAS = ("pressure_pu", 1, 0.5)
BOUND_HEAT = ("t_pu", 1, 0.5)

bounds_vm = (
    BOUND_EL[1] * (1 - BOUND_EL[2]),
    BOUND_EL[1] * (1 + BOUND_EL[2]),
)
bounds_t = (
    BOUND_HEAT[1] * (1 - BOUND_HEAT[2]),
    BOUND_HEAT[1] * (1 + BOUND_HEAT[2]),
)
bounds_pressure = (
    BOUND_GAS[1] * (1 - BOUND_GAS[2]),
    BOUND_GAS[1] * (1 + BOUND_GAS[2]),
)

bounds_ext_el = (0, 100)
bounds_ext_gas = (0, 100)
# The benchmark net's water slack needs ~15 kg/s circulation for its heat
# demand; the problem default of (-10, 10) makes the model provably
# infeasible (verified with Gurobi DualReductions=0).
bounds_ext_heat = (-100, 100)


def test_scaled_example_gas_incident_pyo():
    # GIVEN
    net_multi: mm.Network = create_monee_benchmark_net()
    net_multi.apply_formulation(EL_MISOCP_FORMULATION)
    optimization_problem = mp.create_min_load_shedding_problem(
        bounds_vm=bounds_vm,
        bounds_t=bounds_t,
        bounds_pressure=bounds_pressure,
        bounds_ext_el=bounds_ext_el,
        bounds_ext_gas=bounds_ext_gas,
        bounds_ext_heat=bounds_ext_heat,
        include_ext_grids=True,
        debug=True,
    )

    # WHEN
    flow_result = monee.run_energy_flow(net_multi, solver=PyomoSolver())
    result = monee.run_energy_flow_optimization(
        net_multi, optimization_problem=optimization_problem, solver=PyomoSolver()
    )
    resilience = mp.calc_general_resilience_performance(result.network)

    # THEN
    assert flow_result.success
    assert result.success

    assert resilience[0] == 0
    assert math.isclose(resilience[2], 0, abs_tol=0.01)
    assert result is not None


def create_multi_chp():
    pn = mm.Network()

    # WATER
    w_node_0 = pn.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[pn.child(mm.ExtHydrGrid(t_k=359))],
    )
    w_node_1 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_2 = pn.node(mm.Junction(), grid=mm.WATER_KEY)
    w_node_3 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.ConsumeHydrGrid(1))],
        grid=mm.WATER_KEY,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=100),
        w_node_0,
        w_node_1,
    )
    pn.branch(
        mm.WaterPipe(diameter_m=0.15, length_m=200),
        w_node_2,
        w_node_3,
    )

    # GAS
    gas_grid = mm.create_gas_grid("gas", type="lgas")
    g_node_0 = pn.node(
        mm.Junction(),
        child_ids=[pn.child(mm.Source(mass_flow_kgs=0.1))],
        grid=gas_grid,
    )
    g_node_1 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.ExtHydrGrid())], grid=gas_grid
    )
    g_node_2 = pn.node(
        mm.Junction(), child_ids=[pn.child(mm.Sink(mass_flow_kgs=1))], grid=gas_grid
    )

    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=100, temperature_ext_k=300, roughness_m=0.01
        ),
        g_node_0,
        g_node_1,
    )
    pn.branch(
        mm.GasPipe(
            diameter_m=0.35, length_m=150, temperature_ext_k=300, roughness_m=0.01
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
        mm.CHP(0.1, 0.6, 0.4, 0.1, regulation=1),
        gas_node_id=g_node_2,
        heat_node_id=w_node_1,
        heat_return_node_id=w_node_2,
        power_node_id=el_node_2,
    )

    return pn


def test_simple_chp():
    # GIVEN
    multi_energy_network = create_multi_chp()
    multi_energy_network.apply_formulation(EL_MISOCP_FORMULATION)

    # WHEN
    result = ms.PyomoSolver().solve(multi_energy_network)

    # THEN
    assert result.success

    assert len(result.dataframes) == 15
    assert math.isclose(
        result.dataframes["ExtPowerGrid"]["p_mw"][0],
        3.1985501723755214,
        abs_tol=0.001,
    )
    assert math.isclose(
        result.dataframes["ExtHydrGrid"]["mass_flow_kgs"][1],
        -1.0,
    )
    assert math.isclose(
        result.dataframes["Junction"]["t_k"][1], 358.98302500340054, abs_tol=0.05
    )


def test_monee_visu():
    import plotly.graph_objects as go

    # GIVEN
    net_multi: mm.Network = create_monee_benchmark_net()
    net_multi.apply_formulation(EL_MISOCP_FORMULATION)
    result = monee.run_energy_flow(net_multi, solver=PyomoSolver())

    from monee.visualization import plot_result

    # WHEN
    fig = plot_result(result)

    # THEN
    assert result.success

    assert isinstance(fig, go.Figure)
    assert len(fig.data) > 0
