import math

import monee
import monee.model as mm
import monee.problem as mp
from monee.model.formulation import MISOCP_NETWORK_FORMULATION
from monee.network.mes import create_monee_benchmark_net

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

ext_grid_el_bounds = (0, 10)
ext_grid_gas_bounds = (0, 10)


def test_scaled_example_gas_incident_pyo():
    net_multi: mm.Network = create_monee_benchmark_net()
    net_multi.apply_formulation(MISOCP_NETWORK_FORMULATION)
    # net_multi.childs_by_type(mm.Source)[0].model.mass_flow = -1.3

    print(monee.run_energy_flow(net_multi))

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
        net_multi, optimization_problem=optimization_problem
    )

    resilience = mp.calc_general_resilience_performance(result.network)

    print(result)
    print(result.objective)
    print(resilience)

    assert resilience[0] == 0
    assert math.isclose(resilience[2], 66.09600000018102, abs_tol=0.01)
    assert result is not None
