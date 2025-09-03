import monee.model as mm
import monee.solver as ms
import monee.problem as mp
import monee.express as mx

from monee.model import Network
from monee.simulation import (
    solve,
    TimeseriesData,
    run_timeseries,
    TimeseriesResult,
    StepHook,
)


def run_energy_flow(net: mm.Network, solver=None, **kwargs):
    return run_energy_flow_optimization(net, None, solver=solver, **kwargs)


def run_energy_flow_optimization(
    net: mm.Network, optimization_problem: mp.OptimizationProblem, solver=None, **kwargs
):
    return solve(net, optimization_problem, solver, **kwargs)


def solve_load_shedding_problem(
    network: Network,
    bounds_vm: tuple,
    bounds_t: tuple,
    bounds_pressure: tuple,
    bounds_ext_el: tuple,
    bounds_ext_gas: tuple,
    debug=False,
    **kwargs
):
    optimization_problem = mp.create_load_shedding_optimization_problem(
        bounds_el=bounds_vm,
        bounds_heat=bounds_t,
        bounds_gas=bounds_pressure,
        ext_grid_el_bounds=bounds_ext_el,
        ext_grid_gas_bounds=bounds_ext_gas,
        debug=debug,
    )

    return run_energy_flow_optimization(
        network, optimization_problem=optimization_problem, **kwargs
    )
