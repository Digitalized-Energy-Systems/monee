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


def run_energy_flow(net: mm.Network, solver=None):
    return run_energy_flow_optimization(net, None, solver=solver)


def run_energy_flow_optimization(
    net: mm.Network, optimization_problem: mp.OptimizationProblem, solver=None
):
    return solve(net, optimization_problem, solver)
