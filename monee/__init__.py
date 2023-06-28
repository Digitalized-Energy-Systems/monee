import monee.model as mm
import monee.solver as ms
import monee.problem as mp


def run_energy_flow(net: mm.Network, solver=None):
    return run_energy_flow_optimization(net, None, solver=solver)


def run_energy_flow_optimization(
    net: mm.Network, optimization_problem: mp.OptimizationProblem, solver=None
):
    actual_solver = solver
    if actual_solver is None:
        actual_solver = ms.GEKKOSolver()
    return actual_solver.solve(net, optimization_problem=optimization_problem)
