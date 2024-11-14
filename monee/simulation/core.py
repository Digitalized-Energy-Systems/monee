import monee.solver as ms
from monee.model import Network
from monee.problem import OptimizationProblem


def solve(net: Network, optimization_problem: OptimizationProblem, solver=None):
    actual_solver = solver
    if actual_solver is None:
        actual_solver = ms.GEKKOSolver()
    return actual_solver.solve(net, optimization_problem=optimization_problem)
