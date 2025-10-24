import monee.solver as ms
from monee.model import Network
from monee.problem import OptimizationProblem


def solve(
    net: Network, optimization_problem: OptimizationProblem, solver=None, **kwargs
):
    """
    No docstring provided.
    """
    actual_solver = solver
    if actual_solver is None:
        solver_impl_id = 1 if optimization_problem is None else 3
        actual_solver = ms.GEKKOSolver(solver=solver_impl_id)
    return actual_solver.solve(net, optimization_problem=optimization_problem, **kwargs)
