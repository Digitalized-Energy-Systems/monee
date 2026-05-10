from monee.model import Network
from monee.problem import OptimizationProblem
from monee.solver.dispatch import resolve_solver


def solve(
    net: Network,
    optimization_problem: OptimizationProblem = None,
    solver=None,
    backend: str | None = None,
    **kwargs,
):
    """Solve *net*.

    Args:
        net: The network to solve.
        optimization_problem: Optional optimisation problem.
        solver: Either a solver-name string (``"ipopt"``, ``"gurobi"``,
            ``"scip"``, …), a concrete :class:`SolverInterface` instance, or
            ``None`` (default — GEKKO+IPOPT).
        backend: ``"gekko"`` / ``"pyomo"`` to force the modelling backend.  When
            ``None`` (default), the backend is auto-routed from *solver*.
        **kwargs: Forwarded to ``solver.solve(...)``.
    """
    actual_solver = resolve_solver(solver, backend=backend)
    return actual_solver.solve(net, optimization_problem=optimization_problem, **kwargs)
