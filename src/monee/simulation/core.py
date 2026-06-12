from monee.model import Network
from monee.problem import OptimizationProblem
from monee.solver.dispatch import resolve_solver


def solve(
    net: Network,
    optimization_problem: OptimizationProblem = None,
    solver=None,
    backend: str | None = None,
    formulation=None,
    **kwargs,
):
    """Solve *net*.

    Args:
        net: The network to solve.
        optimization_problem: Optional optimisation problem.
        solver: Either a solver-name string (``"ipopt"``, ``"gurobi"``,
            ``"scip"``, …), a concrete :class:`SolverInterface` instance, or
            ``None`` (default - GEKKO+IPOPT).
        backend: ``"gekko"`` / ``"pyomo"`` to force the modelling backend.  When
            ``None`` (default), the backend is auto-routed from *solver*.
        formulation: Solve-time formulation - a registry key string
            (``"smooth_nlp"``, ``"convex_miqcqp"``, ``"el_misocp"``, …), a
            :class:`~monee.model.formulation.core.NetworkFormulation`, or a
            sequence of either (merged left to right). Overrides the network's
            ``apply_formulation`` choice for this solve; components without any
            choice use ``DEFAULT_SIMULATION_FORMULATION``.
        **kwargs: Forwarded to ``solver.solve(...)``.
    """
    actual_solver = resolve_solver(solver, backend=backend)
    return actual_solver.solve(
        net,
        optimization_problem=optimization_problem,
        formulation=formulation,
        **kwargs,
    )
