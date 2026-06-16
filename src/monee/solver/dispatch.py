"""Solver dispatch - resolve ``(solver, backend)`` to a concrete solver.

Concrete instances pass through unchanged. IPOPT (the default solver) routes to
the in-process **CasADi** backend when installed (falling back to GEKKO's bundled
IPOPT otherwise); the discrete GEKKO solvers (APOPT/BPOPT) route to GEKKO; any
other name is forwarded to Pyomo. Default is therefore CasADi/IPOPT (GEKKO/IPOPT
without casadi).

``backend='gurobipy'`` selects the native in-memory Gurobi backend
(:class:`~monee.solver.gurobipy.GurobipySolver`) instead of driving Gurobi
through Pyomo's file round-trip; it is single-period only and the solver name
must be ``'gurobi'`` (the sole solver it provides).

``backend='casadi'`` selects the in-process CasADi/IPOPT backend
(:class:`~monee.solver.casadi.CasADiSolver`), which builds the NLP once as an
in-memory expression graph and calls IPOPT in-process (no subprocess). The
solver name must be ``'ipopt'`` (the sole solver it provides). It supports
single-period solves, temporal timeseries (storage/linepack/LTC) via the
per-step loop, and multi-period
(:class:`~monee.solver.casadi.CasADiMultiPeriodSolver`); for memory-less
timeseries it additionally enables a build-once / re-solve graph-reuse fast path
in :func:`monee.run_timeseries`.
"""

from __future__ import annotations

import pyomo.environ as pyo

from .core import SolverInterface

GEKKO_SOLVERS: dict[str, int] = {
    "apopt": 1,
    "bpopt": 2,
    "ipopt": 3,
}


def _casadi_available() -> bool:
    """Whether the optional CasADi backend can be imported. IPOPT requests
    (including the default) route to CasADi when it is, and fall back to GEKKO's
    bundled IPOPT when it is not - so ``import monee`` and the default solve path
    never hard-require casadi."""
    try:
        import casadi  # noqa: F401
    except ImportError:
        return False
    return True


def _pyomo_known_plugin(name: str) -> bool:
    """Plugin-registry membership; no subprocess spawn."""
    return name in set(pyo.SolverFactory)


def _pyomo_available_names() -> list[str]:
    """Installed Pyomo solvers; only called when building an error message."""
    return sorted(
        n
        for n in pyo.SolverFactory
        if not n.startswith("_")
        and pyo.SolverFactory(n).available(exception_flag=False)
    )


def _validate_pyomo(name: str) -> None:
    if not _pyomo_known_plugin(name):
        raise ValueError(
            f"Pyomo has no solver plugin named {name!r}.  "
            f"Installed solvers on this system: {_pyomo_available_names()}"
        )
    if (
        not pyo.SolverFactory(name).available(exception_flag=False)
        and name.lower() != "scip"
    ):
        raise ValueError(
            f"Pyomo solver {name!r} is registered but its executable / "
            f"Python API is not available on this system. Installed: "
            f"{_pyomo_available_names()}"
        )


def _is_solver_instance(obj) -> bool:
    return isinstance(obj, SolverInterface)


def _is_multi_period_solver_instance(obj) -> bool:
    return obj is not None and hasattr(obj, "solve_multi_period")


def _dispatch_backend(
    solver,
    backend,
    gekko_factory,
    pyomo_factory,
    gurobipy_factory=None,
    casadi_factory=None,
):
    """Shared (name → backend → construct) routing for the single- and
    multi-period resolvers.  *gekko_factory* / *pyomo_factory* take the resolved
    GEKKO solver code / Pyomo solver name respectively and return the concrete
    solver instance.  *gurobipy_factory* / *casadi_factory* (no arguments) build
    the native Gurobi / CasADi solvers; pass ``None`` where those backends are
    unavailable (e.g. multi-period)."""
    name = (solver or "ipopt").lower() if isinstance(solver, str) else "ipopt"
    chosen_backend = backend or _auto_backend(name)

    if chosen_backend == "gekko":
        if name not in GEKKO_SOLVERS:
            raise ValueError(
                f"GEKKO has no solver named {name!r}; choose from "
                f"{sorted(GEKKO_SOLVERS)} or pass backend='pyomo'."
            )
        return gekko_factory(GEKKO_SOLVERS[name])

    if chosen_backend == "pyomo":
        _validate_pyomo(name)
        return pyomo_factory(name)

    if chosen_backend == "casadi":
        if casadi_factory is None:
            raise ValueError(
                "The casadi backend is single-period only; use backend='pyomo' "
                "or backend='gekko' for multi-period solves."
            )
        if isinstance(solver, str) and name != "ipopt":
            raise ValueError(
                "The casadi backend only provides IPOPT; got "
                f"{solver!r}. Pass solver='ipopt' (or omit it) with "
                "backend='casadi'."
            )
        return casadi_factory()

    if chosen_backend == "gurobipy":
        if gurobipy_factory is None:
            raise ValueError(
                "The gurobipy backend is single-period only; use "
                "backend='pyomo' or backend='gekko' for multi-period solves."
            )
        if isinstance(solver, str) and name != "gurobi":
            raise ValueError(
                "The gurobipy backend only provides the 'gurobi' solver; got "
                f"{solver!r}. Pass solver='gurobi' (or omit it) with "
                "backend='gurobipy'."
            )
        return gurobipy_factory()

    raise ValueError(
        f"Unknown backend {chosen_backend!r}; expected 'gekko', 'pyomo' or "
        "'gurobipy'."
    )


def resolve_solver(
    solver=None,
    backend: str | None = None,
) -> SolverInterface:
    """Single-step :class:`SolverInterface` for ``(solver, backend)``."""
    if _is_solver_instance(solver):
        if backend is not None:
            raise ValueError(
                "backend= cannot be specified when solver= is already a "
                "concrete SolverInterface instance."
            )
        return solver

    def _gekko_factory(code):
        # Lazy import so a missing gekko install doesn't break Pyomo callers.
        from .gekko import GEKKOSolver

        return GEKKOSolver(solver=code)

    def _pyomo_factory(name):
        from .pyo import PyomoSolver

        return PyomoSolver(solver_name=name)

    def _gurobipy_factory():
        from .gurobipy import GurobipySolver

        return GurobipySolver()

    def _casadi_factory():
        from .casadi import CasADiSolver

        return CasADiSolver()

    if solver is None and backend is None:
        # Default solver is IPOPT -> CasADi when available, else GEKKO's IPOPT.
        if _casadi_available():
            return _casadi_factory()
        return _gekko_factory(GEKKO_SOLVERS["ipopt"])

    return _dispatch_backend(
        solver,
        backend,
        _gekko_factory,
        _pyomo_factory,
        _gurobipy_factory,
        _casadi_factory,
    )


def resolve_multi_period_solver(
    solver=None,
    backend: str | None = None,
):
    """Multi-period analogue of :func:`resolve_solver`."""
    if _is_multi_period_solver_instance(solver):
        if backend is not None:
            raise ValueError(
                "backend= cannot be specified when solver= is already a "
                "concrete multi-period solver instance."
            )
        return solver

    from monee.simulation.multi_period import (
        GekkoMultiPeriodSolver,
        PyomoMultiPeriodSolver,
    )

    def _casadi_mp_factory():
        from .casadi import CasADiMultiPeriodSolver

        return CasADiMultiPeriodSolver()

    if solver is None and backend is None:
        # Default solver is IPOPT -> CasADi when available, else GEKKO's IPOPT.
        if _casadi_available():
            return _casadi_mp_factory()
        return GekkoMultiPeriodSolver(solver=GEKKO_SOLVERS["ipopt"])

    return _dispatch_backend(
        solver,
        backend,
        lambda code: GekkoMultiPeriodSolver(solver=code),
        lambda name: PyomoMultiPeriodSolver(solver_name=name),
        casadi_factory=_casadi_mp_factory,
    )


def _auto_backend(name: str) -> str:
    """Route a solver name to a default backend.

    ``ipopt`` (the default solver) routes to the in-process **CasADi** backend
    when it is installed: CasADi solves the same continuous NLP as GEKKO/IPOPT
    but in-process (no subprocess / text round-trip) and is typically much
    faster, with full coverage of the smooth formulations (incl. gas/heat splines
    and temporal/multi-period coupling). It falls back to GEKKO's bundled IPOPT
    when casadi is absent. The discrete-capable GEKKO solvers (``apopt`` /
    ``bpopt``) stay on GEKKO; any other name goes to Pyomo.
    """
    if name == "ipopt":
        return "casadi" if _casadi_available() else "gekko"
    if name in GEKKO_SOLVERS:
        return "gekko"
    return "pyomo"
