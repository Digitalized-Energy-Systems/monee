"""Solver dispatch - resolve ``(solver, backend)`` to a concrete solver.

Concrete instances pass through unchanged. GEKKO names (APOPT/BPOPT/IPOPT)
route to GEKKO; anything else is forwarded to Pyomo. Default is GEKKO+IPOPT.
"""

from __future__ import annotations

import pyomo.environ as pyo

from .core import SolverInterface

GEKKO_SOLVERS: dict[str, int] = {
    "apopt": 1,
    "bpopt": 2,
    "ipopt": 3,
}

# Names in both backends - default to GEKKO (bundled IPOPT, faster on NLPs).
_DUAL_AVAILABLE = frozenset({"ipopt"})


def _pyomo_known_plugin(name: str) -> bool:
    """Plugin-registry membership; no subprocess spawn."""
    return name in pyo.SolverFactory.__dict__["_cls"]


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
    if not pyo.SolverFactory(name).available(exception_flag=False):
        raise ValueError(
            f"Pyomo solver {name!r} is registered but its executable / "
            f"Python API is not available on this system.  Installed: "
            f"{_pyomo_available_names()}"
        )


def _is_solver_instance(obj) -> bool:
    return isinstance(obj, SolverInterface)


def _is_multi_period_solver_instance(obj) -> bool:
    return obj is not None and hasattr(obj, "solve_multi_period")


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

    if solver is None and backend is None:
        # Lazy import so a missing gekko install doesn't break Pyomo callers.
        from .gekko import GEKKOSolver

        return GEKKOSolver(solver=GEKKO_SOLVERS["ipopt"])

    name = (solver or "ipopt").lower() if isinstance(solver, str) else "ipopt"
    chosen_backend = backend or _auto_backend(name)

    if chosen_backend == "gekko":
        if name not in GEKKO_SOLVERS:
            raise ValueError(
                f"GEKKO has no solver named {name!r}; choose from "
                f"{sorted(GEKKO_SOLVERS)} or pass backend='pyomo'."
            )
        from .gekko import GEKKOSolver

        return GEKKOSolver(solver=GEKKO_SOLVERS[name])

    if chosen_backend == "pyomo":
        _validate_pyomo(name)
        from .pyo import PyomoSolver

        return PyomoSolver(solver_name=name)

    raise ValueError(
        f"Unknown backend {chosen_backend!r}; expected 'gekko' or 'pyomo'."
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

    if solver is None and backend is None:
        return GekkoMultiPeriodSolver(solver=GEKKO_SOLVERS["ipopt"])

    name = (solver or "ipopt").lower() if isinstance(solver, str) else "ipopt"
    chosen_backend = backend or _auto_backend(name)

    if chosen_backend == "gekko":
        if name not in GEKKO_SOLVERS:
            raise ValueError(
                f"GEKKO has no solver named {name!r}; choose from "
                f"{sorted(GEKKO_SOLVERS)} or pass backend='pyomo'."
            )
        return GekkoMultiPeriodSolver(solver=GEKKO_SOLVERS[name])

    if chosen_backend == "pyomo":
        _validate_pyomo(name)
        return PyomoMultiPeriodSolver(solver_name=name)

    raise ValueError(
        f"Unknown backend {chosen_backend!r}; expected 'gekko' or 'pyomo'."
    )


def _auto_backend(name: str) -> str:
    """GEKKO names → ``"gekko"``; everything else → ``"pyomo"``."""
    if name in _DUAL_AVAILABLE:
        return "gekko"
    if name in GEKKO_SOLVERS:
        return "gekko"
    return "pyomo"
