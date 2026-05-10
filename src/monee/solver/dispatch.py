"""
Solver dispatch — resolve a ``(solver, backend)`` pair to a concrete solver
instance.

Goals:

- Single-step: ``(solver, backend) → SolverInterface``
  via :func:`resolve_solver`.
- Multi-period: ``(solver, backend) → MultiPeriodSolver``
  via :func:`resolve_multi_period_solver`.

Routing rules
-------------
1. If *solver* is already a backend instance (``SolverInterface`` for the
   single-step case, or has a ``solve_multi_period`` method for the multi-period
   case), return it unchanged.  This preserves the legacy
   ``solve(net, solver=PyomoSolver(...))`` and ``run_multi_period(...,
   solver=GekkoMultiPeriodSolver(...))`` callsites.
2. ``solver`` defaults to ``"ipopt"`` and ``backend`` defaults to auto-routed.
3. Auto-routing: solver names from the (small, fixed) GEKKO set route to GEKKO;
   any other name is forwarded to Pyomo's :class:`pyo.SolverFactory`, with
   availability checked at dispatch time.
4. ``backend="gekko"`` or ``"pyomo"`` forces the backend; the solver name is
   then validated against that backend's accepted names.

GEKKO has a 3-entry fixed solver table (APOPT/BPOPT/IPOPT); kept inline.
Pyomo's solver namespace is dynamic — we read it from
``pyo.SolverFactory.__dict__['_cls']`` for the cheap "is this a known plugin
name?" check, then fall back to ``solver.available()`` for the
"is the executable / Python API actually present?" check.
"""

from __future__ import annotations

import pyomo.environ as pyo

from .core import SolverInterface

# GEKKO solver-name → solver int.  GEKKO's solver set is fixed in upstream
# (APOPT / BPOPT / IPOPT) so a tiny inline table is fine.
GEKKO_SOLVERS: dict[str, int] = {
    "apopt": 1,
    "bpopt": 2,
    "ipopt": 3,
}

# Solver names that are dual-availability (present in both backends).  When the
# user passes one of these without an explicit ``backend``, we prefer GEKKO —
# its IPOPT is bundled with the gekko wheel and tends to be the faster default
# on monee's nonlinear formulations.
_DUAL_AVAILABLE = frozenset({"ipopt"})


def _pyomo_known_plugin(name: str) -> bool:
    """Cheap presence check against Pyomo's plugin registry — does NOT spawn
    subprocesses or import solver-specific deps."""
    return name in pyo.SolverFactory.__dict__["_cls"]


def _pyomo_available_names() -> list[str]:
    """Lazy enumeration of *installed* Pyomo solvers.  Spawns availability
    probes; only call when constructing an error message."""
    return sorted(
        n
        for n in pyo.SolverFactory
        if not n.startswith("_")
        and pyo.SolverFactory(n).available(exception_flag=False)
    )


def _validate_pyomo(name: str) -> None:
    """Raise :exc:`ValueError` if *name* is not a Pyomo plugin or its
    executable / Python API is unavailable on this system."""
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
    """``True`` if *obj* is a ready-to-use single-step solver."""
    return isinstance(obj, SolverInterface)


def _is_multi_period_solver_instance(obj) -> bool:
    """``True`` if *obj* is a ready-to-use multi-period solver."""
    return obj is not None and hasattr(obj, "solve_multi_period")


def resolve_solver(
    solver=None,
    backend: str | None = None,
) -> SolverInterface:
    """Return a single-step :class:`SolverInterface` for *(solver, backend)*.

    Args:
        solver: Either a solver name (str), a ready-made
            :class:`SolverInterface`, or ``None`` (uses the default GEKKO+IPOPT).
        backend: ``"gekko"`` or ``"pyomo"`` to force a backend, or ``None`` to
            auto-route from the solver name.
    """
    if _is_solver_instance(solver):
        if backend is not None:
            raise ValueError(
                "backend= cannot be specified when solver= is already a "
                "concrete SolverInterface instance."
            )
        return solver

    if solver is None and backend is None:
        # Default path: GEKKO+IPOPT.  Imported lazily so callers without
        # gekko installed who pass solver="gurobi" don't hit an import error.
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
    """Return a multi-period solver for *(solver, backend)*.

    Same semantics as :func:`resolve_solver` but returns
    :class:`~monee.simulation.multi_period.GekkoMultiPeriodSolver` /
    :class:`~monee.simulation.multi_period.PyomoMultiPeriodSolver`.
    """
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
    """Pick a backend for *name* without an explicit ``backend=`` argument.

    Rules:
    * GEKKO-only names (APOPT/BPOPT) → GEKKO.
    * Dual-available names (IPOPT) → GEKKO (default; faster on nonlinear).
    * Anything else → Pyomo.
    """
    if name in _DUAL_AVAILABLE:
        return "gekko"
    if name in GEKKO_SOLVERS:
        return "gekko"
    return "pyomo"
