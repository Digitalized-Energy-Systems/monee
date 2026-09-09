"""Solver dispatch - resolve ``(solver, backend)`` to a concrete solver.

Concrete instances pass through unchanged. IPOPT (the default solver) routes to
the in-process **CasADi** backend when installed (falling back to GEKKO's bundled
IPOPT otherwise); the discrete GEKKO solvers (APOPT/BPOPT) route to GEKKO;
``gurobi`` routes to the native **gurobipy** backend for single-period solves
(falling back to Pyomo's Gurobi plugin for multi-period or when ``gurobipy`` is
not importable); any other name is forwarded to Pyomo. Default is therefore
CasADi/IPOPT (GEKKO/IPOPT without casadi).

``backend='gurobipy'`` selects the native in-memory Gurobi backend
(:class:`~monee.solver.gurobipy.GurobipySolver`) instead of driving Gurobi
through Pyomo's file round-trip; it is single-period only and the solver name
must be ``'gurobi'`` (the sole solver it provides). Single-period ``gurobi``
already resolves here by default; pass ``backend='pyomo'`` to force the Pyomo
round-trip instead.

``solver='auto'`` wraps the default in-process backend in :class:`AutoSolver`:
it solves as usual and, on a numerics failure only, retries once on SCIP,
recording on the result which backend produced it. It is single-period
(``run_energy_flow`` / ``run_timeseries`` / the Stepper), it is not the default,
and it cannot be combined with ``backend=``.

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

import logging

import pyomo.environ as pyo

from .core import ResultWarning, SolverInterface

_log = logging.getLogger(__name__)

GEKKO_SOLVERS: dict[str, int] = {
    "apopt": 1,
    "bpopt": 2,
    "ipopt": 3,
}

#: Solver name that asks for the in-process default plus one automatic retry on
#: :data:`AUTO_FALLBACK_SOLVER` (see :class:`AutoSolver`).
AUTO_SOLVER = "auto"

#: Back-end the ``'auto'`` solver retries a numerics failure on. SCIP is a
#: spatial branch-and-bound global solver that does not depend on the flat start
#: the interior-point default breaks on, and it attaches a structured
#: infeasibility report when the model really is infeasible.
AUTO_FALLBACK_SOLVER = "scip"

#: Pyomo ``TerminationCondition`` strings that state a verdict about the model
#: rather than a numerical breakdown. Used to classify a failed result that did
#: not come from the CasADi back-end (which reports it directly, see
#: :func:`monee.solver.casadi.is_numerics_failure`).
_VERDICT_TERMINATIONS = frozenset(
    {"infeasible", "infeasibleOrUnbounded", "unbounded", "invalidProblem"}
)


def _casadi_available() -> bool:
    try:
        import casadi  # noqa: F401
    except ImportError:
        return False
    return True


def _gurobipy_available() -> bool:
    try:
        import gurobipy  # noqa: F401
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


def _solve_error_types() -> tuple[type, ...]:
    """Backend solve exceptions to intercept; CasADi's is optional."""
    from .infeasibility import GekkoSolveError

    try:
        from .casadi import CasADiSolveError
    except ImportError:  # pragma: no cover - only without casadi installed
        return (GekkoSolveError,)
    return (GekkoSolveError, CasADiSolveError)


def _numerics_failure(outcome) -> bool:
    """Whether a failed solve outcome is a numerical breakdown.

    *outcome* is a ``success=False`` result or a raised backend solve error. A
    CasADi failure carries the classification itself (its report's
    ``numerics_failure``, see :func:`monee.solver.casadi.is_numerics_failure`);
    anything else is read off the termination condition, and an outcome that
    says nothing at all counts as a numerics failure so the retry still happens.
    """
    report = getattr(outcome, "report", None) or getattr(
        outcome, "infeasibility_report", None
    )
    flag = getattr(report, "numerics_failure", None)
    if flag is not None:
        return bool(flag)
    termination = getattr(outcome, "termination_condition", None)
    return str(termination).split(".")[-1] not in _VERDICT_TERMINATIONS


class AutoSolver(SolverInterface):
    """``solver='auto'``: the default in-process back-end, with one retry.

    The first attempt runs the back-end :func:`resolve_solver` picks by default
    (CasADi/IPOPT, or GEKKO/IPOPT without casadi). If it ends in a *numerics*
    failure - a breakdown of the interior-point search, not a proven
    infeasibility - the same solve is retried once on
    :data:`AUTO_FALLBACK_SOLVER` (SCIP through Pyomo), which is the retry the
    failure message already advises by hand. A proven infeasibility, an
    unbounded model or a badly posed one is returned or raised straight away:
    another back-end reproduces it, and only that verdict should end a study.

    The returned result records where it came from: ``backend_used`` /
    ``solver_used`` name the back-end that actually produced it, and
    ``fallback_from`` is ``None`` on a first-attempt result or the label of the
    back-end that failed when the fallback produced it. A fallback also appends
    a ``backend_fallback`` entry to ``result.warnings``.

    Not the default, deliberately: switching back-ends changes what a published
    number means, so it stays an explicit opt-in. It is the recommended setting
    for sweeps (contingency loops, Monte Carlo), where one unsolvable case in
    ten should not stop the run. See the concepts/solvers and
    how-to/troubleshooting docs pages.
    """

    _backend_name = AUTO_SOLVER
    _solver_name = AUTO_SOLVER

    def __init__(self, primary=None, fallback: str = AUTO_FALLBACK_SOLVER):
        self._primary = primary if primary is not None else resolve_solver()
        self._fallback_name = fallback
        self._fallback = None

    @property
    def backend_name(self) -> str:
        return self._primary.backend_name

    @property
    def solver_name(self) -> str | None:
        return self._primary.solver_name

    @property
    def benefits_from_warm_start(self) -> bool:
        return self._primary.benefits_from_warm_start

    def set_warm_start_hint(self, active: bool) -> None:
        self._primary.set_warm_start_hint(active)

    def _add_equations(self, solver_obj, eqs):
        self._primary._add_equations(solver_obj, eqs)

    def _fallback_solver(self):
        if self._fallback is None:
            self._fallback = resolve_solver(self._fallback_name, backend="pyomo")
        return self._fallback

    def _primary_label(self) -> str:
        solver = self._primary.solver_name
        return (
            f"{self._primary.backend_name}/{solver}"
            if solver
            else self._primary.backend_name
        )

    def solve(self, input_network, **kwargs):
        failure = None
        result = None
        try:
            result = self._primary.solve(input_network, **kwargs)
        except _solve_error_types() as exc:
            failure = exc
        else:
            if not result.success:
                failure = result
        if failure is None:
            result.fallback_from = None
            return result
        if not _numerics_failure(failure):
            _log.error(
                "solver='auto': %s reported a verdict about the model, not a "
                "numerical failure (%s); not retrying on %r, because another "
                "back-end reproduces it. See the how-to/troubleshooting docs "
                "page.",
                self._primary_label(),
                getattr(failure, "termination_condition", None)
                or type(failure).__name__,
                self._fallback_name,
            )
            if result is None:
                raise failure
            result.fallback_from = None
            return result
        _log.warning(
            "solver='auto': %s ended in a numerics failure, retrying once on "
            "%r. This is a recovery step, not a failed run; the result records "
            "which back-end produced it in backend_used / fallback_from.",
            self._primary_label(),
            self._fallback_name,
        )
        # solver_options are back-end specific (``ipopt.*`` keys mean nothing to
        # SCIP), so the retry runs on the fallback's own defaults.
        fallback_kwargs = {k: v for k, v in kwargs.items() if k != "solver_options"}
        fallback_result = self._fallback_solver().solve(
            input_network, **fallback_kwargs
        )
        if result is None and not fallback_result.success:
            raise failure
        fallback_result.fallback_from = self._primary_label()
        fallback_result.warnings.append(
            ResultWarning(
                "backend_fallback",
                f"solver='auto': {self._primary_label()} ended in a numerics "
                f"failure, this result was produced by "
                f"{self._fallback_name!r} instead",
            )
        )
        _log.warning(
            "solver='auto': the retry on %r ended with success=%s.",
            self._fallback_name,
            fallback_result.success,
        )
        return fallback_result


def _is_solver_instance(obj) -> bool:
    return isinstance(obj, SolverInterface)


def _is_multi_period_solver_instance(obj) -> bool:
    return obj is not None and hasattr(obj, "solve_multi_period")


def _dispatch_casadi(solver, name, casadi_factory):
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


def _dispatch_gurobipy(solver, name, gurobipy_factory):
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


def _dispatch_backend(
    solver,
    backend,
    gekko_factory,
    pyomo_factory,
    gurobipy_factory=None,
    casadi_factory=None,
):
    """Shared (name -> backend -> construct) routing for the single- and
    multi-period resolvers.  *gekko_factory* / *pyomo_factory* take the resolved
    GEKKO solver code / Pyomo solver name respectively and return the concrete
    solver instance.  *gurobipy_factory* / *casadi_factory* (no arguments) build
    the native Gurobi / CasADi solvers; pass ``None`` where those backends are
    unavailable (e.g. multi-period)."""
    name = (solver or "ipopt").lower() if isinstance(solver, str) else "ipopt"

    chosen_backend = backend or _auto_backend(
        name, gurobipy_supported=gurobipy_factory is not None
    )

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
        return _dispatch_casadi(solver, name, casadi_factory)

    if chosen_backend == "gurobipy":
        return _dispatch_gurobipy(solver, name, gurobipy_factory)

    raise ValueError(
        f"Unknown backend {chosen_backend!r}; expected 'gekko', 'pyomo', "
        "'gurobipy' or 'casadi'."
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

    if isinstance(solver, str) and solver.lower() == AUTO_SOLVER:
        if backend is not None:
            raise ValueError(
                "solver='auto' picks the back-end itself, so backend= cannot be "
                "combined with it; drop backend=, or name the solver you want. "
                "See the concepts/solvers docs page."
            )
        return AutoSolver()

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
    if isinstance(solver, str) and solver.lower() == AUTO_SOLVER:
        raise ValueError(
            "solver='auto' is single-period only (run_energy_flow, run_timeseries "
            "and the Stepper); a multi-period horizon has no partial result to "
            f"fall back with. Pass solver={AUTO_FALLBACK_SOLVER!r} to run the "
            "whole horizon on the fallback back-end instead. See the "
            "concepts/solvers docs page."
        )
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

    return _dispatch_backend(
        solver,
        backend,
        lambda code: GekkoMultiPeriodSolver(solver=code),
        lambda name: PyomoMultiPeriodSolver(solver_name=name),
        casadi_factory=_casadi_mp_factory,
    )


def _auto_backend(name: str, gurobipy_supported: bool = True) -> str:
    """Route a solver name to a default backend.

    ``ipopt`` (the default solver) routes to the in-process **CasADi** backend
    when it is installed: CasADi solves the same continuous NLP as GEKKO/IPOPT
    but in-process (no subprocess / text round-trip) and is typically much
    faster, with full coverage of the smooth formulations (incl. gas/heat splines
    and temporal/multi-period coupling). It falls back to GEKKO's bundled IPOPT
    when casadi is absent. The discrete-capable GEKKO solvers (``apopt`` /
    ``bpopt``) stay on GEKKO.

    ``gurobi`` routes to the native in-memory **gurobipy** backend (no Pyomo
    file round-trip) when that backend is usable in this context -
    ``gurobipy_supported`` (single-period only) and the ``gurobipy`` module is
    importable - and falls back to Pyomo's Gurobi plugin otherwise (multi-period
    or no gurobipy). Any other name goes to Pyomo.
    """
    if name == "ipopt":
        return "casadi" if _casadi_available() else "gekko"
    if name in GEKKO_SOLVERS:
        return "gekko"
    if name == "gurobi" and gurobipy_supported and _gurobipy_available():
        return "gurobipy"
    return "pyomo"
