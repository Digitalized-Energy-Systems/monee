from .core import SolverResult
from .dispatch import (
    AUTO_FALLBACK_SOLVER,
    AUTO_SOLVER,
    GEKKO_SOLVERS,
    AutoSolver,
    resolve_multi_period_solver,
    resolve_solver,
)
from .gekko import GEKKOSolver
from .infeasibility import (
    GekkoInfeasibilityReport,
    GekkoSolveError,
    InfeasibilityReport,
    collect_constraint_residuals,
    collect_variable_bound_violations,
    collect_variables_at_bounds,
    compute_mis,
    diagnose_gekko_infeasibility,
    diagnose_infeasibility,
)
from .pyo import PyomoSolver

# The CasADi backend is optional (casadi may not be installed); expose its
# classes lazily so importing monee.solver never hard-requires casadi.
__all_lazy__ = (
    "CasADiSolver",
    "CasADiTimeseries",
    "CasADiMultiPeriodSolver",
    "CasADiSolveError",
    "CasADiFailureReport",
    "is_numerics_failure",
)


def __getattr__(name):
    if name in __all_lazy__:
        from . import casadi as _casadi

        return getattr(_casadi, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
