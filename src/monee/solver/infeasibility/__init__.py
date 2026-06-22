"""Solver-specific infeasibility diagnostics.

:mod:`.pyo` analyses a Pyomo model after an infeasible solve (constraint
residuals, bound violations, Minimal Infeasible Subsystem); :mod:`.apm`
parses the APMonitor run-directory artifacts of a failed GEKKO solve
(IPOPT/APOPT). Both produce structured reports with a uniform ``summary()``.
"""

from .apm import (
    GekkoInfeasibilityReport,
    GekkoSolveError,
    collect_gekko_bound_violations,
    collect_gekko_variables_at_bounds,
    diagnose_gekko_infeasibility,
    parse_infeasibilities,
    sanitize_apm_name,
)
from .pyo import (
    BoundViolation,
    ConstraintResidual,
    InfeasibilityReport,
    collect_constraint_residuals,
    collect_variable_bound_violations,
    collect_variables_at_bounds,
    compute_mis,
    diagnose_infeasibility,
)

__all__ = [
    "BoundViolation",
    "ConstraintResidual",
    "GekkoInfeasibilityReport",
    "GekkoSolveError",
    "InfeasibilityReport",
    "collect_constraint_residuals",
    "collect_gekko_bound_violations",
    "collect_gekko_variables_at_bounds",
    "collect_variable_bound_violations",
    "collect_variables_at_bounds",
    "compute_mis",
    "diagnose_gekko_infeasibility",
    "diagnose_infeasibility",
    "parse_infeasibilities",
    "sanitize_apm_name",
]
