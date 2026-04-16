"""
Pyomo infeasibility diagnostic tools for monee networks.

When a Pyomo solve returns infeasible, these tools help identify which
constraints conflict and which variable bounds are violated.  They wrap
Pyomo's built-in analysis utilities and translate Pyomo variable names
(``child_3__p_mw``) back to monee component descriptions.

Typical usage — called automatically on solve failure::

    from monee.solver.infeasibility import diagnose_infeasibility
    report = diagnose_infeasibility(pm, solver_name="scip")
    print(report)

Or for manual inspection of a solved (or failed) model::

    from monee.solver.infeasibility import (
        collect_constraint_residuals,
        collect_variable_bound_violations,
    )
    residuals = collect_constraint_residuals(pm, tol=1e-4)
    violations = collect_variable_bound_violations(pm, tol=1e-4)
"""

import logging
import re
from dataclasses import dataclass, field

import pyomo.environ as pyo

_log = logging.getLogger(__name__)

# Pattern for monee-generated Pyomo variable names:
#   {category}_{component_id}__{attribute}   (single-period)
#   {category}_{component_id}_t{period}__{attribute}   (multi-period)
_VAR_NAME_RE = re.compile(
    r"^(?P<cat>\w+?)_(?P<id>\d+)(?:_t(?P<t>\d+))?__(?P<attr>\w+)$"
)


def _parse_var_name(name: str) -> dict | None:
    """Parse a monee Pyomo variable name into its components.

    Returns a dict with keys ``cat``, ``id``, ``t`` (or None), ``attr``,
    or ``None`` if the name doesn't match the expected pattern.
    """
    m = _VAR_NAME_RE.match(name)
    if m is None:
        return None
    return {
        "cat": m.group("cat"),
        "id": int(m.group("id")),
        "t": int(m.group("t")) if m.group("t") else None,
        "attr": m.group("attr"),
    }


def _var_display_name(name: str) -> str:
    """Return a human-readable label for a Pyomo variable.

    Translates ``child_3__p_mw`` → ``child[3].p_mw`` and
    ``child_3_t2__p_mw`` → ``child[3].p_mw (t=2)``.
    """
    parsed = _parse_var_name(name)
    if parsed is None:
        return name
    label = f"{parsed['cat']}[{parsed['id']}].{parsed['attr']}"
    if parsed["t"] is not None:
        label += f" (t={parsed['t']})"
    return label


# Constraint residual analysis


@dataclass
class ConstraintResidual:
    """A single constraint that is violated or close to violation."""

    index: int | str
    body_value: float
    lower: float | None
    upper: float | None
    residual: float
    expression: str | None = None


def collect_constraint_residuals(
    pm: pyo.ConcreteModel, tol: float = 1e-4
) -> list[ConstraintResidual]:
    """Return a list of constraints whose residual exceeds *tol*.

    Each entry includes the constraint index, its evaluated body value,
    bounds, residual magnitude, and (where available) the expression string.
    """
    violated = []
    for idx in pm.cons:
        con = pm.cons[idx]
        if con.body is None:
            continue
        try:
            body_val = pyo.value(con.body, exception=False)
        except Exception:
            continue
        if body_val is None:
            continue

        lb = pyo.value(con.lower) if con.lower is not None else None
        ub = pyo.value(con.upper) if con.upper is not None else None

        residual = 0.0
        if lb is not None and body_val < lb - tol:
            residual = lb - body_val
        if ub is not None and body_val > ub + tol:
            residual = max(residual, body_val - ub)
        if lb is not None and ub is not None and lb == ub:
            residual = abs(body_val - lb)

        if residual > tol:
            try:
                expr_str = str(con.expr)
            except Exception:
                expr_str = None
            violated.append(
                ConstraintResidual(
                    index=idx,
                    body_value=body_val,
                    lower=lb,
                    upper=ub,
                    residual=residual,
                    expression=expr_str,
                )
            )
    violated.sort(key=lambda r: r.residual, reverse=True)
    return violated


# Variable bound analysis


@dataclass
class BoundViolation:
    """A variable whose value violates or sits at its bounds."""

    name: str
    display_name: str
    value: float
    lower: float | None
    upper: float | None
    violation: float


def collect_variable_bound_violations(
    pm: pyo.ConcreteModel, tol: float = 1e-4
) -> list[BoundViolation]:
    """Return variables whose current value violates their bounds."""
    violations = []
    for var in pm.component_objects(pyo.Var, active=True):
        name = var.name
        try:
            val = pyo.value(var, exception=False)
        except Exception:
            continue
        if val is None:
            continue
        lb = var.lb
        ub = var.ub

        viol = 0.0
        if lb is not None and val < lb - tol:
            viol = lb - val
        if ub is not None and val > ub + tol:
            viol = max(viol, val - ub)

        if viol > tol:
            violations.append(
                BoundViolation(
                    name=name,
                    display_name=_var_display_name(name),
                    value=val,
                    lower=lb,
                    upper=ub,
                    violation=viol,
                )
            )
    violations.sort(key=lambda v: v.violation, reverse=True)
    return violations


def collect_variables_at_bounds(pm: pyo.ConcreteModel, tol: float = 1e-4) -> list[dict]:
    """Return variables whose current value is at (or within *tol* of) a bound.

    Useful for identifying which bounds are active / binding in the solution.
    """
    at_bounds = []
    for var in pm.component_objects(pyo.Var, active=True):
        name = var.name
        try:
            val = pyo.value(var, exception=False)
        except Exception:
            continue
        if val is None:
            continue
        lb = var.lb
        ub = var.ub

        at_lb = lb is not None and abs(val - lb) < tol
        at_ub = ub is not None and abs(val - ub) < tol

        if at_lb or at_ub:
            at_bounds.append(
                {
                    "name": name,
                    "display_name": _var_display_name(name),
                    "value": val,
                    "lower": lb,
                    "upper": ub,
                    "at_lower": at_lb,
                    "at_upper": at_ub,
                }
            )
    return at_bounds


# MIS (Minimal Infeasible Subsystem) via Pyomo


def compute_mis(pm: pyo.ConcreteModel, solver_name: str = "scip") -> list[str]:
    """Compute a Minimal Infeasible Subsystem (MIS) for the model.

    Returns the names/indices of constraints and variable bounds that form
    the smallest conflicting set.  Removing any single element from the MIS
    would make the remaining system feasible.

    Requires a working MIP solver (SCIP, Gurobi, CPLEX).
    """
    import contextlib
    import io

    from pyomo.contrib.iis import compute_infeasibility_explanation

    # Capture the output from compute_infeasibility_explanation
    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    iis_logger = logging.getLogger("pyomo.contrib.iis")
    original_level = iis_logger.level
    iis_logger.setLevel(logging.INFO)
    iis_logger.addHandler(handler)

    try:
        # Suppress Pyomo warnings about loading infeasible results
        with contextlib.redirect_stderr(io.StringIO()):
            compute_infeasibility_explanation(pm, solver=solver_name)
    except Exception as e:
        _log.warning("MIS computation failed: %s", e)
        return []
    finally:
        iis_logger.removeHandler(handler)
        iis_logger.setLevel(original_level)

    # Parse the output for MIS constraints
    output = log_capture.getvalue()
    mis_constraints = []
    in_mis_section = False
    for line in output.splitlines():
        if "Constraints / bounds in MIS:" in line:
            in_mis_section = True
            continue
        if "Constraints / bounds in guards" in line:
            in_mis_section = False
            continue
        if in_mis_section and line.strip().startswith("constraint:"):
            mis_constraints.append(line.strip())
        if in_mis_section and line.strip().startswith("variable bound:"):
            mis_constraints.append(line.strip())

    return mis_constraints


# Top-level diagnostic report


@dataclass
class InfeasibilityReport:
    """Structured report of infeasibility diagnostics."""

    constraint_residuals: list[ConstraintResidual] = field(default_factory=list)
    bound_violations: list[BoundViolation] = field(default_factory=list)
    variables_at_bounds: list[dict] = field(default_factory=list)
    mis_constraints: list[str] = field(default_factory=list)

    def summary(self, max_items: int = 10) -> str:
        """Return a human-readable summary of the infeasibility."""
        lines = []

        if self.constraint_residuals:
            lines.append(
                f"=== Violated constraints ({len(self.constraint_residuals)} total) ==="
            )
            for r in self.constraint_residuals[:max_items]:
                bound_str = ""
                if r.lower is not None and r.upper is not None and r.lower == r.upper:
                    bound_str = f"== {r.lower}"
                elif r.lower is not None and r.upper is not None:
                    bound_str = f"in [{r.lower}, {r.upper}]"
                elif r.lower is not None:
                    bound_str = f">= {r.lower}"
                elif r.upper is not None:
                    bound_str = f"<= {r.upper}"
                lines.append(
                    f"  cons[{r.index}]: body={r.body_value:.6g} {bound_str} "
                    f"(residual={r.residual:.4g})"
                )
            if len(self.constraint_residuals) > max_items:
                lines.append(
                    f"  ... and {len(self.constraint_residuals) - max_items} more"
                )
        else:
            lines.append("=== No violated constraints ===")

        lines.append("")

        if self.bound_violations:
            lines.append(
                f"=== Variable bound violations ({len(self.bound_violations)} total) ==="
            )
            for v in self.bound_violations[:max_items]:
                lines.append(
                    f"  {v.display_name}: value={v.value:.6g} "
                    f"bounds=[{v.lower}, {v.upper}] "
                    f"(violation={v.violation:.4g})"
                )
            if len(self.bound_violations) > max_items:
                lines.append(f"  ... and {len(self.bound_violations) - max_items} more")
        else:
            lines.append("=== No variable bound violations ===")

        if self.variables_at_bounds:
            lines.append("")
            n_at_lb = sum(1 for v in self.variables_at_bounds if v["at_lower"])
            n_at_ub = sum(1 for v in self.variables_at_bounds if v["at_upper"])
            lines.append(
                f"=== Variables at bounds: {n_at_lb} at lower, {n_at_ub} at upper ==="
            )
            for v in self.variables_at_bounds[:max_items]:
                which = (
                    "lower"
                    if v["at_lower"] and not v["at_upper"]
                    else "upper"
                    if v["at_upper"] and not v["at_lower"]
                    else "both"
                )
                lines.append(
                    f"  {v['display_name']}: value={v['value']:.6g} "
                    f"bounds=[{v['lower']}, {v['upper']}] (at {which})"
                )
            if len(self.variables_at_bounds) > max_items:
                lines.append(
                    f"  ... and {len(self.variables_at_bounds) - max_items} more"
                )

        if self.mis_constraints:
            lines.append("")
            lines.append(
                f"=== Minimal Infeasible Subsystem ({len(self.mis_constraints)} elements) ==="
            )
            for c in self.mis_constraints:
                lines.append(f"  {c}")

        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return (
            f"InfeasibilityReport("
            f"constraints={len(self.constraint_residuals)}, "
            f"bound_violations={len(self.bound_violations)}, "
            f"at_bounds={len(self.variables_at_bounds)}, "
            f"mis={len(self.mis_constraints)})"
        )


def diagnose_infeasibility(
    pm: pyo.ConcreteModel,
    solver_name: str = "scip",
    tol: float = 1e-4,
    compute_mis_flag: bool = True,
) -> InfeasibilityReport:
    """Run all infeasibility diagnostics on a Pyomo model.

    Args:
        pm: The (infeasible) Pyomo ConcreteModel.
        solver_name: Solver to use for MIS computation.
        tol: Feasibility tolerance for residual/bound checks.
        compute_mis_flag: Whether to compute the Minimal Infeasible Subsystem
            (requires re-solving with relaxed constraints, can be slow).

    Returns:
        An :class:`InfeasibilityReport` with all diagnostic results.
    """
    report = InfeasibilityReport(
        constraint_residuals=collect_constraint_residuals(pm, tol=tol),
        bound_violations=collect_variable_bound_violations(pm, tol=tol),
        variables_at_bounds=collect_variables_at_bounds(pm, tol=tol),
    )

    if compute_mis_flag:
        try:
            report.mis_constraints = compute_mis(pm, solver_name=solver_name)
        except Exception as e:
            _log.warning("MIS computation failed: %s", e)

    return report
