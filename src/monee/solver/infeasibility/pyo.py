"""Pyomo infeasibility diagnostics. Translates Pyomo var names like
``child_3__p_mw`` back to monee component descriptions and wraps the
Pyomo analysis utilities."""

import contextlib
import io
import logging
import re
from dataclasses import dataclass, field

import pyomo.environ as pyo

_log = logging.getLogger(__name__)

# Pyomo var name: {cat}_{id}__{attr} or {cat}_{id}_t{period}__{attr}.
_VAR_NAME_RE = re.compile(
    r"^(?P<cat>\w+?)_(?P<id>\d+)(?:_t(?P<t>\d+))?__(?P<attr>\w+)$"
)


def _parse_var_name(name: str) -> dict | None:
    """Return ``{cat, id, t|None, attr}`` or None if *name* doesn't match."""
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
    """Pyomo name → ``cat[id].attr`` (with ``(t=...)`` for multi-period)."""
    parsed = _parse_var_name(name)
    if parsed is None:
        return name
    label = f"{parsed['cat']}[{parsed['id']}].{parsed['attr']}"
    if parsed["t"] is not None:
        label += f" (t={parsed['t']})"
    return label


@dataclass
class ConstraintResidual:
    """A constraint that is violated or close to violation."""

    index: int | str
    body_value: float
    lower: float | None
    upper: float | None
    residual: float
    expression: str | None = None


def collect_constraint_residuals(
    pm: pyo.ConcreteModel, tol: float = 1e-4
) -> list[ConstraintResidual]:
    """List constraints whose residual exceeds *tol*, sorted by magnitude."""
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


@dataclass
class BoundViolation:
    """A variable violating or sitting on its bounds."""

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
    """List variables whose value sits within *tol* of a bound (active bounds)."""
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


# Per-solver option key for a wall-clock cap on each internal solve in
# ``compute_infeasibility_explanation`` (it iterates several solves).
_MIS_TIME_LIMIT_OPTION = {
    "gurobi": "TimeLimit",
    "cplex": "timelimit",
    "scip": "limits/time",
    "appsi_highs": "time_limit",
    "cbc": "seconds",
}

# Cap on every internal MIS solve.  The MIS routine elastically relaxes
# the LP (≈3× model size) and runs a fresh MILP for each candidate; on
# pathological cases (large slack envelope + 100%-gap-from-zero) those
# never converge.  10 s is enough to return a useful MIS on small cases
# and to time out gracefully on hard ones.
_MIS_TIME_LIMIT_S: float = 10.0


def compute_mis(pm: pyo.ConcreteModel, solver_name: str = "scip") -> list[str]:
    """Names/indices of constraints/bounds forming a Minimal Infeasible Subsystem.
    Needs SCIP/Gurobi/CPLEX.  Each internal solve is capped at
    ``_MIS_TIME_LIMIT_S`` so a hard MIS computation can't stall the caller."""
    from pyomo.contrib.iis import compute_infeasibility_explanation

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    iis_logger = logging.getLogger("pyomo.contrib.iis")
    original_level = iis_logger.level
    iis_logger.setLevel(logging.INFO)
    iis_logger.addHandler(handler)

    solver = pyo.SolverFactory(solver_name)
    time_limit_key = _MIS_TIME_LIMIT_OPTION.get(solver_name)
    if time_limit_key is not None:
        solver.options[time_limit_key] = _MIS_TIME_LIMIT_S

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            compute_infeasibility_explanation(pm, solver=solver)
    except Exception as e:
        _log.warning("MIS computation failed: %s", e)
        return []
    finally:
        iis_logger.removeHandler(handler)
        iis_logger.setLevel(original_level)

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


@dataclass
class InfeasibilityReport:
    """Structured infeasibility report."""

    constraint_residuals: list[ConstraintResidual] = field(default_factory=list)
    bound_violations: list[BoundViolation] = field(default_factory=list)
    variables_at_bounds: list[dict] = field(default_factory=list)
    mis_constraints: list[str] = field(default_factory=list)

    def summary(self, max_items: int = 10) -> str:
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
                if v["at_lower"] and not v["at_upper"]:
                    which = "lower"
                elif v["at_upper"] and not v["at_lower"]:
                    which = "upper"
                else:
                    which = "both"
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
    """Build an :class:`InfeasibilityReport`. ``compute_mis_flag`` triggers the
    Minimal Infeasible Subsystem analysis (slow)."""
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
