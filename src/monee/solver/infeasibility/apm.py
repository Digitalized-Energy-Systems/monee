"""GEKKO/APMonitor infeasibility diagnostics (IPOPT and APOPT).

On a failed local solve APMonitor leaves two artifacts in the run directory
(``m._path``) before GEKKO raises its bare ``@error: Solution Not Found``:

* ``infeasibilities.txt`` - the equations with the largest residuals at the
  final iterate, each with the involved variables (value, bounds, marginal),
* ``results.json`` - the final (failed) iterate for every variable.

This module parses both into a structured :class:`GekkoInfeasibilityReport`
analogous to the Pyomo :class:`~monee.solver.infeasibility.pyo.InfeasibilityReport`.
APMonitor sanitises variable names with ``re.sub(r"\\W+", "_", name).lower()``
(``powergenerator-5.p_mw`` → ``powergenerator_5_p_mw``), which is not
reversible; callers therefore pass a *name_map* of sanitised → original monee
names, built while injecting the GEKKO variables.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field

_log = logging.getLogger(__name__)


def sanitize_apm_name(name: str) -> str:
    """Replicate GEKKO's variable-name sanitisation (gekko.Var)."""
    return re.sub(r"\W+", "_", name).lower()


class GekkoSolveError(RuntimeError):
    """Failed GEKKO solve with attached :class:`GekkoInfeasibilityReport`."""

    def __init__(self, message: str, report: "GekkoInfeasibilityReport | None" = None):
        super().__init__(message)
        self.report = report


@dataclass
class ApmVariableState:
    """A variable as listed under an infeasible equation."""

    name: str
    display_name: str
    value: float
    lower: float
    upper: float
    marginal: float


@dataclass
class ApmEquationResidual:
    """An equation flagged by APMonitor as possibly infeasible."""

    number: int
    lower: float
    residual: float
    upper: float
    infeasibility: float
    equation: str
    variables: list[ApmVariableState] = field(default_factory=list)


# ``<idx> <num> <num> <num> <num> <name...>`` - shared shape of the equation
# rows (lower/residual/upper/infeas.) and variable rows (lower/value/upper/$value).
_APM_NUM = r"[-+]?(?:[\d.]+(?:E[-+]?\d+)?|NaN|Inf(?:inity)?)"
_APM_ROW_RE = re.compile(
    rf"^\s*(\d+)\s+({_APM_NUM})\s+({_APM_NUM})\s+({_APM_NUM})\s+({_APM_NUM})\s+(\S.*)$",
    re.IGNORECASE,
)
# APMonitor prefixes names with the mode ("ss.", "est.", ...).
_APM_MODE_PREFIX_RE = re.compile(r"^\w+\.")


def _strip_mode_prefix(name: str) -> str:
    return _APM_MODE_PREFIX_RE.sub("", name, count=1)


def _build_display_translator(name_map: dict[str, str] | None):
    """Callable replacing sanitised APM tokens with original monee names."""
    if not name_map:
        return lambda text: text
    alternation = "|".join(
        re.escape(k) for k in sorted(name_map, key=len, reverse=True)
    )
    token_re = re.compile(rf"\b({alternation})\b")
    return lambda text: token_re.sub(lambda m: name_map[m.group(1)], text)


def parse_infeasibilities(
    text: str, name_map: dict[str, str] | None = None
) -> list[ApmEquationResidual]:
    """Parse the 'POSSIBLE INFEASIBLE EQUATIONS' section of infeasibilities.txt."""
    translate = _build_display_translator(name_map)
    equations: list[ApmEquationResidual] = []
    in_section = False
    row_kind = None  # "eq" | "var" - set by the last seen column header
    for line in text.splitlines():
        # The header is misspelled "INFEASBILE" in APMonitor - match loosely.
        if "POSSIBLE INFEAS" in line:
            in_section = True
            continue
        if not in_section:
            continue
        if "ACTIVE OBJECTIVE" in line or "ACTIVE EQUATIONS" in line:
            break
        if line.lstrip().startswith("EQ Number"):
            row_kind = "eq"
            continue
        if line.lstrip().startswith("Variable"):
            row_kind = "var"
            continue
        m = _APM_ROW_RE.match(line)
        if m is None or row_kind is None:
            continue
        idx, a, b, c, d, name = m.groups()
        if row_kind == "eq":
            equations.append(
                ApmEquationResidual(
                    number=int(idx),
                    lower=float(a),
                    residual=float(b),
                    upper=float(c),
                    infeasibility=float(d),
                    equation=translate(_strip_mode_prefix(name.strip())),
                )
            )
        elif equations:
            apm_name = _strip_mode_prefix(name.strip())
            equations[-1].variables.append(
                ApmVariableState(
                    name=apm_name,
                    display_name=(name_map or {}).get(apm_name, apm_name),
                    lower=float(a),
                    value=float(b),
                    upper=float(c),
                    marginal=float(d),
                )
            )
    # APMonitor reports signed infeasibility values - rank by magnitude.
    equations.sort(key=lambda e: abs(e.infeasibility), reverse=True)
    return equations


def _load_failed_iterate(run_path: str) -> dict[str, float]:
    """Final iterate from results.json (APMonitor writes it even on failure)."""
    try:
        with open(os.path.join(run_path, "results.json")) as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return {}
    values = {}
    for key, val in raw.items():
        if isinstance(val, list) and val:
            values[key] = val[0]
        elif isinstance(val, (int, float)):
            values[key] = val
    return values


def collect_gekko_bound_violations(
    m, tol: float = 1e-4, name_map: dict[str, str] | None = None
) -> list[dict]:
    """Variables whose failed-iterate value violates their declared bounds."""
    values = _load_failed_iterate(getattr(m, "_path", "") or "")
    violations = []
    for var in getattr(m, "_variables", []):
        val = values.get(var.name)
        if val is None:
            continue
        lb, ub = var.LOWER, var.UPPER
        viol = 0.0
        if lb is not None and val < lb - tol:
            viol = lb - val
        if ub is not None and val > ub + tol:
            viol = max(viol, val - ub)
        if viol > tol:
            violations.append(
                {
                    "name": var.name,
                    "display_name": (name_map or {}).get(var.name, var.name),
                    "value": val,
                    "lower": lb,
                    "upper": ub,
                    "violation": viol,
                }
            )
    violations.sort(key=lambda v: v["violation"], reverse=True)
    return violations


def collect_gekko_variables_at_bounds(
    m, tol: float = 1e-4, name_map: dict[str, str] | None = None
) -> list[dict]:
    """Variables whose failed-iterate value sits within *tol* of a bound."""
    values = _load_failed_iterate(getattr(m, "_path", "") or "")
    at_bounds = []
    for var in getattr(m, "_variables", []):
        val = values.get(var.name)
        if val is None:
            continue
        lb, ub = var.LOWER, var.UPPER
        at_lb = lb is not None and abs(val - lb) < tol
        at_ub = ub is not None and abs(val - ub) < tol
        if at_lb or at_ub:
            at_bounds.append(
                {
                    "name": var.name,
                    "display_name": (name_map or {}).get(var.name, var.name),
                    "value": val,
                    "lower": lb,
                    "upper": ub,
                    "at_lower": at_lb,
                    "at_upper": at_ub,
                }
            )
    return at_bounds


@dataclass
class GekkoInfeasibilityReport:
    """Structured report of a failed GEKKO/APMonitor solve."""

    solver_message: str = ""
    infeasible_equations: list[ApmEquationResidual] = field(default_factory=list)
    bound_violations: list[dict] = field(default_factory=list)
    variables_at_bounds: list[dict] = field(default_factory=list)

    def summary(self, max_items: int = 10) -> str:
        lines = []
        if self.solver_message:
            lines.append(f"=== Solver message ===\n  {self.solver_message}")
            lines.append("")

        if self.infeasible_equations:
            lines.append(
                f"=== Possibly infeasible equations "
                f"({len(self.infeasible_equations)} total) ==="
            )
            for eq in self.infeasible_equations[:max_items]:
                lines.append(
                    f"  {eq.equation}\n"
                    f"    residual={eq.residual:.6g} in [{eq.lower:.6g}, "
                    f"{eq.upper:.6g}] (infeasibility={eq.infeasibility:.4g})"
                )
                for v in eq.variables:
                    lines.append(
                        f"    {v.display_name} = {v.value:.6g} "
                        f"in [{v.lower:.6g}, {v.upper:.6g}]"
                    )
            if len(self.infeasible_equations) > max_items:
                lines.append(
                    f"  ... and {len(self.infeasible_equations) - max_items} more"
                )
        else:
            lines.append("=== No infeasible equations reported by APMonitor ===")

        lines.append("")

        if self.bound_violations:
            lines.append(
                f"=== Variable bound violations "
                f"({len(self.bound_violations)} total) ==="
            )
            for v in self.bound_violations[:max_items]:
                lines.append(
                    f"  {v['display_name']}: value={v['value']:.6g} "
                    f"bounds=[{v['lower']}, {v['upper']}] "
                    f"(violation={v['violation']:.4g})"
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

        return "\n".join(lines)

    def __str__(self):
        return self.summary()

    def __repr__(self):
        return (
            f"GekkoInfeasibilityReport("
            f"equations={len(self.infeasible_equations)}, "
            f"bound_violations={len(self.bound_violations)}, "
            f"at_bounds={len(self.variables_at_bounds)})"
        )


def diagnose_gekko_infeasibility(
    m,
    name_map: dict[str, str] | None = None,
    solver_message: str = "",
    tol: float = 1e-4,
) -> GekkoInfeasibilityReport | None:
    """Build a :class:`GekkoInfeasibilityReport` from the run directory of a
    failed local solve. Best-effort: returns ``None`` when no artifacts exist
    (e.g. remote solve) and never raises."""
    try:
        run_path = getattr(m, "_path", None)
        if not run_path or not os.path.isdir(run_path):
            return None
        report = GekkoInfeasibilityReport(
            solver_message=solver_message.strip(),
            bound_violations=collect_gekko_bound_violations(
                m, tol=tol, name_map=name_map
            ),
            variables_at_bounds=collect_gekko_variables_at_bounds(
                m, tol=tol, name_map=name_map
            ),
        )
        infeas_file = os.path.join(run_path, "infeasibilities.txt")
        if os.path.isfile(infeas_file):
            with open(infeas_file) as f:
                report.infeasible_equations = parse_infeasibilities(
                    f.read(), name_map=name_map
                )
        return report
    except Exception as e:
        _log.warning("GEKKO infeasibility diagnosis failed: %s", e)
        return None
