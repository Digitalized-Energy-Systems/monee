"""Pyomo infeasibility diagnostics. Translates Pyomo var names like
``child_3__p_mw`` back to monee component descriptions and wraps the
Pyomo analysis utilities."""

import contextlib
import io
import logging
import re
from collections import Counter
from dataclasses import dataclass, field

import pyomo.environ as pyo

_log = logging.getLogger(__name__)

# Pyomo var name: {cat}_{id}__{attr} or {cat}_{id}_t{period}__{attr}.
_VAR_NAME_RE = re.compile(
    r"^(?P<cat>\w+?)_(?P<id>\d+)(?:_t(?P<t>\d+))?__(?P<attr>\w+)$"
)
# Branch ids are (from, to, key) tuples, sanitized to three segments.
_BRANCH_VAR_NAME_RE = re.compile(
    r"^branch_(?P<f>\d+)_(?P<to>\d+)_(?P<k>\d+)(?:_t(?P<t>\d+))?__(?P<attr>\w+)$"
)
# Constraint names from the solver's _add_equations: {prefix}[_t{period}]_eq_{i}.
_CON_NAME_RE = re.compile(
    r"^(?P<cat>node|child|compound)_(?P<id>\d+)(?:_t(?P<t>\d+))?_eq_(?P<i>\d+)$"
)
_BRANCH_CON_NAME_RE = re.compile(
    r"^branch_(?P<f>\d+)_(?P<to>\d+)_(?P<k>\d+)(?:_t(?P<t>\d+))?_eq_(?P<i>\d+)$"
)


def _parse_var_name(name: str) -> dict | None:
    """Return ``{cat, id, t|None, attr}`` or None if *name* doesn't match."""
    m = _BRANCH_VAR_NAME_RE.match(name)
    if m is not None:
        return {
            "cat": "branch",
            "id": (int(m.group("f")), int(m.group("to")), int(m.group("k"))),
            "t": int(m.group("t")) if m.group("t") else None,
            "attr": m.group("attr"),
        }
    m = _VAR_NAME_RE.match(name)
    if m is None:
        return None
    return {
        "cat": m.group("cat"),
        "id": int(m.group("id")),
        "t": int(m.group("t")) if m.group("t") else None,
        "attr": m.group("attr"),
    }


def _parse_con_name(name: str) -> dict | None:
    """Return ``{cat, id, t|None, i}`` for a monee constraint name, else None."""
    m = _BRANCH_CON_NAME_RE.match(name)
    if m is not None:
        return {
            "cat": "branch",
            "id": (int(m.group("f")), int(m.group("to")), int(m.group("k"))),
            "t": int(m.group("t")) if m.group("t") else None,
            "i": int(m.group("i")),
        }
    m = _CON_NAME_RE.match(name)
    if m is None:
        return None
    return {
        "cat": m.group("cat"),
        "id": int(m.group("id")),
        "t": int(m.group("t")) if m.group("t") else None,
        "i": int(m.group("i")),
    }


class ComponentIndex:
    """Id-to-component lookup for translating solver names into physical
    terms (component type, user-given name, carrier)."""

    def __init__(self, network):
        self.nodes: dict = {}
        self.childs: dict = {}
        self.compounds: dict = {}
        self.branches: dict = {}
        node_carrier: dict = {}
        for node in network.nodes:
            carrier = getattr(node.grid, "name", None)
            node_carrier[node.id] = carrier
            self.nodes[node.id] = (type(node.model).__name__, carrier, node.name)
        for child in network.childs:
            carrier = getattr(child.grid, "name", None) or node_carrier.get(
                child.node_id
            )
            self.childs[child.id] = (type(child.model).__name__, carrier, child.name)
        for compound in network.compounds:
            self.compounds[compound.id] = (
                type(compound.model).__name__,
                getattr(compound.grid, "name", None),
                compound.name,
            )
        for branch in network.branches:
            self.branches[branch.id] = (
                type(branch.model).__name__,
                getattr(branch.grid, "name", None),
                branch.name,
            )

    @staticmethod
    def from_network(network) -> "ComponentIndex | None":
        if network is None:
            return None
        try:
            return ComponentIndex(network)
        except Exception:
            return None

    def _entry(self, cat: str, comp_id):
        table = {
            "node": self.nodes,
            "child": self.childs,
            "compound": self.compounds,
            "branch": self.branches,
        }.get(cat)
        if table is None:
            return None
        return table.get(comp_id)

    def carrier_of(self, cat: str, comp_id) -> str | None:
        entry = self._entry(cat, comp_id)
        return entry[1] if entry is not None else None

    def component_label(self, cat: str, comp_id) -> str | None:
        """``ExtPowerGrid 'grid' (child 0, el)`` style description, or None."""
        entry = self._entry(cat, comp_id)
        if entry is None:
            return None
        type_name, carrier, name = entry
        name_part = f" '{name}'" if name else ""
        if cat == "branch":
            f, to, _k = comp_id
            loc = f"branch {f}->{to}"
        else:
            loc = f"{cat} {comp_id}"
        carrier_part = f", {carrier}" if carrier else ""
        return f"{type_name}{name_part} ({loc}{carrier_part})"

    def constraint_display(self, raw_name: str) -> tuple[str, str | None, int | None]:
        """``(display, carrier, period)`` for a constraint name."""
        parsed = _parse_con_name(raw_name)
        if parsed is None:
            return raw_name, None, None
        label = self.component_label(parsed["cat"], parsed["id"])
        if label is None:
            return raw_name, None, parsed["t"]
        kind = "balance eq" if parsed["cat"] == "node" else "eq"
        display = f"{kind} {parsed['i']} of {label}"
        if parsed["t"] is not None:
            display += f" at period t={parsed['t']}"
        display += f" [{raw_name}]"
        return display, self.carrier_of(parsed["cat"], parsed["id"]), parsed["t"]

    def var_display(self, raw_name: str) -> tuple[str, str | None, int | None]:
        """``(display, carrier, period)`` for a variable name."""
        parsed = _parse_var_name(raw_name)
        if parsed is None:
            return raw_name, None, None
        label = self.component_label(parsed["cat"], parsed["id"])
        if label is None:
            return _var_display_name(raw_name), None, parsed["t"]
        display = f"{label}.{parsed['attr']}"
        if parsed["t"] is not None:
            display += f" (t={parsed['t']})"
        return display, self.carrier_of(parsed["cat"], parsed["id"]), parsed["t"]


def _var_display_name(name: str) -> str:
    """Pyomo name -> ``cat[id].attr`` (with ``(t=...)`` for multi-period)."""
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
    display_name: str | None = None
    carrier: str | None = None
    period: int | None = None
    component: str | None = None


def collect_constraint_residuals(  # NOSONAR
    pm: pyo.ConcreteModel, tol: float = 1e-4
) -> list[ConstraintResidual]:
    """List constraints whose residual exceeds *tol*, sorted by magnitude.

    Scans every active constraint on the model - both the anonymous
    ``pm.cons`` ConstraintList and the named physics constraints
    ``PyomoSolver._add_equation`` attaches as model attributes."""
    violated = []
    for con in pm.component_data_objects(pyo.Constraint, active=True):
        idx = con.name
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
    carrier: str | None = None
    period: int | None = None


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
# the LP (~=3x model size) and runs a fresh MILP for each candidate; on
# pathological cases (large slack envelope + 100%-gap-from-zero) those
# never converge.  10 s is enough to return a useful MIS on small cases
# and to time out gracefully on hard ones.
_MIS_TIME_LIMIT_S: float = 10.0


def _make_mis_solver(solver_name: str):
    """Same solver resolution the main Pyomo solve uses, so the MIS works
    wherever the solve itself did (notably the pyscipopt bridge on machines
    without a standalone scip executable)."""
    if solver_name == "scip":
        try:
            from monee.solver.pyo import PyomoSolver

            return PyomoSolver._make_solver(solver_name)
        except Exception:
            pass
    return pyo.SolverFactory(solver_name)


def _describe_error(e: Exception) -> str:
    text = str(e).strip()
    desc = f"{type(e).__name__}: {text}" if text else type(e).__name__
    tb = e.__traceback__
    if tb is not None:
        while tb.tb_next is not None:
            tb = tb.tb_next
        filename = tb.tb_frame.f_code.co_filename.replace("\\", "/").rsplit("/", 1)[-1]
        desc += f" (raised at {filename}:{tb.tb_lineno})"
    return desc


class _RecordingSolver:
    """Wraps the MIS solver to record how each internal solve terminated, so a
    missing MIS can be explained by what actually happened (a per-solve
    time-out reads very differently from a genuinely unstable model)."""

    def __init__(self, solver):
        self._solver = solver
        self.terminations: list[str] = []

    def solve(self, *args, **kwargs):
        results = self._solver.solve(*args, **kwargs)
        try:
            self.terminations.append(str(results.solver.termination_condition))
        except Exception:
            self.terminations.append("unknown")
        return results

    def __getattr__(self, name):
        return getattr(self._solver, name)


def compute_mis(
    pm: pyo.ConcreteModel,
    solver_name: str = "scip",
    mis_time_limit: float | None = None,
) -> list[str]:
    """Names/indices of constraints/bounds forming a Minimal Infeasible Subsystem.
    Needs SCIP/Gurobi/CPLEX.  Each internal solve is capped at *mis_time_limit*
    seconds (default ``_MIS_TIME_LIMIT_S``) so a hard MIS computation can't
    stall the caller; raise it when the report says the cap was hit."""
    return _compute_mis_with_reason(
        pm, solver_name=solver_name, mis_time_limit=mis_time_limit
    )[0]


def _parse_mis_output(output: str) -> tuple[list[str], list[str]]:
    """``(mis_lines, relaxation_lines)`` from the pyomo.contrib.iis log: the MIS
    section proper, and the elastic-phase explanation ('feasible with only the
    following relaxed') that exists even when the MIS refinement fails."""
    mis_lines: list[str] = []
    relax_lines: list[str] = []
    bucket = None
    for line in output.splitlines():
        if "relaxed:" in line:
            bucket = relax_lines
            continue
        if "Constraints / bounds in MIS:" in line:
            bucket = mis_lines
            continue
        if "Constraints / bounds in guards" in line:
            bucket = None
            continue
        stripped = line.strip()
        if bucket is not None and stripped.startswith(
            ("constraint:", "variable bound:", "lb of var", "ub of var")
        ):
            bucket.append(stripped)
        elif bucket is not None and not line.startswith(("\t", " ")):
            bucket = None
    return mis_lines, relax_lines


def _termination_note(terminations: list[str], time_limit: float) -> str:
    """Sentence naming the internal solves that did not reach optimality, plus
    the knob to raise when they timed out.  Empty when every solve was optimal
    (then the routine's own diagnosis stands)."""
    if not terminations:
        return ""
    non_optimal = [t for t in terminations if t.lower() != "optimal"]
    if not non_optimal:
        return ""
    shown = ", ".join(f"{n}x {cond}" for cond, n in Counter(non_optimal).most_common())
    note = (
        f" {len(non_optimal)} of the {len(terminations)} internal MIS solves "
        f"did not reach optimality ({shown})"
    )
    if any("limit" in cond.lower() for cond in non_optimal):
        note += (
            f"; the per-solve cap is {time_limit:g} s, so retry with "
            "diagnose_infeasibility(..., mis_time_limit=<seconds>)"
        )
    return note + "."


def _no_mis_reason(
    output: str,
    solver_name: str,
    has_relaxation: bool,
    terminations: list[str] | None = None,
    time_limit: float = _MIS_TIME_LIMIT_S,
) -> str:
    note = _termination_note(terminations or [], time_limit)
    if "A feasible solution was found!" in output:
        return (
            "the MIS solver found the model feasible, so no MIS exists; the "
            "original solve failure may be numerical or solver-specific" + note
        )
    if "likely unstable" in output:
        base = (
            "the elastic phase could not relax any constraint into "
            "feasibility, so nothing could be isolated"
        )
        if not note:
            return (
                base + "; every internal solve terminated optimally, so the model "
                "is likely numerically unstable rather than time-limited"
            )
        return base + "." + note
    if "Could not determine Minimal Intractable System" in output:
        if has_relaxation:
            return (
                "the MIS refinement did not isolate a minimal subsystem; "
                "the infeasibility explanation section is the best "
                "available localization" + note
            )
        return "the MIS refinement could not isolate a minimal subsystem" + note
    return (
        "the MIS routine finished but reported no MIS section "
        f"(solver {solver_name}, per-solve cap {time_limit:g} s)" + note
    )


def _compute_mis_with_reason(
    pm: pyo.ConcreteModel,
    solver_name: str = "scip",
    mis_time_limit: float | None = None,
) -> tuple[list[str], list[str], str | None]:
    """``(mis_constraints, relaxation_explanation, failure_reason)``; the reason
    is None when a MIS was found and never empty otherwise (exception type name
    at minimum), and names what the internal solves did when they did not reach
    optimality."""
    from pyomo.contrib.iis import compute_infeasibility_explanation

    time_limit = _MIS_TIME_LIMIT_S if mis_time_limit is None else float(mis_time_limit)

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    handler.setLevel(logging.INFO)

    iis_logger = logging.getLogger("pyomo.contrib.iis")
    original_level = iis_logger.level
    iis_logger.setLevel(logging.INFO)
    iis_logger.addHandler(handler)

    solver = _make_mis_solver(solver_name)
    time_limit_key = _MIS_TIME_LIMIT_OPTION.get(solver_name)
    if time_limit_key is not None:
        solver.options[time_limit_key] = time_limit
    solver = _RecordingSolver(solver)

    try:
        available = solver.available(exception_flag=False)
    except Exception:
        available = False
    if not available:
        reason = (
            f"solver '{solver_name}' is not available to Pyomo (no executable "
            "found), and the MIS computation needs one; install SCIP, Gurobi "
            "or CPLEX or pass a different solver_name"
        )
        _log.warning("MIS computation failed: %s", reason)
        iis_logger.removeHandler(handler)
        iis_logger.setLevel(original_level)
        return [], [], reason

    try:
        with contextlib.redirect_stderr(io.StringIO()):
            compute_infeasibility_explanation(pm, solver=solver)
    except Exception as e:
        reason = _describe_error(e) + _termination_note(solver.terminations, time_limit)
        _log.warning("MIS computation failed: %s", reason)
        mis_lines, relax_lines = _parse_mis_output(log_capture.getvalue())
        return mis_lines, relax_lines, reason
    finally:
        iis_logger.removeHandler(handler)
        iis_logger.setLevel(original_level)

    output = log_capture.getvalue()
    mis_lines, relax_lines = _parse_mis_output(output)

    if not mis_lines:
        return (
            [],
            relax_lines,
            _no_mis_reason(
                output,
                solver_name,
                bool(relax_lines),
                solver.terminations,
                time_limit,
            ),
        )
    return mis_lines, relax_lines, None


@dataclass
class InfeasibilityReport:
    """Structured infeasibility report."""

    constraint_residuals: list[ConstraintResidual] = field(default_factory=list)
    bound_violations: list[BoundViolation] = field(default_factory=list)
    variables_at_bounds: list[dict] = field(default_factory=list)
    mis_constraints: list[str] = field(default_factory=list)
    mis_message: str | None = None
    relaxation_explanation: list[str] = field(default_factory=list)

    def first_violated_period(self) -> int | None:
        """Smallest period index carrying a violated constraint or bound, or
        None when the problem is single-period (no period-tagged entries)."""
        periods = [
            r.period for r in self.constraint_residuals if r.period is not None
        ] + [v.period for v in self.bound_violations if v.period is not None]
        return min(periods) if periods else None

    def _grouped_residuals(self) -> list[tuple[str | None, list[ConstraintResidual]]]:
        groups: dict = {}
        for r in self.constraint_residuals:
            groups.setdefault(r.carrier, []).append(r)
        return sorted(
            groups.items(),
            key=lambda kv: max(r.residual for r in kv[1]),
            reverse=True,
        )

    def violation_hotspots(self, limit: int = 10) -> list[dict]:
        """Violated constraints collapsed onto the components that own them,
        ranked by how many of a component's equations are violated and then by
        the largest residual. When no minimal infeasible subsystem is available
        this is the closest thing to a localization the report has: a few
        hundred equation residuals become a handful of named components."""
        groups: dict = {}
        for r in self.constraint_residuals:
            key = r.component or r.display_name or str(r.index)
            entry = groups.setdefault(
                key,
                {
                    "component": key,
                    "carrier": r.carrier,
                    "count": 0,
                    "max_residual": 0.0,
                },
            )
            entry["count"] += 1
            entry["max_residual"] = max(entry["max_residual"], r.residual)
        ranked = sorted(
            groups.values(),
            key=lambda e: (e["count"], e["max_residual"]),
            reverse=True,
        )
        return ranked[:limit] if limit is not None else ranked

    def _hotspot_lines(self, max_items: int) -> list[str]:
        if not self.constraint_residuals:
            return []
        all_spots = self.violation_hotspots(limit=None)
        if len(all_spots) < 2:
            return []
        lines = [
            f"=== Violation hotspots ({len(self.constraint_residuals)} violated "
            f"equations across {len(all_spots)} components) ==="
        ]
        for spot in all_spots[:max_items]:
            lines.append(
                f"  {spot['component']}: {spot['count']} violated "
                f"eq(s), largest residual {spot['max_residual']:.4g}"
            )
        if len(all_spots) > max_items:
            lines.append(f"  ... and {len(all_spots) - max_items} more components")
        return lines

    def _cause_lines(self) -> list[str]:
        lines: list[str] = []
        if self.mis_constraints:
            lines.append(
                f"=== Minimal Infeasible Subsystem ({len(self.mis_constraints)} elements) ==="
            )
            lines.extend(f"  {c}" for c in self.mis_constraints)
        elif self.mis_message:
            lines.append(
                f"=== Minimal Infeasible Subsystem: not available "
                f"({self.mis_message}) ==="
            )
        if self.relaxation_explanation:
            if lines:
                lines.append("")
            lines.append(
                "=== Infeasibility explanation: feasible after relaxing "
                f"({len(self.relaxation_explanation)} elements) ==="
            )
            lines.extend(f"  {c}" for c in self.relaxation_explanation)
        return lines

    def _bound_violation_lines(self, max_items: int) -> list[str]:
        if not self.bound_violations:
            return ["=== No variable bound violations ==="]
        lines = [
            f"=== Variable bound violations ({len(self.bound_violations)} total) ==="
        ]
        for v in self.bound_violations[:max_items]:
            lines.append(
                f"  {v.display_name}: value={v.value:.6g} "
                f"bounds=[{v.lower}, {v.upper}] "
                f"(violation={v.violation:.4g})"
            )
        if len(self.bound_violations) > max_items:
            lines.append(f"  ... and {len(self.bound_violations) - max_items} more")
        return lines

    def _at_bounds_lines(self, max_items: int) -> list[str]:
        if not self.variables_at_bounds:
            return []
        n_at_lb = sum(1 for v in self.variables_at_bounds if v["at_lower"])
        n_at_ub = sum(1 for v in self.variables_at_bounds if v["at_upper"])
        lines = [f"=== Variables at bounds: {n_at_lb} at lower, {n_at_ub} at upper ==="]
        # Child/compound variables carry the operational caps and setpoints
        # (import limits, storage power, ...); list them before the node and
        # branch state variables so an active cap is not buried.
        _cat_rank = {"child": 0, "compound": 1, "node": 2, "branch": 3}
        at_bounds = sorted(
            self.variables_at_bounds,
            key=lambda v: _cat_rank.get(str(v["name"]).split("_", 1)[0], 4),
        )
        for v in at_bounds[:max_items]:
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
            lines.append(f"  ... and {len(self.variables_at_bounds) - max_items} more")
        return lines

    def _residual_lines(self, max_items: int) -> list[str]:
        if not self.constraint_residuals:
            return ["=== No violated constraints ==="]
        lines = [
            f"=== Violated constraints ({len(self.constraint_residuals)} total), "
            f"residuals at the last iterate (not a solution) ==="
        ]
        groups = self._grouped_residuals()
        if len(groups) > 1 or groups[0][0] is not None:
            counts = ", ".join(
                f"{carrier or 'other'} {len(rs)}" for carrier, rs in groups
            )
            lines.append(f"  By carrier: {counts}.")
            lines.append(
                "  Reading guide: these are residuals at the point the solver "
                "stopped, not causes; read the sections above first. They are "
                "grouped by carrier, largest first; a violated cap or bound in "
                "one carrier often shows up as balance residuals in another, so "
                "check every group before chasing the largest number."
            )
        first_t = self.first_violated_period()
        if first_t is not None:
            lines.append(f"  First violated period: t={first_t}.")
        for carrier, rs in groups:
            if len(groups) > 1 or carrier is not None:
                lines.append(f"  --- carrier: {carrier or 'other'} ---")
            for r in rs[:max_items]:
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
                    f"  {r.display_name or r.index}: body={r.body_value:.6g} "
                    f"{bound_str} (residual={r.residual:.4g})"
                )
            if len(rs) > max_items:
                lines.append(f"  ... and {len(rs) - max_items} more")
        return lines

    def summary(self, max_items: int = 10) -> str:
        """Report text, causes first: the minimal infeasible subsystem and the
        relaxation explanation, then the per-component hotspot ranking, then
        the binding bounds, then the raw iterate residuals (which are a symptom
        of wherever the solver stopped). Every list is capped at *max_items*
        entries, the residual list per carrier group."""
        sections = [
            self._cause_lines(),
            self._hotspot_lines(max_items),
            self._bound_violation_lines(max_items),
            self._at_bounds_lines(max_items),
            self._residual_lines(max_items),
        ]
        blocks = ["\n".join(sec) for sec in sections if sec]
        return "\n\n".join(blocks)

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


def _translate_mis_line(line: str, index: ComponentIndex) -> str:
    m = re.match(r"^(lb|ub) of var (\S+)(.*)$", line)
    if m is not None:
        display, _carrier, _period = index.var_display(m.group(2))
        if display != m.group(2):
            return f"{m.group(1)} of var {m.group(2)} [{display}]{m.group(3)}"
        return line
    m = re.match(r"^(constraint|variable bound): (\S+)(.*)$", line)
    if m is not None:
        raw = m.group(2)
        display, _carrier, _period = index.constraint_display(raw)
        if display != raw:
            return f"{m.group(1)}: {display}{m.group(3)}"
        display, _carrier, _period = index.var_display(raw)
        if display != raw:
            return f"{m.group(1)}: {raw} [{display}]{m.group(3)}"
    return line


def _annotate_report(report: InfeasibilityReport, index: ComponentIndex) -> None:
    for r in report.constraint_residuals:
        display, carrier, period = index.constraint_display(str(r.index))
        r.display_name, r.carrier, r.period = display, carrier, period
        parsed = _parse_con_name(str(r.index))
        if parsed is not None:
            r.component = index.component_label(parsed["cat"], parsed["id"])
    for v in report.bound_violations:
        display, carrier, period = index.var_display(v.name)
        v.display_name, v.carrier, v.period = display, carrier, period
    for entry in report.variables_at_bounds:
        display, carrier, period = index.var_display(entry["name"])
        entry["display_name"] = display
        entry["carrier"] = carrier
        entry["period"] = period
    report.mis_constraints = [
        _translate_mis_line(line, index) for line in report.mis_constraints
    ]
    report.relaxation_explanation = [
        _translate_mis_line(line, index) for line in report.relaxation_explanation
    ]


def diagnose_infeasibility(
    pm: pyo.ConcreteModel,
    solver_name: str = "scip",
    tol: float = 1e-4,
    compute_mis_flag: bool = True,
    network=None,
    mis_time_limit: float | None = None,
) -> InfeasibilityReport:
    """Build an :class:`InfeasibilityReport`. ``compute_mis_flag`` triggers the
    Minimal Infeasible Subsystem analysis (slow); ``mis_time_limit`` caps each
    of its internal solves in seconds (default 10). *network* (or a
    ``pm._monee_network`` attribute the solver left on the model) translates
    solver names into component terms and groups residuals by carrier.
    When no MIS could be isolated, ``mis_message`` says why and names the
    internal solves that did not reach optimality. See the
    how-to/diagnose_infeasibility docs page."""
    report = InfeasibilityReport(
        constraint_residuals=collect_constraint_residuals(pm, tol=tol),
        bound_violations=collect_variable_bound_violations(pm, tol=tol),
        variables_at_bounds=collect_variables_at_bounds(pm, tol=tol),
    )

    if compute_mis_flag:
        try:
            (
                report.mis_constraints,
                report.relaxation_explanation,
                report.mis_message,
            ) = _compute_mis_with_reason(
                pm, solver_name=solver_name, mis_time_limit=mis_time_limit
            )
        except Exception as e:
            report.mis_message = _describe_error(e)
            _log.warning("MIS computation failed: %s", report.mis_message)
    else:
        report.mis_message = "not requested (compute_mis_flag=False)"

    if network is None:
        network = getattr(pm, "_monee_network", None)
    index = ComponentIndex.from_network(network)
    if index is not None:
        _annotate_report(report, index)

    return report
