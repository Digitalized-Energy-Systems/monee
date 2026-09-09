"""In-process CasADi/IPOPT backend for monee.

Motivation
----------
GEKKO's per-solve wall-clock is dominated not by the numerical solve but by
APMonitor re-serialising + re-parsing + re-compiling the whole symbolic model
into a subprocess on *every* solve (~1.9 s for mv_oberrhein, vs ~0.07 s of
actual IPOPT). CasADi builds the NLP once as an in-memory expression graph and
calls IPOPT in-process - no text model, no files, no subprocess.

How it reuses monee
-------------------
monee's formulations are backend-agnostic: they build equations from the
*injected* variable objects and take math hooks (``sin_impl``, ``cos_impl``,
...). So only three pieces are CasADi-specific and everything else (network
prep, ``process_equations_*``, post-processing, result frames) is reused
unchanged:

* :class:`CasSym` - wraps a CasADi ``SX``. Its arithmetic operators build CasADi
  expressions; its ``==`` / ``<=`` / ``>=`` capture a constraint record
  (residual + op) instead of collapsing to a bool. Injected as each model
  variable, so monee's unchanged ``equations()`` bodies produce CasADi
  expressions.
* :class:`CasModel` - quacks like the GEKKO model object for the subset of the
  API monee calls (``sin`` / ``cos`` / ``sqrt`` / ..., ``Equation(s)``, ``Obj``,
  ``Intermediate``). This lets us call monee's existing ``process_equations_*``
  passes verbatim.
* :class:`CasADiSolver` - mirrors :meth:`GEKKOSolver.solve`'s network prep, then
  assembles a single ``nlpsol('ipopt', ...)`` and writes the solution back.

Temporal coupling
-----------------
:class:`CasADiSolver` processes inter-step / inter-temporal equations when a
``step_state`` is supplied, so timeseries with storage SOC, ramp limits and
linepack work through the standard :func:`monee.run_timeseries` per-step loop.
:class:`CasADiMultiPeriodSolver` builds one NLP spanning all T periods with
``inter_period`` coupling (``backend="casadi"`` in
:func:`monee.run_multi_period`).

:class:`CasADiTimeseries` exploits the in-process compiled graph for the
headline win on *memory-less* timeseries: it assembles the NLP + IPOPT solver
**once**, with every time-varying input declared as a CasADi *parameter*, then
re-solves each step by only setting the parameter vector and warm-starting from
the previous step - no rebuild, no recompile, no subprocess per step. (This
build-once reuse does not carry inter-step state, so :func:`monee.run_timeseries`
only takes it for networks without temporal coupling and otherwise uses the
per-step loop above.)

Scope
-----
Electricity AC, gas and heat NLP - power flow, OPF, storage/temporal coupling and
multi-period. Gas/heat tabulated relations (the ``pwl_impl`` hook) are realised
as a smooth cubic B-spline ``interpolant`` (:class:`CasADiCubicSplineImpl`) - the
smooth analogue of the MILP backends' piecewise-linear form and the direct
counterpart of GEKKO's ``cspline``. Integer variables are relaxed (IPOPT is
continuous, exactly like the GEKKO/IPOPT default); cases that must *enforce*
integrality (MINLP) need APOPT or a MIP backend (gurobi/pyomo).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass

import numpy as np

from monee.model import Const, Intermediate, MultiGridCompoundModel, Network, Var
from monee.model.formulation.registry import warn_relaxed_integrality
from monee.problem.core import OptimizationProblem

from .core import (
    OperatorEquationAssembly,
    ResultWarning,
    SolverInterface,
    SolverResult,
    apply_child_overwrites,
    apply_post_process_all,
    compute_bound_violations,
    finalize_failed_solution,
    finalize_solution,
    inject_vars,
    iter_active_models,
    mark_slacks_and_prescriptions,
    prepare_solve_network,
    uncovered_free_var_entry,
    validate_result,
    warn_false_equation,
    warn_on_residuals,
)

try:  # CasADi is an optional backend; keep monee importable without it.
    import casadi as ca
except ImportError:  # pragma: no cover - exercised only when casadi is absent
    ca = None

INF = float("inf")

_log = logging.getLogger(__name__)

_CASADI_MISSING = (
    "The CasADi backend requires the 'casadi' package, which is not installed. "
    "Install it with `pip install casadi` (or `pip install monee[casadi]`)."
)


def _require_casadi():
    if ca is None:  # pragma: no cover - trivial guard
        raise ImportError(_CASADI_MISSING)


class CasADiSolveError(RuntimeError):
    """Raised when a CasADi/IPOPT solve does not converge under ``strict=True``.

    :meth:`CasADiSolver.solve` (and therefore :func:`monee.run_energy_flow`)
    returns a ``success=False`` :class:`~monee.solver.core.SolverResult` instead
    by default, matching the Pyomo/Gurobi backends: the failure carries the
    status, the message and the diagnostics on the result rather than in a
    traceback. ``strict=True`` restores the raise, and the timeseries /
    multi-period drivers on this backend still raise per step or per horizon
    (their loops have nowhere to put a partial result). See the
    how-to/troubleshooting docs page.
    """

    #: :class:`CasADiFailureReport` for the failed solve; ``None`` on errors
    #: raised before diagnostics could be collected.
    report = None


_FALLBACK_HINT = (
    " Multi-energy compounds (CHP, P2H, G2H) next to AC branches are known to "
    "defeat the IPOPT flat start; retry with solver='scip'."
)

_FALLBACK_HINT_STATUSES = frozenset({"Restoration_Failed", "Error_In_Step_Computation"})

#: IPOPT return status of a *proven* infeasibility: the barrier method reduced
#: the constraint violation as far as it can and the point it reached is a local
#: infeasible stationary point. Retrying on another back-end does not turn this
#: into a solution; the model has to change.
_PROVEN_INFEASIBLE_STATUSES = frozenset({"Infeasible_Problem_Detected"})

#: Return statuses that report a badly posed model or a bad option rather than a
#: numerical breakdown. Another back-end sees the same defect, so they do not
#: qualify for the ``solver='auto'`` fallback either.
_MODEL_ERROR_STATUSES = frozenset(
    {
        "Invalid_Problem_Definition",
        "Invalid_Option",
        "Not_Enough_Degrees_Of_Freedom",
    }
)


def is_numerics_failure(return_status: str | None) -> bool:
    """Whether *return_status* is a numerical breakdown rather than a verdict.

    ``True`` for the statuses that say IPOPT could not finish the search
    (restoration failure, a failed step computation, an iteration or time limit,
    an invalid number, ...): the model may well be feasible and another back-end
    or another start point can still solve it. ``False`` for
    ``Infeasible_Problem_Detected`` (a proven local infeasibility) and for the
    badly-posed-model statuses, which every back-end reproduces. Drives the
    ``solver='auto'`` fallback and is reported as
    :attr:`CasADiFailureReport.numerics_failure`; see the how-to/troubleshooting
    docs page.
    """
    return (
        return_status not in _PROVEN_INFEASIBLE_STATUSES
        and return_status not in _MODEL_ERROR_STATUSES
    )


def _no_convergence_message(what: str, stats, network: Network | None = None) -> str:
    msg = (
        f"CasADi/IPOPT {what} did not converge "
        f"(return status: {stats.get('return_status')!r})."
    )
    if stats.get("return_status") in _FALLBACK_HINT_STATUSES and (
        network is not None
        and any(isinstance(c.model, MultiGridCompoundModel) for c in network.compounds)
    ):
        msg += _FALLBACK_HINT
    return (
        msg + " See docs how-to/diagnose_infeasibility for what each return "
        "status means and what to check."
    )


def _sx(x):
    return x.e if isinstance(x, CasSym) else x


class _Rel:
    __slots__ = ("r", "op")

    def __init__(self, r, op):
        self.r = r
        self.op = op


class CasSym:  # NOSONAR
    """Thin wrapper over a CasADi ``SX`` so monee's operator-based equation
    bodies build a CasADi graph and ``==`` / ``<=`` / ``>=`` capture
    constraints."""

    __slots__ = ("e",)

    def __init__(self, e):
        self.e = e

    # arithmetic -> CasSym
    def __add__(self, o):
        return CasSym(self.e + _sx(o))

    def __radd__(self, o):
        return CasSym(_sx(o) + self.e)

    def __sub__(self, o):
        return CasSym(self.e - _sx(o))

    def __rsub__(self, o):
        return CasSym(_sx(o) - self.e)

    def __mul__(self, o):
        return CasSym(self.e * _sx(o))

    def __rmul__(self, o):
        return CasSym(_sx(o) * self.e)

    def __truediv__(self, o):
        return CasSym(self.e / _sx(o))

    def __rtruediv__(self, o):
        return CasSym(_sx(o) / self.e)

    def __pow__(self, o):
        return CasSym(self.e ** _sx(o))

    def __rpow__(self, o):
        return CasSym(_sx(o) ** self.e)

    def __neg__(self):
        return CasSym(-self.e)

    # relations -> constraint records (residual {==,<=} 0)
    def __eq__(self, o):
        return _Rel(self.e - _sx(o), "eq")

    def __le__(self, o):
        return _Rel(self.e - _sx(o), "le")

    def __ge__(self, o):
        return _Rel(_sx(o) - self.e, "le")

    def __hash__(self):
        return id(self)


class CasModel:
    """Minimal stand-in for the GEKKO model object used by monee's equation
    passes. Collects CasADi constraints and objective terms."""

    def __init__(self):
        self.cons: list[_Rel] = []
        self.obj_terms: list = []
        self.user_obj_terms: list = []
        self._pwl_count = 0  # unique-name counter for spline interpolants

    # --- math hooks (mirror m.sin / m.cos / ...) ---
    @staticmethod
    def sin(x):
        return CasSym(ca.sin(_sx(x)))

    @staticmethod
    def cos(x):
        return CasSym(ca.cos(_sx(x)))

    @staticmethod
    def sqrt(x):
        return CasSym(ca.sqrt(_sx(x)))

    @staticmethod
    def exp(x):
        return CasSym(ca.exp(_sx(x)))

    @staticmethod
    def log10(x):
        return CasSym(ca.log10(_sx(x)))

    @staticmethod
    def abs3(x):
        return CasSym(ca.fabs(_sx(x)))

    @staticmethod
    def max2(a, b):
        return CasSym(ca.fmax(_sx(a), _sx(b)))

    @staticmethod
    def sign2(x):
        return CasSym(ca.sign(_sx(x)))

    @staticmethod
    def if2(cond, a, b):
        return CasSym(ca.if_else(_sx(cond), _sx(a), _sx(b)))

    sign3 = sign2
    if3 = if2

    # --- model assembly hooks ---
    def Intermediate(self, expr):  # NOSONAR
        # An intermediate is just a named sub-expression; inline it.
        return expr if isinstance(expr, CasSym) else CasSym(_sx(expr))

    def Equation(self, eq):  # NOSONAR
        if isinstance(eq, _Rel):
            self.cons.append(eq)
        elif eq is False:
            # Mirrors the Pyomo/Gurobipy backends (and core.filter_bool_eqs):
            # a False sentinel is a structurally infeasible constraint from an
            # over-deactivated component - drop it, but surface the error.
            warn_false_equation(logger=_log)

    def Equations(self, eqs):  # NOSONAR
        for eq in eqs:
            self.Equation(eq)

    def Obj(self, expr):  # NOSONAR
        self.obj_terms.append(_sx(expr))

    def ObjUser(self, expr):  # NOSONAR
        """Objective term coming from the user's problem (not from a
        formulation's tightening hooks); tracked twice so the solve can report
        the user objective apart from the combined one."""
        sx = _sx(expr)
        self.obj_terms.append(sx)
        self.user_obj_terms.append(sx)


class CasADiCubicSplineImpl:
    """``pwl_impl`` for the CasADi backend: binds ``y == spline(x)`` through the
    tabulated samples ``(xs, ys)`` using a CasADi B-spline ``interpolant``.

    This is the *smooth* analogue of the MILP/PWL backends' piecewise-linear
    ``addGenConstrPWL`` and the direct counterpart of GEKKO's ``cspline`` - a
    cubic spline is twice-differentiable, which is what IPOPT (a smooth-NLP
    solver) needs; a kinked PWL would give a discontinuous Jacobian. A cubic
    B-spline needs >= 4 knots; for the rare coarse tables (2-3 breakpoints) it
    drops to the highest feasible degree, falling back to linear interpolation
    where even that is rejected. (The bundled gas/heat PWL formulations default
    to 12 breakpoints, so the cubic path is the normal one.)"""

    def __init__(self, model: CasModel):
        self.m = model

    def piecewise_eq(self, y, x, xs, ys, _name=None):
        xs = [float(v) for v in xs]
        ys = [float(v) for v in ys]
        nm = f"monee_cspline_{self.m._pwl_count}"
        self.m._pwl_count += 1
        degree = min(3, max(1, len(xs) - 1))
        try:
            interp = ca.interpolant(nm, "bspline", [xs], ys, {"degree": [degree]})
        except Exception:
            # CasADi rejects some small-knot/degree combinations (e.g. exactly
            # 3 points at degree 2); fall back to a linear interpolant there.
            interp = ca.interpolant(nm, "linear", [xs], ys)
        self.m.Equation(CasSym(_sx(y)) == CasSym(interp(_sx(x))))


# Default IPOPT options shared by the single-shot solver and the timeseries
# driver; overridable per instance via ``solver_options`` (merged over these).
_IPOPT_OPTS = {
    "print_time": 0,
    "ipopt.print_level": 0,
    "ipopt.sb": "yes",
    "ipopt.tol": 1e-6,
    "ipopt.max_iter": 3000,
}

# draw_debug/verbose overlay: show the IPOPT log instead of silencing it.
_IPOPT_DEBUG_OPTS = {
    "print_time": 1,
    "ipopt.print_level": 5,
    "ipopt.sb": "no",
}

# For solves whose x0 is a previous solution (see set_warm_start_hint). The
# flat-start defaults are actively harmful there: the default initial barrier
# (mu_init=0.1) pushes a near-optimal point far back into the interior and can
# fail the step computation outright. Full warm-start mode keeps the point
# (and its near-zero bound distances) as-is: the only tested set that never
# failed nor drifted (<=1e-8 vs cold) from a near-solution start. Trade-off:
# on LARGE smooth gas/heat NLPs (degenerate pos/neg flow bounds get near-zero
# multipliers) it roughly doubles iterations vs a cold start - there, either
# pass warm_start=False to the Stepper or override per instance with
# solver_options={"ipopt.warm_start_init_point": "no", "ipopt.mu_init": 1e-5}
# (~1.5x faster than cold, at the cost of occasional flat-start retries and
# result staleness up to the warm constr_viol_tol).
_IPOPT_WARM_OPTS = {
    "ipopt.warm_start_init_point": "yes",
    "ipopt.mu_init": 1e-6,
    "ipopt.warm_start_bound_push": 1e-9,
    "ipopt.warm_start_mult_bound_push": 1e-9,
}


def _merged_ipopt_opts(
    solver_options: dict | None, debug: bool = False, warm: bool = False
) -> dict:
    opts = dict(_IPOPT_OPTS)
    if warm:
        opts.update(_IPOPT_WARM_OPTS)
    if debug:
        opts.update(_IPOPT_DEBUG_OPTS)
    if solver_options:
        opts.update(solver_options)
    if warm and "ipopt.constr_viol_tol" not in (solver_options or {}):
        # A near-feasible start may be accepted after 0 iterations when the
        # parameter change since the last solve is small; the default unscaled
        # feasibility check (1e-4) would let per-constraint residuals 100x
        # looser than ipopt.tol pass as converged. Cap it at the effective tol
        # (staleness bound); slightly stricter than cold acceptance, and a warm
        # solve that cannot reach it fails into the Stepper's flat-start retry.
        opts["ipopt.constr_viol_tol"] = min(1e-6, opts.get("ipopt.tol", 1e-6))
    return opts


def _assemble_nlp_vectors(reg, m: CasModel):
    """``(X, x0, lbx, ubx, g, lbg, ubg, f)`` from the injected variable register
    and the collected constraints/objective terms."""
    X = ca.vertcat(*[r["sx"] for r in reg])
    x0 = ca.DM([r["x0"] for r in reg])
    lbx = ca.DM([r["lb"] for r in reg])
    ubx = ca.DM([r["ub"] for r in reg])
    g = ca.vertcat(*[c.r for c in m.cons]) if m.cons else ca.SX.zeros(0)
    lbg = ca.DM([0.0 if c.op == "eq" else -INF for c in m.cons])
    ubg = ca.DM([0.0 for _ in m.cons])
    f = ca.SX(0)
    for t in m.obj_terms:
        f = f + t
    return X, x0, lbx, ubx, g, lbg, ubg, f


_IPOPT_RETRY_OPTS = {"ipopt.mu_strategy": "adaptive"}


def _consistent_flow_x0(reg, x0):
    """Alternative start vector, or ``None`` when the given one is already
    consistent.

    The smooth pipe formulations (also the binary-free substitutes a continuous
    backend swaps in, see :func:`_substitute_relaxed_defaults`) seed
    ``mass_flow_mag_kgs`` at the design magnitude while leaving
    ``mass_flow_kgs`` at exactly 0. That start violates ``mag == |signed|`` and,
    worse, sits on the kink of the smooth absolute value, where
    ``mass_flow_pos_kgs == mass_flow_neg_kgs`` makes the smooth-upwinding
    temperature rows a rank-deficient block; IPOPT then reports
    Error_In_Step_Computation on a perfectly feasible model. Seeding the signed
    flow at the magnitude, in the branch's stored from -> to direction, breaks
    that symmetry. Only ever used as a retry start (see
    :func:`_solve_with_retry`): the first attempt keeps the established start
    point, so a solve that converges today keeps its result.
    """
    mag_index = {
        id(r["model"]): i for i, r in enumerate(reg) if r["key"] == "mass_flow_mag_kgs"
    }
    alt = None
    for i, r in enumerate(reg):
        if r["key"] != "mass_flow_kgs" or r["x0"] != 0.0:
            continue
        # Const(1), the from -> to pin of the smooth pipe formulations, is a
        # plain float by the time the variables are injected; a free direction
        # binary is a symbol here and must not be seeded.
        direction = r["model"].__dict__.get("direction")
        if not isinstance(direction, (int, float)) or direction != 1:
            continue
        j = mag_index.get(id(r["model"]))
        if j is None:
            continue
        magnitude = reg[j]["x0"] * (reg[j]["scale"] or 1.0)
        if magnitude <= 0:
            continue
        seed = magnitude / (r["scale"] or 1.0)
        if seed < r["lb"] or seed > r["ub"]:
            continue
        if alt is None:
            alt = [float(v) for v in np.array(x0).flatten()]
        alt[i] = seed
    return None if alt is None else ca.DM(alt)


def _solve_with_retry(solver, nlp, opts, args, name, what, x0_alt=None):
    """Solve, and on anything short of ``Solve_Succeeded`` rebuild once with an
    adaptive barrier strategy and re-solve, keeping the better of the two.

    IPOPT's monotone default barrier can fail the step computation or the
    restoration phase from monee's flat start on problems that are perfectly
    feasible - widening a variable bound is enough to trigger it - and it can
    stall at an acceptable-level point far from the optimum, while the adaptive
    strategy converges on the same model.

    ``x0_alt`` (see :func:`_consistent_flow_x0`) adds a last resort for the
    remaining case, a flat start that is itself degenerate: both barrier
    strategies are tried again from that start, and only a success replaces the
    outcome. Everything here runs after the established path has already
    failed, so a solve that converges today is untouched.
    """
    sol = solver(**args)
    stats = solver.stats()
    if stats.get("return_status") == "Solve_Succeeded" or "ipopt.mu_strategy" in opts:
        return sol, stats
    _log.warning(
        "CasADi/IPOPT %s: first attempt stopped at return status %r, now "
        "retrying automatically with ipopt.mu_strategy='adaptive'. This is a "
        "recovery step, not a failed run; a closing line reports the outcome. "
        "The how-to/diagnose_infeasibility docs page explains the statuses.",
        what,
        stats.get("return_status"),
    )
    retry = ca.nlpsol(f"{name}_retry", "ipopt", nlp, {**opts, **_IPOPT_RETRY_OPTS})
    retry_sol = retry(**args)
    retry_stats = retry.stats()
    better = retry_stats.get("return_status") == "Solve_Succeeded" or (
        not stats.get("success", False) and retry_stats.get("success", False)
    )
    best = (retry_sol, retry_stats) if better else (sol, stats)
    if best[1].get("success", False):
        _log.warning(
            "CasADi/IPOPT %s: recovered at return status %r, the run continues "
            "with that solution.",
            what,
            best[1].get("return_status"),
        )
        return best
    if x0_alt is None:
        return best
    _log.warning(
        "CasADi/IPOPT %s still at return status %r; retrying from a consistent "
        "pipe-flow start point. The how-to/diagnose_infeasibility docs page "
        "explains the start-point failure statuses.",
        what,
        best[1].get("return_status"),
    )
    alt_args = {**args, "x0": x0_alt}
    for attempt in (solver, retry):
        alt_sol = attempt(**alt_args)
        alt_stats = attempt.stats()
        if alt_stats.get("success", False):
            _log.warning(
                "CasADi/IPOPT %s: recovered at return status %r from the "
                "consistent pipe-flow start point, the run continues with that "
                "solution.",
                what,
                alt_stats.get("return_status"),
            )
            return alt_sol, alt_stats
    return best


def _status_fields(stats) -> tuple[str, str | None]:
    """``(solver_status, termination_condition)`` from IPOPT's return status.

    ``stats['success']`` is True for ``Solved_To_Acceptable_Level`` too, so a
    solve that stopped short of the requested tolerance is otherwise
    indistinguishable from a converged one.
    """
    rs = stats.get("return_status")
    if rs == "Solve_Succeeded":
        return "ok", rs
    _log.warning(
        "IPOPT stopped with return status %r instead of 'Solve_Succeeded': the "
        "point satisfies the acceptable tolerance only and its objective can be "
        "far from optimal. Cross-check with another back-end if the objective "
        "value matters.",
        rs,
    )
    return "warning", rs


def _residual_from_values(gv, lbg, ubg) -> float | None:
    if gv.size == 0:
        return None
    return float(np.max(_residual_array(gv, lbg, ubg)))


def _residual_array(gv, lbg, ubg):
    lo = np.array(lbg).flatten()
    hi = np.array(ubg).flatten()
    return np.maximum(np.maximum(lo - gv, gv - hi), 0.0)


_LIMIT_STATUSES = frozenset(
    {
        "Maximum_Iterations_Exceeded",
        "Maximum_CpuTime_Exceeded",
        "Maximum_WallTime_Exceeded",
    }
)


def _bound_hint_lines(reg, x_last, reg_range=None, period=None, tol=1e-6, limit=5):
    """Controllable (child/compound) variables sitting on a bound at the final
    iterate, in physical terms; the usual carriers of caps and setpoints."""
    start, end = reg_range if reg_range is not None else (0, len(reg))
    lines = []
    for i in range(start, min(end, len(reg))):
        r = reg[i]
        if r.get("cat") not in ("child", "compound"):
            continue
        val = float(x_last[i]) * (r.get("scale", 1.0) or 1.0)
        vmin, vmax = r.get("vmin"), r.get("vmax")
        if vmin is not None and val <= vmin + tol:
            which, bound = "lower", vmin
        elif vmax is not None and val >= vmax - tol:
            which, bound = "upper", vmax
        else:
            continue
        name = f" '{r['comp_name']}'" if r.get("comp_name") else ""
        carrier = f", {r['carrier']}" if r.get("carrier") else ""
        suffix = f" (t={period})" if period is not None else ""
        lines.append(
            f"  {r.get('comp_type', '?')}{name} ({r['cat']} {r['comp_id']}"
            f"{carrier}).{r['key']} = {val:.6g} at {which} bound "
            f"{bound:g}{suffix}"
        )
        if len(lines) >= limit:
            break
    return lines


def _failure_diagnostics(  # NOSONAR
    stats, X, x_last, g, lbg, ubg, reg, cons_spans=None, reg_spans=None, tol=1e-4
):
    """``(extra_message, info)`` for a failed solve: infeasibility detection at
    the final iterate, first infeasible period (multi-period, via *cons_spans*
    ``(t, start, end)`` slices of the constraint vector) and at-bound
    controllable variables. Never raises."""
    info = {"max_residual": None, "first_infeasible_period": None}
    lines = []
    status = stats.get("return_status")
    resid = None
    if x_last is not None and g.numel() > 0:
        try:
            gv = np.array(
                ca.Function("monee_diag_g", [X], [g])(ca.DM(x_last))
            ).flatten()
            resid = _residual_array(gv, lbg, ubg)
        except Exception:
            resid = None
    if resid is not None and resid.size and float(np.max(resid)) > tol:
        max_resid = float(np.max(resid))
        info["max_residual"] = max_resid
        if status in _LIMIT_STATUSES:
            lines.append(
                f"The final iterate still violates constraints (largest "
                f"residual {max_resid:.4g}): the problem is likely infeasible, "
                "not merely unconverged - raising the iteration limit will "
                "not help."
            )
        else:
            lines.append(
                f"Largest constraint residual at the final iterate: {max_resid:.4g}."
            )
        first_t = None
        if cons_spans:
            viol = []
            for t, a, b in cons_spans:
                if b > a:
                    mr = float(np.max(resid[a:b]))
                    if mr > tol:
                        viol.append((t, mr))
            if viol:
                first_t, first_resid = viol[0]
                info["first_infeasible_period"] = first_t
                worst_t, worst_resid = max(viol, key=lambda p: p[1])
                line = (
                    f"First infeasible period: t={first_t} "
                    f"(largest residual {first_resid:.4g}"
                )
                if worst_t != first_t:
                    line += f"; worst period t={worst_t}, residual {worst_resid:.4g}"
                lines.append(line + ").")
        reg_range = None
        if first_t is not None and reg_spans:
            for t, a, b in reg_spans:
                if t == first_t:
                    reg_range = (a, b)
                    break
        hints = _bound_hint_lines(reg, x_last, reg_range=reg_range, period=first_t)
        if hints:
            where = f" in period t={first_t}" if first_t is not None else ""
            lines.append(f"Controllable variables at their bounds{where}:")
            lines.extend(hints)
    lines.append(
        "Retry on solver='scip' (or solver='auto', which does it for you) to "
        "find out whether this is a numerics failure or a real infeasibility: "
        "SCIP either solves the model or attaches a structured infeasibility "
        "report (residuals grouped by carrier, minimal infeasible subsystem) to "
        "result.infeasibility_report. See the how-to/troubleshooting docs page."
    )
    return "\n".join(lines), info


@dataclass
class CasADiFailureReport:
    """Diagnostics of a failed CasADi/IPOPT solve.

    Attached to :attr:`SolverResult.infeasibility_report` when
    :meth:`CasADiSolver.solve` returns a ``success=False`` result, and to
    :attr:`CasADiSolveError.report` when it raises instead. ``message`` is the
    text the exception carries, ``summary()`` returns it unchanged so the
    Pyomo/SCIP report and this one are read the same way, and
    ``numerics_failure`` says whether the run may still be worth retrying
    elsewhere (see :func:`is_numerics_failure`). ``max_residual`` and
    ``first_infeasible_period`` are measured at the final iterate, which is not
    a solution.
    """

    message: str
    return_status: str | None
    numerics_failure: bool
    max_residual: float | None = None
    first_infeasible_period: int | None = None

    def summary(self, **_kwargs) -> str:
        return self.message

    def __str__(self) -> str:
        return self.message

    def as_error(self) -> CasADiSolveError:
        err = CasADiSolveError(self.message)
        err.report = self
        err.max_residual = self.max_residual
        err.first_infeasible_period = self.first_infeasible_period
        return err


def _no_convergence_report(  # NOSONAR
    what,
    stats,
    network,
    X,
    x_last,
    g,
    lbg,
    ubg,
    reg,
    cons_spans=None,
    reg_spans=None,
) -> CasADiFailureReport:
    extra, info = _failure_diagnostics(
        stats, X, x_last, g, lbg, ubg, reg, cons_spans=cons_spans, reg_spans=reg_spans
    )
    msg = _no_convergence_message(what, stats, network)
    if extra:
        msg += "\n" + extra
    status = stats.get("return_status")
    return CasADiFailureReport(
        message=msg,
        return_status=status,
        numerics_failure=is_numerics_failure(status),
        max_residual=info["max_residual"],
        first_infeasible_period=info["first_infeasible_period"],
    )


def _raise_no_convergence(  # NOSONAR
    what,
    stats,
    network,
    X,
    x_last,
    g,
    lbg,
    ubg,
    reg,
    cons_spans=None,
    reg_spans=None,
):
    raise _no_convergence_report(
        what,
        stats,
        network,
        X,
        x_last,
        g,
        lbg,
        ubg,
        reg,
        cons_spans=cons_spans,
        reg_spans=reg_spans,
    ).as_error()


def _solution_residual(X, x_opt, g, lbg, ubg) -> float | None:
    """Largest violation of the assembled constraint system at the solution."""
    if g.numel() == 0:
        return None
    gv = np.array(ca.Function("monee_g", [X], [g])(ca.DM(x_opt))).flatten()
    return _residual_from_values(gv, lbg, ubg)


def _phantom_dof_entry(reg, m) -> ResultWarning | None:
    """In simulation mode, report injected variables that no equation pins.

    ``simulation=True`` only reaches the formulations (which square their own
    variables); a Var on a custom child model that appears in no equation stays
    a free degree of freedom and the solver returns an arbitrary point inside
    its bounds.
    """
    n_eq = sum(1 for c in m.cons if c.op == "eq")
    if len(reg) == n_eq:
        return None
    used = (
        {s.name() for s in ca.symvar(ca.vertcat(*[c.r for c in m.cons]))}
        if m.cons
        else set()
    )
    names = sorted(
        {
            f"{type(r['model']).__name__}.{r['key']}"
            for r in reg
            if r["sx"].name() not in used
        }
    )
    return ResultWarning(
        "dof",
        f"simulation requested but the model is not square ({len(reg)} "
        f"variables, {n_eq} equations), so the solved values are not a unique "
        "setpoint: the solver returns whichever feasible point it lands on. "
        "Variables in no equation: "
        + ((", ".join(names[:10]) + (", ..." if len(names) > 10 else "")) or "none")
        + ". Declare a fixed setpoint as a plain float or Const, or add an "
        "equation that pins the Var",
        value=float(len(reg) - n_eq),
    )


def _eval_obj_terms(X, x_opt, terms) -> float | None:
    """Value of ``sum(terms)`` at the solution, or ``None`` when empty."""
    if not terms:
        return None
    f = ca.SX(0)
    for t in terms:
        f = f + t
    return float(ca.Function("obj_part", [X], [f])(ca.DM(x_opt)))


def _write_back_solution(reg, x_opt):
    """Scatter the solved vector into the models as plain monee Vars."""
    for i, r in enumerate(reg):
        r["model"].__dict__[r["key"]] = Var(
            value=float(x_opt[i]) * r.get("scale", 1.0),
            min=r["vmin"],
            max=r["vmax"],
            integer=r.get("integer", False),
            name=r["name"],
        )


def _eval_leftover_cassyms(models, X, x_opt, fname="intval"):
    """Evaluate remaining CasSym attributes (inlined intermediates) on *models*
    in one batched call and materialise them as :class:`Intermediate`."""
    leftover = [
        (model, key, val.e)
        for model in models
        for key, val in model.__dict__.items()
        if isinstance(val, CasSym)
    ]
    if not leftover:
        return
    F = ca.Function(fname, [X], [ca.vertcat(*[e for _, _, e in leftover])])
    vals = np.array(F(ca.DM(x_opt))).flatten()
    for (model, key, _), v in zip(leftover, vals):
        model.__dict__[key] = Intermediate(value=float(v))


def _substitute_relaxed_defaults(network, formulation_spec, simulation):
    """Swap the MIQCQP-shaped default branch formulations for their binary-free
    smooth substitutes (gas and water/heat). IPOPT relaxes the ``direction``
    binary: on gas the epigraph relaxation stays slack and reports a pressure
    drop several times the Weymouth value, on water/heat a fractional direction
    blends both flow directions into meaningless temperatures. A formulation
    the caller asked for explicitly is left alone (and reported by
    :func:`monee.model.formulation.registry.warn_relaxed_integrality` and the
    result validation's relaxation check)."""
    from monee.model.formulation.bundles import (
        DEFAULT_SIMULATION_FORMULATION,
        SMOOTH_SUBSTITUTE_FORMULATION,
    )
    from monee.model.formulation.registry import resolve_formulation

    requested = resolve_formulation(formulation_spec)
    for branch in network.branches:
        model, grid = branch.model, branch.grid
        default = DEFAULT_SIMULATION_FORMULATION.lookup(model, grid)
        smooth = SMOOTH_SUBSTITUTE_FORMULATION.lookup(model, grid)
        if branch.formulation is not default or smooth is None:
            continue
        if getattr(branch, "formulation_pinned", False):
            continue
        if requested is not None and requested.lookup(model, grid) is not None:
            continue
        if network.lookup_formulation(model, grid) is not None:
            continue
        branch.formulation = smooth
        # ensure_var re-declares the branch's Vars, and the optimization problem
        # has already been applied at this point: carry the bounds over.
        bounds = {
            key: (val.min, val.max)
            for key, val in vars(model).items()
            if isinstance(val, Var)
        }
        smooth.ensure_var(model, simulation=simulation, grid=grid)
        for key, (lo, hi) in bounds.items():
            var = getattr(model, key, None)
            if isinstance(var, Var):
                var.min, var.max = lo, hi


class CasADiSolver(OperatorEquationAssembly, SolverInterface):
    """In-process CasADi/IPOPT backend. Reuses the shared
    :class:`~monee.solver.core.OperatorEquationAssembly` equation passes; only
    variable injection, the solve, and the write-back are CasADi-specific."""

    def __init__(self, solver_options: dict | None = None):
        """``solver_options`` are CasADi/IPOPT options merged over the module
        defaults (e.g. ``{"ipopt.max_iter": 500, "ipopt.tol": 1e-8}``)."""
        _require_casadi()
        self._backend_name = "casadi"
        self._solver_name = "ipopt"
        self.solver_options = dict(solver_options) if solver_options else {}

        self._simulation: bool = False
        self._reg: list = []
        self._warm_start_hint: bool = False

        self.last_build_s = None  # nlpsol construction: graph -> derivative fns
        self.last_solve_s = None  # the IPOPT solve() call
        self.last_engine_s = None  # build + solve (the GEKKO "engine" analogue)
        self.last_iters = None

    def set_warm_start_hint(self, active: bool) -> None:
        """Declare that the next solve's variable values are a previous
        solution (e.g. the :class:`~monee.simulation.Stepper` persisted the
        last step's result). That solve then runs IPOPT with warm-start
        options (small initial barrier) instead of the flat-start defaults.
        One-shot: consumed and reset by the next :meth:`solve`."""
        self._warm_start_hint = bool(active)

    def _add_equations(self, m, eqs):
        m.Equations(eqs)

    def _pwl_impl(self, m):
        return CasADiCubicSplineImpl(m)

    def _inject(self, model, comp, cat):  # NOSONAR
        for key, val in list(model.__dict__.items()):  # NOSONAR
            if isinstance(val, Var):
                # Integer Vars are relaxed: IPOPT is continuous.
                sx = ca.SX.sym(f"v{len(self._reg)}")
                lo = -INF if val.min is None else float(val.min)
                hi = INF if val.max is None else float(val.max)
                v0 = val.value
                x0 = (
                    0.0
                    if (v0 is None or (isinstance(v0, float) and math.isnan(v0)))
                    else float(v0)
                )

                scale = getattr(val, "scale", 1.0) or 1.0
                phys = sx if math.isclose(scale, 1.0) else scale * sx
                if not math.isclose(scale, 1.0):
                    lo = lo / scale
                    hi = hi / scale
                    # A negative scale flips the interval; reorder after division.
                    if lo > hi:
                        lo, hi = hi, lo
                    x0 = x0 / scale
                self._reg.append(
                    {
                        "model": model,
                        "key": key,
                        "sx": sx,
                        "lb": lo,
                        "ub": hi,
                        "x0": x0,
                        "scale": scale,
                        "name": val.name,
                        "vmin": val.min,
                        "vmax": val.max,
                        "integer": val.integer,
                        "cat": cat,
                        "comp_id": getattr(comp, "id", None),
                        "comp_type": type(model).__name__,
                        "comp_name": getattr(comp, "name", None),
                        "carrier": getattr(getattr(comp, "grid", None), "name", None),
                    }
                )
                setattr(model, key, CasSym(phys))
            elif isinstance(val, Const):
                v = val.value
                if isinstance(v, CasSym):
                    setattr(model, key, v)
                elif isinstance(v, (ca.SX, ca.MX)):
                    setattr(model, key, CasSym(v))
                else:
                    setattr(model, key, float(v))

    def _active_models(self, network, nodes, branches, compounds, ignored_nodes):
        return iter_active_models(network, nodes, branches, compounds, ignored_nodes)

    def _failed_result(  # NOSONAR
        self,
        network,
        input_network,
        nodes,
        branches,
        compounds,
        ignored_nodes,
        reg,
        X,
        x_last,
        report: CasADiFailureReport,
        simulation: bool,
    ) -> SolverResult:
        """``success=False`` result for a solve that did not converge.

        Mirrors the Pyomo backend: the final iterate is scattered only to
        replace the injected symbols with numbers, then
        :func:`finalize_failed_solution` restores the pre-solve values from
        *input_network* - the frames therefore report the inputs, not a
        solution, and the caller's network keeps its warm start and its carried
        storage state. Every step is best-effort: a failed solve must not turn
        into an exception from the reporting path itself.
        """
        try:
            _write_back_solution(reg, x_last)
            _eval_leftover_cassyms(
                self._active_models(network, nodes, branches, compounds, ignored_nodes),
                X,
                x_last,
            )
            violations = finalize_failed_solution(
                nodes, branches, compounds, network, input_network
            )
            dataframes = network.as_result_dataframe_dict()
        except Exception as exc:  # noqa: BLE001 - the failure is what we report
            _log.warning(
                "Could not assemble result frames for the failed solve (%s); "
                "returning an empty-frame result. The failure itself is "
                "reported on result.infeasibility_report.",
                exc,
            )
            violations, dataframes = {}, {}
        result = SolverResult(
            network,
            dataframes,
            None,
            False,
            violations,
            solver_status="error",
            termination_condition=report.return_status,
            residuals=report.max_residual,
            mode_used="simulation" if simulation else "optimization",
            backend_used=self.backend_name,
            solver_used=self.solver_name,
            infeasibility_report=report,
        )
        return validate_result(
            network, result, ignored_nodes=ignored_nodes, simulation=simulation
        )

    def solve(  # NOSONAR
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
        step_state=None,
        simulation=False,
        formulation=None,
        solver_options: dict | None = None,
        strict: bool = False,
    ) -> SolverResult:
        """``solver_options`` are per-call CasADi/IPOPT options merged over the
        instance options and the module defaults (e.g.
        ``{"ipopt.max_iter": 500}``); the top-level ``run_*`` helpers forward
        the kwarg here.

        A solve that does not converge returns a ``success=False``
        :class:`~monee.solver.core.SolverResult` whose ``solver_status`` is
        ``'error'``, whose ``termination_condition`` is the IPOPT return status
        and whose ``infeasibility_report`` is a :class:`CasADiFailureReport`
        with the message and the final-iterate diagnostics; the frames report
        the pre-solve inputs and *input_network* is left untouched. Check
        ``result.success`` before reading any number off it.

        ``strict=True`` restores the raise: a non-converged solve raises
        :class:`CasADiSolveError`, and a converged one whose result validation
        attaches warnings raises
        :class:`~monee.solver.core.ValidationError` (see the
        how-to/diagnose_infeasibility and how-to/troubleshooting docs pages)."""
        self._simulation = simulation
        self._reg = []
        warm = self._warm_start_hint
        self._warm_start_hint = False
        m = CasModel()

        # --- network prep (mirrors GEKKOSolver.solve) ---
        network, ignored_nodes, _islanding_config = prepare_solve_network(
            input_network,
            optimization_problem=optimization_problem,
            formulation=formulation,
            simulation=simulation,
            exclude_unconnected_nodes=exclude_unconnected_nodes,
        )
        _substitute_relaxed_defaults(network, formulation, simulation)
        relaxed_components = warn_relaxed_integrality(network, "IPOPT")

        nodes = network.nodes
        apply_child_overwrites(network, nodes, ignored_nodes)

        branches = network.branches
        compounds = network.compounds
        mark_slacks_and_prescriptions(network, ignored_nodes)

        # --- variable injection (CasADi symbols) ---
        inject_vars(self._inject, nodes, branches, compounds, network, ignored_nodes)

        # Inter-step temporal coupling (storage SOC, ramp limits, linepack, ...):
        # let extensions/formulations react to the previous step's solved state
        # and flag temporal models so their static-only equations are suppressed.
        if step_state is not None:
            for ext in network.extensions:
                ext.activate_timeseries(network, ignored_nodes, step_state=step_state)
            self.mark_temporal_components(network, ignored_nodes)

        # --- equation assembly (reuses monee's unchanged passes) ---
        objs_exprs: list = []
        self.init_branches(branches)
        self.process_equations_nodes_childs(m, network, nodes, ignored_nodes)
        self.process_equations_branches(m, network, branches, ignored_nodes, objs_exprs)
        self.process_equations_compounds(m, network, compounds, ignored_nodes)
        if optimization_problem is not None:
            self.process_oxf_components(m, network, optimization_problem)
        else:
            self.process_internal_oxf_components(m, network)

        # Inter-step + inter-temporal equations couple this step's variables to
        # the previous step's solved (float) values held in step_state.
        if step_state is not None:
            self.process_inter_step_equations(
                m,
                network,
                nodes,
                branches,
                compounds,
                ignored_nodes,
                step_state,
                optimization_problem=optimization_problem,
            )
            for ext in network.extensions:
                m.Equations(
                    ext.inter_step_equations(network, ignored_nodes, step_state)
                )
                m.Equations(
                    ext.inter_temporal_equations(network, ignored_nodes, step_state)
                )

        for ext in network.extensions:
            m.Equations(ext.equations(network, ignored_nodes))
        for expr in objs_exprs:
            m.Obj(expr)

        # --- assemble & solve the NLP in-process ---
        reg = self._reg
        X, x0, lbx, ubx, g, lbg, ubg, f = _assemble_nlp_vectors(reg, m)
        dof_entry = (
            _phantom_dof_entry(reg, m)
            if simulation
            else uncovered_free_var_entry(reg, m)
        )

        t0 = time.perf_counter()
        # Common-subexpression elimination: monee's per-element equation bodies
        # rebuild identical sub-terms many times (e.g. the AC flow equations
        # reconstruct ``vm_from*vm_to``, ``cos(va_from-va_to)``, ``vm**2`` across
        # all four P/Q flow functions of every branch). ``ca.cse`` merges those
        # duplicate graph nodes before the autodiff build, which both shrinks the
        # nlpsol construction and speeds per-iteration evaluation.
        f, g = ca.cse([f, g])
        nlp = {"x": X, "f": f, "g": g}
        opts = _merged_ipopt_opts(
            {**self.solver_options, **(solver_options or {})},
            debug=draw_debug,
            warm=warm,
        )
        solver = ca.nlpsol("monee_casadi", "ipopt", nlp, opts)
        self.last_build_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        sol, stats = _solve_with_retry(
            solver,
            nlp,
            opts,
            {"x0": x0, "lbx": lbx, "ubx": ubx, "lbg": lbg, "ubg": ubg},
            "monee_casadi",
            "solve",
            x0_alt=None if warm else _consistent_flow_x0(reg, x0),
        )
        self.last_solve_s = time.perf_counter() - t0
        self.last_engine_s = self.last_build_s + self.last_solve_s
        success = bool(stats.get("success", False))
        self.last_iters = stats.get("iter_count")
        if not success:
            report = _no_convergence_report(
                "solve",
                stats,
                network,
                X,
                np.array(sol["x"]).flatten(),
                g,
                lbg,
                ubg,
                reg,
            )
            if strict:
                raise report.as_error()
            _log.error("%s", report.message)
            return self._failed_result(
                network,
                input_network,
                nodes,
                branches,
                compounds,
                ignored_nodes,
                reg,
                X,
                np.array(sol["x"]).flatten(),
                report,
                simulation,
            )
        x_opt = np.array(sol["x"]).flatten()

        _write_back_solution(reg, x_opt)
        _eval_leftover_cassyms(
            self._active_models(network, nodes, branches, compounds, ignored_nodes),
            X,
            x_opt,
        )

        violations = finalize_solution(
            nodes, branches, compounds, network, input_network
        )
        objective = float(sol["f"]) if m.obj_terms else 0.0
        user_objective = _eval_obj_terms(X, x_opt, m.user_obj_terms)
        solver_status, termination_condition = _status_fields(stats)
        residuals = warn_on_residuals(_solution_residual(X, x_opt, g, lbg, ubg))
        result = SolverResult(
            network,
            network.as_result_dataframe_dict(),
            objective,
            success,
            violations,
            solver_status=solver_status,
            termination_condition=termination_condition,
            residuals=residuals,
            mode_used="simulation" if simulation else "optimization",
            backend_used=self.backend_name,
            solver_used=self.solver_name,
            user_objective=user_objective,
            aux_objective=(
                None if user_objective is None else objective - user_objective
            ),
        )
        return validate_result(
            network,
            result,
            ignored_nodes=ignored_nodes,
            simulation=simulation,
            strict=strict,
            relaxed_integrality=relaxed_components,
            extra_warnings=[dof_entry] if dof_entry is not None else None,
        )


class CasADiTimeseries:
    """Build-once / re-solve-per-step timeseries driver on the CasADi backend.

    The whole point of an in-process compiled backend: the model *structure* is
    identical across timesteps, so we assemble the CasADi graph and the
    ``nlpsol`` solver **once**, with every time-varying input declared as a
    CasADi *parameter* (``p``). Each step then only sets the parameter vector and
    re-solves with a warm start from the previous step - no rebuild, no
    re-compile, no subprocess. Contrast :func:`monee.run_timeseries`, which
    rebuilds and recompiles the entire model every step.

    Scope mirrors :class:`CasADiSolver`: electricity AC, plain power flow
    (``optimization_problem=None``). Time-varying attributes come from a
    :class:`~monee.simulation.timeseries.TimeseriesData` (child/node/branch id
    series). Inter-step temporal coupling is *not* supported - only memory-less
    per-step inputs (loads, generation).
    """

    def __init__(  # NOSONAR
        self,
        input_network,
        timeseries_data,
        formulation=None,
        simulation=True,
        steps=None,
        solver_options: dict | None = None,
    ):
        _require_casadi()
        timeseries_data.validate_bound(input_network, strict=False)
        self._td = timeseries_data
        self._simulation = simulation
        cs = CasADiSolver(solver_options=solver_options)
        cs._simulation = simulation
        cs._reg = []
        m = CasModel()

        network, ignored, _islanding_config = prepare_solve_network(
            input_network,
            optimization_problem=None,
            formulation=formulation,
            simulation=simulation,
        )
        _substitute_relaxed_defaults(network, formulation, simulation)
        warn_relaxed_integrality(network, "IPOPT")
        nodes = network.nodes

        # Declare every time-varying attribute as a CasADi parameter BEFORE the
        # overwrite/inject passes. A boundary child whose pinned setpoint is itself
        # time-varying (e.g. an ExtHydrGrid supply ``t_k``, an ExtPowerGrid
        # ``vm_pu``) folds that attribute into the node boundary inside
        # ``overwrite``; declaring the parameter first means the boundary captures
        # the parameter *symbol* rather than a value frozen at step 0.
        self._params = []  # (model, key, param_sx, series)
        self._declare_params(network, timeseries_data)

        apply_child_overwrites(network, nodes, ignored)
        branches, compounds = network.branches, network.compounds
        mark_slacks_and_prescriptions(network, ignored)

        inject_vars(cs._inject, nodes, branches, compounds, network, ignored)

        objs: list = []
        cs.init_branches(branches)
        cs.process_equations_nodes_childs(m, network, nodes, ignored)
        cs.process_equations_branches(m, network, branches, ignored, objs)
        cs.process_equations_compounds(m, network, compounds, ignored)
        cs.process_internal_oxf_components(m, network)
        for ext in network.extensions:
            m.Equations(ext.equations(network, ignored))
        for expr in objs:
            m.Obj(expr)

        reg = cs._reg
        self._network, self._nodes = network, nodes
        self._branches, self._compounds, self._ignored = branches, compounds, ignored
        self._var_entries = reg
        P = (
            ca.vertcat(*[p[2] for p in self._params])
            if self._params
            else ca.SX.zeros(0)
        )
        X, x0, lbx, ubx, g, lbg, ubg, f = _assemble_nlp_vectors(reg, m)
        self._lbx, self._ubx = lbx, ubx
        self._x = x0  # warm-start state
        self._lbg, self._ubg = lbg, ubg
        self._has_obj = bool(m.obj_terms)

        t0 = time.perf_counter()
        # Merge duplicate sub-terms before the autodiff build (see CasADiSolver.solve).
        f, g = ca.cse([f, g])
        self._nlp = {"x": X, "f": f, "g": g, "p": P}
        self._opts = _merged_ipopt_opts(cs.solver_options)
        self._solver = ca.nlpsol("monee_ts", "ipopt", self._nlp, self._opts)
        self._g_fun = ca.Function("monee_ts_g", [X, P], [g]) if g.numel() > 0 else None

        # A child overwrite may replace a declared parameter with its own value
        # (the equations then never see the series); only the parameters still
        # installed on their model may be scattered back for reporting.
        self._scatter_params = [
            (model, key, series)
            for model, key, psx, series in self._params
            if getattr(model.__dict__.get(key), "e", None) is psx
        ]

        # One compiled function (X, P) -> all inlined-intermediate values, for
        # cheap per-step result extraction.
        var_ids = {(id(r["model"]), r["key"]) for r in reg}
        par_ids = {(id(m), k) for m, k, _ in self._scatter_params}
        self._inter = [
            (model, key, val.e)
            for model in self._active_models()
            for key, val in model.__dict__.items()
            if isinstance(val, CasSym)
            and (id(model), key) not in var_ids
            and (id(model), key) not in par_ids
        ]
        self._f_inter = (
            ca.Function("inter", [X, P], [ca.vertcat(*[e for _, _, e in self._inter])])
            if self._inter
            else None
        )
        self.build_s = time.perf_counter() - t0
        self.steps = steps if steps is not None else timeseries_data.length
        self.last_solve_total_s = None
        self.last_iters_total = None
        self.last_step_success = None

    # ------------------------------------------------------------------ #
    def _declare_params(self, network, td):  # NOSONAR
        # Mirror TimeseriesData.apply_to_network: iterate the network's components
        # and gather the series that target each one by id AND by name (and the
        # compound dicts). Iterating td's id-only dicts (as before) silently
        # dropped name-addressed child/branch series and all compound series.
        def add(model, attr, series):
            psx = ca.SX.sym(f"p{len(self._params)}")
            self._params.append((model, attr, psx, series))
            setattr(model, attr, CasSym(psx))

        def merged(comp, id_dict, name_dict):
            # id series first, then name series (name wins on attr conflict) so
            # each attr yields exactly one parameter - matches apply_to_*.
            out = {}
            if comp.id in id_dict:
                out.update(id_dict[comp.id])
            name = getattr(comp, "name", None)
            if name is not None and name in name_dict:
                out.update(name_dict[name])
            return out

        for node in network.nodes:
            for attr, series in td._node_id_to_series.get(node.id, {}).items():
                add(node.model, attr, series)
            for child in network.childs_by_ids(node.child_ids):
                for attr, series in merged(
                    child, td._child_id_to_series, td._child_name_to_series
                ).items():
                    add(child.model, attr, series)
        for branch in network.branches:
            for attr, series in merged(
                branch, td._branch_id_to_series, td._branch_name_to_series
            ).items():
                add(branch.model, attr, series)
        for compound in network.compounds:
            for attr, series in merged(
                compound, td._compound_id_to_series, td._compound_name_to_series
            ).items():
                add(compound.model, attr, series)

    def _active_models(self):
        return iter_active_models(
            self._network, self._nodes, self._branches, self._compounds, self._ignored
        )

    # ------------------------------------------------------------------ #
    def _solve_step(self, t):
        """Set the parameter vector for step *t*, re-solve (warm-started), and
        scatter the solution into the network models. Returns the solve stats."""
        pvals = ca.DM([float(p[3][t]) for p in self._params])
        self._p = pvals
        t0 = time.perf_counter()
        sol, stats = _solve_with_retry(
            self._solver,
            self._nlp,
            self._opts,
            {
                "x0": self._x,
                "lbx": self._lbx,
                "ubx": self._ubx,
                "lbg": self._lbg,
                "ubg": self._ubg,
                "p": pvals,
            },
            "monee_ts",
            f"timeseries step {t}",
        )
        solve_s = time.perf_counter() - t0
        self._x = sol["x"]  # warm-start the next step
        self._last_objective = float(sol["f"]) if self._has_obj else 0.0

        # scatter into models for result extraction
        xv = np.array(sol["x"]).flatten()
        _write_back_solution(self._var_entries, xv)
        for model, key, series in self._scatter_params:
            model.__dict__[key] = float(series[t])
        if self._f_inter is not None:
            ivals = np.array(self._f_inter(sol["x"], pvals)).flatten()
            for (model, key, _), v in zip(self._inter, ivals):
                model.__dict__[key] = Intermediate(value=float(v))
        apply_post_process_all(
            self._nodes, self._branches, self._compounds, self._network
        )
        return solve_s, stats

    def step_result(self, t) -> SolverResult:
        """Solve step *t* and return a :class:`SolverResult` for the network at
        that step (used by the :func:`monee.run_timeseries` reuse fast path)."""
        _solve_s, stats = self._solve_step(t)
        success = bool(stats.get("success", False))
        self.last_step_success = success
        if not success:
            raise CasADiSolveError(
                _no_convergence_message(f"timeseries step {t}", stats, self._network)
            )
        violations = compute_bound_violations(
            self._nodes, self._branches, self._compounds, self._network
        )
        solver_status, termination_condition = _status_fields(stats)
        residuals = None
        if self._g_fun is not None:
            gv = np.array(self._g_fun(self._x, self._p)).flatten()
            residuals = warn_on_residuals(
                _residual_from_values(gv, self._lbg, self._ubg)
            )
        return SolverResult(
            self._network,
            self._network.as_result_dataframe_dict(),
            self._last_objective,
            success,
            violations,
            solver_status=solver_status,
            termination_condition=termination_condition,
            residuals=residuals,
            mode_used="simulation" if self._simulation else "optimization",
            backend_used="casadi",
            solver_used="ipopt",
        )

    def run(self):
        """Solve every timestep, reusing the compiled solver. Returns a list of
        per-step ``{type_name: DataFrame}`` dicts (like
        :attr:`SolverResult.dataframes`). A non-converged step raises
        :class:`CasADiSolveError` (same failure contract as
        :meth:`step_result`)."""
        results = []
        solve_total = 0.0
        iters_total = 0
        for t in range(self.steps):
            solve_s, stats = self._solve_step(t)
            solve_total += solve_s
            iters_total += stats.get("iter_count") or 0
            success = bool(stats.get("success", False))
            self.last_step_success = success
            if not success:
                self.last_solve_total_s = solve_total
                self.last_iters_total = iters_total
                raise CasADiSolveError(
                    _no_convergence_message(
                        f"timeseries step {t}", stats, self._network
                    )
                )
            results.append(self._network.as_result_dataframe_dict())
        self.last_solve_total_s = solve_total
        self.last_iters_total = iters_total
        return results


class CasADiMultiPeriodSolver:
    """Multi-period optimizer on in-process CasADi/IPOPT.

    Two-pass, mirroring :class:`~monee.simulation.multi_period.GekkoMultiPeriodSolver`:
    inject CasADi symbols for all T periods into one register, then assemble
    per-period and ``inter_period`` equations with a
    :class:`~monee.simulation.step_state.PeriodState` that sees every period
    (so cross-period constraints become algebraic links between the periods'
    injected variables), and solve the whole horizon as a single ``nlpsol`` NLP.
    """

    def __init__(self):
        _require_casadi()
        self._backend_name = "casadi"
        self._solver_name = "ipopt"
        self.last_build_s = None
        self.last_solve_s = None
        self.last_iters = None

    def solve_multi_period(  # NOSONAR
        self,
        network: Network,
        timeseries_data=None,
        steps: int = None,
        optimization_problem=None,
        dt_h=None,
        datetime_index=None,
        initial_state: dict = None,
        terminal_state: dict = None,
        formulation=None,
    ):
        from monee.simulation.multi_period import (
            MultiPeriodResult,
            _find_component_var,
            _prepare_period,
            _resolve_dt_h,
            _resolve_steps,
        )
        from monee.simulation.step_state import PeriodState

        steps = _resolve_steps(steps, timeseries_data)
        dt_h_list = _resolve_dt_h(dt_h, datetime_index, steps)

        cs = CasADiSolver()
        cs._simulation = False
        cs._reg = []
        m = CasModel()

        # Pass 1: prepare networks and inject CasADi symbols for all periods.
        net_copies: list[Network] = []
        ignored_list: list[set] = []
        reg_spans: list[tuple[int, int, int]] = []
        for t in range(steps):
            net_t, ignored_t = _prepare_period(
                network, timeseries_data, t, optimization_problem, formulation
            )
            _substitute_relaxed_defaults(net_t, formulation, simulation=False)
            reg_start = len(cs._reg)
            inject_vars(
                cs._inject,
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                net_t,
                ignored_t,
            )
            for ext in net_t.extensions:
                ext.activate_timeseries(net_t, ignored_t)
            cs.mark_temporal_components(net_t, ignored_t)
            reg_spans.append((t, reg_start, len(cs._reg)))
            net_copies.append(net_t)
            ignored_list.append(ignored_t)

        # Pass 2: build per-period and inter-period equations.
        cons_spans: list[tuple[int, int, int]] = []
        for t in range(steps):
            cons_start = len(m.cons)
            net_t, ignored_t = net_copies[t], ignored_list[t]
            period_state = PeriodState(
                net_copies,
                current_t=t,
                dt_h=dt_h_list[t],
                initial_state=initial_state,
            )
            cs.init_branches(net_t.branches)
            per_objs: list = []
            cs.process_equations_nodes_childs(m, net_t, net_t.nodes, ignored_t)
            cs.process_equations_branches(m, net_t, net_t.branches, ignored_t, per_objs)
            cs.process_equations_compounds(m, net_t, net_t.compounds, ignored_t)
            if optimization_problem is not None:
                cs.process_oxf_components(
                    m, net_t, optimization_problem, period_index=t
                )
            else:
                cs.process_internal_oxf_components(m, net_t)
            for expr in per_objs:
                m.Obj(expr)
            cs.process_inter_period_equations(
                m,
                net_t,
                net_t.nodes,
                net_t.branches,
                net_t.compounds,
                ignored_t,
                period_state,
                optimization_problem=optimization_problem,
                period_index=t,
            )
            for ext in net_t.extensions:
                m.Equations(ext.inter_period_equations(net_t, ignored_t, period_state))
                m.Equations(
                    ext.inter_temporal_equations(net_t, ignored_t, period_state)
                )
                m.Equations(ext.equations(net_t, ignored_t))
            if terminal_state and t == steps - 1:
                for (comp_id, attr), target in terminal_state.items():
                    var = _find_component_var(net_t, comp_id, attr)
                    if isinstance(var, (int, float)):
                        # A plain constant would make ``var == target`` a Python
                        # bool; only a solver variable / expression can be pinned.
                        _log.warning(
                            "terminal_state target (%r, %r) is a constant, not "
                            "a solver variable; the terminal constraint was "
                            "skipped. Make the attribute controllable to pin it.",
                            comp_id,
                            attr,
                        )
                    elif var is not None:
                        m.Equation(var == target)
            cons_spans.append((t, cons_start, len(m.cons)))

        # Assemble and solve the whole horizon as one NLP.
        reg = cs._reg
        X, x0, lbx, ubx, g, lbg, ubg, f = _assemble_nlp_vectors(reg, m)

        # run_mpc drives every rolling window through the same solver instance,
        # so this counter is the window index in an MPC run.
        self._mp_solve_index = getattr(self, "_mp_solve_index", 0) + 1
        _log.info("Multi-period CasADi solve: T=%d periods", steps)
        t0 = time.perf_counter()
        nlp = {"x": X, "f": f, "g": g}
        solver = ca.nlpsol("monee_casadi_mp", "ipopt", nlp, _IPOPT_OPTS)
        self.last_build_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        sol, stats = _solve_with_retry(
            solver,
            nlp,
            _IPOPT_OPTS,
            {"x0": x0, "lbx": lbx, "ubx": ubx, "lbg": lbg, "ubg": ubg},
            "monee_casadi_mp",
            f"multi-period solve {self._mp_solve_index} "
            f"(T={steps} periods; window {self._mp_solve_index} under run_mpc)",
        )
        self.last_solve_s = time.perf_counter() - t0
        success = bool(stats.get("success", False))
        self.last_iters = stats.get("iter_count")
        if not success:
            _raise_no_convergence(
                "multi-period solve",
                stats,
                network,
                X,
                np.array(sol["x"]).flatten(),
                g,
                lbg,
                ubg,
                reg,
                cons_spans=cons_spans,
                reg_spans=reg_spans,
            )
        x_opt = np.array(sol["x"]).flatten()

        # Scatter the solution back into every period's models.
        _write_back_solution(reg, x_opt)
        _eval_leftover_cassyms(
            (
                model
                for net_t, ignored_t in zip(net_copies, ignored_list)
                for model in cs._active_models(
                    net_t, net_t.nodes, net_t.branches, net_t.compounds, ignored_t
                )
            ),
            X,
            x_opt,
            fname="intval_mp",
        )
        for net_t in net_copies:
            apply_post_process_all(net_t.nodes, net_t.branches, net_t.compounds, net_t)

        objective = float(sol["f"]) if m.obj_terms else 0.0
        user_objective = _eval_obj_terms(X, x_opt, m.user_obj_terms)
        solver_status, termination_condition = _status_fields(stats)
        return MultiPeriodResult(
            net_copies,
            objective=objective,
            success=success,
            solver_status=solver_status,
            termination_condition=termination_condition,
            datetime_index=datetime_index,
            backend_used=self._backend_name,
            solver_used=self._solver_name,
            user_objective=user_objective,
        )
