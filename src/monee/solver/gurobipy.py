"""Native gurobipy solver backend.

Gurobi is reachable through Pyomo (``PyomoSolver(solver_name="gurobi")``), but
that path serialises the model to an LP/NL file and shells out.  This backend
builds the model in-memory with the ``gurobipy`` API, which both avoids the
file round-trip and exposes Gurobi features Pyomo hides - notably IIS-based
infeasibility diagnosis (``computeIIS``) and direct warm starts (``Var.Start``).

It mirrors :class:`~monee.solver.pyo.PyomoSolver`'s structure (same calls into
the backend-agnostic helpers of :mod:`monee.solver.core`) and supports the same
formulation families:

* **MIP / (MI)QCQP / MISOCP** (``convex_miqcqp``, ``el_misocp``,
  ``gas_convex_miqcqp``, ``heat_convex_milp``, ``gas_milp_pwl`` ...): linear,
  quadratic and second-order-cone constraints map straight onto gurobipy's
  ``LinExpr`` / ``QuadExpr`` and ``addGenConstrPWL``.
* **smooth NLP** (``smooth_nlp``, ``*_nlp``): the ``sqrt``/``sin``/``cos``/
  ``log``/``exp`` and divide-by-variable terms are built with ``gurobipy.nlfunc``
  and attached as general nonlinear constraints (Gurobi >= 12).

The one structural wrinkle: Gurobi only accepts a nonlinear constraint in the
form ``y = f(x)`` (a single resultant variable equals a nonlinear expression).
Monee's ``equations()`` bodies return arbitrary relationals (``lhs <= rhs`` with
nonlinear terms on either side).  :meth:`GurobipySolver._add_temp_constr`
therefore rewrites every nonlinear relational ``lhs (==|<=|>=) rhs`` as a fresh
free auxiliary ``a`` with ``a == lhs - rhs`` (via ``addGenConstrNL``) plus the
linear ``a (==|<=|>=) 0`` - which also lets nonlinear *inequalities* through.
Purely linear/quadratic relationals are passed to ``addConstr`` untouched so the
QCP/SOCP machinery still recognises the cones.
"""

import logging
import math
import time
import types

from monee.model import (
    Const,
    GenericModel,
    Intermediate,
    IntermediateEq,
    Network,
    Var,
)
from monee.problem.core import OptimizationProblem

from .core import (
    InterStepState,
    SolverInterface,
    SolverResult,
    StepState,
    apply_child_overwrites,
    apply_post_process_all,
    as_iter,
    compute_bound_violations,
    filter_bool_eqs,
    filter_intermediate_eqs,
    finalize_solution,
    ignore_branch,
    ignore_child,
    ignore_compound,
    ignore_node,
    inject_vars,
    mark_slacks_and_prescriptions,
    prepare_solve_network,
    withdraw_vars,
)

_log = logging.getLogger(__name__)

# Gurobi parameter defaults for the native backend.  Deliberately minimal -
# ~kW precision at MW scale (MIPGap) and a wall-clock cap (TimeLimit) - letting
# Gurobi auto-tune everything else.  Note: the ScaleFlag=2 / MIPFocus=2 pair
# tuned for the *pyomo* file path does NOT transfer here; on the native model
# MIPFocus=2 sent the MISOCP load-shedding solve from ~0.6 s to ~9 s (it forces
# bound-proving work the auto strategy avoids).  Override per solve via
# ``GurobipySolver(params=...)``.
DEFAULT_GUROBI_PARAMS: dict = {
    "MIPGap": 1e-3,
    "TimeLimit": 300,
}


def _require_gurobipy():
    """Import gurobipy lazily so a missing install doesn't break other backends."""
    try:
        import gurobipy as gp
        from gurobipy import GRB, nlfunc
    except ImportError as exc:  # pragma: no cover - exercised only without gurobipy
        raise ImportError(
            "The gurobipy backend requires the 'gurobipy' package (Gurobi >= 12 "
            "for the nonlinear NLP formulations). Install it with "
            "`pip install gurobipy` and a valid Gurobi licence."
        ) from exc
    return gp, GRB, nlfunc


class GurobiIISReport:
    """Lightweight infeasibility report built from Gurobi's IIS.

    Exposes the same ``summary()`` surface as the Pyomo/GEKKO reports so it
    can be carried on :attr:`SolverResult.infeasibility_report` uniformly.
    """

    def __init__(self, constraints: list[str], bounds: list[str]):
        self.constraints = constraints
        self.bounds = bounds

    def summary(self, max_items: int = 50) -> str:
        lines = ["Irreducible Inconsistent Subsystem (Gurobi IIS):"]
        if self.bounds:
            lines.append(f"  Variable bounds in IIS ({len(self.bounds)}):")
            lines += [f"    {b}" for b in self.bounds[:max_items]]
        if self.constraints:
            lines.append(f"  Constraints in IIS ({len(self.constraints)}):")
            lines += [f"    {c}" for c in self.constraints[:max_items]]
        if not self.bounds and not self.constraints:
            lines.append("  (empty IIS - model may be unbounded, not infeasible)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GurobiIISReport(constraints={len(self.constraints)}, "
            f"bounds={len(self.bounds)})"
        )


class GurobiPWLImpl:
    """SOS2/piecewise-linear equality ``y == pwl(x; xs, ys)`` via
    ``addGenConstrPWL``.  When *x* is an expression rather than a plain ``Var``
    a free auxiliary variable is introduced to hold it (Gurobi's PWL constraint
    requires both arguments to be variables)."""

    def __init__(self, solver: "GurobipySolver", gm):
        self.solver = solver
        self.gm = gm

    def piecewise_eq(self, y, x, xs, ys, name=None):
        gp = self.solver._gp
        xv = x if isinstance(x, gp.Var) else self.solver._free_aux(self.gm, eq=x)
        yv = y if isinstance(y, gp.Var) else self.solver._free_aux(self.gm, eq=y)
        self.gm.addGenConstrPWL(
            xv, yv, list(xs), list(ys), name=self.solver._sanitize_name(name or "pwl")
        )


class GurobipySolver(SolverInterface):
    """gurobipy-backed solver.

    Args:
        params: extra Gurobi parameters merged over :data:`DEFAULT_GUROBI_PARAMS`
            (e.g. ``{"TimeLimit": 60, "MIPGap": 1e-4}``).
    """

    _BOUND_SNAP_TOL = 1e-9

    _LEX_REL_TOL = 1e-6
    _LEX_ABS_TOL = 1e-9

    _LEX_AUX_MIPGAP = 1e-2
    _LEX_AUX_TIMELIMIT = 15.0

    def __init__(self, params: dict | None = None):
        self._backend_name = "gurobipy"
        self._solver_name = "gurobi"
        self._params = dict(DEFAULT_GUROBI_PARAMS)
        if params:
            self._params.update(params)
        # Per-solve state, (re)initialised at the top of solve().
        self._simulation: bool = False
        self._gp = None
        self._GRB = None
        self._nlfunc = None

    # --------- variable injection / withdrawal ---------

    def inject_gurobi_vars_attr(self, gm, target: GenericModel, prefix: str):  # NOSONAR
        """Replace Var/Const with a gurobipy Var / float.  The stale solved
        value (clamped into the current bounds, NaN scrubbed) seeds ``Var.Start``
        for a warm start."""
        GRB = self._GRB
        for key, value in target.__dict__.items():
            if isinstance(value, Var):
                lb = value.min if value.min is not None else -GRB.INFINITY
                ub = value.max if value.max is not None else GRB.INFINITY
                v = gm.addVar(
                    lb=lb,
                    ub=ub,
                    vtype=GRB.INTEGER if value.integer else GRB.CONTINUOUS,
                    name=self._sanitize_name(f"{prefix}__{key}"),
                )
                init = value.value
                if init is not None and isinstance(init, float) and math.isnan(init):
                    init = None
                if init is not None:
                    if value.min is not None and init < value.min:
                        init = value.min
                    if value.max is not None and init > value.max:
                        init = value.max
                    v.Start = init
                setattr(target, key, v)
            elif type(value) is Const:
                setattr(target, key, float(value.value))

    def withdraw_gurobi_vars_attr(self, target: GenericModel):
        """gurobipy Var -> :class:`Var`; gurobipy expression (a passive
        :class:`Intermediate`, see :meth:`_process_intermediate_eqs`) ->
        :class:`Intermediate`.  Mirrors the Pyomo backend: integer rounding,
        NaN/None -> 0, bound-noise snapping."""
        gp = self._gp
        for key, value in list(target.__dict__.items()):  # NOSONAR
            if isinstance(value, (gp.LinExpr, gp.QuadExpr, gp.NLExpr)):
                setattr(target, key, Intermediate(value=self._expr_value(value)))
                continue
            if not isinstance(value, gp.Var):
                continue
            setattr(target, key, self._var_from_gurobi(value))

    def _var_from_gurobi(self, value) -> Var:
        """Convert a solved gurobipy ``Var`` into a monee :class:`Var` with
        integer rounding, NaN/None -> 0 and bound-noise snapping."""
        GRB = self._GRB
        val = self._var_value(value)
        is_integer = value.VType in (GRB.INTEGER, GRB.BINARY)
        lb = value.LB
        ub = value.UB
        lb = None if lb <= -GRB.INFINITY else lb
        ub = None if ub >= GRB.INFINITY else ub
        if is_integer:
            val = int(round(val))
        else:
            tol = self._BOUND_SNAP_TOL
            if lb is not None and lb - tol <= val < lb:
                val = lb
            if ub is not None and ub < val <= ub + tol:
                val = ub
        return Var(value=val, min=lb, max=ub, integer=is_integer)

    @staticmethod
    def _var_value(v) -> float:
        """``Var.X`` if a solution exists, else 0 (so a failed solve still hands
        a usable warm start to the next solve)."""
        try:
            x = v.X
        except Exception:
            return 0.0
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return 0.0
        return x

    def _expr_value(self, expr) -> float:
        """Numeric value of a passive intermediate expression after a solve.
        Linear/quadratic expressions evaluate via ``getValue``; nonlinear ones
        (no ``getValue``) are evaluated over their array representation."""
        if isinstance(expr, self._gp.NLExpr):
            try:
                return self._eval_nlexpr(*expr._to_array_repr())
            except Exception:
                return float("nan")
        try:
            return expr.getValue()
        except Exception:
            return float("nan")

    def _eval_nlexpr(self, opcodes, data, parent) -> float:  # NOSONAR
        """Evaluate a Gurobi nonlinear-expression tree at the solved point.

        The tree is ``(opcodes, data, parent)``: ``opcodes[i]`` is the operation
        at node ``i``, ``data[i]`` carries a constant or :class:`Var` for the
        leaf opcodes, and ``parent[i]`` is the parent node (-1 at the root).
        Children appear in ascending node order, which is operand order."""
        GRB = self._GRB
        children: dict[int, list[int]] = {}
        root = 0
        for i, p in enumerate(parent):
            if p < 0:
                root = i
            else:
                children.setdefault(p, []).append(i)

        unary = {
            GRB.OPCODE_SQRT: math.sqrt,
            GRB.OPCODE_SIN: math.sin,
            GRB.OPCODE_COS: math.cos,
            GRB.OPCODE_TAN: math.tan,
            GRB.OPCODE_EXP: math.exp,
            GRB.OPCODE_LOG: math.log,
            GRB.OPCODE_LOG2: math.log2,
            GRB.OPCODE_LOG10: math.log10,
            GRB.OPCODE_TANH: math.tanh,
            GRB.OPCODE_SQUARE: lambda v: v * v,
            GRB.OPCODE_UMINUS: lambda v: -v,
            GRB.OPCODE_LOGISTIC: lambda v: 1.0 / (1.0 + math.exp(-v)),
        }

        def ev(i: int) -> float:
            op = opcodes[i]
            if op == GRB.OPCODE_CONSTANT:
                return float(data[i])
            if op == GRB.OPCODE_VARIABLE:
                return self._var_value(data[i])
            ch = [ev(c) for c in children.get(i, [])]
            if op == GRB.OPCODE_PLUS:
                return sum(ch)
            if op == GRB.OPCODE_MULTIPLY:
                out = 1.0
                for c in ch:
                    out *= c
                return out
            if op == GRB.OPCODE_MINUS:
                return ch[0] - ch[1]
            if op == GRB.OPCODE_DIVIDE:
                return ch[0] / ch[1]
            if op == GRB.OPCODE_POW:
                return ch[0] ** ch[1]
            if op == GRB.OPCODE_SIGNPOW:
                return math.copysign(abs(ch[0]) ** ch[1], ch[0])
            fn = unary.get(op)
            if fn is None:
                raise ValueError(f"unsupported nonlinear opcode {op}")
            return fn(ch[0])

        return ev(root)

    # --------- expression / constraint helpers ---------

    @staticmethod
    def _sanitize_name(name: str) -> str:
        """Make a string safe for a Gurobi component name (no spaces/brackets)."""
        return (
            str(name)
            .replace("(", "")
            .replace(")", "")
            .replace(", ", "_")
            .replace(" ", "_")
            .replace("[", "")
            .replace("]", "")
        )

    def _is_nonlinear(self, expr) -> bool:
        return isinstance(expr, self._gp.NLExpr)

    def _free_aux(self, gm, eq=None):
        """A fresh free (unbounded) auxiliary variable, optionally bound to *eq*
        via the appropriate (linear/quadratic/nonlinear) defining constraint."""
        GRB = self._GRB
        aux = gm.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        if eq is not None:
            if self._is_nonlinear(eq):
                gm.addGenConstrNL(aux, eq)
            else:
                gm.addConstr(aux == eq)
        return aux

    def _add_temp_constr(self, gm, tc, name=None):
        """Register one relational constraint.  Linear/quadratic/SOC go straight
        to ``addConstr``; nonlinear relationals are rewritten through a free
        auxiliary (see module docstring)."""
        GRB = self._GRB
        combined = tc._lhs - tc._rhs
        if not self._is_nonlinear(combined):
            gm.addConstr(tc, name=self._sanitize_name(name) if name else "")
            return
        aux = gm.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY)
        gm.addGenConstrNL(aux, combined)
        sense = tc._sense
        cname = self._sanitize_name(name) if name else ""
        if sense == GRB.EQUAL:
            gm.addConstr(aux == 0, name=cname)
        elif sense == GRB.LESS_EQUAL:
            gm.addConstr(aux <= 0, name=cname)
        else:  # GRB.GREATER_EQUAL
            gm.addConstr(aux >= 0, name=cname)

    def _add_equation(self, gm, expr, name=None):
        # A Python bool sentinel is not a constraint: True is a tautology
        # (no-op); False is structural infeasibility from an over-deactivated
        # component - drop it so load-shedding can resolve the situation but
        # warn so the modelling error surfaces (mirrors the other backends).
        if isinstance(expr, bool):
            if expr is False:
                _log.warning(
                    "Dropping always-false (structurally infeasible) equation%s; "
                    "this usually means a node/branch was over-deactivated.",
                    f" ({name})" if name is not None else "",
                )
            return
        self._add_temp_constr(gm, expr, name=name)

    def _add_equations(self, gm, exprs, name_prefix=None):
        for i, e in enumerate(exprs):
            name = f"{name_prefix}_eq_{i}" if name_prefix is not None else None
            self._add_equation(gm, e, name=name)

    def _process_intermediate_eqs(self, model_obj, equations):
        """Bind each :class:`Intermediate` attribute to its defining *expression*
        (not a variable), exactly like the Pyomo backend's ``pyo.Expression``.

        A passive expression embeds into downstream equations that reference it,
        but - crucially - never enters the model on its own.  Reporting-only
        intermediates (e.g. ``i_from_ka``/``loading_*`` ``= sqrt(...)``) thus add
        no constraints; materialising them as ``addGenConstrNL`` instead turned
        the pure MISOCP into a non-convex MINLP and blew the time limit."""
        for intermediate_eq in [eq for eq in equations if type(eq) is IntermediateEq]:
            attr_val = getattr(model_obj, intermediate_eq.attr)
            if type(attr_val) is not Intermediate:
                continue
            eq = (
                intermediate_eq.eq()
                if isinstance(intermediate_eq.eq, types.FunctionType)
                else intermediate_eq.eq
            )
            setattr(model_obj, intermediate_eq.attr, eq)

    # --------- objective ---------

    def _set_objective(self, gm, exprs):
        """Minimise ``sum(exprs)``; a nonlinear sum is hosted on a free
        auxiliary (Gurobi's objective must be linear/quadratic)."""
        obj = sum(exprs) if exprs else 0
        if self._is_nonlinear(obj):
            obj = self._free_aux(gm, eq=obj)
        gm.setObjective(obj, self._GRB.MINIMIZE)

    def _obj_value(self, gm, exprs) -> float:
        """Combined objective value ``sum(exprs)`` after a solve (matches the
        Pyomo backend, which reports user+aux even in lex mode).  Falls back to
        ``gm.ObjVal`` when the sum is nonlinear (NLExpr has no ``getValue``)."""
        try:
            e = sum(exprs) if exprs else 0.0
            if isinstance(e, (int, float)):
                return float(e)
            return e.getValue()
        except Exception:
            try:
                return gm.ObjVal
            except Exception:
                return float("nan")

    def _linearize_for_multiobj(self, gm, exprs):
        """A linear stand-in for ``sum(exprs)``, usable as a ``setObjectiveN``
        objective.  Linear sums pass through; a quadratic/nonlinear sum is
        hoisted onto a free auxiliary ``a`` (``a == sum``) - the multi-objective
        API requires *linear* objectives, but the model may carry the quadratic
        constraint that defines ``a``."""
        e = sum(exprs) if exprs else 0.0
        if isinstance(e, (int, float)) or not isinstance(
            e, (self._gp.QuadExpr, self._gp.NLExpr)
        ):
            return e
        return self._free_aux(gm, eq=e)

    @classmethod
    def _lex_cap_slack(cls, s_star: float, mip_gap: float) -> float:
        rel = max(float(mip_gap or 0.0), cls._LEX_REL_TOL)
        return cls._LEX_ABS_TOL + rel * max(1.0, abs(s_star))

    # --------- result classification ---------

    def _classify(self, gm, *, phase_label: str):
        """Map a Gurobi status to ``(success, report)``.  OPTIMAL -> silent
        success; INFEASIBLE -> IIS report + error; a limit hit with a feasible
        incumbent -> warning, treated as success."""
        GRB = self._GRB
        status = gm.Status
        if status == GRB.OPTIMAL:
            return True, None
        if status == GRB.INFEASIBLE:
            report = self._compute_iis(gm)
            _log.error(
                "%s infeasible (status=%s).  Diagnostic report:\n%s",
                phase_label,
                status,
                report.summary(max_items=50),
            )
            return False, report
        if gm.SolCount > 0:
            _log.warning(
                "%s returned non-optimal status %s; using witness solution.",
                phase_label,
                status,
            )
            return True, None
        _log.error(
            "%s failed without a usable solution (status=%s).", phase_label, status
        )
        return False, None

    def _compute_iis(self, gm) -> GurobiIISReport:
        constraints: list[str] = []
        bounds: list[str] = []
        try:
            gm.computeIIS()
            for c in gm.getConstrs():
                if c.IISConstr:
                    constraints.append(c.ConstrName or f"c{c.index}")
            for gc in gm.getGenConstrs():
                if getattr(gc, "IISGenConstr", 0):
                    constraints.append(gc.GenConstrName or f"genc{gc.index}")
            for v in gm.getVars():
                if v.IISLB or v.IISUB:
                    side = "/".join(
                        s for s, f in (("LB", v.IISLB), ("UB", v.IISUB)) if f
                    )
                    bounds.append(f"{v.VarName} [{side}]")
        except Exception as exc:  # pragma: no cover - IIS can itself fail
            _log.warning("Gurobi computeIIS failed: %s", exc)
        return GurobiIISReport(constraints, bounds)

    # --------- main solve ---------

    def solve(  # NOSONAR
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes: bool = False,
        step_state: StepState = None,
        simulation: bool = False,
        formulation=None,
        **kwargs,
    ):
        gp, GRB, nlfunc = _require_gurobipy()
        self._gp, self._GRB, self._nlfunc = gp, GRB, nlfunc
        debug = draw_debug or kwargs.pop("debug", False)
        self._simulation = simulation

        gm = gp.Model("monee")
        gm.setParam("OutputFlag", 1 if debug else 0)
        for key, val in self._params.items():
            gm.setParam(key, val)

        # Two objective buckets for lexicographic mode (single sum otherwise).
        user_obj_exprs: list = []
        aux_obj_exprs: list = []

        network, ignored_nodes, _islanding_config = prepare_solve_network(
            input_network,
            optimization_problem=optimization_problem,
            formulation=formulation,
            simulation=simulation,
            exclude_unconnected_nodes=exclude_unconnected_nodes,
        )

        nodes = network.nodes
        apply_child_overwrites(network, nodes, ignored_nodes)

        branches = network.branches
        compounds = network.compounds

        mark_slacks_and_prescriptions(network, ignored_nodes)

        inject_vars(
            lambda model, comp, cat: self.inject_gurobi_vars_attr(
                gm, model, prefix=f"{cat}_{comp.id}"
            ),
            nodes,
            branches,
            compounds,
            network,
            ignored_nodes,
        )
        # Flush pending variables so constraint expressions can reference them.
        gm.update()

        if step_state is not None:
            for ext in network.extensions:
                ext.activate_timeseries(network, ignored_nodes, step_state=step_state)
            self.mark_temporal_components(network, ignored_nodes)

        self.init_branches(branches)

        self.process_equations_nodes_childs(
            gm, network, nodes, ignored_nodes, aux_obj_exprs
        )
        self.process_equations_branches(
            gm, network, branches, ignored_nodes, aux_obj_exprs
        )
        self.process_equations_compounds(
            gm, network, compounds, ignored_nodes, aux_obj_exprs
        )

        if optimization_problem is not None:
            self.process_oxf_components(
                gm, network, optimization_problem, user_obj_exprs
            )
        else:
            self.process_internal_oxf_components(gm, network, user_obj_exprs)

        if step_state is not None:
            self.process_inter_step_equations(
                gm,
                network,
                nodes,
                branches,
                compounds,
                ignored_nodes,
                step_state,
                optimization_problem=optimization_problem,
            )
            for ext in network.extensions:
                self._add_equations(
                    gm, ext.inter_step_equations(network, ignored_nodes, step_state)
                )
                self._add_equations(
                    gm,
                    ext.inter_temporal_equations(network, ignored_nodes, step_state),
                )

        for ext in network.extensions:
            self._add_equations(gm, ext.equations(network, ignored_nodes))

        lex_objectives = (
            optimization_problem is not None
            and optimization_problem.lex_objectives
            and len(user_obj_exprs) > 0
            and len(aux_obj_exprs) > 0
        )

        if lex_objectives:
            success, report = self._solve_lexicographic(
                gm, user_obj_exprs, aux_obj_exprs
            )
        else:
            self._set_objective(gm, user_obj_exprs + aux_obj_exprs)
            gm.optimize()
            success, report = self._classify(gm, phase_label="Gurobi solve")

        all_obj_exprs = user_obj_exprs + aux_obj_exprs

        withdraw_vars(
            self.withdraw_gurobi_vars_attr, nodes, branches, compounds, network
        )
        violations = finalize_solution(
            nodes, branches, compounds, network, input_network
        )

        obj_val = self._obj_value(gm, all_obj_exprs) if success else float("nan")

        return SolverResult(
            network,
            network.as_result_dataframe_dict(),
            obj_val,
            success,
            violations,
            infeasibility_report=report if not success else None,
            backend_used=self.backend_name,
            solver_used=self.solver_name,
        )

    def _solve_lexicographic(self, gm, user_obj_exprs, aux_obj_exprs):
        """Lexicographic solve: optimise the user objective first, then the aux
        (formulation-tightening) objective without degrading the user tier.

        Uses Gurobi's native hierarchical multi-objective (``setObjectiveN``
        with priorities) whenever the objectives are at most quadratic - a
        single solve, no explicit cap constraint - hoisting any quadratic term
        onto an auxiliary so the multi-objective stays linear.  A genuinely
        *nonlinear* objective (NLP formulations) cannot be expressed in the
        multi-objective API and routes to the portable two-phase fallback."""
        if not self._is_nonlinear(sum(user_obj_exprs)) and not self._is_nonlinear(
            sum(aux_obj_exprs)
        ):
            return self._solve_lexicographic_native(gm, user_obj_exprs, aux_obj_exprs)
        return self._solve_lexicographic_two_phase(gm, user_obj_exprs, aux_obj_exprs)

    def _solve_lexicographic_native(self, gm, user_obj_exprs, aux_obj_exprs):
        """Native hierarchical objectives: higher priority = optimised first.

        The user tier is held strict (``reltol=0``, ``abstol=_LEX_ABS_TOL``) so
        optimising the aux tier cannot degrade it - a non-zero ``reltol`` would
        *loosen* Gurobi's default (``ObjNRelTol=0``) and let the primary
        objective drift by that fraction.  (Any residual spread between this and
        the two-phase/Pyomo result is the ``MIPGap`` optimality tolerance on the
        large auto-scaled user objective: ~``MIPGap·|user|``, not lex drift.)"""
        GRB = self._GRB
        gm.ModelSense = GRB.MINIMIZE
        gm.setObjectiveN(
            self._linearize_for_multiobj(gm, user_obj_exprs),
            index=0,
            priority=2,
            reltol=0.0,
            abstol=self._LEX_ABS_TOL,
            name="user",
        )
        gm.setObjectiveN(
            self._linearize_for_multiobj(gm, aux_obj_exprs),
            index=1,
            priority=1,
            name="aux",
        )
        # Loosen only the aux pass (see _LEX_AUX_* docs): per-objective env so
        # the user tier keeps the model's global tolerances.
        if self._LEX_AUX_MIPGAP is not None or self._LEX_AUX_TIMELIMIT is not None:
            env_aux = gm.getMultiobjEnv(1)
            if self._LEX_AUX_MIPGAP is not None:
                env_aux.setParam("MIPGap", self._LEX_AUX_MIPGAP)
            if self._LEX_AUX_TIMELIMIT is not None:
                env_aux.setParam("TimeLimit", self._LEX_AUX_TIMELIMIT)
            try:
                gm.optimize()
            finally:
                gm.discardMultiobjEnvs()
        else:
            gm.optimize()
        return self._classify(gm, phase_label="Lexicographic (native priorities)")

    def _solve_lexicographic_two_phase(self, gm, user_obj_exprs, aux_obj_exprs):
        """Portable two-phase lexicographic: minimise ``Sum(user)`` then
        ``Sum(aux)`` under ``Sum(user) <= S* + slack``.  Handles the
        quadratic/nonlinear objectives ``setObjectiveN`` cannot."""
        # Phase 1: user objective only.
        self._set_objective(gm, user_obj_exprs)
        gm.optimize()
        success, report = self._classify(gm, phase_label="Lexicographic phase 1")
        if not success:
            return success, report

        s_star = gm.ObjVal
        slack = self._lex_cap_slack(s_star, self._params.get("MIPGap", 0.0))
        self._add_temp_constr(gm, sum(user_obj_exprs) <= s_star + slack, name="lex_cap")

        # Phase 2: aux objective under the phase-1 cap.
        self._set_objective(gm, aux_obj_exprs)
        gm.optimize()
        return self._classify(gm, phase_label="Lexicographic phase 2")

    # --------- equation-building passes (mirror PyomoSolver) ---------

    def process_internal_oxf_components(self, gm, network, user_obj_exprs):
        for constraint in network.constraints:
            self._add_equation(gm, constraint(network))
        obj = None
        for objective in network.objectives:
            obj = objective(network) if obj is None else (obj + objective(network))
        if obj is not None:
            user_obj_exprs.append(obj)

    def process_oxf_components(
        self,
        gm,
        network,
        optimization_problem: OptimizationProblem,
        user_obj_exprs,
        period_index=None,
    ):
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            self._add_equations(
                gm,
                optimization_problem.constraints.all(
                    network, period_index=period_index
                ),
            )
        obj = None
        for objective in (
            optimization_problem.objectives.all(network, period_index=period_index)
            if optimization_problem.objectives is not None
            else []
        ):
            obj = objective if obj is None else (obj + objective)
        if obj is not None:
            user_obj_exprs.append(obj)

    def process_equations_compounds(
        self, gm, network, compounds, ignored_nodes, aux_obj_exprs
    ):
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue
            for expr in compound.minimize(network, sqrt_impl=self._nlfunc.sqrt):
                aux_obj_exprs.append(expr)
            equations = compound.equations(network)
            if equations is not None:
                equations = filter_bool_eqs(
                    as_iter(equations), context=f"compound_{compound.id}"
                )
                self._process_intermediate_eqs(compound, equations)
                self._add_equations(
                    gm,
                    filter_intermediate_eqs(equations),
                    name_prefix=f"compound_{compound.id}",
                )

    def process_equations_nodes_childs(
        self, gm, network: Network, nodes, ignored_nodes, aux_obj_exprs
    ):
        nf = self._nlfunc
        impls = {
            "sin_impl": nf.sin,
            "cos_impl": nf.cos,
            "abs_impl": self._abs_impl,
            "sqrt_impl": nf.sqrt,
            "log_impl": nf.log10,
            "exp_impl": nf.exp,
        }
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue

            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid

            from_branches = [
                network.branch_by_id(bid).model
                for bid in node.from_branch_ids
                if not ignore_branch(network.branch_by_id(bid), network, ignored_nodes)
            ]
            to_branches = [
                network.branch_by_id(bid).model
                for bid in node.to_branch_ids
                if not ignore_branch(network.branch_by_id(bid), network, ignored_nodes)
            ]
            connected_childs = [
                child.model
                for child in node_childs
                if not ignore_child(child, ignored_nodes)
            ]

            equations = as_iter(
                node.equations(
                    grid, from_branches, to_branches, connected_childs, **impls
                )
            )

            for expr in node.minimize(
                grid, from_branches, to_branches, connected_childs, sqrt_impl=nf.sqrt
            ):
                aux_obj_exprs.append(expr)

            node_eqs = filter_bool_eqs(equations, context=f"node_{node.id}")
            self._process_intermediate_eqs(node.model, node_eqs)
            self._add_equations(
                gm, filter_intermediate_eqs(node_eqs), name_prefix=f"node_{node.id}"
            )

            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                for expr in child.minimize(grid, node, sqrt_impl=nf.sqrt):
                    aux_obj_exprs.append(expr)
                child_eqs = filter_bool_eqs(
                    as_iter(child.equations(grid, node)), context=f"child_{child.id}"
                )
                self._process_intermediate_eqs(child.model, child_eqs)
                self._add_equations(
                    gm,
                    filter_intermediate_eqs(child_eqs),
                    name_prefix=f"child_{child.id}",
                )

    def process_equations_branches(
        self, gm, network, branches, ignored_nodes, aux_obj_exprs
    ):
        nf = self._nlfunc
        pwl_impl = GurobiPWLImpl(self, gm)

        def _unsupported(what, hint):
            def _impl(*args, **kwargs):
                raise NotImplementedError(
                    f"GEKKO {what} has no gurobipy equivalent; {hint}."
                )

            return _impl

        impls = {
            "sin_impl": nf.sin,
            "cos_impl": nf.cos,
            "abs_impl": self._abs_impl,
            "sqrt_impl": nf.sqrt,
            "log_impl": nf.log10,
            "exp_impl": nf.exp,
            "pwl_impl": pwl_impl,
            "if_impl": _unsupported("if2/if3", "use a Piecewise / big-M formulation"),
            "max_impl": _unsupported("max2", "use explicit constraints"),
            "sign_impl": _unsupported("sign2/sign3", "use a binary formulation"),
        }

        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue
            grid = branch.grid

            branch_eqs = as_iter(
                branch.equations(
                    grid,
                    network.node_by_id(branch.from_node_id).model,
                    network.node_by_id(branch.to_node_id).model,
                    simulation=self._simulation,
                    **impls,
                )
            )

            for expr in branch.minimize(
                grid,
                network.node_by_id(branch.from_node_id).model,
                network.node_by_id(branch.to_node_id).model,
                sqrt_impl=nf.sqrt,
            ):
                aux_obj_exprs.append(expr)

            branch_eqs = filter_bool_eqs(branch_eqs, context=f"branch_{branch.id}")
            self._process_intermediate_eqs(branch.model, branch_eqs)
            self._add_equations(
                gm,
                filter_intermediate_eqs(branch_eqs),
                name_prefix=f"branch_{branch.id}",
            )

    def _abs_impl(self, e):
        """Smooth |e| as sqrt(e^2) for gurobipy expressions; builtin abs for
        plain numbers.  (Unused by the bundled formulations, but kept so a
        custom model passing ``abs_impl`` doesn't crash.)"""
        if isinstance(e, (int, float)):
            return abs(e)
        return self._nlfunc.sqrt(self._nlfunc.square(e))


# Temporal-coupling hooks a model / formulation / extension may expose for the
# sequential timeseries (the persistent driver wires these against a parameter
# step-state so the previous-step value enters as a per-step parameter).
_TEMPORAL_METHODS = ("inter_temporal_equations", "inter_step_equations")


class _ParamStepState(InterStepState):
    """A :class:`~monee.solver.core.InterStepState` whose :meth:`get` returns a
    *fixed* Gurobi variable (``lb == ub``, i.e. a parameter) instead of the
    previous step's float.

    The inter-step equations (e.g. ``e_mwh == prev_e + dt*p``) then reference
    that parameter, so the model is built **once** and the carried state is
    updated per step by re-bounding the parameter - the sequential analogue of
    multi-period's live-variable coupling (``PeriodState``).
    """

    def __init__(self, gm, initial_vals, dt_h: float = 1.0):
        self._gm = gm
        self._initial = initial_vals  # {(component_id, attr): value}
        self.dt_h = dt_h
        self.params: dict = {}  # {(component_id, attr): gurobi Var}

    def get(self, component_id, attr: str, step: int = -1):
        key = (component_id, attr)
        if key not in self.params:
            init = float(self._initial.get(key, 0.0))
            self.params[key] = self._gm.addVar(
                lb=init, ub=init, name=f"prev__{component_id}__{attr}"
            )
        return self.params[key]


class GurobipyTimeseries:
    """Build-once / re-bound-per-step Gurobi timeseries driver (MILP/MIQCQP/
    MISOCP/smooth-NLP).

    Unlike :func:`monee.run_timeseries`, which reconstructs the whole model on
    every step, this driver assembles the gurobipy model **once** and, across the
    timeseries, only mutates the time-varying inputs in place
    (``var.LB``/``var.UB``) and re-:meth:`~gurobipy.Model.optimize`. Two levers:

    1. **Model reuse** - every time-varying attribute (from a
       :class:`~monee.simulation.timeseries.TimeseriesData`) is turned into a
       fixed decision variable (``lb == ub == value``) so it can be re-bounded
       per step without touching the constraint structure.
    2. **Carried state as a parameter** - inter-step coupling
       (``state[t] == prev_state + dt*flow[t]``) makes ``prev_state`` a fixed
       Gurobi variable too, fed from the prior step's solved value. This handles
       storage SoC, the :class:`LumpedThermalCapacitance` extension's thermal
       inertia, linepack, ... exactly as the rebuild loop does, but without the
       rebuild.

    The integer solution is additionally carried forward as a MIP start
    (``Var.Start``) so Gurobi reuses the previous step's discrete decisions - the
    big lever for the temporally-correlated binaries in storage / DHS models.

    This reuses :class:`GurobipySolver`'s build passes verbatim; only the
    persistent-model bookkeeping is added.

    Args:
        carry_mip_start: seed each step's integer variables with the previous
            step's solution (``Var.Start``). Turn off to isolate the model-reuse
            win from the warm-start win.
        params: extra Gurobi parameters merged over
            :data:`DEFAULT_GUROBI_PARAMS` (same semantics as
            :class:`GurobipySolver`).
    """

    def __init__(  # NOSONAR
        self,
        input_network,
        timeseries_data,
        optimization_problem=None,
        formulation=None,
        simulation=False,
        steps=None,
        carry_mip_start=True,
        params: dict | None = None,
    ):
        self._td = timeseries_data
        self.carry_mip_start = carry_mip_start
        gp, GRB, nlfunc = _require_gurobipy()
        self._gp, self._GRB = gp, GRB

        gs = GurobipySolver(params=params)
        gs._gp, gs._GRB, gs._nlfunc = gp, GRB, nlfunc
        gs._simulation = simulation
        self._gs = gs

        gm = gp.Model("monee_ts")
        gm.setParam("OutputFlag", 0)
        for key, val in gs._params.items():
            gm.setParam(key, val)
        self._gm = gm

        network, ignored, _islanding = prepare_solve_network(
            input_network,
            optimization_problem=optimization_problem,
            formulation=formulation,
            simulation=simulation,
        )

        nodes = network.nodes
        apply_child_overwrites(network, nodes, ignored)
        branches, compounds = network.branches, network.compounds
        mark_slacks_and_prescriptions(network, ignored)

        # Capture initial values of all Var attributes (before they become
        # Gurobi vars) so carried-state parameters can seed step 0.
        self._initial_vals: dict = {}
        for node in nodes:
            if ignore_node(node, network, ignored):
                continue
            self._cap_initials(node.id, node.model)
            for child in network.childs_by_ids(node.child_ids):
                if not ignore_child(child, ignored):
                    self._cap_initials(child.id, child.model)
        for branch in branches:
            if not ignore_branch(branch, network, ignored):
                self._cap_initials(branch.id, branch.model)

        # Turn every time-varying attribute into a *fixed* decision variable
        # (lb == ub == value) so it injects as a Gurobi Var we can re-bound per
        # step without rebuilding any constraint.
        self._param_targets: list = []  # (model, attr, series)
        self._declare_param_targets(network, timeseries_data)

        inject_vars(
            lambda model, comp, cat: gs.inject_gurobi_vars_attr(
                gm, model, prefix=f"{cat}_{comp.id}"
            ),
            nodes,
            branches,
            compounds,
            network,
            ignored,
        )
        gm.update()

        # Temporal setup must precede equation building: extensions like LTC use
        # activate_timeseries() to mark components (e.g. _ltc_active) so the
        # canonical balance is dropped in favour of the inter-temporal equation.
        # Pass step_state=None so only the marking runs (no float warm-start
        # assignment, which would clash with our parameter variables).
        self._carried: list = []  # (prev_param_gvar, current_state_gvar)
        has_temporal = any(
            hasattr(m, meth)
            for m in self._iter_models(network, nodes, branches, compounds, ignored)
            for meth in _TEMPORAL_METHODS
        ) or any(
            hasattr(ext, meth)
            for ext in network.extensions
            for meth in (*_TEMPORAL_METHODS, "activate_timeseries")
        )
        self._param_state = None
        if has_temporal:
            gs.mark_temporal_components(network, ignored)
            for ext in network.extensions:
                if hasattr(ext, "activate_timeseries"):
                    ext.activate_timeseries(network, ignored, step_state=None)

        aux, user = [], []
        gs.init_branches(branches)
        gs.process_equations_nodes_childs(gm, network, nodes, ignored, aux)
        gs.process_equations_branches(gm, network, branches, ignored, aux)
        gs.process_equations_compounds(gm, network, compounds, ignored, aux)
        if optimization_problem is not None:
            gs.process_oxf_components(gm, network, optimization_problem, user)
        else:
            gs.process_internal_oxf_components(gm, network, user)

        # --- inter-step (carried-state) coupling ---
        # Wire the inter-step equations against a parameter step-state so the
        # previous-step state (storage SoC, LTC junction temperature, ...)
        # enters as a per-step parameter rather than a baked-in constant.
        if has_temporal:
            self._param_state = _ParamStepState(gm, self._initial_vals)
            # model-level coupling (e.g. storage)
            gs.process_inter_step_equations(
                gm,
                network,
                nodes,
                branches,
                compounds,
                ignored,
                self._param_state,
                optimization_problem=optimization_problem,
            )
            # extension-level coupling (e.g. LTC thermal inertia, linepack)
            for ext in network.extensions:
                for meth in _TEMPORAL_METHODS:
                    if hasattr(ext, meth):
                        gs._add_equations(
                            gm,
                            getattr(ext, meth)(network, ignored, self._param_state),
                        )

        for ext in network.extensions:
            gs._add_equations(gm, ext.equations(network, ignored))
        gs._set_objective(gm, user + aux)
        gm.update()
        self._obj_exprs = user + aux

        # Link each carried-state parameter to the live variable that produces
        # its next value (e.g. prev_e parameter <- this step's e_mwh variable).
        if self._param_state is not None:
            for (cid, attr), pvar in self._param_state.params.items():
                model = self._model_by_id(network, cid, attr)
                if model is not None:
                    self._carried.append((pvar, model.__dict__[attr]))

        self._network, self._nodes = network, nodes
        self._branches, self._compounds, self._ignored = branches, compounds, ignored

        # Registry of every solved Gurobi object on the models (vars + passive
        # intermediate expressions), so per-step extraction reads the live model
        # rather than the (mutated) model attributes.
        self._reg: list = []  # (model, key, gobj, kind, is_int)
        self._int_vars: list = []  # gurobi Var objects that are integer/binary
        self._params: list = []  # (gurobi Var, series)
        ptargets = {(id(m), a): s for (m, a, s) in self._param_targets}
        for model in self._active_models():
            for key, val in model.__dict__.items():
                if isinstance(val, gp.Var):
                    is_int = val.VType in (GRB.INTEGER, GRB.BINARY)
                    self._reg.append((model, key, val, "var", is_int))
                    if is_int:
                        self._int_vars.append(val)
                    s = ptargets.get((id(model), key))
                    if s is not None:
                        self._params.append((val, s))
                elif isinstance(val, (gp.LinExpr, gp.QuadExpr, gp.NLExpr)):
                    self._reg.append((model, key, val, "expr", False))

        self.n_vars = gm.NumVars
        self.n_int = gm.NumIntVars + gm.NumBinVars
        self.n_constr = gm.NumConstrs
        length = timeseries_data.length
        length = length() if callable(length) else length
        self.steps = steps if steps is not None else length
        # Per-step / cumulative state.
        self._prev_int = None
        self._last_objective = None
        self.objectives: list = []
        self.last_solve_total_s = None
        self.last_status_ok = None
        self.last_step_success = None

    # ------------------------------------------------------------------ #
    def _cap_initials(self, comp_id, model):
        for key, val in model.__dict__.items():
            if isinstance(val, (Var, Const, Intermediate)):
                self._initial_vals[(comp_id, key)] = val.value

    @staticmethod
    def _iter_models(network, nodes, branches, compounds, ignored):
        for branch in branches:
            if not ignore_branch(branch, network, ignored):
                yield branch.model
        for node in nodes:
            if ignore_node(node, network, ignored):
                continue
            yield node.model
            for child in network.childs_by_ids(node.child_ids):
                if not ignore_child(child, ignored):
                    yield child.model
        for compound in compounds:
            if not ignore_compound(compound, ignored):
                yield compound.model

    def _model_by_id(self, network, cid, attr):
        """Find the component (child / node / branch) with id *cid* whose *attr*
        is a live Gurobi Var - disambiguates node vs child id collisions."""
        for accessor in ("child_by_id", "node_by_id", "branch_by_id"):
            try:
                comp = getattr(network, accessor)(cid)
            except Exception:
                comp = None
            if comp is not None and isinstance(
                comp.model.__dict__.get(attr), self._gp.Var
            ):
                return comp.model
        return None

    # ------------------------------------------------------------------ #
    def _declare_param_targets(self, network, td):
        def add(model, attr):
            cur = getattr(model, attr)
            cur = (
                cur.value if isinstance(cur, (Var, Const, Intermediate)) else float(cur)
            )
            setattr(model, attr, Var(value=float(cur), min=float(cur), max=float(cur)))

        for cid, attrs in td._child_id_to_series.items():
            model = network.child_by_id(cid).model
            for attr, series in attrs.items():
                add(model, attr)
                self._param_targets.append((model, attr, series))
        for nid, attrs in td._node_id_to_series.items():
            model = network.node_by_id(nid).model
            for attr, series in attrs.items():
                add(model, attr)
                self._param_targets.append((model, attr, series))
        for bid, attrs in td._branch_id_to_series.items():
            model = network.branch_by_id(bid).model
            for attr, series in attrs.items():
                add(model, attr)
                self._param_targets.append((model, attr, series))

    def _active_models(self):
        net, ign = self._network, self._ignored
        for branch in self._branches:
            if not ignore_branch(branch, net, ign):
                yield branch.model
        for node in self._nodes:
            if ignore_node(node, net, ign):
                continue
            yield node.model
            for child in net.childs_by_ids(node.child_ids):
                if not ignore_child(child, ign):
                    yield child.model
        for compound in self._compounds:
            if not ignore_compound(compound, ign):
                yield compound.model

    # ------------------------------------------------------------------ #
    def _solve_step(self, t: int) -> bool:  # NOSONAR
        """Re-bound the time-varying inputs (and carried state) for step *t*,
        re-solve the persistent model (warm-started), and scatter the solution
        into the network models. Returns whether the solve succeeded."""
        gs, gm = self._gs, self._gm
        for gvar, series in self._params:
            v = float(series[t])
            gvar.LB = v
            gvar.UB = v
        # carry state forward: set prev-state parameters from the last step's
        # solved values (step 0 keeps the captured initial state).
        if t > 0:
            for pvar, cur in self._carried:
                val = gs._var_value(cur)
                pvar.LB = val
                pvar.UB = val
        if self.carry_mip_start and self._prev_int is not None:
            for gvar, val in zip(self._int_vars, self._prev_int):
                gvar.Start = val

        gm.optimize()
        ok = gm.Status in (self._GRB.OPTIMAL, self._GRB.SUBOPTIMAL) or gm.SolCount > 0
        self._last_objective = (
            self._gs._obj_value(gm, self._obj_exprs) if ok else float("nan")
        )

        # extract solution for results (reads live Gurobi objects, then
        # overwrites the model attrs with plain monee values)
        for model, key, gobj, kind, is_int in self._reg:
            if kind == "var":
                v = gs._var_value(gobj)
                model.__dict__[key] = Var(
                    value=int(round(v)) if is_int else v,
                    min=None if gobj.LB <= -self._GRB.INFINITY else gobj.LB,
                    max=None if gobj.UB >= self._GRB.INFINITY else gobj.UB,
                    integer=is_int,
                )
            else:
                model.__dict__[key] = Intermediate(value=gs._expr_value(gobj))
        apply_post_process_all(
            self._nodes, self._branches, self._compounds, self._network
        )
        if ok:
            self._prev_int = [gs._var_value(v) for v in self._int_vars]
        return ok

    def step_result(self, t: int) -> SolverResult:
        """Solve step *t* and return a :class:`SolverResult` for the network at
        that step (used by the :func:`monee.run_timeseries` reuse fast path)."""
        ok = self._solve_step(t)
        self.last_step_success = ok
        violations = compute_bound_violations(
            self._nodes, self._branches, self._compounds, self._network
        )
        return SolverResult(
            self._network,
            self._network.as_result_dataframe_dict(),
            self._last_objective,
            ok,
            violations,
            mode_used="optimization",
            backend_used="gurobipy",
            solver_used="gurobi",
        )

    def run(self):
        """Solve every timestep, reusing the persistent model. Returns a list of
        per-step ``{type_name: DataFrame}`` dicts (like
        :attr:`SolverResult.dataframes`)."""
        results, solve_total, ok_all = [], 0.0, True
        self.objectives = []
        for t in range(self.steps):
            t0 = time.perf_counter()
            ok = self._solve_step(t)
            solve_total += time.perf_counter() - t0
            ok_all = ok_all and ok
            self.objectives.append(self._last_objective)
            results.append(self._network.as_result_dataframe_dict())
        self.last_solve_total_s = solve_total
        self.last_status_ok = ok_all
        return results
