import logging
import math
import os
import shutil
import tempfile
import types

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition

from monee.model import (
    Const,
    GenericModel,
    Intermediate,
    IntermediateEq,
    Network,
    Var,
)
from monee.model.extension.islanding.core import NetworkIslandingConfig
from monee.problem.core import OptimizationProblem
from monee.simulation.step_state import StepState
from monee.solver.infeasibility import diagnose_infeasibility

from .core import (
    SolverInterface,
    SolverResult,
    apply_post_process_all,
    as_iter,
    compute_bound_violations,
    filter_intermediate_eqs,
    find_ignored_nodes,
    ignore_branch,
    ignore_child,
    ignore_compound,
    ignore_node,
    inject_vars,
    mark_he_flow_prescription,
    mark_heat_balance_slacks,
    mark_ignored_components,
    persist_solution,
    pin_floating_hydraulic_gauges,
    withdraw_vars,
)

_log = logging.getLogger(__name__)

DEFAULT_SOLVER_OPTIONS = {}

# Gurobi defaults tuned for McCormick-DHS + MISOCP electrical:
#   ScaleFlag=2 - aggressive geometric scaling for the SOC/bilinear blocks;
#   MIPFocus=2  - close the gap (the LP root bound is already at the optimum);
#   MIPGap=1e-3 - ~kW precision at MW scale (default 1e-4 wastes B&B).
# NumericFocus stays at 0: was net-negative once Reynolds/pressure_pa scaling fixed.
PER_SOLVER_OPTIONS = {
    "gurobi": {
        "ScaleFlag": 2,
        "MIPFocus": 2,
        "MIPGap": 1e-3,
        "TimeLimit": 300,
    },
    # Dual presolve reductions make SCIP spuriously prove infeasibility on the
    # ill-conditioned MISOCP load-shedding models (verified: identical model
    # optimal with these off and with Gurobi, 'infeasible' with them on).
    "scip": {
        "misc/allowstrongdualreds": False,
        "misc/allowweakdualreds": False,
        "limits/time": 300,
    },
}


def _classic_scip_available() -> bool:
    """True iff the standalone ``scip``/``scipampl`` executable is on PATH.

    Checked via ``shutil.which`` rather than ``SolverFactory("scip").available()``
    so we don't trigger Pyomo's noisy "Could not locate the 'scip' executable"
    warning just to decide which backend to use.
    """
    return bool(shutil.which("scip") or shutil.which("scipampl"))


def _pyscipopt_available() -> bool:
    """True iff the ``pyscipopt`` Python bindings (PyPI ``pyscipopt``) import."""
    try:
        import pyscipopt  # noqa: F401
    except ImportError:
        return False
    return True


class PyscipoptSolver:
    """Drive SCIP through the ``pyscipopt`` bindings when the classic ``scip``
    executable is absent..
    """

    def __init__(self):
        self.options: dict = {}

    def warm_start_capable(self) -> bool:
        return False

    def available(self, exception_flag: bool = False) -> bool:
        return _pyscipopt_available()

    def solve(self, model, tee: bool = False, **kwargs):
        from pyscipopt import Model as ScipModel

        tmpdir = tempfile.mkdtemp(prefix="monee_scip_")
        nl_path = os.path.join(tmpdir, "model.nl")
        try:
            model.write(
                nl_path,
                format="nl",
                io_options={"symbolic_solver_labels": True},
            )

            sm = ScipModel()
            if not tee:
                sm.hideOutput()
            sm.readProblem(nl_path)
            for key, value in self.options.items():
                try:
                    sm.setParam(key, value)
                except Exception:
                    _log.debug("pyscipopt: ignoring unknown SCIP option %r", key)

            sm.optimize()

            status = sm.getStatus()
            has_solution = sm.getNSols() > 0
            if has_solution:
                # Strip whitespace from SCIP variable names: on Windows the
                # pyomo-written .col label file has CRLF line endings, so
                # SCIP's NL reader keeps a trailing '\r' in every name.
                scip_vals = {var.name.strip(): sm.getVal(var) for var in sm.getVars()}
                matched = 0
                for var in model.component_data_objects(pyo.Var, active=True):
                    val = scip_vals.get(var.name)
                    if val is not None:
                        var.set_value(val, skip_validation=True)
                        matched += 1
                if matched == 0 and scip_vals:
                    _log.warning(
                        "pyscipopt bridge: no SCIP variable names matched the "
                        "pyomo model; solution write-back failed and the "
                        "reported values are the variable initializations."
                    )

            return self._build_result(status, has_solution)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    @staticmethod
    def _build_result(status: str, has_solution: bool):
        """Map a SCIP status string onto the ``(status, termination_condition)``
        pair that :func:`_classify_solve_result` inspects."""
        if status == "optimal":
            solver_status, tc = SolverStatus.ok, TerminationCondition.optimal
        elif status == "infeasible":
            solver_status, tc = SolverStatus.warning, TerminationCondition.infeasible
        elif status == "unbounded":
            solver_status, tc = SolverStatus.warning, TerminationCondition.unbounded
        elif has_solution:
            # A limit was hit (time/node/gap) but SCIP has a feasible incumbent;
            # _classify_solve_result treats this as a usable witness solution.
            solver_status, tc = SolverStatus.aborted, TerminationCondition.maxTimeLimit
        else:
            # Limit hit with no incumbent -> report as infeasible so the caller
            # surfaces a diagnostic rather than reading unset Vars as a solution.
            solver_status, tc = SolverStatus.warning, TerminationCondition.infeasible
        return types.SimpleNamespace(
            solver=types.SimpleNamespace(status=solver_status, termination_condition=tc)
        )


class PyomoPWLImpl:
    def __init__(self, pm: pyo.ConcreteModel, pw_repn: str = "SOS2"):
        self.pm = pm
        self.pw_repn = pw_repn
        # counter to ensure unique component names
        if not hasattr(pm, "_pwl_counter"):
            pm._pwl_counter = 0

    def piecewise_eq(self, y, x, xs, ys, name: str | None = None):
        pm = self.pm
        pm._pwl_counter += 1
        k = pm._pwl_counter
        if name is None:
            name = f"pwl_{k}"

        xs = list(xs)
        ys = list(ys)

        pw = pyo.Piecewise(
            y,
            x,
            pw_pts=xs,
            f_rule=ys,  # list form: y[i] at xs[i]
            pw_constr_type="EQ",
            pw_repn=self.pw_repn,  # "SOS2" (tight), or "BIGM"
            warn_domain_coverage=False,
        )
        setattr(pm, name, pw)


def _classify_solve_result(result, pm, solver_name: str, *, phase_label: str):
    """Map a Pyomo solve outcome to ``(success, report)``.

    OK → silent; infeasible → error + MIS report; other non-OK (limits with a
    feasible incumbent) → warning, treated as success.
    """
    status = result.solver.status
    tc = result.solver.termination_condition
    if status == SolverStatus.ok:
        return True, None
    if tc == TerminationCondition.infeasible:
        report = diagnose_infeasibility(
            pm, solver_name=solver_name, compute_mis_flag=True, tol=0.001
        )
        _log.error(
            "%s infeasible (status=%s, termination=%s).  Diagnostic report:\n%s",
            phase_label,
            status,
            tc,
            report.summary(max_items=50),
        )
        return False, report
    # Only limit-type terminations come with a feasible incumbent worth using.
    # Anything else (infeasibleOrUnbounded, unbounded, error, solverFailure...)
    # has no witness: declaring success there means downstream code reads the
    # variable *initializations* as a solution.
    witness_terminations = {
        TerminationCondition.maxTimeLimit,
        TerminationCondition.maxIterations,
        TerminationCondition.maxEvaluations,
        TerminationCondition.feasible,
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,
        TerminationCondition.globallyOptimal,
    }
    if tc in witness_terminations:
        _log.warning(
            "%s returned non-ok status (status=%s, termination=%s); "
            "using witness solution.",
            phase_label,
            status,
            tc,
        )
        return True, None
    _log.error(
        "%s failed without a usable solution (status=%s, termination=%s).",
        phase_label,
        status,
        tc,
    )
    return False, None


class PyomoSolver(SolverInterface):
    """Pyomo-backed solver. ``solver_name`` is overridable per :meth:`solve`."""

    def __init__(self, solver_name: str = "scip"):
        self._solver_name = solver_name
        # Per-solve simulation flag (set at the top of solve()); read by the
        # equation-building passes to drop operational flow limits.
        self._simulation: bool = False

    @staticmethod
    def inject_pyomo_vars_attr(
        pm: pyo.ConcreteModel, target: GenericModel, prefix: str
    ):
        """Replace Var/Const with Pyomo Var / numeric. Clamps stale ``initialize``
        into ``[min, max]`` (suppresses W1002 when bounds tightened since last solve)."""
        for key, value in target.__dict__.items():
            if isinstance(value, Var):
                init = value.value
                # NaN survives into Gurobi's warmstart file and crashes the solve.
                if init is not None and isinstance(init, float) and math.isnan(init):
                    init = None
                if init is not None:
                    if value.min is not None and init < value.min:
                        init = value.min
                    if value.max is not None and init > value.max:
                        init = value.max
                v = pyo.Var(
                    domain=pyo.Integers if value.integer else pyo.Reals,
                    bounds=(value.min, value.max),
                    initialize=init,
                )
                setattr(pm, PyomoSolver._sanitize_name(f"{prefix}__{key}"), v)
                setattr(target, key, v)
            elif type(value) is Const:
                setattr(target, key, float(value.value))

    # Snap-to-bound tolerance; sub-epsilon noise re-injected as ``initialize``
    # otherwise triggers W1002.
    _BOUND_SNAP_TOL = 1e-9

    @staticmethod
    def withdraw_pyomo_vars_attr(target: GenericModel):
        """Pyomo Var → :class:`Var`. Restores ``integer``, snaps bound-noise to
        bounds, replaces NaN/None with 0 so the next solve's warmstart survives."""
        for key, value in target.__dict__.items():
            if isinstance(value, pyo.Var):
                lb, ub = value.bounds if value.bounds is not None else (None, None)
                is_integer = value.is_integer()
                val = pyo.value(value, exception=False)
                if val is None or (isinstance(val, float) and math.isnan(val)):
                    val = 0
                if is_integer:
                    val = int(round(val))
                else:
                    tol = PyomoSolver._BOUND_SNAP_TOL
                    if lb is not None and lb - tol <= val < lb:
                        val = lb
                    if ub is not None and ub < val <= ub + tol:
                        val = ub
                setattr(target, key, Var(value=val, min=lb, max=ub, integer=is_integer))
            elif isinstance(value, pyo.Expression):
                expr_val = pyo.value(value, exception=False)
                if expr_val is None or (
                    isinstance(expr_val, float) and math.isnan(expr_val)
                ):
                    expr_val = 0
                setattr(target, key, Intermediate(value=expr_val))

    # --------- Constraint helpers ---------

    @staticmethod
    def _add_equation(pm, expr, name=None):
        # Trivial bool equations are not valid Pyomo Constraint expressions:
        # True = tautology (no-op); False = structural infeasibility from an
        # over-deactivated node/branch.  Skip both so the load-shedding
        # objective can drive the solution instead of crashing construction.
        if isinstance(expr, bool):
            return
        if name is not None:
            setattr(pm, name, pyo.Constraint(expr=expr))
        else:
            pm.cons.add(expr)

    @staticmethod
    def _sanitize_name(name):
        """Make a string safe for use as a Pyomo component name."""
        return (
            name.replace("(", "").replace(")", "").replace(", ", "_").replace(" ", "_")
        )

    def _add_equations(self, pm, exprs, name_prefix=None):
        for i, e in enumerate(exprs):
            if name_prefix is not None:
                name = self._sanitize_name(f"{name_prefix}_eq_{i}")
            else:
                name = None
            self._add_equation(pm, e, name=name)

    @staticmethod
    def _process_intermediate_eqs(pm, model_obj, equations):
        """
        GEKKO Intermediate: two cases in your original code:
        - If attr is not Intermediate: enforce equality constraint
        - If attr is Intermediate: create an Intermediate and assign it

        In Pyomo, use Expression for intermediates.
        """
        for intermediate_eq in [eq for eq in equations if type(eq) is IntermediateEq]:
            attr_val = getattr(model_obj, intermediate_eq.attr)
            eq = (
                intermediate_eq.eq()
                if isinstance(intermediate_eq.eq, types.FunctionType)
                else intermediate_eq.eq
            )

            if type(attr_val) is Intermediate:
                e = pyo.Expression(expr=eq)
                name = f"expr__{id(model_obj)}__{intermediate_eq.attr}"
                setattr(pm, name, e)
                setattr(model_obj, intermediate_eq.attr, e)

    @staticmethod
    def _make_solver(solver_name: str):
        """Resolve ``solver_name`` to a solver object.

        For ``scip``, transparently fall back to the :class:`PyscipoptSolver`
        bridge when the standalone ``scip`` executable is missing but the
        ``pyscipopt`` bindings are installed. Any other case defers to
        :func:`pyo.SolverFactory` (which raises the usual informative error at
        solve time if the solver is genuinely unavailable)."""
        if (
            solver_name == "scip"
            and not _classic_scip_available()
            and _pyscipopt_available()
        ):
            _log.info(
                "Standalone 'scip' executable not found; solving via pyscipopt "
                "(SCIP Python bindings) over an AMPL .nl round-trip."
            )
            return PyscipoptSolver()
        return pyo.SolverFactory(solver_name)

    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        exclude_unconnected_nodes: bool = False,
        solver_name: str | None = None,
        debug=False,
        step_state: StepState = None,
        simulation: bool = False,
    ):
        self._simulation = simulation
        if solver_name is None:
            solver_name = self._solver_name
        pm = pyo.ConcreteModel()
        pm.cons = pyo.ConstraintList()
        # Two buckets for lex mode (single sum in legacy mode).
        pm.user_obj_exprs = []  # user objectives
        pm.aux_obj_exprs = []  # formulation minimize() tightening terms

        network = input_network.copy()

        for ext in network.extensions:
            ext.prepare(network)

        # Re-declare each formulation's vars in simulation mode so the model is
        # squared (phantom DOFs pinned to Const, |m| warm-started, vm_pu_squared
        # demoted to a PostProcess). Pyomo has no IMODE=1; a square system simply
        # solves as the objective-free feasibility problem, yielding the same
        # unique steady state GEKKO gets via IMODE=1 - kept consistent across
        # backends so pyomo+ipopt is an equivalent NLP path. Runs on the copy
        # only, and BEFORE pin_floating_hydraulic_gauges / mark_heat_balance_slacks
        # (which set pressure Consts a re-declare would overwrite). No-op for
        # formulations that don't square.
        if simulation:
            for component in network.all_components():
                if component.formulation is not None:
                    component.formulation.ensure_var(component.model, simulation=True, grid=component.grid)

        islanding_config = next(
            (e for e in network.extensions if isinstance(e, NetworkIslandingConfig)),
            None,
        )

        # ignored_nodes computed before _apply so controllable filters checking
        # component.ignored exclude disconnected components.
        ignored_nodes = set()
        if optimization_problem is None or exclude_unconnected_nodes:
            ignored_nodes = find_ignored_nodes(network, islanding_config)
            if ignored_nodes:
                mark_ignored_components(network, ignored_nodes)

        if optimization_problem is not None:
            optimization_problem._apply(network)

        nodes = network.nodes
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            for child in network.childs_by_ids(node.child_ids):
                if child.active:
                    child.model.overwrite(node.model, node.grid)

        branches = network.branches
        compounds = network.compounds

        # Pin the pressure gauge of any hydraulic island without a grid-forming
        # source - removes a rank-deficient free DOF.
        pin_floating_hydraulic_gauges(network, ignored_nodes)

        # Drop each heat island's dependent nodal balance at its grid-forming
        # node (heat slack) - result-preserving for the exact balance. Under
        # mccormick the Junction balance is already trivial and the relaxed
        # eq. 9a is kept, so this is a no-op there.
        mark_heat_balance_slacks(network, ignored_nodes)

        # Decide per compound-internal SubHE whether the design flow prescribes
        # the through-flow (supply/return islands with free-mass-flow slacks)
        # or yields to a network-determined flow (e.g. a fixed sink downstream).
        mark_he_flow_prescription(network, ignored_nodes)

        inject_vars(
            lambda model, comp, cat: PyomoSolver.inject_pyomo_vars_attr(
                pm, model, prefix=f"{cat}_{comp.id}"
            ),
            nodes,
            branches,
            compounds,
            network,
            ignored_nodes,
        )

        if step_state is not None:
            for ext in network.extensions:
                ext.activate_timeseries(network, ignored_nodes, step_state=step_state)
            self.mark_temporal_components(network, ignored_nodes)

        self.init_branches(branches)

        self.process_equations_nodes_childs(pm, network, nodes, ignored_nodes)
        self.process_equations_branches(pm, network, branches, ignored_nodes)
        self.process_equations_compounds(pm, network, compounds, ignored_nodes)

        if optimization_problem is not None:
            self.process_oxf_components(pm, network, optimization_problem)
        else:
            self.process_internal_oxf_components(pm, network)

        if step_state is not None:
            self.process_inter_step_equations(
                pm,
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
                    pm, ext.inter_step_equations(network, ignored_nodes, step_state)
                )
                self._add_equations(
                    pm, ext.inter_temporal_equations(network, ignored_nodes, step_state)
                )

        for ext in network.extensions:
            self._add_equations(pm, ext.equations(network, ignored_nodes))

        lex_objectives = (
            optimization_problem is not None
            and optimization_problem.lex_objectives
            and len(pm.user_obj_exprs) > 0
            and len(pm.aux_obj_exprs) > 0
        )

        solver = self._make_solver(solver_name)
        for k, v in DEFAULT_SOLVER_OPTIONS.items():
            solver.options[k] = v
        for k, v in PER_SOLVER_OPTIONS.get(solver_name, {}).items():
            solver.options[k] = v

        solve_kwargs = {"tee": debug}
        if getattr(solver, "warm_start_capable", lambda: False)():
            solve_kwargs["warmstart"] = True

        if lex_objectives:
            result, success, report = self._solve_lexicographic(
                pm, solver, solver_name, solve_kwargs
            )
        else:
            pm.obj = pyo.Objective(
                expr=sum(pm.user_obj_exprs) + sum(pm.aux_obj_exprs),
                sense=pyo.minimize,
            )
            result = solver.solve(pm, **solve_kwargs)
            success, report = _classify_solve_result(
                result,
                pm,
                solver_name,
                phase_label="Pyomo solve",
            )

        withdraw_vars(
            PyomoSolver.withdraw_pyomo_vars_attr, nodes, branches, compounds, network
        )
        apply_post_process_all(nodes, branches, compounds, network)
        persist_solution(network, input_network)
        violations = compute_bound_violations(nodes, branches, compounds, network)

        # NaN fallback: pyo.value raises on unset Vars (e.g. McCormick aux on a
        # freshly-activated branch); SolverResult.success is the real outcome.
        obj_val = pyo.value(pm.obj, exception=False)
        if obj_val is None:
            obj_val = float("nan")

        solver_result = SolverResult(
            network,
            network.as_result_dataframe_dict(),
            obj_val,
            success,
            violations,
        )
        if not success:
            solver_result.infeasibility_report = report
        return solver_result

    # Slack added on top of solver MIPGap so the phase-2 cap never excludes
    # a phase-1 incumbent the solver accepted within tolerance.
    _LEX_REL_TOL = 1e-6
    _LEX_ABS_TOL = 1e-9

    @classmethod
    def _lex_cap_slack(cls, s_star: float, solver_options: dict) -> float:
        rel_gap = float(solver_options.get("MIPGap", 0.0) or 0.0)
        rel = max(rel_gap, cls._LEX_REL_TOL)
        return cls._LEX_ABS_TOL + rel * max(1.0, abs(s_star))

    def _solve_lexicographic(self, pm, solver, solver_name, solve_kwargs):
        """Two-phase lex: minimise ``Σ user`` then ``Σ aux`` under
        ``Σ user ≤ S* + slack``. Combined ``pm.obj`` attached (deactivated)
        for backwards-compatible ``pyo.value(pm.obj)``."""
        pm.obj_user = pyo.Objective(expr=sum(pm.user_obj_exprs), sense=pyo.minimize)
        pm.obj_aux = pyo.Objective(expr=sum(pm.aux_obj_exprs), sense=pyo.minimize)
        pm.obj_aux.deactivate()

        # Phase 1: user objective only.
        result = solver.solve(pm, **solve_kwargs)
        success, report = _classify_solve_result(
            result,
            pm,
            solver_name,
            phase_label="Lexicographic phase 1",
        )
        if not success:
            pm.obj = pyo.Objective(
                expr=sum(pm.user_obj_exprs) + sum(pm.aux_obj_exprs),
                sense=pyo.minimize,
            )
            pm.obj.deactivate()
            return result, success, report

        s_star = pyo.value(pm.obj_user, exception=False)
        if s_star is None:
            s_star = 0.0

        slack = self._lex_cap_slack(s_star, solver.options)
        pm.lex_cap = pyo.Constraint(expr=sum(pm.user_obj_exprs) <= s_star + slack)
        pm.obj_user.deactivate()
        pm.obj_aux.activate()

        result = solver.solve(pm, **solve_kwargs)
        success, report = _classify_solve_result(
            result,
            pm,
            solver_name,
            phase_label="Lexicographic phase 2",
        )

        pm.obj = pyo.Objective(
            expr=sum(pm.user_obj_exprs) + sum(pm.aux_obj_exprs),
            sense=pyo.minimize,
        )
        pm.obj.deactivate()
        return result, success, report

    def process_internal_oxf_components(self, pm, network):
        for constraint in network.constraints:
            self._add_equation(pm, constraint(network))

        obj = None
        for objective in network.objectives:
            obj = objective(network) if obj is None else (obj + objective(network))
        if obj is not None:
            pm.user_obj_exprs.append(obj)

    def process_oxf_components(
        self,
        pm,
        network,
        optimization_problem: OptimizationProblem,
        period_index=None,
    ):
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            self._add_equations(
                pm,
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
            pm.user_obj_exprs.append(obj)

    def process_equations_compounds(self, pm, network, compounds, ignored_nodes):
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue

            equations = compound.equations(network)

            if equations is not None:
                equations = as_iter(equations)
                self._process_intermediate_eqs(pm, compound, equations)
                self._add_equations(
                    pm,
                    filter_intermediate_eqs(equations),
                    name_prefix=f"compound_{compound.id}",
                )

    def process_equations_nodes_childs(
        self, pm, network: Network, nodes, ignored_nodes
    ):
        sin_impl = pyo.sin
        cos_impl = pyo.cos
        abs_impl = abs
        sqrt_impl = pyo.sqrt
        log_impl = pyo.log

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
                    grid,
                    from_branches,
                    to_branches,
                    connected_childs,
                    sin_impl=sin_impl,
                    cos_impl=cos_impl,
                    abs_impl=abs_impl,
                    sqrt_impl=sqrt_impl,
                    log_impl=log_impl,
                )
            )

            for expr in node.minimize(
                grid, from_branches, to_branches, connected_childs, sqrt_impl=sqrt_impl
            ):
                pm.aux_obj_exprs.append(expr)

            node_eqs = [eq for eq in equations if not isinstance(eq, bool)]
            self._process_intermediate_eqs(pm, node.model, node_eqs)
            self._add_equations(
                pm, filter_intermediate_eqs(node_eqs), name_prefix=f"node_{node.id}"
            )

            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                for expr in child.minimize(grid, node, sqrt_impl=sqrt_impl):
                    pm.aux_obj_exprs.append(expr)
                child_eqs = as_iter(child.equations(grid, node))
                self._process_intermediate_eqs(pm, child.model, child_eqs)
                self._add_equations(
                    pm,
                    filter_intermediate_eqs(child_eqs),
                    name_prefix=f"child_{child.id}",
                )

    def process_equations_branches(self, pm, network, branches, ignored_nodes):
        sin_impl = pyo.sin
        cos_impl = pyo.cos
        abs_impl = abs
        sqrt_impl = pyo.sqrt
        log_impl = pyo.log
        pwl_impl = PyomoPWLImpl(pm, pw_repn="SOS2")

        def if_impl(*args, **kwargs):
            raise NotImplementedError(
                "Replace GEKKO if2/if3 with a Pyomo Piecewise / big-M formulation."
            )

        def max_impl(*args, **kwargs):
            raise NotImplementedError(
                "Replace GEKKO max2 with Pyomo max_ (or explicit constraints)."
            )

        def sign_impl(*args, **kwargs):
            raise NotImplementedError(
                "Replace GEKKO sign2/sign3 with a Pyomo formulation (often binary)."
            )

        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue

            grid = branch.grid

            branch_eqs = as_iter(
                branch.equations(
                    grid,
                    network.node_by_id(branch.from_node_id).model,
                    network.node_by_id(branch.to_node_id).model,
                    sin_impl=sin_impl,
                    cos_impl=cos_impl,
                    if_impl=if_impl,
                    abs_impl=abs_impl,
                    max_impl=max_impl,
                    sign_impl=sign_impl,
                    log_impl=log_impl,
                    sqrt_impl=sqrt_impl,
                    pwl_impl=pwl_impl,
                    simulation=self._simulation,
                )
            )

            for expr in branch.minimize(
                grid,
                network.node_by_id(branch.from_node_id).model,
                network.node_by_id(branch.to_node_id).model,
                sqrt_impl=sqrt_impl,
            ):
                pm.aux_obj_exprs.append(expr)

            self._process_intermediate_eqs(pm, branch.model, branch_eqs)
            self._add_equations(
                pm,
                filter_intermediate_eqs(branch_eqs),
                name_prefix=f"branch_{branch.id}",
            )
