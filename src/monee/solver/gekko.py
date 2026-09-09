import contextlib
import logging
import math

import networkx as nx
from gekko import GEKKO
from gekko.gk_operators import GK_Intermediate, GK_Operators
from gekko.gk_variable import GKVariable

from monee.model import (
    Const,
    GenericModel,
    Intermediate,
    Network,
    Var,
)
from monee.problem.core import OptimizationProblem

from .core import (
    OperatorEquationAssembly,
    ResultWarning,
    SolverInterface,
    SolverResult,
    StepState,
    apply_child_overwrites,
    finalize_solution,
    generate_real_topology,
    inject_vars,
    mark_slacks_and_prescriptions,
    persist_solution,
    prepare_solve_network,
    remove_cps,
    validate_result,
    warn_on_residuals,
    withdraw_vars,
)
from .infeasibility.apm import (
    GekkoSolveError,
    diagnose_gekko_infeasibility,
    sanitize_apm_name,
)

# Reverse of dispatch.GEKKO_SOLVERS (name -> code), so a constructed GEKKOSolver
# can report which solver it runs on SolverResult.solver_used.
_GEKKO_CODE_TO_NAME: dict[int, str] = {1: "apopt", 2: "bpopt", 3: "ipopt"}

_EVAL_NS = {
    name: getattr(math, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "log10",
        "sqrt",
        "erf",
    )
}
_EVAL_NS["abs"] = abs


def _gekko_status(m) -> tuple[str, str]:
    """``(solver_status, termination_condition)`` from GEKKO's application and
    solver status flags."""
    app = int(m.options.APPSTATUS)
    solve = int(m.options.SOLVESTATUS)
    if app == 1 and solve == 1:
        return "ok", "optimal"
    logging.warning(
        "GEKKO reported APPSTATUS=%d, SOLVESTATUS=%d: the returned point is not "
        "a clean successful solution.",
        app,
        solve,
    )
    return "warning", f"other (APPSTATUS={app}, SOLVESTATUS={solve})"


def _gekko_max_residual(m) -> float | None:
    """Largest equation violation at GEKKO's returned point, or ``None`` when it
    cannot be evaluated.

    APOPT can report a successful solution for a point that misses an equality
    by a physically relevant amount, and GEKKO only writes its own residual
    report (``infeasibilities.txt``) on failure, so re-evaluate the equations it
    was given against the values it returned.
    """

    def scalar(obj):
        # GEKKO wraps values in GK_Value, whose payload is either a scalar or a
        # one-entry trajectory; unwrap until a number comes out.
        v = obj.value
        for _ in range(4):
            if isinstance(v, (int, float)):
                return float(v)
            nxt = getattr(v, "value", None)
            if nxt is None:
                with contextlib.suppress(TypeError, IndexError, KeyError):
                    nxt = v[0]
            if nxt is None or nxt is v:
                break
            v = nxt
        return None

    ns = dict(_EVAL_NS)
    for group in (m._variables, m._parameters, m._intermediates, m._constants):
        for obj in group:
            val = scalar(obj)
            if val is not None:
                ns[obj.name] = val

    worst = 0.0
    for eq in m._equations:
        # GEKKO's own rendering of the equations monee handed it: names, numbers
        # and math calls only, evaluated without builtins.
        expr = str(eq.value).replace("^", "**")
        for op in (">=", "<=", "="):
            if op in expr:
                break
        else:
            return None
        lhs, _, rhs = expr.partition(op)
        try:
            diff = eval(lhs, {"__builtins__": {}}, ns) - eval(  # noqa: S307
                rhs, {"__builtins__": {}}, ns
            )
        except Exception:
            return None
        if op == "=":
            worst = max(worst, abs(diff))
        elif op == "<=":
            worst = max(worst, diff)
        else:
            worst = max(worst, -diff)
    return worst


# The builtin, aliased because inject_gekko_vars_attr's legacy signature shadows
# ``id``. GKVariable overrides ``__eq__`` (builds equations) so identity keys are
# the only safe dict keys for the var-metadata registry.
id_ = id

# APOPT (SOLVER=1) MINLP options. IPOPT rejects the minlp_* keys, so they are
# applied only for the APOPT path (see _solver_options).
DEFAULT_SOLVER_OPTIONS = [
    "minlp_maximum_iterations 1000",
    "minlp_max_iter_with_int_sol 500",
    "minlp_as_nlp 0",
    "nlp_maximum_iterations 1000",
    "minlp_branch_method 3",
    "minlp_gap_tol 1.0e-3",
    "minlp_integer_tol 1.0e-4",
    "minlp_integer_max 2.0e5",
    "minlp_integer_leaves 150",
    "minlp_print_level 1",
    "objective_convergence_tolerance 1.0e-4",
    "constraint_convergence_tolerance 1.0e-4",
]

# IPOPT (SOLVER=3) is a pure-NLP solver, the smooth gas/heat formulations target
# it. Only IPOPT-valid option keys here.
IPOPT_SOLVER_OPTIONS = [
    "max_iter 3000",
    "tol 1.0e-6",
    "constr_viol_tol 1.0e-6",
]


def _solver_options(solver: int):
    """APOPT keeps its MINLP options; IPOPT gets NLP-only options it accepts."""
    return IPOPT_SOLVER_OPTIONS if solver == GEKKO_IPOPT else DEFAULT_SOLVER_OPTIONS


GEKKO_IPOPT = 3


class GekkoCubicSplineImpl:
    def __init__(self, m):
        self.m = m

    def piecewise_eq(self, y, x, xs, ys, _name=None):
        xs = list(xs)
        ys = list(ys)
        return self.m.cspline(x, y, xs, ys)


class GEKKOSolver(OperatorEquationAssembly, SolverInterface):
    """GEKKO/APMonitor back-end.

    Failure contract: unlike the CasADi, Pyomo and gurobipy back-ends, which
    return a ``success=False`` :class:`~monee.solver.core.SolverResult`, a failed
    GEKKO solve raises :class:`~monee.solver.GekkoSolveError` carrying the parsed
    APMonitor report. The failure arrives as an exception out of the APMonitor
    subprocess with no iterate to report on a result, so there is nothing to hand
    back. ``solver='auto'`` treats that exception as a numerics failure and
    retries on SCIP like any other. See the concepts/solvers docs page.
    """

    def __init__(self, solver=1):
        self.solver: int = solver
        self._backend_name = "gekko"
        self._solver_name = _GEKKO_CODE_TO_NAME.get(solver, str(solver))
        self._simulation: bool = False

    @staticmethod
    def inject_gekko_vars_attr(
        gekko: GEKKO, target: GenericModel, id, name_map=None, var_meta=None
    ):
        i = 0
        for key, value in target.__dict__.items():
            if isinstance(value, Var):
                name = f"{id}.{value.name}" if value.name is not None else f"{id}.{i}"
                if name_map is not None:
                    # APMonitor sanitises names irreversibly; remember the
                    # original for the infeasibility diagnostics.
                    name_map[sanitize_apm_name(name)] = name
                gk_var = gekko.Var(
                    value.value,
                    lb=value.min,
                    ub=value.max,
                    integer=value.integer,
                    name=name,
                )
                if var_meta is not None:
                    # GEKKO mangles names irreversibly (lowercased, "int_"
                    # prefix, non-word chars -> "_"); keep the original Var so
                    # withdraw can restore name/integer/scale exactly.
                    var_meta[id_(gk_var)] = value
                setattr(target, key, gk_var)
                i += 1
            if type(value) is Const:
                setattr(target, key, gekko.Const(value.value))

    @staticmethod
    def withdraw_gekko_vars_attr(target: GenericModel, var_meta=None):
        for key, value in target.__dict__.items():
            if type(value) is GKVariable:
                orig = var_meta.get(id_(value)) if var_meta is not None else None
                val = value.VALUE.value[0]
                if orig is not None:
                    if orig.integer:
                        val = int(round(val))
                    new_var = Var(
                        value=val,
                        min=value.LOWER,
                        max=value.UPPER,
                        integer=orig.integer,
                        name=orig.name,
                        scale=orig.scale,
                    )
                else:
                    # No registry (legacy callers): GEKKO's stored NAME is a
                    # mangled artefact, best-effort only.
                    new_var = Var(
                        value=val,
                        min=value.LOWER,
                        max=value.UPPER,
                        integer=value.NAME.startswith("int_"),
                        name=value.NAME.split("_")[-1],
                    )
                setattr(target, key, new_var)
            if type(value) is GK_Operators:
                setattr(target, key, Const(value.VALUE.value))
            if type(value) is GK_Intermediate:
                setattr(target, key, Intermediate(value=value.VALUE.value[0]))

    def _add_equations(self, m, eqs):
        m.Equations(eqs)

    @staticmethod
    def _solve_with_fallback(m, use_sim):
        if use_sim:
            try:
                m.solve(disp=False)
                return 1
            except Exception as exc:
                logging.warning(
                    "Simulation (IMODE=1) solve failed; falling back to "
                    "IMODE=3 - the fast square steady-state path was NOT used. "
                    "The model is likely not square (check for phantom degrees "
                    "of freedom / non-simulation formulations). Solver said: %s",
                    exc,
                )
                m.options.IMODE = 3
        m.solve(disp=False)
        return m.options.IMODE

    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
        step_state: StepState = None,
        simulation=False,
        formulation=None,
        strict: bool = False,
    ):
        self._simulation = simulation
        m = GEKKO(remote=False)
        try:
            return self._solve_with_model(
                m,
                input_network,
                optimization_problem=optimization_problem,
                draw_debug=draw_debug,
                exclude_unconnected_nodes=exclude_unconnected_nodes,
                step_state=step_state,
                simulation=simulation,
                formulation=formulation,
                strict=strict,
            )
        finally:
            # APMonitor leaves its run directory behind on every solve; the
            # diagnostics that read it (diagnose_gekko_infeasibility) have
            # already run by the time we get here.
            with contextlib.suppress(Exception):
                m.cleanup()

    def _solve_with_model(  # NOSONAR
        self,
        m: GEKKO,
        input_network: Network,
        *,
        optimization_problem,
        draw_debug,
        exclude_unconnected_nodes,
        step_state,
        simulation,
        formulation,
        strict=False,
    ):
        m.options.SOLVER = self.solver
        m.options.WEB = 0
        m.options.IMODE = 1 if simulation else 3
        m.solver_options = _solver_options(self.solver)

        # Copy the network, run extension prepare/attach, locate the islanding
        # config and compute ignored nodes BEFORE _apply so controllable filters
        # checking component.ignored correctly exclude disconnected components.
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

        # Pin floating hydraulic gauges, recognise each heat island's
        # grid-forming node as the heat slack (dropping its dependent nodal heat
        # balance, required for a square IMODE=1 solve) and decide each
        # compound-internal SubHE's flow prescription.
        mark_slacks_and_prescriptions(network, ignored_nodes)

        apm_name_map: dict[str, str] = {}
        var_meta: dict[int, Var] = {}
        withdraw_fn = lambda target: GEKKOSolver.withdraw_gekko_vars_attr(  # noqa: E731
            target, var_meta=var_meta
        )
        inject_vars(
            lambda model, comp, cat: GEKKOSolver.inject_gekko_vars_attr(
                m,
                model,
                comp.nid if cat == "branch" else comp.tid,
                name_map=apm_name_map,
                var_meta=var_meta,
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

        objs_exprs = []
        self.init_branches(branches)
        self.process_equations_nodes_childs(m, network, nodes, ignored_nodes)
        self.process_equations_branches(m, network, branches, ignored_nodes, objs_exprs)
        self.process_equations_compounds(m, network, compounds, ignored_nodes)
        if optimization_problem is not None:
            self.process_oxf_components(m, network, optimization_problem)
        else:
            self.process_internal_oxf_components(m, network)

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

        # IMODE=1 (square simulation) only applies to a plain flow: no objective
        # of any kind (else IMODE=1 silently ignores it). m._objectives covers
        # every m.Obj() call, including the node/child/compound minimize terms
        # added inside the shared equation passes.
        use_sim = (
            simulation
            and optimization_problem is None
            and not m._objectives
            and not network.objectives
        )
        m.options.IMODE = 1 if use_sim else 3

        try:
            imode_used = self._solve_with_fallback(m, use_sim)
        except Exception as exc:
            # APMonitor's exception is just "@error: Solution Not Found";
            # build a proper report from the run-directory artifacts instead.
            report = diagnose_gekko_infeasibility(
                m, name_map=apm_name_map, solver_message=str(exc)
            )
            if report is not None:
                logging.error(
                    "Solver not converged. Diagnostic report:\n%s",
                    report.summary(max_items=25),
                )
            else:
                logging.error("Solver not converged.")
            if draw_debug:
                import matplotlib.pyplot as plt

                remove_cps(network)
                nx.draw_networkx(
                    generate_real_topology(network._network_internal),
                    node_size=5,
                    font_size=2,
                    width=0.4,
                )
                plt.savefig("debug-network.pdf")
            # Best-effort warm-start handoff from the partial iterate.
            try:
                withdraw_vars(
                    withdraw_fn,
                    nodes,
                    branches,
                    compounds,
                    network,
                )
                persist_solution(network, input_network)
            except Exception:
                pass
            if report is not None:
                raise GekkoSolveError(
                    "GEKKO solve failed.\n\nDiagnostic report:\n"
                    + report.summary(max_items=25),
                    report=report,
                ) from exc
            raise
        withdraw_vars(withdraw_fn, nodes, branches, compounds, network)
        violations = finalize_solution(
            nodes, branches, compounds, network, input_network
        )
        solver_status, termination_condition = _gekko_status(m)
        solver_result = SolverResult(
            network,
            network.as_result_dataframe_dict(),
            m.options.OBJFCNVAL,
            m.options.APPSTATUS == 1,
            violations,
            solver_status=solver_status,
            termination_condition=termination_condition,
            residuals=warn_on_residuals(_gekko_max_residual(m)),
            mode_used="simulation" if imode_used == 1 else "optimization",
            backend_used=self.backend_name,
            solver_used=self.solver_name,
        )
        extra = None
        if use_sim and imode_used != 1:
            extra = [
                ResultWarning(
                    "dof",
                    "simulation requested but the square IMODE=1 solve failed "
                    "and fell back to IMODE=3; the model is likely not square "
                    "(phantom degrees of freedom or non-simulation "
                    "formulations), so the solved values are one feasible "
                    "point, not a unique setpoint",
                )
            ]
        return validate_result(
            network,
            solver_result,
            ignored_nodes=ignored_nodes,
            simulation=simulation,
            strict=strict,
            extra_warnings=extra,
        )

    def _pwl_impl(self, m):
        # spline outperforms GEKKO's native pwl
        return GekkoCubicSplineImpl(m)
