import logging

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
from monee.model.extension.islanding.core import NetworkIslandingConfig
from monee.model.formulation.registry import attach_formulations
from monee.problem.core import OptimizationProblem

from .core import (
    OperatorEquationAssembly,
    SolverInterface,
    SolverResult,
    StepState,
    apply_post_process_all,
    compute_bound_violations,
    find_ignored_nodes,
    generate_real_topology,
    ignore_node,
    inject_vars,
    mark_he_flow_prescription,
    mark_heat_balance_slacks,
    mark_ignored_components,
    persist_solution,
    pin_floating_hydraulic_gauges,
    remove_cps,
    withdraw_vars,
)
from .infeasibility.apm import (
    GekkoSolveError,
    diagnose_gekko_infeasibility,
    sanitize_apm_name,
)

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

    def piecewise_eq(self, y, x, xs, ys, name=None):
        xs = list(xs)
        ys = list(ys)
        return self.m.cspline(x, y, xs, ys)


class GEKKOSolver(OperatorEquationAssembly, SolverInterface):
    def __init__(self, solver=1):
        self.solver: int = solver
        self._simulation: bool = False

    @staticmethod
    def inject_gekko_vars_attr(gekko: GEKKO, target: GenericModel, id, name_map=None):
        i = 0
        for key, value in target.__dict__.items():
            if isinstance(value, Var):
                name = f"{id}.{value.name}" if value.name is not None else f"{id}.{i}"
                if name_map is not None:
                    # APMonitor sanitises names irreversibly; remember the
                    # original for the infeasibility diagnostics.
                    name_map[sanitize_apm_name(name)] = name
                setattr(
                    target,
                    key,
                    gekko.Var(
                        value.value,
                        lb=value.min,
                        ub=value.max,
                        integer=value.integer,
                        name=name,
                    ),
                )
                i += 1
            if type(value) is Const:
                setattr(target, key, gekko.Const(value.value))

    @staticmethod
    def withdraw_gekko_vars_attr(target: GenericModel):
        for key, value in target.__dict__.items():
            if type(value) is GKVariable:
                setattr(
                    target,
                    key,
                    Var(
                        value=value.VALUE.value[0],
                        min=value.LOWER,
                        max=value.UPPER,
                        name=value.NAME.split("_")[-1],
                    ),
                )
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
    ):
        self._simulation = simulation
        m = GEKKO(remote=False)
        m.options.SOLVER = self.solver
        m.options.WEB = 0
        m.options.IMODE = 1 if simulation else 3
        m.solver_options = _solver_options(self.solver)
        network = input_network.copy()

        for ext in network.extensions:
            ext.prepare(network)

        # Attach the effective formulations and declare their vars on the
        # solve-time copy (simulation=True additionally squares the model:
        # phantom DOFs pinned, |m| warm-started, vm_pu_squared demoted).
        attach_formulations(network, formulation, simulation=simulation)

        islanding_config = next(
            (e for e in network.extensions if isinstance(e, NetworkIslandingConfig)),
            None,
        )

        # Compute ignored_nodes BEFORE _apply so controllable filters checking
        # component.ignored correctly exclude disconnected components.
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
        # source (e.g. an HE-fed return loop) - removes a rank-deficient free DOF
        # IPOPT would otherwise have to regularise.
        pin_floating_hydraulic_gauges(network, ignored_nodes)

        # Recognise each heat island's grid-forming node as the heat slack and
        # drop its (dependent) nodal heat balance - removes the heat carrier's
        # redundant constraint and is required for a square IMODE=1 solve.
        mark_heat_balance_slacks(network, ignored_nodes)

        # Decide per compound-internal SubHE whether the design flow prescribes
        # the through-flow (supply/return islands with free-mass-flow slacks)
        # or yields to a network-determined flow (e.g. a fixed sink downstream).
        mark_he_flow_prescription(network, ignored_nodes)

        apm_name_map: dict[str, str] = {}
        inject_vars(
            lambda model, comp, cat: GEKKOSolver.inject_gekko_vars_attr(
                m,
                model,
                comp.nid if cat == "branch" else comp.tid,
                name_map=apm_name_map,
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

        if objs_exprs:
            for expr in objs_exprs:
                m.Obj(expr)

        # IMODE=1 (square simulation) only applies to a plain flow: no objective
        # of any kind (else IMODE=1 silently ignores it).
        use_sim = (
            simulation
            and optimization_problem is None
            and not objs_exprs
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
                    GEKKOSolver.withdraw_gekko_vars_attr,
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
        withdraw_vars(
            GEKKOSolver.withdraw_gekko_vars_attr, nodes, branches, compounds, network
        )
        apply_post_process_all(nodes, branches, compounds, network)
        persist_solution(network, input_network)
        violations = compute_bound_violations(nodes, branches, compounds, network)
        solver_result = SolverResult(
            network,
            network.as_result_dataframe_dict(),
            m.options.OBJFCNVAL,
            m.options.APPSTATUS == 1,
            violations,
            mode_used="simulation" if imode_used == 1 else "optimization",
        )
        return solver_result

    def _pwl_impl(self, m):
        # spline outperforms GEKKO's native pwl
        return GekkoCubicSplineImpl(m)
