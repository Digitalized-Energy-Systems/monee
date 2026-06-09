import logging

import networkx as nx
from gekko import GEKKO
from gekko.gk_operators import GK_Intermediate, GK_Operators
from gekko.gk_variable import GKVariable

from monee.model import (
    Const,
    GenericModel,
    Intermediate,
    IntermediateEq,
    Network,
    Var,
)
from monee.problem.core import OptimizationProblem
from monee.simulation.step_state import StepState

from .core import (
    SolverInterface,
    SolverResult,
    as_iter,
    compute_bound_violations,
    filter_intermediate_eqs,
    find_ignored_nodes,
    generate_real_topology,
    ignore_branch,
    ignore_child,
    ignore_compound,
    ignore_node,
    inject_vars,
    mark_heat_balance_slacks,
    mark_ignored_components,
    persist_solution,
    remove_cps,
    withdraw_vars,
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

# IPOPT (SOLVER=3) is a pure-NLP solver — the smooth gas/heat formulations target
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


def _process_intermediate_eqs(m, model, equations):
    for intermediate_eq in [eq for eq in equations if type(eq) is IntermediateEq]:
        attr_intermediate_var = getattr(model, intermediate_eq.attr)
        eq = (
            intermediate_eq.eq() if callable(intermediate_eq.eq) else intermediate_eq.eq
        )

        if type(attr_intermediate_var) is Intermediate:
            i = m.Intermediate(eq)
            setattr(model, intermediate_eq.attr, i)


class GEKKOSolver(SolverInterface):
    def __init__(self, solver=1, simulation=False):
        self.solver: int = solver
        # simulation=True runs a plain energy flow as a square steady-state
        # simulation (IMODE=1) — far faster than optimising the feasibility
        # problem — and falls back to IMODE=3 if the model is not square.
        self.simulation: bool = simulation

    @staticmethod
    def inject_gekko_vars_attr(gekko: GEKKO, target: GenericModel, id):
        i = 0
        for key, value in target.__dict__.items():
            if isinstance(value, Var):
                name = f"{id}.{value.name}" if value.name is not None else f"{id}.{i}"
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
        """Solve, trying IMODE=1 first in simulation mode and falling back to
        IMODE=3 if the model is not square (``Degrees of Freedom``) or otherwise
        fails as a simulation. The IMODE=3 retry runs on the same model."""
        if use_sim:
            try:
                m.solve(disp=False)
                return
            except Exception:
                m.options.IMODE = 3
        m.solve(disp=False)

    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
        step_state: StepState = None,
    ):
        m = GEKKO(remote=False)
        m.options.SOLVER = self.solver
        m.options.WEB = 0
        m.options.IMODE = 1 if self.simulation else 3
        m.solver_options = _solver_options(self.solver)
        network = input_network.copy()

        for ext in network.extensions:
            ext.prepare(network)

        from monee.model.extension.islanding.core import NetworkIslandingConfig

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

        # Recognise each heat island's grid-forming node as the heat slack and
        # drop its (dependent) nodal heat balance — removes the heat carrier's
        # redundant constraint and is required for a square IMODE=1 solve.
        mark_heat_balance_slacks(network, ignored_nodes)

        inject_vars(
            lambda model, comp, cat: GEKKOSolver.inject_gekko_vars_attr(
                m, model, comp.nid if cat == "branch" else comp.tid
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
            m.Obj(sum(objs_exprs))

        # IMODE=1 (square simulation) only applies to a plain flow: no objective
        # of any kind (else IMODE=1 silently ignores it).
        use_sim = (
            self.simulation
            and optimization_problem is None
            and not objs_exprs
            and not network.objectives
        )
        m.options.IMODE = 1 if use_sim else 3

        try:
            self._solve_with_fallback(m, use_sim)
        except Exception:
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
            raise
        withdraw_vars(
            GEKKOSolver.withdraw_gekko_vars_attr, nodes, branches, compounds, network
        )
        persist_solution(network, input_network)
        violations = compute_bound_violations(nodes, branches, compounds, network)
        solver_result = SolverResult(
            network,
            network.as_result_dataframe_dict(),
            m.options.OBJFCNVAL,
            m.options.APPSTATUS == 1,
            violations,
        )
        return solver_result

    def process_internal_oxf_components(self, m, network):
        for constraint in network.constraints:
            m.Equation(constraint(network))
        obj = None
        for objective in network.objectives:
            if obj is not None:
                obj = obj + objective(network)
            else:
                obj = objective(network)
        if obj is not None:
            m.Obj(obj)

    def process_oxf_components(
        self,
        m,
        network: Network,
        optimization_problem: OptimizationProblem,
        period_index=None,
    ):
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            m.Equations(
                optimization_problem.constraints.all(network, period_index=period_index)
            )
        obj = None
        for objective in (
            optimization_problem.objectives.all(network, period_index=period_index)
            if optimization_problem.objectives is not None
            else []
        ):
            if obj is not None:
                obj = obj + objective
            else:
                obj = objective
        if obj is not None:
            m.Obj(obj)

    def process_equations_compounds(self, m, network, compounds, ignored_nodes):
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                continue

            equations = compound.equations(network)

            for expr in compound.minimize(network, sqrt_impl=m.sqrt):
                m.Obj(expr)

            if equations is not None:
                _process_intermediate_eqs(m, compound, equations)
                m.Equations(filter_intermediate_eqs(as_iter(equations)))

    def process_equations_nodes_childs(self, m, network: Network, nodes, ignored_nodes):
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid

            from_branches = [
                network.branch_by_id(branch_id).model
                for branch_id in node.from_branch_ids
                if not ignore_branch(
                    network.branch_by_id(branch_id), network, ignored_nodes
                )
            ]
            to_branches = [
                network.branch_by_id(branch_id).model
                for branch_id in node.to_branch_ids
                if not ignore_branch(
                    network.branch_by_id(branch_id), network, ignored_nodes
                )
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
                    sin_impl=m.sin,
                    cos_impl=m.cos,
                    if_impl=m.if2,
                    abs_impl=m.abs3,
                    max_impl=m.max2,
                    sign_impl=m.sign2,
                    sqrt_impl=m.sqrt,
                )
            )
            for expr in node.minimize(
                grid, from_branches, to_branches, connected_childs, sqrt_impl=m.sqrt
            ):
                m.Obj(expr)

            node_eqs = [eq for eq in equations if type(eq) is not bool or not eq]
            _process_intermediate_eqs(m, node.model, node_eqs)
            m.Equations(filter_intermediate_eqs(node_eqs))

            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                child_eqs = as_iter(child.equations(grid, node))

                for expr in child.minimize(grid, node, sqrt_impl=m.sqrt):
                    m.Obj(expr)

                _process_intermediate_eqs(m, child.model, child_eqs)
                m.Equations(filter_intermediate_eqs(child_eqs))

    def process_equations_branches(
        self, m, network, branches, ignored_nodes, objs_exprs
    ):
        # spline outperforms GEKKO's native pwl
        pwl_impl = GekkoCubicSplineImpl(m)
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue
            grid = branch.grid

            branch_eqs = as_iter(
                branch.equations(
                    grid,
                    network.node_by_id(branch.from_node_id).model,
                    network.node_by_id(branch.to_node_id).model,
                    sin_impl=m.sin,
                    cos_impl=m.cos,
                    if_impl=m.if3,
                    abs_impl=m.abs3,
                    max_impl=m.max2,
                    sign_impl=m.sign3,
                    log_impl=m.log10,
                    sqrt_impl=m.sqrt,
                    exp_impl=m.exp,
                    pwl_impl=pwl_impl,
                )
            )

            for expr in branch.minimize(
                grid,
                network.node_by_id(branch.from_node_id).model,
                network.node_by_id(branch.to_node_id).model,
                sqrt_impl=m.sqrt,
            ):
                objs_exprs.append(expr)

            _process_intermediate_eqs(m, branch.model, branch_eqs)
            m.Equations(filter_intermediate_eqs(branch_eqs))
