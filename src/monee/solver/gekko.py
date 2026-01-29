import logging

import networkx as nx
from gekko import GEKKO
from gekko.gk_operators import GK_Intermediate, GK_Operators
from gekko.gk_variable import GKVariable
from .core import *

from monee.model import (
    Branch,
    Compound,
    Const,
    GenericModel,
    Intermediate,
    IntermediateEq,
    Network,
    Node,
    Var,
)
from monee.problem.core import OptimizationProblem

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


def _process_intermediate_eqs(m, model, equations):
    """
    No docstring provided.
    """
    for intermediate_eq in [eq for eq in equations if type(eq) is IntermediateEq]:
        attr_intermediate_var = getattr(model, intermediate_eq.attr)
        if type(attr_intermediate_var) is not Intermediate:
            m.Equation(attr_intermediate_var == intermediate_eq.eq)
        else:
            i = m.Intermediate(intermediate_eq.eq)
            setattr(model, intermediate_eq.attr, i)

class GEKKOSolver(SolverInterface):
    """
    No docstring provided.
    """

    def __init__(self, solver=1):
        self.solver: int = solver

    @staticmethod
    def inject_gekko_vars_attr(gekko: GEKKO, target: GenericModel, id):
        """
        No docstring provided.
        """
        i = 0
        for key, value in target.__dict__.items():
            if type(value) is Var:
                if value.name is not None:
                    name = f"{id}.{value.name}"
                else:
                    name = f"{id}.{i}"
                setattr(
                    target,
                    key,
                    gekko.Var(
                        value.value, lb=value.min, ub=value.max, integer=value.integer, name=name
                    ),
                )
                i += 1
            if type(value) is Const:
                setattr(target, key, gekko.Const(value.value))

    @staticmethod
    def inject_nans(target: GenericModel):
        """
        No docstring provided.
        """
        for key, value in target.__dict__.items():
            if isinstance(value, Const):
                setattr(target, key, Const(float("nan")))
            if isinstance(value, Var):
                setattr(target, key, Var(float("nan"), max=value.max, min=value.min, name=value.name))

    @staticmethod
    def inject_gekko_vars(
        gekko_model: GEKKO,
        nodes: list[Node],
        branches: list[Branch],
        compounds: list[Compound],
        network: Network,
        ignored_nodes: set,
    ):
        """
        No docstring provided.
        """
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                branch.ignored = True
                GEKKOSolver.inject_nans(branch.model)
                continue
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, branch.model, branch.nid)
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                node.ignored = True
                for child in network.childs_by_ids(node.child_ids):
                    child.ignored = True
                    GEKKOSolver.inject_nans(child.model)
                GEKKOSolver.inject_nans(node.model)
                continue
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, node.model, node.tid)
            for child in network.childs_by_ids(node.child_ids):
                if ignore_child(child, ignored_nodes):
                    child.ignored = True
                    GEKKOSolver.inject_nans(child.model)
                    continue
                GEKKOSolver.inject_gekko_vars_attr(gekko_model, child.model, child.tid)
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                compound.ignored = True
                GEKKOSolver.inject_nans(compound.model)
                continue
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, compound.model, compound.tid)

    @staticmethod
    def withdraw_gekko_vars_attr(target: GenericModel):
        """
        No docstring provided.
        """
        for key, value in target.__dict__.items():
            if type(value) is GKVariable:
                setattr(
                    target,
                    key,
                    Var(value=value.VALUE.value[0], min=value.LOWER, max=value.UPPER, name=value.NAME.split("_")[-1]),
                )
            if type(value) is GK_Operators:
                setattr(target, key, Const(value.VALUE.value))
            if type(value) is GK_Intermediate:
                setattr(target, key, Intermediate(value=value.VALUE.value[0]))

    @staticmethod
    def withdraw_gekko_vars(nodes, branches, compounds, network):
        """
        No docstring provided.
        """
        for branch in branches:
            GEKKOSolver.withdraw_gekko_vars_attr(branch.model)
        for node in nodes:
            GEKKOSolver.withdraw_gekko_vars_attr(node.model)
            for child in network.childs_by_ids(node.child_ids):
                GEKKOSolver.withdraw_gekko_vars_attr(child.model)
        for compound in compounds:
            GEKKOSolver.withdraw_gekko_vars_attr(compound.model)

    def solve(
        self,
        input_network: Network,
        optimization_problem: OptimizationProblem = None,
        draw_debug=False,
        exclude_unconnected_nodes=False,
    ):
        """
        No docstring provided.
        """
        GKVariable.max = property(lambda self: self.UPPER)
        GKVariable.min = property(lambda self: self.LOWER)
        m = GEKKO(remote=False)
        m.options.SOLVER = self.solver
        m.options.WEB = 0
        m.options.IMODE = 3
        m.solver_options = DEFAULT_SOLVER_OPTIONS
        network = input_network.copy()

        if optimization_problem is not None:
            optimization_problem._apply(network)

        ignored_nodes = set()
        if optimization_problem is None or exclude_unconnected_nodes:
            ignored_nodes = find_ignored_nodes(network)

        nodes = network.nodes
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            for child in network.childs_by_ids(node.child_ids):
                if child.active:
                    child.model.overwrite(node.model)

        branches = network.branches
        compounds = network.compounds
        
        GEKKOSolver.inject_gekko_vars(
            m, nodes, branches, compounds, network, ignored_nodes
        )
        objs_exprs = []
        self.init_branches(branches)
        self.process_equations_nodes_childs(m, network, nodes, ignored_nodes)
        self.process_equations_branches(m, network, branches, ignored_nodes, objs_exprs)
        self.process_equations_compounds(m, network, compounds, ignored_nodes)
        if optimization_problem is not None:
            self.process_oxf_components(m, network, optimization_problem)
        else:
            self.process_internal_oxf_components(m, network)
        m.Obj(sum(objs_exprs))
        try:
            m.solve(disp=False)
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
            raise
        GEKKOSolver.withdraw_gekko_vars(nodes, branches, compounds, network)
        solver_result = SolverResult(
            network, network.as_result_dataframe_dict(), m.options.OBJFCNVAL
        )
        return solver_result

    def process_internal_oxf_components(self, m, network):
        """
        No docstring provided.
        """
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
        self, m, network: Network, optimization_problem: OptimizationProblem
    ):
        """
        No docstring provided.
        """
        if optimization_problem.constraints is not None and (
            not optimization_problem.constraints.empty
        ):
            m.Equations(optimization_problem.constraints.all(network))
        obj = 0
        for objective in optimization_problem.objectives.all(network):
            if obj is not None:
                obj = obj + objective
            else:
                obj = objective
        if obj is not None:
            m.Obj(obj)

    def process_equations_compounds(self, m, network, compounds, ignored_nodes):
        """
        No docstring provided.
        """
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
        """
        No docstring provided.
        """
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
                    sqrt_impl=m.sqrt
                )
            )
            for expr in node.minimize(grid, 
                                              from_branches, 
                                              to_branches,
                                              connected_childs,
                                              sqrt_impl=m.sqrt):
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

    def init_branches(self, branches):
        for branch in branches:
            branch.model.init(branch.grid)

    def process_equations_branches(self, m, network, branches, ignored_nodes, objs_exprs):
        """
        No docstring provided.
        """
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
                    sqrt_impl=m.sqrt
                )
            )

            for expr in branch.minimize(grid, 
                                       network.node_by_id(branch.from_node_id).model, 
                                       network.node_by_id(branch.to_node_id).model,
                                       sqrt_impl=m.sqrt):
                objs_exprs.append(expr)

            _process_intermediate_eqs(m, branch.model, branch_eqs)
            m.Equations(filter_intermediate_eqs(branch_eqs))
