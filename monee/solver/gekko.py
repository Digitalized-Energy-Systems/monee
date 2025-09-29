import logging
from dataclasses import dataclass

import networkx as nx
import pandas
from gekko import GEKKO
from gekko.gk_operators import GK_Intermediate, GK_Operators
from gekko.gk_variable import GKVariable

from monee.model import (
    CHP,
    Branch,
    Compound,
    Const,
    ExtHydrGrid,
    ExtPowerGrid,
    GasToHeat,
    GenericModel,
    Intermediate,
    IntermediateEq,
    MultiGridBranchModel,
    Network,
    Node,
    PowerToHeat,
    Var,
    WaterPipe,
)
from monee.problem.core import OptimizationProblem

DEFAULT_SOLVER_OPTIONS = [
    "minlp_maximum_iterations 250",
    "minlp_max_iter_with_int_sol 500",
    "minlp_as_nlp 0",
    "nlp_maximum_iterations 250",
    "minlp_branch_method 3",
    "minlp_gap_tol 1.0e-3",
    "minlp_integer_tol 1.0e-4",
    "minlp_integer_max 2.0e5",
    "minlp_integer_leaves 50",
    "minlp_print_level 1",
    "objective_convergence_tolerance 1.0e-4",
    "constraint_convergence_tolerance 1.0e-4",
]


@dataclass
class SolverResult:
    """
    No docstring provided.
    """

    network: Network
    dataframes: dict[str, pandas.DataFrame]
    objective: float

    def __str__(self) -> str:
        """
        No docstring provided.
        """
        result_str = str(self.network)
        result_str += "\n"
        for cls_str, dataframe in self.dataframes.items():
            result_str += cls_str
            result_str += "\n"
            result_str += dataframe.to_string()
            result_str += "\n"
            result_str += "\n"
        return result_str


def _as_iter(possible_iter):
    """
    No docstring provided.
    """
    if possible_iter is None:
        raise Exception("None as result for 'equations' is not allowed!")
    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


def _filter_intermediate_eqs(eqs):
    """
    No docstring provided.
    """
    return [eq for eq in eqs if type(eq) is not IntermediateEq]


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


def ignore_branch(branch, network: Network, ignored_nodes):
    """
    No docstring provided.
    """
    ig = (
        not branch.active
        or ignore_node(network.node_by_id(branch.id[0]), network, ignored_nodes)
        or ignore_node(network.node_by_id(branch.id[1]), network, ignored_nodes)
    )
    return ig


def ignore_node(node, network: Network, ignored_nodes):
    """
    No docstring provided.
    """
    ig = not node.active or node.id in ignored_nodes
    if not node.independent:
        ig = ig or ignore_compound(network.compound_of_node(node.id), ignored_nodes)
    return ig


def ignore_child(child, ignored_nodes):
    """
    No docstring provided.
    """
    ig = not child.active or child.node_id in ignored_nodes
    return ig


def ignore_compound(compound, ignored_nodes):
    """
    No docstring provided.
    """
    ig = not compound.active
    if any([value in ignored_nodes for value in compound.connected_to.values()]):
        if hasattr(compound.model, "set_active"):
            compound.model.set_active(False)
        else:
            ig = True
    elif hasattr(compound.model, "set_active"):
        compound.model.set_active(True)
    return ig


def generate_real_topology(nx_net):
    """
    No docstring provided.
    """
    net_copy = nx_net.copy()
    for edge in nx_net.edges.data():
        branch = edge[2]["internal_branch"]
        if not branch.active or (
            type(branch.model.on_off) is not Var and branch.model.on_off == 0
        ):
            net_copy.remove_edge(edge[0], edge[1], 0)
    return net_copy


COMPOUND_TYPES_TO_REMOVE = [PowerToHeat, GasToHeat, CHP]


def remove_cps(network: Network):
    """
    No docstring provided.
    """
    relevant_compounds = [
        compound
        for compound in network.compounds
        if type(compound.model) in COMPOUND_TYPES_TO_REMOVE
    ]
    for comp in relevant_compounds:
        network.remove_compound(comp.id)
        if type(comp.model) in COMPOUND_TYPES_TO_REMOVE:
            heat_return_node = network.node_by_id(
                comp.connected_to["heat_return_node_id"]
            )
            heat_node = network.node_by_id(comp.connected_to["heat_node_id"])
            network.branch(WaterPipe(0, 0), heat_return_node.id, heat_node.id)
    for branch in network.branches:
        if isinstance(branch.model, MultiGridBranchModel):
            network.remove_branch(branch.id)


def find_ignored_nodes(network: Network):
    """
    No docstring provided.
    """
    ignored_nodes = set()
    without_cps = network.copy()
    remove_cps(without_cps)
    real_topology = generate_real_topology(without_cps._network_internal)
    components = nx.connected_components(real_topology)
    for component in components:
        component_leading = False
        for node in component:
            int_node: Node = real_topology.nodes[node]["internal_node"]
            for child_id in int_node.child_ids:
                child = without_cps.child_by_id(child_id)
                if isinstance(child.model, ExtPowerGrid | ExtHydrGrid):
                    component_leading = True
                    break
            if component_leading:
                break
        if not component_leading:
            ignored_nodes.update(component)
    return ignored_nodes


class GEKKOSolver:
    """
    No docstring provided.
    """

    def __init__(self, solver=1):
        self.solver: int = solver

    @staticmethod
    def inject_gekko_vars_attr(gekko: GEKKO, target: GenericModel):
        """
        No docstring provided.
        """
        for key, value in target.__dict__.items():
            if type(value) is Var:
                setattr(
                    target,
                    key,
                    gekko.Var(
                        value.value, lb=value.min, ub=value.max, integer=value.integer
                    ),
                )
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
            if isinstance(value, Var | Const):
                setattr(target, key, Var(float("nan"), max=value.max, min=value.min))

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
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, branch.model)
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                node.ignored = True
                for child in network.childs_by_ids(node.child_ids):
                    child.ignored = True
                    GEKKOSolver.inject_nans(child.model)
                GEKKOSolver.inject_nans(node.model)
                continue
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, node.model)
            for child in network.childs_by_ids(node.child_ids):
                if ignore_child(child, ignored_nodes):
                    child.ignored = True
                    GEKKOSolver.inject_nans(child.model)
                    continue
                GEKKOSolver.inject_gekko_vars_attr(gekko_model, child.model)
        for compound in compounds:
            if ignore_compound(compound, ignored_nodes):
                compound.ignored = True
                GEKKOSolver.inject_nans(compound.model)
                continue
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, compound.model)

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
                    Var(value=value.VALUE.value[0], min=value.LOWER, max=value.UPPER),
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
        else:
            m.Obj(0)
        ignored_nodes = set()
        if optimization_problem is None:
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
        self.init_branches(branches)
        self.process_equations_nodes_childs(m, network, nodes, ignored_nodes)
        self.process_equations_branches(m, network, branches, ignored_nodes)
        self.process_equations_compounds(m, network, compounds, ignored_nodes)
        if optimization_problem is not None:
            self.process_oxf_components(m, network, optimization_problem)
        else:
            self.process_internal_oxf_components(m, network)
        try:
            m.options.COLDSTART = 1
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
            for constraint in compound.constraints:
                m.Equation(constraint(compound.model))
            equations = compound.model.equations(network)
            if equations is not None:
                _process_intermediate_eqs(m, compound, equations)
                m.Equations(_filter_intermediate_eqs(_as_iter(equations)))

    def process_equations_nodes_childs(self, m, network: Network, nodes, ignored_nodes):
        """
        No docstring provided.
        """
        for node in nodes:
            if ignore_node(node, network, ignored_nodes):
                continue
            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid
            for constraint in node.constraints:
                m.Equation(
                    constraint(
                        grid,
                        [
                            network.branch_by_id(branch_id).model
                            for branch_id in node.from_branch_ids
                        ],
                        [
                            network.branch_by_id(branch_id).model
                            for branch_id in node.to_branch_ids
                        ],
                        node_childs,
                    )
                )
            equations = _as_iter(
                node.model.equations(
                    grid,
                    [
                        network.branch_by_id(branch_id).model
                        for branch_id in node.from_branch_ids
                        if not ignore_branch(
                            network.branch_by_id(branch_id), network, ignored_nodes
                        )
                    ],
                    [
                        network.branch_by_id(branch_id).model
                        for branch_id in node.to_branch_ids
                        if not ignore_branch(
                            network.branch_by_id(branch_id), network, ignored_nodes
                        )
                    ],
                    [
                        child.model
                        for child in node_childs
                        if not ignore_child(child, ignored_nodes)
                    ],
                    sin_impl=m.sin,
                    cos_impl=m.cos,
                    if_impl=m.if2,
                    abs_impl=m.abs3,
                    max_impl=m.max2,
                    sign_impl=m.sign2,
                )
            )
            node_eqs = [eq for eq in equations if type(eq) is not bool or not eq]
            _process_intermediate_eqs(m, node.model, node_eqs)
            m.Equations(_filter_intermediate_eqs(node_eqs))
            for child in node_childs:
                if ignore_child(child, ignored_nodes):
                    continue
                child_eqs = _as_iter(child.model.equations(grid, node))
                _process_intermediate_eqs(m, child.model, child_eqs)
                m.Equations(_filter_intermediate_eqs(child_eqs))

    def init_branches(self, branches):
        """
        No docstring provided.
        """
        for branch in branches:
            branch.model.init(branch.grid)

    def process_equations_branches(self, m, network, branches, ignored_nodes):
        """
        No docstring provided.
        """
        for branch in branches:
            if ignore_branch(branch, network, ignored_nodes):
                continue
            grid = branch.grid
            for constraint in branch.constraints:
                m.Equation(
                    constraint(
                        branch.model,
                        grid,
                        network.node_by_id(branch.from_node_id).model,
                        network.node_by_id(branch.to_node_id).model,
                    )
                )
            branch_eqs = _as_iter(
                branch.model.equations(
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
                )
            )
            _process_intermediate_eqs(m, branch.model, branch_eqs)
            m.Equations(_filter_intermediate_eqs(branch_eqs))
