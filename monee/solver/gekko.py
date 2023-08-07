from typing import List, Dict
from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Operators
from monee.model.core import Network, Var, GenericModel, Node, Branch, Const, Compound
from monee.problem.core import OptimizationProblem
import pandas

DEFAULT_SOLVER_OPTIONS = [
    "minlp_maximum_iterations 500",
    "minlp_max_iter_with_int_sol 10",
    "minlp_as_nlp 0",
    "nlp_maximum_iterations 50",
    "minlp_branch_method 1",
    "minlp_integer_tol 0.05",
    "minlp_gap_tol 0.01",
]


@dataclass
class SolverResult:
    network: Network
    dataframes: Dict[str, pandas.DataFrame]

    def __str__(self) -> str:
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
    if possible_iter is None:
        raise Exception(f"None as result for 'equations' is not allowed!")

    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


class GEKKOSolver:
    @staticmethod
    def inject_gekko_vars_attr(gekko: GEKKO, target: GenericModel):
        for key, value in target.__dict__.items():
            if type(value) == Var:
                setattr(
                    target,
                    key,
                    gekko.Var(value.value, lb=value.min, ub=value.max),
                )
            if type(value) == Const:
                setattr(
                    target,
                    key,
                    gekko.Const(value.value),
                )

    @staticmethod
    def inject_gekko_vars(
        gekko_model: GEKKO,
        nodes: List[Node],
        branches: List[Branch],
        compounds: List[Compound],
        network: Network,
    ):
        for branch in branches:
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, branch.model)
        for node in nodes:
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, node.model)
            for child in network.childs_by_ids(node.child_ids):
                GEKKOSolver.inject_gekko_vars_attr(gekko_model, child.model)

        for compound in compounds:
            GEKKOSolver.inject_gekko_vars_attr(gekko_model, compound.model)

    @staticmethod
    def withdraw_gekko_vars_attr(target: GenericModel):
        for key, value in target.__dict__.items():
            if type(value) == GKVariable:
                setattr(
                    target,
                    key,
                    Var(value=value.VALUE.value[0], min=value.LOWER, max=value.UPPER),
                )
            if type(value) == GK_Operators:
                setattr(
                    target,
                    key,
                    Const(value.VALUE.value),
                )

    @staticmethod
    def withdraw_gekko_vars(nodes, branches, compounds, network):
        for branch in branches:
            GEKKOSolver.withdraw_gekko_vars_attr(branch.model)
        for node in nodes:
            GEKKOSolver.withdraw_gekko_vars_attr(node.model)
            for child in network.childs_by_ids(node.child_ids):
                GEKKOSolver.withdraw_gekko_vars_attr(child.model)

        for compound in compounds:
            GEKKOSolver.withdraw_gekko_vars_attr(compound.model)

    def solve(
        self, input_network: Network, optimization_problem: OptimizationProblem = None
    ):
        # ensure compatibility of gekko models with own models
        # for creating objectives and constraitns
        GKVariable.max = property(lambda self: self.UPPER)
        GKVariable.min = property(lambda self: self.LOWER)

        m = GEKKO(remote=False)
        m.open_folder()
        m.options.SOLVER = 1
        m.solver_options = DEFAULT_SOLVER_OPTIONS

        network = input_network.copy()
        nodes = network.nodes

        # prepare for overwritting default node behaviors with
        # childs
        for node in nodes:
            for child in network.childs_by_ids(node.child_ids):
                if child.active:
                    child.model.overwrite(node.model)

        branches = network.branches
        compounds = network.compounds

        if optimization_problem is not None:
            optimization_problem._apply(network)
        else:
            m.Obj(0)

        GEKKOSolver.inject_gekko_vars(m, nodes, branches, compounds, network)

        self.process_equations_branches(m, network, branches)
        self.process_equations_nodes_childs(m, network, nodes)
        self.process_equations_compounds(m, network, compounds)

        if optimization_problem is not None:
            self.process_oxf_components(m, network, optimization_problem)
        else:
            self.process_internal_oxf_components(m, network)

        try:
            m.options.COLDSTART = 0
            m.solve(disp=True)
        except:
            m.options.COLDSTART = 2
            m.solve(disp=True)
            print("Solver not converged. Using Presolve Solution.")

        GEKKOSolver.withdraw_gekko_vars(nodes, branches, compounds, network)

        solver_result = SolverResult(network, network.as_result_dataframe_dict())
        return solver_result

    def process_internal_oxf_components(self, m, network):
        for constraint in network.constraints:
            m.Equation(constraint(network))

        obj = None
        for objective in network.objectives:
            if obj != None:
                obj = obj + objective(network)
            else:
                obj = objective(network)
        if obj is not None:
            m.Obj(obj)

    def process_oxf_components(
        self, m, network: Network, optimization_problem: OptimizationProblem
    ):
        if (
            optimization_problem.constraints is not None
            and not optimization_problem.constraints.empty
        ):
            m.Equations(optimization_problem.constraints.all(network))

        obj = None
        for objective in optimization_problem.objectives.all(network):
            if obj != None:
                obj = obj + objective
            else:
                obj = objective
        if obj is not None:
            m.Obj(obj)

    def process_equations_compounds(self, m, network, compounds):
        for compound in compounds:
            if not compound.active:
                continue
            for constraint in compound.constraints:
                m.Equation(constraint(compound.model))
            equations = compound.model.equations(network)
            if equations is not None:
                m.Equations(_as_iter(equations))

    def process_equations_nodes_childs(self, m, network: Network, nodes):
        for node in nodes:
            if not node.active:
                continue
            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid or network.default_grid_model
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
            m.Equations(
                _as_iter(
                    node.model.equations(
                        grid,
                        [
                            network.branch_by_id(branch_id).model
                            for branch_id in node.from_branch_ids
                        ],
                        [
                            network.branch_by_id(branch_id).model
                            for branch_id in node.to_branch_ids
                        ],
                        [child.model for child in node_childs],
                    )
                )
            )
            for child in node_childs:
                if not child.active:
                    continue
                m.Equations(_as_iter(child.model.equations(grid, node)))

    def process_equations_branches(self, m, network, branches):
        for branch in branches:
            if not branch.active:
                continue

            grid = branch.grid or network.default_grid_model
            for constraint in branch.constraints:
                m.Equation(
                    constraint(
                        grid,
                        network.node_by_id(branch.from_node_id).model,
                        network.node_by_id(branch.to_node_id).model,
                    )
                )
            m.Equations(
                _as_iter(
                    branch.model.equations(
                        grid,
                        network.node_by_id(branch.from_node_id).model,
                        network.node_by_id(branch.to_node_id).model,
                        sin_impl=m.sin,
                        cos_impl=m.cos,
                        if_impl=m.if3,
                        abs_impl=m.abs3,
                    )
                )
            )
