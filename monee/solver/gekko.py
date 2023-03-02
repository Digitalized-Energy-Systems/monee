from typing import List, Dict
from copy import deepcopy
from dataclasses import dataclass

from gekko import GEKKO
from gekko.gk_variable import GKVariable
from gekko.gk_operators import GK_Operators
from monee.model.core import Network, Var, GenericModel, Node, Branch, Const, Compound
import pandas


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
    return possible_iter if hasattr(possible_iter, "__iter__") else [possible_iter]


class GEKKOSolver:
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

    def solve(self, input_network: Network):
        m = GEKKO()
        m.options.SOLVER = 1
        m.solver_options = [
            "minlp_maximum_iterations 500",
            "minlp_max_iter_with_int_sol 10",
            "minlp_as_nlp 0",
            "nlp_maximum_iterations 50",
            "minlp_branch_method 1",
            "minlp_integer_tol 0.05",
            "minlp_gap_tol 0.01",
        ]

        network = deepcopy(input_network)
        nodes = network.nodes

        # prepare for overwritting default node behaviors with
        # childs
        for node in nodes:
            for child in network.childs_by_ids(node.child_ids):
                child.model.overwrite(node.model)

        branches = network.branches
        compounds = network.compounds

        GEKKOSolver.inject_gekko_vars(m, nodes, branches, compounds, network)

        for branch in branches:
            grid = branch.grid or network.default_grid_model
            for constraint in branch.constraints:
                m.Equation(
                    constraint(
                        grid,
                        branch.from_node.model,
                        branch.to_node.model,
                    )
                )
            m.Equations(
                _as_iter(
                    branch.model.equations(
                        grid,
                        branch.from_node.model,
                        branch.to_node.model,
                        sin_impl=m.sin,
                        cos_impl=m.cos,
                    )
                )
            )
        for node in nodes:
            node_childs = network.childs_by_ids(node.child_ids)
            grid = node.grid or network.default_grid_model
            for constraint in node.constraints:
                m.Equation(
                    constraint(
                        grid,
                        [branch.model for branch in node.from_branches],
                        [branch.model for branch in node.to_branches],
                        node_childs,
                    )
                )
            m.Equations(
                _as_iter(
                    node.model.equations(
                        grid,
                        [branch.model for branch in node.from_branches],
                        [branch.model for branch in node.to_branches],
                        [child.model for child in node_childs],
                    )
                )
            )
            for child in node_childs:
                m.Equations(_as_iter(child.model.equations(grid, node)))

        for compound in compounds:
            for constraint in compound.constraints:
                m.Equation(constraint(compound.model))
            m.Equations(_as_iter(compound.model.equations(network)))

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
        m.solve()

        GEKKOSolver.withdraw_gekko_vars(nodes, branches, compounds, network)

        solver_result = SolverResult(network, network.as_result_dataframe_dict())
        return solver_result
