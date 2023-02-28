from typing import List
from abc import ABC, abstractmethod
import networkx as nx
import pandas

component_list = []


def model(cls):
    component_list.append(cls)
    return cls


class Var:
    def __init__(self, value, max=None, min=None) -> None:
        self.value = value
        self.max = max
        self.min = min
    
    def __neg__(self):
        return Var(value=-self.value, max=self.max, min=self.min)


class Const:
    def __init__(self, value) -> None:
        self.value = value


class GenericModel(ABC):
    @property
    def vars(self):
        return {k: v for (k, v) in self.__dict__.items() if k[0] != "_"}


class NodeModel(GenericModel):
    @abstractmethod
    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        pass


class BranchModel(GenericModel):
    @abstractmethod
    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        pass


class MultiGridBranchModel(GenericModel):
    @abstractmethod
    def equations(self, grids, from_models, to_models, **kwargs):
        pass


class ChildModel(GenericModel):
    def overwrite(self, node_model):
        pass

    @abstractmethod
    def equations(self, grid, node_model, **kwargs):
        pass


class Child:
    def __init__(self, child_id, model, constraints) -> None:
        self.id = child_id
        self.model = model
        self.constraints = constraints


class Node:
    def __init__(
        self, node_id, model, child_ids=None, constraints=None, grid=None
    ) -> None:
        self.id = node_id
        self.model = model
        self.child_ids = [] if child_ids is None else child_ids
        self.constraints = [] if constraints is None else constraints
        self.grid = grid
        self.from_branches = []
        self.to_branches = []

    def add_from_branch(self, branch):
        self.from_branches.append(branch)

    def add_to_branch(self, branch):
        self.to_branches.append(branch)


class Branch:
    def __init__(self, model, from_node, to_node, constraints=None, grid=None) -> None:
        self.id = None
        self.model = model
        self.from_node = from_node
        self.to_node = to_node
        self.constraints = [] if constraints is None else constraints
        self.grid = grid


class Network:
    def __init__(self, model) -> None:
        self.default_grid_model = model

        self._network_internal = nx.MultiGraph()
        self._childs = []
        self._constraints = []
        self._objectives = []

    @property
    def constraints(self):
        return self._constraints

    @property
    def objectives(self):
        return self._objectives

    @property
    def childs(self):
        return self._childs

    def childs_by_ids(self, child_ids):
        return [self._childs[child_id] for child_id in child_ids]

    @property
    def nodes(self) -> List[Node]:
        return [
            self._network_internal.nodes[node]["internal_node"]
            for node in self._network_internal.nodes
        ]

    @property
    def branches(self) -> List[Branch]:
        return [
            self._network_internal.edges[edge]["internal_branch"]
            for edge in self._network_internal.edges
        ]

    def node_by_id(self, node_id):
        return self._network_internal.nodes[node_id]["internal_node"]

    def branch_by_id(self, branch_id):
        return self._network_internal.edges[branch_id]["internal_branch"]

    def child(self, model, constraints=None, overwrite_id=None):
        child_id = overwrite_id or len(self.childs)
        child = Child(child_id, model, constraints)
        self.childs.append(child)
        return child_id

    def node(
        self, model, child_ids=None, constraints=None, grid=None, overwrite_id=None
    ):
        node_id = overwrite_id or len(self._network_internal)
        node = Node(
            node_id, model, child_ids, constraints, grid or self.default_grid_model
        )

        self._network_internal.add_node(node_id, internal_node=node)
        return node_id

    def branch(self, model, from_node_id, to_node_id, constraints=None, grid=None):
        from_node = self.node_by_id(from_node_id)
        to_node = self.node_by_id(to_node_id)
        branch = Branch(
            model, from_node, to_node, constraints, grid or self.default_grid_model
        )
        branch_id = (
            from_node_id,
            to_node_id,
            self._network_internal.add_edge(
                from_node_id, to_node_id, internal_branch=branch
            ),
        )
        branch.id = branch_id
        to_node.add_to_branch(branch)
        from_node.add_from_branch(branch)
        return branch_id

    def constraint(self, constraint_equation):
        self._constraints.append(constraint_equation)

    def objective(self, objective_function):
        self._objectives.append(objective_function)

    def model_dict_to_input(self, model_dict):
        input_dict = {}
        for k, v in model_dict.items():
            input_value = v
            if isinstance(v, (Var)):
                continue
            if isinstance(v, (Const)):
                input_value = v.value
            input_dict[k] = input_value
        return input_dict

    def as_dataframe_dict(self):
        input_dict_list_dict = {}
        model_containers = self.nodes + self.childs + self.branches
        for container in model_containers:
            model_type_name = type(container.model).__name__
            if model_type_name not in input_dict_list_dict:
                input_dict_list_dict[model_type_name] = []
            input_dict_list_dict[model_type_name].append(
                self.model_dict_to_input(container.model.__dict__)
            )
        dataframe_dict = {}
        for result_type, dict_list in input_dict_list_dict.items():
            dataframe_dict[result_type] = pandas.DataFrame(dict_list)
        return dataframe_dict