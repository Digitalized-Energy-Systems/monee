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


class MultiGridBranchModel(BranchModel):
    @abstractmethod
    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        pass


class CompoundModel(GenericModel):
    @abstractmethod
    def create(self, network):
        pass

    def equations(self, network, **kwargs):
        # optional override
        pass


class ChildModel(GenericModel):
    def overwrite(self, node_model):
        # optional override
        pass

    @abstractmethod
    def equations(self, grid, node_model, **kwargs):
        pass


class Child:
    def __init__(self, child_id, model, constraints) -> None:
        self.id = child_id
        self.model = model
        self.constraints = constraints


class Compound:
    def __init__(
        self,
        compound_id,
        model: CompoundModel,
        constraints,
        connected_to,
    ) -> None:
        self.id = compound_id
        self.model = model
        self.constraints = constraints
        self.connected_to = connected_to


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
        self._compounds = []
        self._constraints = []
        self._objectives = []
        self.__blacklist = []
        self.__force_blacklist = False

    @property
    def compounds(self):
        return self._compounds

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

    def is_blacklisted(self, obj):
        return obj in self.__blacklist

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
        if node_id not in self._network_internal.nodes:
            raise ValueError(
                f"The node id '{node_id}' is not valid. The valid ids are {self._network_internal.nodes.keys()}"
            )
        return self._network_internal.nodes[node_id]["internal_node"]

    def branch_by_id(self, branch_id):
        if branch_id not in self._network_internal.edges:
            raise ValueError(f"The node id '{branch_id}' is not valid.")
        return self._network_internal.edges[branch_id]["internal_branch"]

    def __insert_to_blacklist_if_forced(self, obj):
        if self.__force_blacklist:
            self.__blacklist.append(obj)

    def child(self, model, constraints=None, overwrite_id=None):
        child_id = overwrite_id or len(self.childs)
        child = Child(child_id, model, constraints)
        self.__insert_to_blacklist_if_forced(child)
        self.childs.append(child)
        return child_id

    def node(
        self, model, child_ids=None, constraints=None, grid=None, overwrite_id=None
    ):
        node_id = overwrite_id or len(self._network_internal)
        node = Node(
            node_id, model, child_ids, constraints, grid or self.default_grid_model
        )
        self.__insert_to_blacklist_if_forced(node)

        self._network_internal.add_node(node_id, internal_node=node)
        return node_id

    def branch(self, model, from_node_id, to_node_id, constraints=None, grid=None):
        from_node = self.node_by_id(from_node_id)
        to_node = self.node_by_id(to_node_id)
        branch = Branch(
            model,
            from_node,
            to_node,
            constraints,
            grid
            or (
                from_node.grid
                if from_node.grid == to_node.grid
                else {
                    type(from_node.grid): from_node.grid,
                    type(to_node.grid): to_node.grid,
                }
            ),
        )
        self.__insert_to_blacklist_if_forced(branch)
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

    def compound(self, model: CompoundModel, constraints=None, **connected_nodes):
        compound_id = len(self._compounds)
        compound = Compound(
            compound_id=compound_id,
            model=model,
            constraints=constraints,
            connected_to=connected_nodes,
        )
        self._compounds.append(compound)
        self.__force_blacklist = True
        model.create(
            self, **{k: self.node_by_id(v) for (k, v) in connected_nodes.items()}
        )
        self.__force_blacklist = False
        return compound_id

    def constraint(self, constraint_equation):
        self._constraints.append(constraint_equation)

    def objective(self, objective_function):
        self._objectives.append(objective_function)

    @staticmethod
    def _model_dict_to_input(model_dict):
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
                Network._model_dict_to_input(container.model.__dict__)
            )
        dataframe_dict = {}
        for result_type, dict_list in input_dict_list_dict.items():
            dataframe_dict[result_type] = pandas.DataFrame(dict_list)
        return dataframe_dict

    @staticmethod
    def _model_dict_to_results(model_dict):
        result_dict = {}
        for k, v in model_dict.items():
            result_value = v
            if isinstance(v, (Var, Const)):
                result_value = v.value
            result_dict[k] = result_value
        return result_dict

    def as_result_dataframe_dict(self):
        result_dict_list_dict = {}
        model_containers = self.nodes + self.childs + self.branches
        for container in model_containers:
            model_type_name = type(container.model).__name__
            if model_type_name not in result_dict_list_dict:
                result_dict_list_dict[model_type_name] = []
            result_dict_list_dict[model_type_name].append(
                Network._model_dict_to_results(container.model.vars)
            )
        dataframe_dict = {}
        for result_type, dict_list in result_dict_list_dict.items():
            dataframe_dict[result_type] = pandas.DataFrame(dict_list)
        return dataframe_dict
