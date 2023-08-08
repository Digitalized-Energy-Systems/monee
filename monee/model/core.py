from typing import List, Dict
from abc import ABC, abstractmethod
import networkx as nx
import pandas
import copy

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

    def __mul__(self, other):
        return Var(value=self.value * other, max=self.max, min=self.min)


class Const:
    def __init__(self, value) -> None:
        self.value = value


class GenericModel(ABC):
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self._ext_data = kwargs

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
        return []


class ChildModel(GenericModel):
    def overwrite(self, node_model):
        # optional override
        pass

    @abstractmethod
    def equations(self, grid, node_model, **kwargs):
        pass


class Child:
    def __init__(self, child_id, model, constraints, name=None, active=True) -> None:
        self.id = child_id
        self.model = model
        self.constraints = constraints
        self.name = name
        self.active = active


class Compound:
    def __init__(
        self,
        compound_id,
        model: CompoundModel,
        constraints,
        connected_to,
        subcomponents,
        name=None,
        active=True,
    ) -> None:
        self.id = compound_id
        self.model = model
        self.constraints = [] if constraints is None else constraints
        self.connected_to = connected_to
        self.name = name
        self.active = active
        self.subcomponents = subcomponents

    def component_of_type(self, comp_type):
        return [
            component
            for component in self.subcomponents
            if type(component) == comp_type
        ]


class Node:
    def __init__(
        self,
        node_id,
        model,
        child_ids=None,
        constraints=None,
        grid=None,
        name=None,
        position=None,
        active=True,
    ) -> None:
        self.id = node_id
        self.model = model
        self.child_ids = [] if child_ids is None else child_ids
        self.constraints = [] if constraints is None else constraints
        self.grid = grid
        self.from_branch_ids = []
        self.to_branch_ids = []
        self.name = name
        self.position = position
        self.active = active

    def add_from_branch_id(self, branch_id):
        self.from_branch_ids.append(branch_id)

    def add_to_branch_id(self, branch_id):
        self.to_branch_ids.append(branch_id)


class Branch:
    def __init__(
        self,
        model,
        from_node_id,
        to_node_id,
        constraints=None,
        grid=None,
        name=None,
        active=True,
    ) -> None:
        self.id = None
        self.model = model
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.constraints = [] if constraints is None else constraints
        self.grid = grid
        self.name = name
        self.active = active


class Network:
    def __init__(self, model=None) -> None:
        self.default_grid_model = model

        self._network_internal = nx.MultiGraph()
        self._child_dict = {}
        self._compound_dict = {}
        self._constraints = []
        self._objectives = []
        self.__blacklist = []
        self.__collected_components = []
        self.__force_blacklist = False
        self.__collect_components = False

    @property
    def graph(self):
        return self._network_internal

    def _set_active(self, cls, id, active):
        if cls == Node:
            self.node_by_id(id).active = active
        elif cls == Branch:
            self.branch_by_id(id).active = active
        elif cls == Compound:
            compound: Compound = self.compound_by_id(id)
            for component in compound.subcomponents:
                component.active = active
            self.compound_by_id(id).active = active
        elif cls == Child:
            self.child_by_id(id).active = active

    def deactivate_by_id(self, cls, id):
        self._set_active(cls, id, False)

    def activate_by_id(self, cls, id):
        self._set_active(cls, id, True)

    def activate(self, component):
        self.activate_by_id(type(component), component.id)

    def deactivate(self, component):
        self.deactivate_by_id(type(component), component.id)

    def all_models(self):
        model_container_list = self.childs + self.compounds + self.branches + self.nodes
        return [model_container.model for model_container in model_container_list]

    def all_models_with_grid(self):
        model_container_list = self.childs + self.compounds + self.branches + self.nodes
        return [
            (
                model_container.model,
                (model_container.grid if hasattr(model_container, "grid") else None),
            )
            for model_container in model_container_list
        ]

    @property
    def constraints(self):
        return self._constraints

    @property
    def objectives(self):
        return self._objectives

    @property
    def compounds(self):
        return list(self._compound_dict.values())

    @property
    def childs(self):
        return list(self._child_dict.values())

    def has_child(self, child_id):
        return child_id in self._child_dict

    def remove_child(self, child_id):
        del self._child_dict[child_id]

    def remove_compound(self, compound_id):
        del self._compound_dict[compound_id]

    def child_by_id(self, child_id):
        return self._child_dict[child_id]

    def compound_by_id(self, compound_id):
        return self._compound_dict[compound_id]

    def compounds_by_type(self, cls):
        return [compound for compound in self.compounds if type(compound.model) == cls]

    def childs_by_ids(self, child_ids):
        return [self.child_by_id(child_id) for child_id in child_ids]

    def is_blacklisted(self, obj):
        return obj in self.__blacklist

    def has_node(self, node_id):
        return node_id in self._network_internal.nodes

    def has_branch(self, branch_id):
        return branch_id in self._network_internal.edges

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

    def __insert_to_container_if_collect_toggled(self, obj):
        if self.__collect_components:
            self.__collected_components.append(obj)

    def child(
        self,
        model,
        attach_to_node_id=None,
        constraints=None,
        overwrite_id=None,
        name=None,
    ):
        child_id = overwrite_id or (
            0 if len(self._child_dict) == 0 else max(self._child_dict.keys()) + 1
        )
        child = Child(child_id, model, constraints, name=name)
        self.__insert_to_blacklist_if_forced(child)
        self.__insert_to_container_if_collect_toggled(child)
        self._child_dict[child_id] = child
        if attach_to_node_id is not None:
            self.node_by_id(attach_to_node_id).child_ids.append(child_id)
        return child_id

    def child_to(self, model, node_id, constraints=None, overwrite_id=None, name=None):
        return self.child(
            model,
            attach_to_node_id=node_id,
            constraints=constraints,
            overwrite_id=overwrite_id,
            name=name,
        )

    def first_node(self):
        return min(self._network_internal)

    def node(
        self,
        model,
        child_ids=None,
        constraints=None,
        grid=None,
        overwrite_id=None,
        name=None,
        position=None,
    ):
        node_id = overwrite_id or (
            0 if len(self._network_internal) == 0 else max(self._network_internal) + 1
        )
        node = Node(
            node_id,
            model,
            child_ids,
            constraints,
            grid or self.default_grid_model,
            name=name,
            position=position,
        )
        self.__insert_to_blacklist_if_forced(node)
        self.__insert_to_container_if_collect_toggled(node)

        self._network_internal.add_node(node_id, internal_node=node)
        return node_id

    def branch(
        self, model, from_node_id, to_node_id, constraints=None, grid=None, name=None
    ):
        from_node = self.node_by_id(from_node_id)
        to_node = self.node_by_id(to_node_id)
        branch = Branch(
            model,
            from_node_id,
            to_node_id,
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
            name=name,
        )
        self.__insert_to_blacklist_if_forced(branch)
        self.__insert_to_container_if_collect_toggled(branch)
        branch_id = (
            from_node_id,
            to_node_id,
            self._network_internal.add_edge(
                from_node_id, to_node_id, internal_branch=branch
            ),
        )
        branch.id = branch_id
        to_node.add_to_branch_id(branch_id)
        from_node.add_from_branch_id(branch_id)
        return branch_id

    def compound(
        self,
        model: CompoundModel,
        constraints=None,
        overwrite_id=None,
        **connected_node_ids,
    ):
        compound_id = overwrite_id or (
            0 if len(self._compound_dict) == 0 else max(self._compound_dict.keys()) + 1
        )
        self.__force_blacklist = True
        self.__collect_components = True
        model.create(
            self,
            **{
                (k.replace("_id", "") if k.endswith("_id") else k): self.node_by_id(v)
                for (k, v) in connected_node_ids.items()
            },
        )
        self.__collect_components = False
        self.__force_blacklist = False
        compound = Compound(
            compound_id=compound_id,
            model=model,
            constraints=constraints,
            connected_to=connected_node_ids,
            subcomponents=self.__collected_components,
        )
        self._compound_dict[compound_id] = compound
        self.__collected_components = []
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
                input_value = "$VAR"
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

    def as_dataframe_dict_str(self):
        dataframes = self.as_dataframe_dict()
        result_str = ""
        for cls_str, dataframe in dataframes.items():
            result_str += cls_str
            result_str += "\n"
            result_str += dataframe.to_string()
            result_str += "\n"
            result_str += "\n"
        return result_str

    def copy(self):
        return copy.deepcopy(self)

    def clear_childs(self):
        self._child_dict = {}
        for node in self.nodes:
            node.child_ids = []


def _clean_up_compound(network: Network, compound):
    node_components = compound.component_of_type(Node)
    fully_intact = True
    for component in node_components:
        if not network.has_node(component.id):
            fully_intact = False
    child_components = compound.component_of_type(Child)
    for component in child_components:
        if not network.has_child(component.id):
            fully_intact = False
    branch_components = compound.component_of_type(Branch)
    for component in branch_components:
        if not network.has_branch(component.id):
            fully_intact = False
    compound_components = compound.component_of_type(Compound)
    for component in compound_components:
        compound_alive = _clean_up_compound(network, compound)
        if not compound_alive:
            fully_intact = False
    network.remove_compound(compound)
    return fully_intact


def to_spanning_tree(network: Network):
    return transform_network(network, nx.minimum_spanning_tree)


def transform_network(network: Network, graph_transform):
    network = network.copy()
    network._network_internal = graph_transform(network.graph)

    for child in list(network.childs):
        referenced = False
        for node in network.nodes:
            if child.id in node.child_ids:
                referenced = True
        if referenced:
            network.remove_child(child.id)

    for compound in list(network.compounds):
        _clean_up_compound(network, compound)

    return network
