import copy

import networkx as nx
import pandas

from .core import (
    EL_KEY,
    GAS_KEY,
    WATER_KEY,
    Branch,
    Child,
    Component,
    Compound,
    CompoundModel,
    Const,
    GenericModel,
    Intermediate,
    Node,
    Var,
)
from .grid import create_gas_grid, create_power_grid, create_water_grid


class Network:
    def __init__(
        self,
        active_grid=None,
        el_model=create_power_grid("power"),
        water_model=create_water_grid("water"),
        gas_model=create_gas_grid("gas"),
    ) -> None:
        self._default_grid_models = {
            EL_KEY: el_model,
            WATER_KEY: water_model,
            GAS_KEY: gas_model,
        }

        self._network_internal = nx.MultiGraph()
        self._child_dict = {}
        self._compound_dict = {}
        self._constraints = []
        self._objectives = []
        self.__blacklist = []
        self.__collected_components = []
        self.__force_blacklist = False
        self.__collect_components = False
        self.__current_grid = active_grid

    def set_default_grid(self, key, grid):
        self._default_grid_models[key] = grid

    def activate_grid(self, grid):
        self.__current_grid = grid

    @property
    def graph(self):
        return self._network_internal

    def _set_active(self, cls, id, active):
        if cls == Node:
            self.node_by_id(id).active = active
        elif cls == Branch:
            branch = self.branch_by_id(id)
            if "active" in branch.model.vars:
                branch.model.active = active
            else:
                branch.active = active
        elif cls == Compound:
            compound: Compound = self.compound_by_id(id)
            if hasattr(compound.model, "set_active"):
                compound.model.set_active(active)
            else:
                for component in compound.subcomponents:
                    self._set_active(type(component), component.id, active)
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
        return [model_container.model for model_container in self.all_components()]

    def all_components(self):
        return self.childs + self.compounds + self.branches + self.nodes

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
    def compounds(self) -> list[Compound]:
        return list(self._compound_dict.values())

    @property
    def childs(self) -> list[Child]:
        return list(self._child_dict.values())

    @property
    def cps(self) -> list[GenericModel]:
        return [comp for comp in self.all_components() if comp.model.is_cp()]

    def has_child(self, child_id):
        return child_id in self._child_dict

    def remove_child(self, child_id):
        del self._child_dict[child_id]

    def compound_of_node(self, node_id):
        for compound in self.compounds:
            for subcomponent in compound.subcomponents:
                if isinstance(subcomponent, Node):
                    if subcomponent.id == node_id:
                        return compound
        return None

    def remove_node(self, node_id):
        self._network_internal.remove_node(node_id)

    def remove_branch(self, branch_id):
        branch: Branch = self.branch_by_id(branch_id)
        self.remove_branch_between(branch.from_node_id, branch.to_node_id)

    def remove_compound(self, compound_id):
        compound: Compound = self.compound_by_id(compound_id)
        del self._compound_dict[compound_id]
        for subcomponent in compound.subcomponents:
            if isinstance(subcomponent, Node):
                self.remove_node(subcomponent.id)
            if isinstance(subcomponent, Branch):
                if self.has_branch(subcomponent.id):
                    self.remove_branch(subcomponent.id)

    def remove_branch_between(self, node_one, node_two, key=0):
        self._network_internal.remove_edge(node_one, node_two, key)
        self.node_by_id(node_one).remove_branch((node_one, node_two, key))
        self.node_by_id(node_two).remove_branch((node_one, node_two, key))

    def move_branch(self, branch_id, new_from_id, new_to_id):
        branch: Branch = self.branch_by_id(branch_id)
        self.remove_branch_between(branch_id[0], branch_id[1], key=branch_id[2])
        return self.branch(
            branch.model,
            new_from_id,
            new_to_id,
            constraints=branch.constraints,
            grid=branch.grid,
            name=branch.name,
        )

    def child_by_id(self, child_id):
        return self._child_dict[child_id]

    def childs_by_type(self, cls):
        return [child for child in self.childs if type(child.model) is cls]

    def compound_by_id(self, compound_id):
        return self._compound_dict[compound_id]

    def compounds_by_type(self, cls):
        return [compound for compound in self.compounds if type(compound.model) is cls]

    def childs_by_ids(self, child_ids) -> list[Child]:
        return [self.child_by_id(child_id) for child_id in child_ids]

    def branches_by_ids(self, branch_ids) -> list[Branch]:
        return [self.branch_by_id(branch_id) for branch_id in branch_ids]

    def is_blacklisted(self, obj):
        return obj in self.__blacklist

    def has_node(self, node_id):
        return node_id in self._network_internal.nodes

    def has_branch(self, branch_id):
        return branch_id in self._network_internal.edges

    def get_branch_between(self, node_id_one, node_id_two):
        return self._network_internal.has_edge(node_id_one, node_id_two)

    def has_branch_between(self, node_id_one, node_id_two):
        return self._network_internal.has_edge(node_id_one, node_id_two)

    def compounds_connected_to(self, node_id) -> list[Component]:
        return [
            compound
            for compound in self.compounds
            if node_id in compound.connected_to.values()
        ]

    def compound_of(self, subcomponent_component_id) -> list[Component]:
        compounds = [
            compound
            for compound in self.compounds
            if subcomponent_component_id in [sc.id for sc in compound.subcomponents]
        ]
        if len(compounds) == 0:
            return None
        return compounds[0]

    def components_connected_to(self, node_id) -> list[Component]:
        node = self.node_by_id(node_id)
        return (
            self.childs_by_ids(node.child_ids)
            + self.compounds_connected_to(node_id)
            + self.branches_by_ids(node.to_branch_ids)
            + self.branches_by_ids(node.from_branch_ids)
        )

    def branches_connected_to(self, node_id) -> list[Branch]:
        node = self.node_by_id(node_id)
        return self.branches_by_ids(node.to_branch_ids) + self.branches_by_ids(
            node.from_branch_ids
        )

    @property
    def nodes(self) -> list[Node]:
        return [
            self._network_internal.nodes[node]["internal_node"]
            for node in self._network_internal.nodes
        ]

    @property
    def branches(self) -> list[Branch]:
        return [
            self._network_internal.edges[edge]["internal_branch"]
            for edge in self._network_internal.edges
        ]

    def node_by_id(self, node_id) -> Node:
        if node_id not in self._network_internal.nodes:
            raise ValueError(
                f"The node id '{node_id}' is not valid. The valid ids are {self._network_internal.nodes.keys()}"
            )
        return self._network_internal.nodes[node_id]["internal_node"]

    def branch_by_id(self, branch_id):
        if branch_id not in self._network_internal.edges:
            raise ValueError(f"The branch id '{branch_id}' is not valid.")
        return self._network_internal.edges[branch_id]["internal_branch"]

    def branches_by_type(self, cls):
        return [branch for branch in self.branches if type(branch.model) is cls]

    def __insert_to_blacklist_if_forced(self, obj):
        if self.__force_blacklist:
            self.__blacklist.append(obj)

    def __insert_to_container_if_collect_toggled(self, obj):
        if self.__collect_components:
            self.__collected_components.append(obj)

    def node_by_id_or_create(self, node_id, auto_node_creator, auto_grid_key):
        if not self.has_node(node_id):
            return self.node_by_id(
                self.node(auto_node_creator(), grid=auto_grid_key, overwrite_id=node_id)
            )
        return self.node_by_id(node_id)

    def child(
        self,
        model,
        attach_to_node_id=None,
        constraints=None,
        overwrite_id=None,
        name=None,
        auto_node_creator=None,
        auto_grid_key=None,
    ):
        child_id = overwrite_id or (
            0 if len(self._child_dict) == 0 else max(self._child_dict.keys()) + 1
        )
        child = Child(
            child_id,
            model,
            constraints,
            name=name,
            independent=not self.__collect_components,
        )
        self.__insert_to_blacklist_if_forced(child)
        self.__insert_to_container_if_collect_toggled(child)
        self._child_dict[child_id] = child
        if attach_to_node_id is not None:
            child.node_id = attach_to_node_id
            attaching_node = self.node_by_id_or_create(
                attach_to_node_id, auto_node_creator, auto_grid_key
            )
            attaching_node.child_ids.append(child_id)
            child.grid = attaching_node.grid
        return child_id

    def child_to(
        self,
        model,
        node_id,
        constraints=None,
        overwrite_id=None,
        name=None,
        auto_node_creator=None,
        auto_grid_key=None,
    ):
        return self.child(
            model,
            attach_to_node_id=node_id,
            constraints=constraints,
            overwrite_id=overwrite_id,
            name=name,
            auto_node_creator=auto_node_creator,
            auto_grid_key=auto_grid_key,
        )

    def first_node(self):
        return min(self._network_internal)

    def _or_default(self, grid_or_name):
        if isinstance(grid_or_name, str):
            return self._default_grid_models[grid_or_name]
        if grid_or_name is None:
            if self.__current_grid is None:
                raise ValueError(
                    "No active grid and no grid was provided. Please provide a grid by using the argument grid= or use activate_grid(grid) to activate a grid for the whole Network object."
                )
            if isinstance(self.__current_grid, str):
                return self._default_grid_models[self.__current_grid]
            return self.__current_grid
        return grid_or_name

    def node(
        self,
        model,
        grid=None,
        child_ids=None,
        constraints=None,
        overwrite_id=None,
        name=None,
        position=None,
    ):
        node_id = (
            0 if len(self._network_internal) == 0 else max(self._network_internal) + 1
        )
        if overwrite_id is not None:
            node_id = overwrite_id

        node = Node(
            node_id,
            model,
            child_ids,
            constraints,
            self._or_default(grid),
            name=name,
            position=position,
            independent=not self.__collect_components,
        )
        if child_ids is not None:
            for child_id in child_ids:
                child = self.child_by_id(child_id)
                child.grid = node.grid
                child.node_id = node_id
        self.__insert_to_blacklist_if_forced(node)
        self.__insert_to_container_if_collect_toggled(node)

        self._network_internal.add_node(node_id, internal_node=node)
        return node_id

    def branch(
        self,
        model,
        from_node_id,
        to_node_id,
        constraints=None,
        grid=None,
        name=None,
        auto_node_creator=None,
        auto_grid_key=None,
        **kwargs,
    ):
        from_node = self.node_by_id_or_create(
            from_node_id,
            auto_node_creator=auto_node_creator,
            auto_grid_key=auto_grid_key,
        )
        to_node = self.node_by_id_or_create(
            to_node_id, auto_node_creator=auto_node_creator, auto_grid_key=auto_grid_key
        )
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
            independent=not self.__collect_components,
            **kwargs,
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
    def _model_dict_to_input(container):
        model_dict = container.model.__dict__
        input_dict = {"active": container.active, 
                      "id": container.id, 
                      "independent": container.independent, 
                      "ignored": container.ignored}
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
                Network._model_dict_to_input(container)
            )
        dataframe_dict = {}
        for result_type, dict_list in input_dict_list_dict.items():
            dataframe_dict[result_type] = pandas.DataFrame(dict_list)
        return dataframe_dict

    @staticmethod
    def _model_dict_to_results(container):
        model_dict = container.model.vars
        result_dict = {"active": container.active, 
                      "id": container.id, 
                      "independent": container.independent, 
                      "ignored": container.ignored}
        for k, v in model_dict.items():
            result_value = v
            if isinstance(v, Var | Const | Intermediate):
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
                Network._model_dict_to_results(container)
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

    def statistics(self):
        type_to_number = {}
        model_containers = self.nodes + self.childs + self.branches + self.compounds
        for container in model_containers:
            if not container.independent:
                continue
            model_type = type(container.model)
            if model_type in type_to_number:
                type_to_number[model_type] += 1
            else:
                type_to_number[model_type] = 1
        return type_to_number

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


def _add_tuple(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _div_tuple(a, div):
    return tuple([a[i] / div for i in range(len(a))])


def calc_coordinates(network: Network, component: Component):
    if type(component) is Node:
        return component.position
    elif type(component) is Branch:
        node_start = network.node_by_id(component.from_node_id)
        node_end = network.node_by_id(component.from_node_id)
        return tuple(
            [
                (node_start.position[i] + node_end.position[i]) / 2
                for i in range(len(node_start.position))
            ]
        )
    elif type(component) is Child:
        return network.node_by_id(component.node_id).position
    elif type(component) is Compound:
        position = (0, 0)
        for connected_node_id in component.connected_to.values():
            node = network.node_by_id(connected_node_id)
            position = _add_tuple(position, node.position)
        return _div_tuple(position, len(component.connected_to))
    raise Exception(f"This should not happen! The component {component} is unknown.")
