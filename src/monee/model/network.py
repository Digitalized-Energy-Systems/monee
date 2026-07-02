from __future__ import annotations

import copy
from collections.abc import Iterator
from typing import TYPE_CHECKING

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
    PostProcess,
    Var,
)
from .grid import create_gas_grid, create_power_grid, create_water_grid

if TYPE_CHECKING:
    from .extension.core import NetworkAspect

    # Imported only for typing: at runtime ``apply_formulation`` resolves the
    # spec lazily via ``formulation.registry.resolve_formulation`` (a local
    # import), so the model package no longer eagerly triggers the
    # formulation -> branch/node import chain at load time.
    from .formulation import (  # noqa: F401
        Formulation,
        NetworkFormulation,
    )


class Network:
    def __init__(
        self,
        active_grid=None,
        el_model=None,
        water_model=None,
        gas_model=None,
    ) -> None:
        self._default_grid_models = {
            EL_KEY: el_model or create_power_grid("power"),
            WATER_KEY: water_model or create_water_grid("water"),
            GAS_KEY: gas_model or create_gas_grid("gas"),
        }
        self._network_internal = nx.MultiGraph()
        self._child_dict = {}
        self._compound_dict = {}
        self._constraints = []
        self._objectives = []
        self._extensions: list[NetworkAspect] = []
        # Identity map id(obj) -> obj: O(1) membership without relying on
        # component __eq__/__hash__ (components are compared by identity).
        self.__blacklist = {}
        # One frame per in-flight compound() call; nesting pushes/pops frames
        # so an inner compound collects into its own frame.
        self.__collection_stack = []
        self.__current_grid = active_grid
        # Declarative network-level formulation choice. No default is seeded
        # here: components without an explicit choice fall back to
        # DEFAULT_SIMULATION_FORMULATION when the solver attaches formulations
        # (see monee.model.formulation.registry.attach_formulations).
        self.__default_formulation: dict[tuple[type, type], Formulation] = {}

    def apply_formulation(self, network_formulation):
        """Record *network_formulation* as the network-level default.

        Accepts the same spec as the solver's ``formulation`` argument
        (:func:`~monee.model.formulation.registry.resolve_formulation`): a
        registry key string (``"smooth_nlp"``), a :class:`NetworkFormulation`,
        or a sequence of either (merged left to right).

        Side-effect free: only the network's formulation map is updated -
        components and their models are untouched. The choice materialises
        when a solver runs ``attach_formulations`` on its solve-time copy. A
        ``formulation`` argument passed to the solver overrides this choice;
        per-component formulations passed to the builder methods override
        both. Repeated calls merge: later registrations win per type key.
        """
        from .formulation.registry import resolve_formulation

        network_formulation = resolve_formulation(network_formulation)
        if network_formulation is None:
            return
        for type_or_tuple, formulation in network_formulation.items():
            tc, tg = None, None
            if isinstance(type_or_tuple, tuple):
                tc, tg = type_or_tuple
            else:
                tc = type_or_tuple

            self.__default_formulation[(tc, tg)] = formulation

    def lookup_formulation(self, model, grid) -> Formulation | None:
        """The network-level formulation for *model* (and *grid*) accumulated
        from ``apply_formulation`` calls, or None. Last matching registration
        wins, mirroring :meth:`NetworkFormulation.lookup`."""
        found = None
        for (tc, tg), formulation in self.__default_formulation.items():
            if isinstance(model, tc) and (tg is None or type(grid) is tg):
                found = formulation
        return found

    def set_default_grid(self, key, grid):
        self._default_grid_models[key] = grid

    def activate_grid(self, grid):
        self.__current_grid = grid

    @property
    def grids(self):
        # Coupling control nodes carry list-valued grids - flatten them.
        seen = set()
        grids = []
        for node in self.nodes:
            node_grids = node.grid if isinstance(node.grid, list) else [node.grid]
            for grid in node_grids:
                if grid not in seen:
                    seen.add(grid)
                    grids.append(grid)
        return grids

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
            # Propagate to subcomponents and the compound's own ``active`` so
            # ignore_*/inject_vars see a fully deactivated compound; model
            # set_active alone wouldn't deactivate subcomponent children.
            for component in compound.subcomponents:
                self._set_active(type(component), component.id, active)
            compound.active = active
            if hasattr(compound.model, "set_active"):
                compound.model.set_active(active)
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

    def iter_all_components(self) -> Iterator[Component]:
        """Yield every component once in canonical traversal order: each node
        immediately followed by its childs, then all branches, then all
        compounds. Mirrors the ad-hoc traversal repeated across the simulation
        layer so iteration order and child handling stay identical."""
        for node in self.nodes:
            yield node
            yield from self.childs_by_ids(node.child_ids)
        yield from self.branches
        yield from self.compounds

    def all_models_with_grid(self):
        model_container_list = self.childs + self.compounds + self.branches + self.nodes
        return [
            (
                model_container.model,
                model_container.grid if hasattr(model_container, "grid") else None,
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
    def extensions(self) -> list[NetworkAspect]:
        """Solver-agnostic network-level extensions."""
        return self._extensions

    def add_extension(self, ext: NetworkAspect) -> None:
        """Register a NetworkAspect extension on this network."""
        self._extensions.append(ext)

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
        # Also drop the parent node's reference; otherwise childs_by_ids
        # raises KeyError walking node.child_ids.
        child = self._child_dict.pop(child_id)
        node_id = getattr(child, "node_id", None)
        if node_id is not None and node_id in self._network_internal.nodes:
            try:
                node = self.node_by_id(node_id)
            except KeyError:
                node = None
            if node is not None and child_id in node.child_ids:
                node.child_ids.remove(child_id)

    def compound_of_node(self, node_id):
        for compound in self.compounds:
            for subcomponent in compound.subcomponents:
                if isinstance(subcomponent, Node) and subcomponent.id == node_id:
                    return compound
        return None

    def remove_node(self, node_id):
        # nx.remove_node drops all incident edges from the graph but leaves
        # the surviving neighbours' from_branch_ids/to_branch_ids pointing at
        # those now-vanished edges. Detach them first so later
        # branches_connected_to / components_connected_to on a neighbour does
        # not call branch_by_id on a missing edge.
        incident = [
            (u, v, key)
            for u, v, key in self._network_internal.edges(node_id, keys=True)
        ]
        for u, v, key in incident:
            self.remove_branch_between(u, v, key=key)
        node = self.node_by_id(node_id)
        for child_id in list(node.child_ids):
            if self.has_child(child_id):
                self.remove_child(child_id)
        self._network_internal.remove_node(node_id)

    def remove_branch(self, branch_id):
        branch: Branch = self.branch_by_id(branch_id)
        self.remove_branch_between(
            branch.from_node_id, branch.to_node_id, key=branch_id[2]
        )

    def has_compound(self, compound_id):
        return compound_id in self._compound_dict

    def remove_compound(self, compound_id):
        compound: Compound = self.compound_by_id(compound_id)
        del self._compound_dict[compound_id]
        for subcomponent in compound.subcomponents:
            if isinstance(subcomponent, Node):
                if self.has_node(subcomponent.id):
                    self.remove_node(subcomponent.id)
            elif isinstance(subcomponent, Branch):
                if self.has_branch(subcomponent.id):
                    self.remove_branch(subcomponent.id)
            elif isinstance(subcomponent, Child):
                if self.has_child(subcomponent.id):
                    self.remove_child(subcomponent.id)
            elif isinstance(subcomponent, Compound):
                if self.has_compound(subcomponent.id):
                    self.remove_compound(subcomponent.id)

    def remove_branch_between(self, node_one, node_two, key=0):
        self._network_internal.remove_edge(node_one, node_two, key)
        self.node_by_id(node_one).remove_branch((node_one, node_two, key))
        self.node_by_id(node_two).remove_branch((node_one, node_two, key))

    def move_branch(self, branch_id, new_from_id, new_to_id):
        branch: Branch = self.branch_by_id(branch_id)
        self.remove_branch_between(branch_id[0], branch_id[1], key=branch_id[2])
        new_branch_id = self.branch(
            branch.model,
            new_from_id,
            new_to_id,
            formulation=branch.formulation,
            constraints=branch.constraints,
            grid=branch.grid,
            name=branch.name,
        )
        new_branch = self.branch_by_id(new_branch_id)
        new_branch.formulation_pinned = branch.formulation_pinned
        new_branch.active = branch.active
        new_branch.independent = branch.independent
        return new_branch_id

    def child_by_id(self, child_id):
        return self._child_dict[child_id]

    def childs_by_type(self, cls):
        return [child for child in self.childs if type(child.model) is cls]

    def compound_by_id(self, compound_id):
        return self._compound_dict[compound_id]

    def compounds_by_type(self, cls):
        return [compound for compound in self.compounds if type(compound.model) is cls]

    def nodes_by_type(self, cls):
        return [node for node in self.nodes if type(node.model) is cls]

    def childs_by_ids(self, child_ids) -> list[Child]:
        return [self.child_by_id(child_id) for child_id in child_ids]

    def has_any_child_of_type(self, branch, cls) -> bool:
        childs = self.get_childs_by_type(branch, cls)
        return len(childs) > 0

    def get_childs_by_type(self, branch, cls) -> list[Child]:
        return [
            child
            for child in self.childs_by_ids(branch.child_ids)
            if isinstance(child.model, cls)
        ]

    def branches_by_ids(self, branch_ids) -> list[Branch]:
        return [self.branch_by_id(branch_id) for branch_id in branch_ids]

    def is_blacklisted(self, obj):
        return id(obj) in self.__blacklist

    def has_node(self, node_id):
        return node_id in self._network_internal.nodes

    def has_branch(self, branch_id):
        return branch_id in self._network_internal.edges

    def get_branch_between(self, node_id_one, node_id_two, bid=0):
        edge_data = self._network_internal.get_edge_data(node_id_one, node_id_two)
        if edge_data is None or bid not in edge_data:
            raise ValueError(
                f"There is no branch between node '{node_id_one}' and "
                f"'{node_id_two}' with key {bid}."
            )
        return edge_data[bid]["internal_branch"]

    def has_branch_between(self, node_id_one, node_id_two):
        return self._network_internal.has_edge(node_id_one, node_id_two)

    def compounds_connected_to(self, node_id) -> list[Component]:
        return [
            compound
            for compound in self.compounds
            if node_id in compound.connected_to.values()
        ]

    def compound_of(self, subcomponent_component_id) -> Component | None:
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
        return [branch for branch in self.branches if isinstance(branch.model, cls)]

    def __insert_to_blacklist_if_forced(self, obj):
        if self.__collection_stack:
            self.__blacklist[id(obj)] = obj

    def __insert_to_container_if_collect_toggled(self, obj):
        if self.__collection_stack:
            self.__collection_stack[-1].append(obj)

    def node_by_id_or_create(self, node_id, auto_node_creator, auto_grid_key):
        if not self.has_node(node_id):
            if auto_node_creator is None:
                raise ValueError(
                    f"The node id '{node_id}' does not exist and no "
                    "auto_node_creator was provided to create it on the fly."
                )
            return self.node_by_id(
                self.node(auto_node_creator(), grid=auto_grid_key, overwrite_id=node_id)
            )
        return self.node_by_id(node_id)

    def child(
        self,
        model,
        attach_to_node_id=None,
        formulation=None,
        constraints=None,
        overwrite_id=None,
        name=None,
        auto_node_creator=None,
        auto_grid_key=None,
    ):
        next_child_id = (
            0 if len(self._child_dict) == 0 else max(self._child_dict.keys()) + 1
        )
        if overwrite_id is not None and overwrite_id in self._child_dict:
            raise ValueError(f"A child with the id '{overwrite_id}' already exists.")
        child_id = overwrite_id if overwrite_id is not None else next_child_id
        child = Child(
            child_id,
            model,
            formulation=formulation,
            constraints=constraints,
            name=name,
            independent=not self.__collection_stack,
        )
        child.formulation_pinned = formulation is not None
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
            child.node_id = attaching_node.id
        return child_id

    def child_to(
        self,
        model,
        node_id,
        formulation=None,
        constraints=None,
        overwrite_id=None,
        name=None,
        auto_node_creator=None,
        auto_grid_key=None,
    ):
        return self.child(
            model,
            formulation=formulation,
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
        formulation=None,
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
            if overwrite_id in self._network_internal.nodes:
                raise ValueError(f"A node with the id '{overwrite_id}' already exists.")
            node_id = overwrite_id

        grid = self._or_default(grid)
        # Apply the grid's voltage floor to the bus voltage variable so the
        # 1/vm term in the AC current equations stays well-conditioned for NLP
        # solvers (IPOPT). No-op for nodes without ``vm_pu`` (e.g. junctions) and
        # for a user-customised lower bound (only the default 0 is raised).
        vm_pu = getattr(model, "vm_pu", None)
        if (
            isinstance(vm_pu, Var)
            and vm_pu.min in (0, None)
            and hasattr(grid, "vm_pu_min")
        ):
            vm_pu.min = grid.vm_pu_min
        node = Node(
            node_id,
            model,
            child_ids,
            formulation=formulation,
            constraints=constraints,
            grid=grid,
            name=name,
            position=position,
            independent=not self.__collection_stack,
        )
        node.formulation_pinned = formulation is not None
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
        formulation=None,
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
        if grid is not None:
            grid = self._or_default(grid)
        branch = Branch(
            model,
            from_node_id,
            to_node_id,
            formulation=formulation,
            constraints=constraints,
            grid=grid
            or (
                from_node.grid
                if from_node.grid == to_node.grid
                else {
                    type(from_node.grid): from_node.grid,
                    type(to_node.grid): to_node.grid,
                }
            ),
            name=name,
            independent=not self.__collection_stack,
            **kwargs,
        )
        branch.formulation_pinned = formulation is not None
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
        formulation=None,
        constraints=None,
        overwrite_id=None,
        **connected_node_ids,
    ):
        # One collection frame per compound() call: nested calls collect into
        # their own frame and an exception in create() discards the frame, so
        # neither nesting nor failures leak components into other compounds.
        self.__collection_stack.append([])
        try:
            model.create(
                self,
                **{
                    k.replace("_id", "") if k.endswith("_id") else k: self.node_by_id(v)
                    for k, v in connected_node_ids.items()
                },
            )
            subcomponents = self.__collection_stack[-1]
        finally:
            self.__collection_stack.pop()
        # Allocate the id only after create() ran: a nested compound() call in
        # create() registers itself first and must not collide with ours.
        next_compound_id = (
            0 if len(self._compound_dict) == 0 else max(self._compound_dict.keys()) + 1
        )
        compound_id = overwrite_id if overwrite_id is not None else next_compound_id
        compound = Compound(
            compound_id=compound_id,
            formulation=formulation,
            model=model,
            constraints=constraints,
            connected_to=connected_node_ids,
            subcomponents=subcomponents,
        )
        compound.formulation_pinned = formulation is not None
        self._compound_dict[compound_id] = compound
        # A nested compound is a subcomponent of the enclosing one and, like
        # any compound-internal component, excluded from native save.
        self.__insert_to_blacklist_if_forced(compound)
        self.__insert_to_container_if_collect_toggled(compound)
        return compound_id

    def constraint(self, constraint_equation):
        self._constraints.append(constraint_equation)

    def objective(self, objective_function):
        self._objectives.append(objective_function)

    @staticmethod
    def _model_dict_to_input(container):
        model_dict = container.model.vars
        input_dict = {
            "active": container.active,
            "id": container.id,
            "independent": container.independent,
            "ignored": container.ignored,
        }
        for k, v in model_dict.items():
            input_value = v
            if isinstance(v, Var):
                input_value = "$VAR"
            if isinstance(v, Intermediate):
                input_value = "$INT"
            if isinstance(v, Const):
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
            input_dict = Network._model_dict_to_input(container)
            if isinstance(container, Child):
                input_dict["node_id"] = container.node_id
            input_dict_list_dict[model_type_name].append(input_dict)
        dataframe_dict = {}
        for result_type, dict_list in input_dict_list_dict.items():
            dataframe_dict[result_type] = pandas.DataFrame(dict_list)
        return dataframe_dict

    @staticmethod
    def _model_dict_to_results(container):
        model_dict = container.model.vars
        result_dict = {
            "active": container.active,
            "id": container.id,
            "independent": container.independent,
            "ignored": container.ignored,
        }
        for k, v in model_dict.items():
            result_value = v
            if isinstance(v, Var | Const | Intermediate | PostProcess):
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
            result_dict = Network._model_dict_to_results(container)
            if isinstance(container, Child):
                result_dict["node_id"] = container.node_id
            result_dict_list_dict[model_type_name].append(result_dict)
        dataframe_dict = {}
        for result_type, dict_list in result_dict_list_dict.items():
            dataframe_dict[result_type] = pandas.DataFrame(dict_list)
        return dataframe_dict

    def as_result_dataframe_dict_str(self):
        dataframes = self.as_result_dataframe_dict()
        result_str = ""
        for cls_str, dataframe in dataframes.items():
            result_str += cls_str
            result_str += "\n"
            result_str += dataframe.to_string()
            result_str += "\n"
            result_str += "\n"
        return result_str

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

    def __repr__(self):
        return self.as_dataframe_dict_str()

    def __str__(self):
        return self.as_dataframe_dict_str()

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

    def __deepcopy__(self, memo):
        new = Network.__new__(Network)
        memo[id(self)] = new

        new._default_grid_models = copy.deepcopy(self._default_grid_models, memo)
        new._child_dict = {
            k: copy.deepcopy(v, memo) for k, v in self._child_dict.items()
        }
        new._compound_dict = {
            k: copy.deepcopy(v, memo) for k, v in self._compound_dict.items()
        }
        # Constraints/objectives are stateless lambdas - share by reference.
        new._constraints = list(self._constraints)
        new._objectives = list(self._objectives)
        new._extensions = copy.deepcopy(self._extensions, memo)
        # Compound-construction transients - deepcopy preserves consistency
        # if the copy ever lands mid-build. The blacklist is keyed by object
        # identity, so rebuild the keys from the copied objects.
        new._Network__blacklist = {
            id(v): v
            for v in copy.deepcopy(list(self._Network__blacklist.values()), memo)
        }
        new._Network__collection_stack = copy.deepcopy(
            self._Network__collection_stack, memo
        )
        new._Network__current_grid = copy.deepcopy(self._Network__current_grid, memo)
        # Default formulations are module-level singletons - share by reference.
        new._Network__default_formulation = dict(self._Network__default_formulation)

        # Manual MultiGraph rebuild - networkx generic deepcopy is much slower.
        g = nx.MultiGraph()
        new._network_internal = g
        for node_id, data in self._network_internal.nodes(data=True):
            new_data = {k: copy.deepcopy(v, memo) for k, v in data.items()}
            g.add_node(node_id, **new_data)
        for u, v, key, data in self._network_internal.edges(keys=True, data=True):
            new_data = {dk: copy.deepcopy(dv, memo) for dk, dv in data.items()}
            g.add_edge(u, v, key=key, **new_data)

        return new

    def clear_childs(self):
        self._child_dict = {}
        for node in self.nodes:
            node.child_ids = []


def _clean_up_compound(network: Network, compound):
    """Return True when every subcomponent of *compound* survived a graph
    transform; otherwise remove the compound (with its remaining parts) so no
    half-alive compound lingers."""
    fully_intact = True
    for component in compound.component_of_type(Node):
        if not network.has_node(component.id):
            fully_intact = False
    for component in compound.component_of_type(Child):
        if not network.has_child(component.id):
            fully_intact = False
    for component in compound.component_of_type(Branch):
        if not network.has_branch(component.id):
            fully_intact = False
    for component in compound.component_of_type(Compound):
        if not network.has_compound(component.id):
            fully_intact = False
        elif not _clean_up_compound(network, component):
            fully_intact = False
    if not fully_intact and network.has_compound(compound.id):
        network.remove_compound(compound.id)
    return fully_intact


def to_spanning_tree(network: Network, *, weight=None):
    """Minimum spanning tree of *network*.

    ``weight=None`` (default) keeps unit edge weights, i.e. a fixed but
    cost-agnostic spanning tree (unchanged historical behaviour). Pass a
    callable ``weight(branch, node_from, node_to) -> float`` to weight edges
    (e.g. by pipe length) and obtain a minimum-weight tree.
    """

    def _mst(g):
        if weight is not None:
            for u, v, _key, data in g.edges(keys=True, data=True):
                data["weight"] = float(
                    weight(
                        data["internal_branch"],
                        g.nodes[u]["internal_node"],
                        g.nodes[v]["internal_node"],
                    )
                )
        return nx.minimum_spanning_tree(g, weight="weight")

    return transform_network(network, _mst)


def to_backbone(
    network: Network,
    *,
    method="span",
    weight=None,
    terminals=None,
    steiner_method="mehlhorn",
):
    """Backbone subgraph of *network* used as the layout skeleton for grid
    generation.

    ``method="span"`` returns a spanning tree over all nodes (see
    :func:`to_spanning_tree`; ``weight`` is forwarded). ``method="steiner"``
    returns an approximate minimum Steiner tree connecting only ``terminals``
    plus the transit nodes needed to link them, dropping nodes no terminal
    needs.
    """
    if method == "span":
        return to_spanning_tree(network, weight=weight)
    if method == "steiner":
        if terminals is None:
            raise ValueError("steiner backbone requires a terminals set")
        return transform_network(
            network, _steiner_transform(set(terminals), weight, steiner_method)
        )
    raise ValueError(
        f"unknown backbone method {method!r}; expected 'span' or 'steiner'"
    )


def _steiner_transform(terminals, weight, steiner_method):
    """Graph transform that reduces a MultiGraph to the Steiner tree over
    *terminals*. Computed on a simple weighted projection (cheapest parallel
    edge per pair) for robustness, then mapped back to the original multi-edges
    so node/branch attributes survive."""
    from networkx.algorithms.approximation import steiner_tree

    def _transform(g):
        simple = nx.Graph()
        for node_id, data in g.nodes(data=True):
            simple.add_node(node_id, **data)
        for u, v, key, data in g.edges(keys=True, data=True):
            w = (
                float(
                    weight(
                        data["internal_branch"],
                        g.nodes[u]["internal_node"],
                        g.nodes[v]["internal_node"],
                    )
                )
                if weight is not None
                else 1.0
            )
            if not simple.has_edge(u, v) or w < simple[u][v]["weight"]:
                simple.add_edge(u, v, weight=w, _mkey=key)

        present = [t for t in terminals if t in simple]
        result = nx.MultiGraph()
        if len(present) <= 1:
            for node_id in present:
                result.add_node(node_id, **g.nodes[node_id])
            return result

        tree = steiner_tree(simple, present, weight="weight", method=steiner_method)
        for node_id in tree.nodes:
            result.add_node(node_id, **g.nodes[node_id])
        for u, v in tree.edges():
            key = simple[u][v]["_mkey"]
            result.add_edge(u, v, key=key, **g[u][v][key])
        return result

    return _transform


def transform_network(network: Network, graph_transform):
    network = network.copy()
    network._network_internal = graph_transform(network.graph)
    # The transform (e.g. minimum_spanning_tree) drops edges, leaving the
    # surviving nodes' from_branch_ids/to_branch_ids referencing branches that
    # no longer exist. Rebuild those lists from the reduced edge set so
    # branches_connected_to / components_connected_to stay consistent.
    for node in network.nodes:
        node.from_branch_ids = []
        node.to_branch_ids = []
    for from_id, to_id, key in network._network_internal.edges(keys=True):
        branch_id = (from_id, to_id, key)
        network.node_by_id(from_id).add_from_branch_id(branch_id)
        network.node_by_id(to_id).add_to_branch_id(branch_id)
    # Clean up compounds first: removing a broken compound also removes its
    # surviving internal nodes, whose children the orphan sweep below catches.
    for compound in network.compounds:
        if network.has_compound(compound.id):
            _clean_up_compound(network, compound)
    for child in network.childs:
        referenced = False
        for node in network.nodes:
            if child.id in node.child_ids:
                referenced = True
        if not referenced:
            network.remove_child(child.id)
    return network


def _add_tuple(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def _div_tuple(a, div):
    return tuple(a[i] / div for i in range(len(a)))


def calc_coordinates(network: Network, component: Component):
    if type(component) is Node:
        return component.position
    elif type(component) is Branch:
        node_start = network.node_by_id(component.from_node_id)
        node_end = network.node_by_id(component.to_node_id)
        return tuple(
            (node_start.position[i] + node_end.position[i]) / 2
            for i in range(len(node_start.position))
        )
    elif type(component) is Child:
        return network.node_by_id(component.node_id).position
    elif type(component) is Compound:
        position = (0, 0)
        for connected_node_id in component.connected_to.values():
            node = network.node_by_id(connected_node_id)
            position = _add_tuple(position, node.position)
        return _div_tuple(position, len(component.connected_to))
    raise ValueError(f"This should not happen! The component {component} is unknown.")
