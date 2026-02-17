from abc import ABC, abstractmethod

EL_KEY = "electricity"
GAS_KEY = "gas"
WATER_KEY = "water"
WATER = WATER_KEY
EL = EL_KEY
GAS = GAS_KEY
component_list = []


def model(cls):
    """
    No docstring provided.
    """
    component_list.append(cls)
    return cls


def upper(var_or_const):
    """
    Returns the upper bound (maximum value) of a variable or constant.

    This function extracts the maximum allowed value from a Var instance, which may represent a decision variable or parameter in an optimization or simulation context. If the input is a Var and its `max` attribute is set, that value is returned; otherwise, the current value is returned. For non-Var inputs, the input is returned unchanged. Use this function to retrieve the upper bound for constraints, reporting, or validation.

    Args:
        var_or_const: A Var instance or a constant value. If a Var, must have a `max` attribute.

    Returns:
        float or Any: The maximum value for a Var, or the input value for constants.

    Examples:
        v = Var(5, max=10)
        upper(v)         # Returns 10
        upper(7)         # Returns 7
        v2 = Var(3)
        upper(v2)        # Returns 3 (since max is None)
    """
    if isinstance(var_or_const, Var):
        if var_or_const.max is None:
            return var_or_const.value
        return var_or_const.max
    return var_or_const


def lower(var_or_const):
    """
    Returns the lower bound (minimum value) of a variable or constant.

    This function is used to extract the minimum allowed value from a Var instance, which may represent a decision variable or parameter in an optimization or simulation context. If the input is a Var and its `min` attribute is set, that value is returned; otherwise, the current value is returned. For non-Var inputs, the input is returned unchanged. Use this function when you need to retrieve the lower bound for constraints, reporting, or validation.

    Args:
        var_or_const: A Var instance or a constant value. If a Var, must have a `min` attribute.

    Returns:
        float or Any: The minimum value for a Var, or the input value for constants.

    Examples:
        v = Var(5, min=2)
        lower(v)         # Returns 2
        lower(10)        # Returns 10
        v2 = Var(7)
        lower(v2)        # Returns 7 (since min is None)
    """
    if isinstance(var_or_const, Var):
        if var_or_const.min is None:
            return var_or_const.value
        return var_or_const.min
    return var_or_const


def value(var_or_const):
    """
    No docstring provided.
    """
    if isinstance(var_or_const, Const | Var | Intermediate):
        return var_or_const.value
    return var_or_const


class Var:
    """
    No docstring provided.
    """

    def __init__(self, value, max=None, min=None, integer=False, name=None) -> None:
        self.value = value
        self.max = max
        self.min = min
        self.integer = integer
        self.name = name

    def __neg__(self):
        """
        No docstring provided.
        """
        return Var(value=-self.value, max=self.max, min=self.min, name=self.name)

    def __mul__(self, other):
        """
        No docstring provided.
        """
        return Var(value=self.value * other, max=self.max, min=self.min, name=self.name)

    def __lt__(self, other):
        """
        No docstring provided.
        """
        if isinstance(other, float | int) and self.max is not None:
            return self.max < other
        return False

    def __le__(self, other):
        """
        No docstring provided.
        """
        if isinstance(other, float | int) and self.max is not None:
            return self.max <= other
        return False

    def __gt__(self, other):
        """
        No docstring provided.
        """
        if isinstance(other, float | int) and self.min is not None:
            return self.min > other
        return False

    def __ge__(self, other):
        """
        No docstring provided.
        """
        if isinstance(other, float | int) and self.min is not None:
            return self.min >= other
        return False

    def __str__(self):
        """
        No docstring provided.
        """
        return f"{self.value} ({self.min}, {self.max}), is int: {self.integer}"


class Const:
    """
    No docstring provided.
    """

    def __init__(self, value) -> None:
        self.value = value


class Intermediate:
    """
    No docstring provided.
    """

    def __init__(self, value=0):
        self.value = value


class IntermediateEq:
    """
    No docstring provided.
    """

    def __init__(self, attr, eq):
        self.attr = attr
        self.eq = eq


class GenericModel(ABC):
    """
    No docstring provided.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._ext_data = kwargs

    @property
    def vars(self):
        """
        No docstring provided.
        """
        return {k: v for k, v in self.__dict__.items() if k[0] != "_"}

    @property
    def values(self):
        """
        No docstring provided.
        """
        return {k: value(v) for k, v in self.__dict__.items() if k[0] != "_"}

    def is_cp(self):
        """
        No docstring provided.
        """
        return False


class NodeModel(GenericModel):
    """
    Abstract base class for node models in a network, defining the interface for nodal equations and behavior.

    NodeModel provides a foundation for representing nodes within various grid domains (e.g., electrical, gas, heat) in network simulations. Subclasses must implement the abstract `equations` method to specify the physical or operational relationships at the node, such as flow conservation, voltage balance, or other nodal constraints. This class is intended for use in extensible, multi-domain network modeling frameworks and supports integration with branch and child component models.

    Example:
        class MyNodeModel(NodeModel):
            def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
                # Implement nodal balance equations
                return [eq1, eq2]

    Methods:
        equations(grid, in_branch_models, out_branch_models, childs, **kwargs): Abstract. Must be implemented by subclasses to define the system of equations for the node.
    """

    @abstractmethod
    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        """
        Defines the system of equations for a node in the network, relating incoming and outgoing branches and attached child components.

        This abstract method must be implemented by subclasses to specify the physical or operational relationships governing the node's behavior within a grid. Use this method to model node constraints such as flow conservation, voltage balance, or other nodal conditions. It is typically called during network simulation or optimization to assemble the overall system equations.

        Args:
            grid: The grid object to which the node belongs (e.g., electrical, gas, or heat grid).
            in_branch_models (list): List of branch model instances representing incoming branches to the node.
            out_branch_models (list): List of branch model instances representing outgoing branches from the node.
            childs (list): List of child component models attached to the node (e.g., loads, generators).
            **kwargs: Additional keyword arguments for solver options or equation customization.

        Returns:
            Any: The equations or constraints representing the node's behavior. The return type and structure depend on the modeling framework and implementation.

        Raises:
            NotImplementedError: If not implemented in a subclass.

        Examples:
            class MyNodeModel(NodeModel):
                def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
                    # Define nodal balance equations
                    return [eq1, eq2]
        """

    def minimize(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        """
        Minimization of an expression as minimization may necessary due to some formulation (when replacing
        constraints with slack variables)

        :param self: Description
        """
        return []


class BranchModel(GenericModel):
    """
    Abstract base class for network branch models, defining the interface for branch-specific equations, loss calculation, and initialization.

    BranchModel provides a common foundation for implementing custom branch types (such as power lines or pipes) in network simulations. Subclasses must implement the abstract `equations` method to define the physical relationships for the branch. Optional methods such as `loss_percent`, `is_cp`, and `init` can be overridden to customize loss calculations, control point designation, and initialization logic. This class inherits from GenericModel and is intended for use within extensible, multi-domain network modeling frameworks.

    Attributes:
        _ext_data (dict): Stores extra data passed during initialization for use in extended models.

    Methods:
        equations(grid, from_node_model, to_node_model, **kwargs): Abstract. Must be implemented by subclasses to define branch equations.
        loss_percent(): Returns the loss percentage for the branch (default 0; override as needed).
        is_cp(): Indicates if the branch is a control point (default False; override as needed).
        init(grid): Optional initialization logic for the branch (default is no-op).

    Example:
        class MyCustomBranch(BranchModel):
            def equations(self, grid, from_node_model, to_node_model, **kwargs):
                # Implement branch-specific equations
                pass

            def loss_percent(self):
                # Custom loss calculation
                return compute_actual_loss(self)

            def is_cp(self):
                # Mark this branch as a control point
                return True

            def init(self, grid):
                # Custom initialization logic
                pass
    """

    @abstractmethod
    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        """
        No docstring provided.
        """

    def minimize(self, grid, from_node_model, to_node_model, **kwargs):
        """
        Minimization of an expression as minimization may necessary due to some formulation (when replacing
        constraints with slack variables)

        :param self: Description
        """
        return []

    def loss_percent(self):
        """
        Returns the percentage of losses for the branch model, defaulting to zero.

        This method provides a placeholder for calculating the loss percentage associated with a branch in the network, such as power or heat losses. By default, it returns 0, indicating no losses are considered. Override this method in subclasses to implement specific loss calculations relevant to the branch type.

        Returns:
            int: Always returns 0, representing zero loss percentage by default.

        Examples:
            # In a custom branch model, override to provide actual loss calculation
            class MyBranchModel(BranchModel):
                def loss_percent(self):
                    return compute_actual_loss(self)

            # In base usage
            loss = branch_model.loss_percent()  # Returns 0 unless overridden
        """
        return 0

    def is_cp(self):
        """
        Indicates whether the branch model represents a control point (CP) in the network.

        This method is used to determine if the current branch model instance should be treated as a control point, which may affect optimization or control strategies in network simulations. By default, this implementation always returns False, meaning the branch is not a control point. Override this method in subclasses if specific branch types should be considered as control points.

        Returns:
            bool: False, indicating the branch model is not a control point by default.

        Examples:
            Check if a branch model is a control point:
                if branch_model.is_cp():
                    # Apply control logic
                    ...
        """
        return False

    def init(self, grid):
        """
        No docstring provided.
        """


class MultiGridBranchModel(BranchModel):
    """
    Abstract base class for branch models that couple multiple grid domains, such as electrical, gas, and heat networks.

    This class extends BranchModel to support branches that interact with more than one grid type, enabling the modeling of multi-energy systems and sector coupling. Subclasses must implement the abstract `equations` method to define the physical or operational relationships across the involved grids. The `is_cp` method returns True by default, indicating that multi-grid branches are treated as control points in the network. The `init` method can be overridden to perform any setup or pre-processing required for multi-grid branches.

    Methods:
        equations(grids, from_node_model, to_node_model, **kwargs): Abstract. Must be implemented by subclasses to define the system of equations for the multi-grid branch.
        is_cp(): Returns True, indicating this branch is a control point by default.
        init(grids): Optional initialization logic for the branch (default is no-op).

    Example:
        class MyMultiGridBranch(MultiGridBranchModel):
            def equations(self, grids, from_node_model, to_node_model, **kwargs):
                # Define equations coupling electrical and gas flows
                return [eq1, eq2, eq3]

            def init(self, grids):
                self.el_grid = grids['el']
                self.gas_grid = grids['gas']
    """

    @abstractmethod
    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        """
        Defines the system of equations for a multi-grid branch, relating variables across multiple grids and connected nodes.

        This abstract method must be implemented by subclasses to specify the physical or operational relationships governing the branch's behavior in a multi-grid context (e.g., coupling between electrical, gas, and heat networks). Use this method when modeling branches that interact with more than one grid domain. The method is typically called during network simulation or optimization to assemble the overall system equations.

        Args:
            grids (dict): Dictionary of grid objects involved in the branch (e.g., {'el': el_grid, 'gas': gas_grid}).
            from_node_model: Model instance representing the source node connected to the branch.
            to_node_model: Model instance representing the destination node connected to the branch.
            **kwargs: Additional keyword arguments for solver options or equation customization.

        Returns:
            Any: The equations or constraints representing the branch's behavior. The return type and structure depend on the modeling framework and implementation.

        Raises:
            NotImplementedError: If not implemented in a subclass.

        Examples:
            class MyMultiGridBranch(MultiGridBranchModel):
                def equations(self, grids, from_node_model, to_node_model, **kwargs):
                    # Define equations coupling electrical and gas flows
                    return [eq1, eq2, eq3]
        """

    def is_cp(self):
        """
        Indicates that the multi-grid branch model is a control point (CP) in the network.

        This method returns True to signal that this branch should be treated as a control point, which may affect optimization, control strategies, or system observability in multi-grid network simulations. Override this method in subclasses if a different control point designation is required.

        Returns:
            bool: Always returns True, indicating the branch is a control point.

        Examples:
            if branch_model.is_cp():
                # Apply control logic specific to control points
                ...
        """
        return True

    def init(self, grids):
        """
        Initializes the multi-grid branch model with the provided grid objects.

        This method is intended to perform any setup or pre-processing required before the branch is used in simulation or optimization, such as caching grid parameters or establishing references to grid-specific data. Override this method in subclasses to implement custom initialization logic for multi-grid branches. It is typically called once during network assembly or before equation evaluation.

        Args:
            grids (dict): Dictionary of grid objects relevant to the branch (e.g., {'el': el_grid, 'gas': gas_grid}).

        Returns:
            None

        Examples:
            class MyMultiGridBranch(MultiGridBranchModel):
                def init(self, grids):
                    self.el_grid = grids['el']
                    self.gas_grid = grids['gas']
        """


class MultiGridNodeModel(NodeModel):
    """
    Represents a node model for multi-grid networks, designating the node as a control point (CP) by default.

    This class extends NodeModel to support nodes that participate in multiple grid domains (such as electrical, gas, and heat networks) and are treated as control points in network simulations. Use this class when modeling nodes that require special handling for optimization, control, or observability in multi-energy systems. The `is_cp` method returns True, indicating the node's control point status, which can influence system-level strategies.

    Example:
        node_model = MultiGridNodeModel(...)
        if node_model.is_cp():
            # Apply control logic for control points
            ...

    Methods:
        is_cp(): Returns True, indicating this node is a control point in the multi-grid network.
    """

    def is_cp(self):
        """
        Indicates that the node model serves as a control point (CP) in a multi-grid network.

        This method returns True to signal that this node should be treated as a control point, which may affect optimization, control strategies, or observability in multi-grid network simulations. Override this method in subclasses if a different control point designation is required.

        Returns:
            bool: Always returns True, indicating the node is a control point.

        Examples:
            if node_model.is_cp():
                # Apply control logic specific to control points
                ...
        """
        return True


class CompoundModel(GenericModel):
    """
    No docstring provided.
    """

    @abstractmethod
    def create(self, network):
        """
        No docstring provided.
        """

    def equations(self, network, **kwargs):
        """
        No docstring provided.
        """
        return []

    def minimize(self, network, **kwargs):
        """
        Minimization of an expression as minimization may necessary due to some formulation (when replacing
        constraints with slack variables)

        :param self: Description
        """
        return []


class MultGridCompoundModel(CompoundModel):
    """
    No docstring provided.
    """

    def is_cp(self):
        """
        No docstring provided.
        """
        return False


class ChildModel(GenericModel):
    """
    No docstring provided.
    """

    def __init__(self, regulation: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.regulation = regulation

    def overwrite(self, node_model, grid):
        """
        No docstring provided.
        """

    @abstractmethod
    def equations(self, grid, node_model, **kwargs):
        """
        No docstring provided.
        """

    def minimize(self, grid, node_model, **kwargs):
        """
        Minimization of an expression as minimization may necessary due to some formulation (when replacing
        constraints with slack variables)

        :param self: Description
        """
        return []


class Component(ABC):
    """
    No docstring provided.
    """

    def __init__(
        self,
        id,
        model,
        formulation=None,
        constraints=None,
        grid=None,
        name=None,
        active=True,
        independent=True,
    ) -> None:
        self.model = model
        self.id = id
        self.constraints = [] if constraints is None else constraints
        self.name = name
        self.active = active
        self.grid = grid
        self.independent = independent
        self.ignored = False

        self.formulation = formulation

    @property
    def tid(self):
        """
        No docstring provided.
        """
        return f"{self.__class__.__name__}-{self.id}".lower()

    @property
    def nid(self):
        """
        No docstring provided.
        """
        return f"{self.model.__class__.__name__}-{self.id}".lower()

    @property
    def formulation(self):
        return self._formulation

    @formulation.setter
    def formulation(self, formulation):
        self._formulation = formulation

        if self._formulation is not None:
            self._formulation.ensure_var(self.model)


class Child(Component):
    """
    No docstring provided.
    """

    def __init__(
        self,
        child_id,
        model,
        formulation=None,
        constraints=None,
        grid=None,
        name=None,
        active=True,
        independent=True,
    ) -> None:
        super().__init__(
            child_id, model, formulation, constraints, grid, name, active, independent
        )
        self.node_id = None
        self.independent = independent

    def equations(self, grid, node_model, **kwargs):
        model_eqs = self.model.equations(grid, node_model, **kwargs)
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.equations(
                self.model, grid, node_model, **kwargs
            )

        return (
            form_eqs
            + model_eqs
            + [c(self.model, grid, node_model, **kwargs) for c in self.constraints]
        )

    def minimize(self, grid, node_model, **kwargs):
        model_eqs = self.model.minimize(grid, node_model, **kwargs)
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.minimize(self.model, grid, node_model, **kwargs)

        return form_eqs + model_eqs


class Compound(Component):
    """
    No docstring provided.
    """

    def __init__(
        self,
        compound_id,
        model: CompoundModel,
        connected_to,
        subcomponents,
        formulation=None,
        constraints=None,
        grid=None,
        name=None,
        active=True,
    ) -> None:
        super().__init__(
            compound_id, model, formulation, constraints, grid, name, active, True
        )
        self.connected_to = connected_to
        self.subcomponents = subcomponents

    def equations(self, network, **kwargs):
        model_eqs = self.model.equations(network, **kwargs)
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.equations(self.model, network, **kwargs)

        return (
            form_eqs
            + model_eqs
            + [c(self.model, network, **kwargs) for c in self.constraints]
        )

    def minimize(self, network, **kwargs):
        model_eqs = self.model.minimize(network, **kwargs)
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.minimize(self.model, network, **kwargs)

        return form_eqs + model_eqs

    def component_of_type(self, comp_type):
        """
        No docstring provided.
        """
        return [
            component
            for component in self.subcomponents
            if type(component) is comp_type
        ]


class Node(Component):
    """
    Represents a node in the network, managing connectivity, child components, and node-specific attributes.

    The Node class extends Component to provide functionality for managing network nodes, including tracking incoming and outgoing branches, associated child components, and node-specific constraints or metadata. Nodes serve as connection points for branches and can represent buses, junctions, or other network entities. Use this class when constructing or modifying network topologies, performing connectivity analysis, or managing nodal attributes in simulation workflows.

    Example:
        node = Node(node_id=1, model=my_node_model, position=(10, 20))
        node.add_from_branch_id((2, 1, 0))
        node.add_to_branch_id((1, 3, 0))
        node.remove_branch((2, 1, 0))

    Parameters:
        node_id: Unique identifier for the node.
        model: The node's associated model object.
        child_ids (list, optional): List of child component IDs attached to the node.
        constraints (list, optional): List of operational constraints for the node.
        grid (optional): The grid or domain to which the node belongs.
        name (str, optional): Human-readable name for the node.
        position (optional): Geographical or logical position of the node.
        active (bool, optional): Whether the node is active in the network. Defaults to True.
        independent (bool, optional): Whether the node is independent in the network. Defaults to True.

    Attributes:
        child_ids (list): IDs of child components attached to the node.
        constraints (list): Operational constraints for the node.
        from_branch_ids (list): Identifiers of incoming branches.
        to_branch_ids (list): Identifiers of outgoing branches.
        position: Geographical or logical position of the node.

    Methods:
        add_from_branch_id(branch_id): Registers a branch as incoming to the node.
        add_to_branch_id(branch_id): Registers a branch as outgoing from the node.
        _remove_branch(branch_id): Removes a branch from incoming or outgoing lists.
        remove_branch(branch_id): Removes both a branch and its reversed counterpart from the node's branch lists.
    """

    def __init__(
        self,
        node_id,
        model,
        child_ids=None,
        formulation=None,
        constraints=None,
        grid=None,
        name=None,
        position=None,
        active=True,
        independent=True,
    ) -> None:
        super().__init__(
            node_id, model, formulation, constraints, grid, name, active, independent
        )
        self.child_ids = [] if child_ids is None else child_ids
        self.constraints = [] if constraints is None else constraints
        self.from_branch_ids = []
        self.to_branch_ids = []
        self.position = position

    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        model_eqs = self.model.equations(
            grid, in_branch_models, out_branch_models, childs, **kwargs
        )
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.equations(
                self.model, grid, in_branch_models, out_branch_models, childs, **kwargs
            )

        return (
            form_eqs
            + model_eqs
            + [
                c(
                    self.model,
                    grid,
                    in_branch_models,
                    out_branch_models,
                    childs,
                    **kwargs,
                )
                for c in self.constraints
            ]
        )

    def minimize(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        model_eqs = self.model.minimize(
            grid, in_branch_models, out_branch_models, childs, **kwargs
        )
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.minimize(
                self.model, grid, in_branch_models, out_branch_models, childs, **kwargs
            )

        return form_eqs + model_eqs

    def add_from_branch_id(self, branch_id):
        """
        Adds a branch identifier to the list of incoming branches for the node.

        Use this method to register a new incoming branch connection to the node, typically during network construction or when dynamically modifying the network topology. This ensures that the node maintains an accurate record of all branches entering it, which is essential for connectivity management and nodal analysis.

        Args:
            branch_id: The identifier of the branch to add as an incoming connection. Must be a hashable object representing the branch.

        Returns:
            None

        Examples:
            node.add_from_branch_id((1, 2, 0))  # Registers branch (1, 2, 0) as incoming to the node.
        """
        self.from_branch_ids.append(branch_id)

    def add_to_branch_id(self, branch_id):
        """
        Adds a branch identifier to the list of outgoing branches for the node.

        This method is used to register a new outgoing branch connection from the node, typically during network construction or when updating the network topology. Maintaining an accurate list of outgoing branches is essential for connectivity management, nodal analysis, and network traversal algorithms.

        Args:
            branch_id: The identifier of the branch to add as an outgoing connection. Should be a hashable object representing the branch.

        Returns:
            None

        Examples:
            node.add_to_branch_id((2, 3, 1))  # Registers branch (2, 3, 1) as outgoing from the node.
        """
        self.to_branch_ids.append(branch_id)

    def _remove_branch(self, branch_id):
        """
        Removes a branch identifier from the node's incoming or outgoing branch lists.

        This method is used internally to update the node's branch connections by removing the specified branch ID from either the `to_branch_ids` (outgoing) or `from_branch_ids` (incoming) lists. Use this method when you need to maintain accurate connectivity information after branch deletions or network reconfiguration. It is typically called by higher-level methods that manage node-branch relationships.

        Args:
            branch_id: The identifier of the branch to remove. Must match an entry in either `to_branch_ids` or `from_branch_ids`.

        Returns:
            None

        Examples:
            node._remove_branch((1, 2, 0))  # Removes branch (1, 2, 0) from the node's branch lists if present.
        """
        if branch_id in self.to_branch_ids:
            self.to_branch_ids.remove(branch_id)
        elif branch_id in self.from_branch_ids:
            self.from_branch_ids.remove(branch_id)

    def remove_branch(self, branch_id):
        """
        Removes a branch identifier and its reversed counterpart from the node's branch lists.

        This method updates the node's connectivity by removing both the specified branch ID and its reversed form (with the from/to nodes swapped) from the incoming and outgoing branch lists. Use this method when deleting a branch or reconfiguring the network to ensure all references to the branch are cleared, regardless of direction. It is typically called during network modification or cleanup operations.

        Args:
            branch_id: The identifier of the branch to remove, as a tuple (from_node, to_node, branch_index).

        Returns:
            None

        Examples:
            node.remove_branch((1, 2, 0))  # Removes both (1, 2, 0) and (2, 1, 0) from the node's branch lists if present.
        """
        switched = (branch_id[1], branch_id[0], branch_id[2])
        self._remove_branch(branch_id)
        self._remove_branch(switched)


class Branch(Component):
    """
    No docstring provided.
    """

    def __init__(
        self,
        model,
        from_node_id,
        to_node_id,
        formulation=None,
        constraints=None,
        grid=None,
        name=None,
        active=True,
        independent=True,
    ) -> None:
        super().__init__(
            None, model, formulation, constraints, grid, name, active, independent
        )
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id

    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        model_eqs = self.model.equations(grid, from_node_model, to_node_model, **kwargs)
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.equations(
                self.model, grid, from_node_model, to_node_model, **kwargs
            )

        return (
            form_eqs
            + model_eqs
            + [
                c(self.model, grid, from_node_model, to_node_model, **kwargs)
                for c in self.constraints
            ]
        )

    def minimize(self, grid, from_node_model, to_node_model, **kwargs):
        model_eqs = self.model.minimize(grid, from_node_model, to_node_model, **kwargs)
        form_eqs = []
        if self.formulation is not None:
            form_eqs = self.formulation.minimize(
                self.model, grid, from_node_model, to_node_model, **kwargs
            )

        return form_eqs + model_eqs

    @property
    def tid(self):
        """
        No docstring provided.
        """
        if self.id[0] > self.id[1]:
            return f"branch-{self.id[0]}-{self.id[1]}"
        else:
            return f"branch-{self.id[1]}-{self.id[0]}"
