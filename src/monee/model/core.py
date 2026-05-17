import copy
from abc import ABC, abstractmethod

EL_KEY = "electricity"
GAS_KEY = "gas"
WATER_KEY = "water"
WATER = WATER_KEY
EL = EL_KEY
GAS = GAS_KEY

component_list = []

_IMMUTABLE_PRIMITIVES = frozenset({int, float, bool, str, bytes, complex, type(None)})


def model(cls):
    """Register a model class in the global ``component_list`` registry."""
    component_list.append(cls)
    return cls


def upper(var_or_const):
    """Return the upper bound of a ``Var`` (or its value if unbounded); pass through otherwise."""
    if isinstance(var_or_const, Var):
        if var_or_const.max is None:
            return var_or_const.value
        return var_or_const.max
    return var_or_const


def lower(var_or_const):
    """Return the lower bound of a ``Var`` (or its value if unbounded); pass through otherwise."""
    if isinstance(var_or_const, Var):
        if var_or_const.min is None:
            return var_or_const.value
        return var_or_const.min
    return var_or_const


def value(var_or_const):
    """Extract ``.value`` from a ``Var``/``Const``/``Intermediate``; pass through plain scalars."""
    if isinstance(var_or_const, Const | Var | Intermediate):
        return var_or_const.value
    return var_or_const


class Var:
    """
    A decision variable (or mutable parameter) in the optimisation model.

    Sign convention for child components: positive = consumption (load),
    negative = generation. ``PowerGenerator(p_mw=5)`` stores ``p_mw=-5``.

    Comparison operators compare against the *bound*, not the value, and return
    ``False`` if the relevant bound is ``None``.
    """

    def __init__(self, value, max=None, min=None, integer=False, name=None) -> None:

        if not isinstance(value, float | int):
            raise ValueError(
                "The initial guess for the variable need to a numeric value!"
            )
        self.value = value
        self.max = max
        self.min = min
        self.integer = integer
        self.name = name

    def __neg__(self):
        actual_max = None if self.max is None else -self.max
        actual_min = None if self.min is None else -self.min
        return type(self)(
            value=-self.value, max=actual_min, min=actual_max, name=self.name
        )

    def __mul__(self, other):
        if isinstance(other, (int, float)) and other < 0:
            new_max = None if self.min is None else self.min * other
            new_min = None if self.max is None else self.max * other
        else:
            new_max = None if self.max is None else self.max * other
            new_min = None if self.min is None else self.min * other
        return type(self)(
            value=self.value * other, max=new_max, min=new_min, name=self.name
        )

    def __lt__(self, other):
        if isinstance(other, float | int) and self.max is not None:
            return self.max < other
        return False

    def __le__(self, other):
        if isinstance(other, float | int) and self.max is not None:
            return self.max <= other
        return False

    def __gt__(self, other):
        if isinstance(other, float | int) and self.min is not None:
            return self.min > other
        return False

    def __ge__(self, other):
        if isinstance(other, float | int) and self.min is not None:
            return self.min >= other
        return False

    def __repr__(self):
        parts = [repr(self.value)]
        if self.min is not None or self.max is not None:
            parts.append(f"min={self.min!r}, max={self.max!r}")
        if self.integer:
            parts.append("integer=True")
        if self.name is not None:
            parts.append(f"name={self.name!r}")
        return f"{type(self).__name__}({', '.join(parts)})"

    def __str__(self):
        return f"{self.value} ({self.min}, {self.max}), is int: {self.integer}"

    def __deepcopy__(self, memo):
        new = type(self).__new__(type(self))
        memo[id(self)] = new
        new.value = self.value
        new.max = self.max
        new.min = self.min
        new.integer = self.integer
        new.name = self.name
        return new


tracked = Var


class Const:
    """
    A fixed (non-optimised) constant that participates in the model attribute protocol.

    Used in ``overwrite()`` to pin a node variable to a fixed setpoint instead
    of turning it into a free decision variable.
    """

    def __init__(self, value) -> None:
        self.value = value

    def __repr__(self):
        return f"Const({self.value!r})"

    def __deepcopy__(self, memo):
        new = Const.__new__(Const)
        memo[id(self)] = new
        new.value = self.value
        return new


class Intermediate:
    """
    A computed (derived) quantity, not a decision variable.

    The solver evaluates the matching :class:`IntermediateEq` and writes the
    result back to ``.value`` after each solve.
    """

    def __init__(self, value=0):
        self.value = value

    def __repr__(self):
        return f"Intermediate({self.value!r})"

    def __deepcopy__(self, memo):
        new = Intermediate.__new__(Intermediate)
        memo[id(self)] = new
        new.value = self.value
        return new


class IntermediateEq:
    """
    Declares how to compute an :class:`Intermediate` attribute from other variables.

    Return an ``IntermediateEq`` from a model's ``equations()`` to register a
    derived quantity; the solver evaluates ``eq`` and stores the result on the
    attribute named by ``attr``.
    """

    def __init__(self, attr, eq):
        self.attr = attr
        self.eq = eq


class GenericModel(ABC):
    """
    Base class for all component models (nodes, branches, children, compounds).

    Public attributes (no leading ``_``) form the solver-visible state. Use
    :class:`Var` for decision variables, :class:`Const` for pinned setpoints,
    plain scalars for parameters.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._ext_data = kwargs

    @property
    def vars(self):
        return {k: v for k, v in self.__dict__.items() if k[0] != "_"}

    @property
    def values(self):
        return {k: value(v) for k, v in self.__dict__.items() if k[0] != "_"}

    def is_cp(self):
        return False

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if v is None or v.__class__ in _IMMUTABLE_PRIMITIVES:
                new.__dict__[k] = v
            else:
                new.__dict__[k] = copy.deepcopy(v, memo)
        return new


class NodeModel(GenericModel):
    """Abstract base class for node models (buses, junctions). Subclasses define nodal equations."""

    @abstractmethod
    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        """Return nodal equations (e.g. flow conservation, voltage balance)."""

    def minimize(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        """Optional objective contribution (e.g. slack penalties). Default: none."""
        return []


class BranchModel(GenericModel):
    """Abstract base class for network branches (lines, pipes). Subclasses define branch equations."""

    @abstractmethod
    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        pass

    def minimize(self, grid, from_node_model, to_node_model, **kwargs):
        return []

    def loss_percent(self):
        return 0

    def is_cp(self):
        return False

    def init(self, grid):
        pass


class MultiGridBranchModel(BranchModel):
    """Branch that couples multiple grid domains (e.g. CHP, P2H). Always a control point."""

    @abstractmethod
    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        pass

    def is_cp(self):
        return True

    def init(self, grids):
        pass


class MultiGridNodeModel(NodeModel):
    """Node that participates in multiple grid domains. Always a control point."""

    def is_cp(self):
        return True


class CompoundModel(GenericModel):
    """
    Composite component spanning multiple nodes/carriers (e.g. P2H unit).

    Subclasses implement :meth:`create` to add sub-components to the network,
    and may override :meth:`equations` for coupling constraints.
    """

    @abstractmethod
    def create(self, network):
        """Add sub-components (nodes, branches, children) to *network*."""

    def equations(self, network, **kwargs):
        return []

    def minimize(self, network, **kwargs):
        return []


class MultiGridCompoundModel(CompoundModel):
    def is_cp(self):
        return False


class ChildModel(GenericModel):
    """
    Leaf component attached to a single node (loads, generators, ext-grids).

    Load convention: positive = consumption, negative = generation. Generator
    subclasses negate the user-supplied magnitude so the public API takes
    positive numbers.

    ``regulation`` in ``[0.0, 1.0]`` scales the setpoint; used by multi-energy
    couplers (P2H, CHP) for partial dispatch.
    """

    def __init__(self, regulation: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.regulation = regulation

    def overwrite(self, node_model, grid):
        """Pin node variables to fixed setpoints (grid-forming children). Default: no-op."""

    @abstractmethod
    def equations(self, grid, node_model, **kwargs):
        pass

    def minimize(self, grid, node_model, **kwargs):
        return []


class Component(ABC):
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
        return f"{self.__class__.__name__}-{self.id}".lower()

    @property
    def nid(self):
        return f"{self.model.__class__.__name__}-{self.id}".lower()

    @property
    def formulation(self):
        return self._formulation

    @formulation.setter
    def formulation(self, formulation):
        self._formulation = formulation

        if self._formulation is not None:
            self._formulation.ensure_var(self.model)

    def __deepcopy__(self, memo):
        new = self.__class__.__new__(self.__class__)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            if v is None or v.__class__ in _IMMUTABLE_PRIMITIVES:
                new.__dict__[k] = v
            else:
                new.__dict__[k] = copy.deepcopy(v, memo)
        return new


class Child(Component):
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
        return [
            component
            for component in self.subcomponents
            if type(component) is comp_type
        ]


class Node(Component):
    """Network node tracking attached child components and incoming/outgoing branches."""

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
        self.from_branch_ids.append(branch_id)

    def add_to_branch_id(self, branch_id):
        self.to_branch_ids.append(branch_id)

    def _remove_branch(self, branch_id):
        if branch_id in self.to_branch_ids:
            self.to_branch_ids.remove(branch_id)
        elif branch_id in self.from_branch_ids:
            self.from_branch_ids.remove(branch_id)

    def remove_branch(self, branch_id):
        """Remove ``branch_id`` and its reversed counterpart from both branch lists."""
        switched = (branch_id[1], branch_id[0], branch_id[2])
        self._remove_branch(branch_id)
        self._remove_branch(switched)


class Branch(Component):
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
        if self.id[0] > self.id[1]:
            return f"branch-{self.id[0]}-{self.id[1]}"
        else:
            return f"branch-{self.id[1]}-{self.id[0]}"
