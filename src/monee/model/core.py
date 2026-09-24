import copy
import functools
import inspect
import os
import sys
import warnings
from abc import ABC, ABCMeta, abstractmethod

EL_KEY = "electricity"
GAS_KEY = "gas"
WATER_KEY = "water"
WATER = WATER_KEY
EL = EL_KEY
GAS = GAS_KEY

component_list = []

_IMMUTABLE_PRIMITIVES = frozenset({int, float, bool, str, bytes, complex, type(None)})


def _deepcopy_skip_immutables(obj, memo):
    new = obj.__class__.__new__(obj.__class__)
    memo[id(obj)] = new
    for k, v in obj.__dict__.items():
        if v is None or v.__class__ in _IMMUTABLE_PRIMITIVES:
            new.__dict__[k] = v
        else:
            new.__dict__[k] = copy.deepcopy(v, memo)
    return new


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


def is_plain_number(value) -> bool:
    """True for a plain int/float (bool excluded), i.e. not a solver Var."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def lower(var_or_const):
    """Return the lower bound of a ``Var`` (or its value if unbounded); pass through otherwise."""
    if isinstance(var_or_const, Var):
        if var_or_const.min is None:
            return var_or_const.value
        return var_or_const.min
    return var_or_const


def value(var_or_const):
    """Extract ``.value`` from a ``Var``/``Const``/``Intermediate``/``PostProcess``; pass through plain scalars."""
    if isinstance(var_or_const, Const | Var | Intermediate | PostProcess):
        return var_or_const.value
    return var_or_const


def set_initial_value(var, value) -> None:
    """Seed a variable's initial guess / warm start, backend-agnostically.

    A model attribute may be a monee :class:`Var` *or*, once a backend has run
    ``inject_vars`` (e.g. in ``activate_timeseries``), the backend's own variable
    object. monee ``Var`` / GEKKO ``GKVariable`` / Pyomo ``Var`` all expose a
    settable ``value``; gurobipy ``Var`` instead carries its warm start in
    ``Start`` (and has no ``value``, so ``var.value = ...`` raises). Duck-type on
    the available attribute so callers in the model layer never need to know
    which solver backend is active.

    A backend symbol that seeds its warm start elsewhere (e.g. the CasADi
    ``CasSym``, which captures ``x0`` at ``inject_vars`` time and exposes neither
    attribute) is deliberately a silent no-op. A plain scalar/string target is
    never a valid warm-start sink, so reject it loudly to catch the wrong-target
    programming error the no-op would otherwise mask.
    """
    if hasattr(var, "value"):
        var.value = value
    elif hasattr(var, "Start"):  # gurobipy.Var
        var.Start = value
    elif isinstance(var, (int, float, complex, str, bool)):
        raise TypeError(
            f"set_initial_value got a non-variable target {var!r}; expected a "
            "backend variable with a settable 'value' or 'Start'"
        )


class Var:
    """
    A decision variable (or mutable parameter) in the optimisation model.

    Sign convention for child components: positive = consumption (load),
    negative = generation. ``PowerGenerator(p_mw=5)`` stores ``p_mw=-5``.

    Comparison operators compare against the *bound*, not the value, and return
    ``False`` if the relevant bound is ``None``.
    """

    def __init__(
        self, value, max=None, min=None, integer=False, name=None, scale=1.0
    ) -> None:

        if not isinstance(value, float | int):
            raise ValueError(
                "The initial guess for the variable need to a numeric value!"
            )
        self.value = value
        self.max = max
        self.min = min

        self.integer = integer
        self.name = name

        self.scale = scale

    def __neg__(self):
        actual_max = None if self.max is None else -self.max
        actual_min = None if self.min is None else -self.min
        return type(self)(
            value=-self.value,
            max=actual_min,
            min=actual_max,
            integer=self.integer,
            name=self.name,
            scale=self.scale,
        )

    def __mul__(self, other):
        if isinstance(other, Var | Const | Intermediate | PostProcess):
            raise TypeError(
                f"Cannot multiply a Var with a {type(other).__name__} before "
                "solver injection. Build such expressions inside equations(), "
                "which run after inject_vars, or use plain numbers."
            )
        if isinstance(other, (int, float)) and other < 0:
            new_max = None if self.min is None else self.min * other
            new_min = None if self.max is None else self.max * other
        else:
            new_max = None if self.max is None else self.max * other
            new_min = None if self.min is None else self.min * other
        return type(self)(
            value=self.value * other,
            max=new_max,
            min=new_min,
            integer=self.integer,
            name=self.name,
            scale=self.scale,
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
        new.scale = self.scale
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


class PostProcess:
    """A report-only quantity computed *after* the solve, outside the solver,
    from the solved model values via ``fn(values)``, where ``values`` is a
    namespace of the model's solved fields (read as ``values.vm_pu``).

    Unlike :class:`Intermediate`, it is never injected as a solver variable nor
    referenced by any equation - it carries no degrees of freedom and cannot
    affect convergence or squareness. The clean home for derived reports (e.g.
    ``vm_pu_squared = vm_pu^2``): physics stays in the solver, reporting outside.
    """

    def __init__(self, fn, value=0):
        self.fn = fn  # callable(values_namespace) -> number
        self.value = value

    def __repr__(self):
        return f"PostProcess({self.value!r})"

    def __deepcopy__(self, memo):
        new = PostProcess.__new__(PostProcess)
        memo[id(self)] = new
        new.fn = self.fn  # a function is atomic - share by reference
        new.value = self.value
        return new


def store_unknown_kwargs(model, kwargs):
    """Warn about keywords no model attribute claims and return them unchanged.

    They are still kept in ``_ext_data`` for one deprecation cycle, but nothing
    ever reads that dict and ``vars``/``values`` skip it, so a typo or a
    misremembered option (``cost=``) is otherwise lost without a trace.
    """
    if kwargs:
        warnings.warn(
            f"{type(model).__name__} ignored unknown keyword argument(s) "
            f"{sorted(kwargs)}: they are not part of the model state and are "
            "neither solved nor serialized. Passing them will become an error.",
            stacklevel=3,
        )
    return kwargs


class UnknownAttributeWarning(UserWarning):
    """Category of the warning emitted when a plain value is assigned to a
    public attribute a model does not declare, e.g.
    ``PowerGenerator(...).p_mw_max = 5``.

    It fires when all of the following hold: the target is a
    :class:`GenericModel` whose construction has finished; the write comes from
    outside monee itself; the name has no leading underscore; the name is not
    already in the instance ``__dict__``, not a class attribute (so a property
    setter never triggers it), not a parameter of the model's ``__init__`` (an
    optional one such as ``cost`` stays settable) and not registered through
    :func:`register_optional_attributes`; and the assigned value is a plain
    scalar, string, sequence or mapping.

    It therefore stays silent for: any write during ``__init__`` or during a
    :meth:`CompoundModel.create`; assignment of a :class:`Var`,
    :class:`Const`, :class:`Intermediate` or :class:`PostProcess`, which
    declares new solver state on purpose (this is how formulations add their
    own variables); and any attribute of a model rebuilt from a saved network
    file, which is restored without running ``__init__`` and so is never armed.

    Silence it with
    ``warnings.filterwarnings("ignore", category=UnknownAttributeWarning)``,
    or turn the whole guard off with ``set_attribute_guard("off")``. The
    concepts/data_model docs page explains which attributes are settable.
    """


ATTRIBUTE_GUARD_MODES = ("warn", "error", "off")

_ATTR_GUARD = os.environ.get("MONEE_ATTRIBUTE_GUARD", "warn").strip().lower()
if _ATTR_GUARD not in ATTRIBUTE_GUARD_MODES:
    warnings.warn(
        f"MONEE_ATTRIBUTE_GUARD={_ATTR_GUARD!r} is not one of "
        f"{list(ATTRIBUTE_GUARD_MODES)}; falling back to 'warn'.",
        stacklevel=2,
    )
    _ATTR_GUARD = "warn"

_LOCK_KEY = "_attrs_locked"
# Ellipsis, not True: the native format skips private attributes it cannot
# encode, so the marker never reaches a saved file, and ``copy`` treats it as
# atomic so model copies stay cheap.
_ARMED = ...

# Values that are inert unless an equation reads them; a Var/Const/Intermediate
# /PostProcess or a back-end symbol is a deliberate state declaration instead.
_PLAIN_VALUE_TYPES = (int, float, complex, bool, str, bytes, list, tuple, dict, set)

# Attributes the library itself writes onto a model after construction and reads
# back with getattr/hasattr, keyed by the model class name that owns them.
_OPTIONAL_ATTRIBUTES: dict[str, set[str]] = {
    "GenericPowerBranch": {"kind"},
}

_optional_cache: dict[type, frozenset] = {}

_building = 0


def register_optional_attributes(model_class, *names: str) -> None:
    """Declare *names* as attributes of *model_class* that may be assigned after
    construction, so the unknown-attribute guard stays quiet about them.

    For attributes a library reads back with ``getattr``/``hasattr`` instead of
    declaring them in ``__init__``. Constructor parameters are exempt already.
    """
    class_name = model_class if isinstance(model_class, str) else model_class.__name__
    _OPTIONAL_ATTRIBUTES.setdefault(class_name, set()).update(names)
    _optional_cache.clear()


def _optional_attribute_names(cls) -> frozenset:
    cached = _optional_cache.get(cls)
    if cached is not None:
        return cached
    names: set[str] = set()
    for klass in cls.__mro__:
        names |= _OPTIONAL_ATTRIBUTES.get(klass.__name__, set())
    try:
        names |= {
            p
            for p in inspect.signature(cls.__init__).parameters
            if p not in ("self", "kwargs", "args")
        }
    except (TypeError, ValueError):
        pass
    result = frozenset(names)
    _optional_cache[cls] = result
    return result


def attribute_guard() -> str:
    """Current mode of the unknown-attribute guard: ``warn``, ``error`` or ``off``."""
    return _ATTR_GUARD


def set_attribute_guard(mode: str) -> str:
    """Set the unknown-attribute guard mode and return the previous one.

    ``warn`` (default) emits :class:`UnknownAttributeWarning`, ``error`` is the
    strict mode that raises :class:`AttributeError` so CI fails on it (also
    reachable without code as ``MONEE_ATTRIBUTE_GUARD=error``), ``off`` removes
    the hook from the model base class so attribute writes run at plain-object
    speed again. See the concepts/data_model docs page.
    """
    global _ATTR_GUARD
    if mode not in ATTRIBUTE_GUARD_MODES:
        raise ValueError(
            f"unknown attribute guard mode {mode!r}; expected one of "
            f"{list(ATTRIBUTE_GUARD_MODES)}"
        )
    previous, _ATTR_GUARD = _ATTR_GUARD, mode
    _install_attribute_guard(mode != "off")
    return previous


def _settable_attribute_names(model) -> list[str]:
    cls = type(model)
    names = {k for k in model.__dict__ if k[0] != "_"}
    names |= {n for n in _optional_attribute_names(cls) if n[0] != "_"}
    for klass in cls.__mro__:
        for name, member in vars(klass).items():
            if name[0] != "_" and isinstance(member, property) and member.fset:
                names.add(name)
    return sorted(names)


def _written_by_monee() -> bool:
    """True when the assignment that reached the guard came from monee itself.

    monee's own layers write attributes a model class cannot declare: a
    formulation adds its variables, an importer marks a branch, the timeseries
    driver pushes the objective parameters the caller named. The guard reports
    what the caller wrote, so library writes are out of scope.
    """
    try:
        # 0 here, 1 _report_unknown_attribute, 2 _guarded_setattr, 3 the write.
        module = sys._getframe(3).f_globals.get("__name__", "")
    except ValueError:
        return False
    return module == "monee" or module.startswith("monee.")


def _report_unknown_attribute(model, name, value) -> None:
    if _ATTR_GUARD == "off" or _building or name[0] == "_":
        return
    if not isinstance(value, _PLAIN_VALUE_TYPES) and value is not None:
        return
    cls = type(model)
    if hasattr(cls, name) or name in _optional_attribute_names(cls):
        return
    if _written_by_monee():
        return
    message = (
        f"{type(model).__name__} does not declare the attribute {name!r}: the "
        f"assignment stores an inert field that no equation reads, so the "
        f"solve silently ignores it. Settable attributes of this model: "
        f"{_settable_attribute_names(model)}. Use a leading underscore "
        f"(_{name}) for your own metadata. The concepts/data_model docs page "
        f"explains which attributes are settable after construction."
    )
    if _ATTR_GUARD == "error":
        raise AttributeError(message)
    warnings.warn(message, UnknownAttributeWarning, stacklevel=3)


def _arming_call(cls, *args, **kwargs):
    # type.__call__ rather than super(): ABCMeta adds no __call__ of its own,
    # and this runs once per model construction.
    instance = type.__call__(cls, *args, **kwargs)
    instance.__dict__[_LOCK_KEY] = _ARMED
    return instance


def _suspending_guard(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global _building
        _building += 1
        try:
            return func(*args, **kwargs)
        finally:
            _building -= 1

    wrapper._suspends_guard = True
    return wrapper


def _guarded_setattr(self, name, value) -> None:
    instance_dict = self.__dict__
    if name in instance_dict:
        # A name already in the instance dict cannot be shadowing a data
        # descriptor, because descriptor writes never reach this branch.
        instance_dict[name] = value
        return
    if _LOCK_KEY in instance_dict:
        _report_unknown_attribute(self, name, value)
    object.__setattr__(self, name, value)


def _install_attribute_guard(enabled: bool) -> None:
    """Attach or detach the guard hooks so ``off`` costs nothing at all: without
    them ``__setattr__`` and instantiation fall back to their C slots."""
    if enabled:
        _ModelMeta.__call__ = _arming_call
        GenericModel.__setattr__ = _guarded_setattr
        return
    for holder, hook in ((_ModelMeta, "__call__"), (GenericModel, "__setattr__")):
        if hook in vars(holder):
            delattr(holder, hook)


class _ModelMeta(ABCMeta):
    """Arms the unknown-attribute guard once the outermost ``__init__`` returns.

    The arming hook is installed on this metaclass by
    :func:`_install_attribute_guard`, not declared in the body, so turning the
    guard off restores plain ``type.__call__`` instantiation.
    """


class GenericModel(ABC, metaclass=_ModelMeta):
    """
    Base class for all component models (nodes, branches, children, compounds).

    Public attributes (no leading ``_``) form the solver-visible state. Use
    :class:`Var` for decision variables, :class:`Const` for pinned setpoints,
    plain scalars for parameters.

    Once ``__init__`` has returned, assigning a plain value to a public name the
    model does not declare emits :class:`UnknownAttributeWarning` (see
    :func:`set_attribute_guard`); names with a leading underscore are free for
    caller metadata.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self._ext_data = store_unknown_kwargs(self, kwargs)

    @property
    def vars(self):
        return {k: v for k, v in self.__dict__.items() if k[0] != "_"}

    @property
    def values(self):
        return {k: value(v) for k, v in self.__dict__.items() if k[0] != "_"}

    def is_cp(self):
        return False

    def __deepcopy__(self, memo):
        return _deepcopy_skip_immutables(self, memo)


_install_attribute_guard(_ATTR_GUARD != "off")


class NodeModel(GenericModel):
    """Abstract base class for node models (buses, junctions). Subclasses define nodal equations."""

    @abstractmethod
    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        """Return nodal equations (e.g. flow conservation, voltage balance)."""

    def minimize(self, _grid, _in_branch_models, _out_branch_models, _childs, **kwargs):
        """Optional objective contribution (e.g. slack penalties). Default: none."""
        return []


class BranchModel(GenericModel):
    """Abstract base class for network branches (lines, pipes). Subclasses define branch equations.

    Custom subclasses must call ``super().__init__()``: it provides the
    ``on_off`` default (1 = in service) that the nodal balances multiply every
    branch flow with. See the branch author contract in
    ``docs/source/concepts/data_model.rst`` for the attributes formulations
    expect on a branch.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.on_off = 1

    @abstractmethod
    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        pass

    def minimize(self, _grid, _from_node_model, _to_node_model, **kwargs):
        return []

    def loss_percent(self):
        return 0

    def init(self, grid):
        """Optional pre-solve initialization hook for the branch. Default: no-op."""


class MultiGridBranchModel(BranchModel):
    """Branch that couples multiple grid domains (e.g. CHP, P2H). Always a control point."""

    @abstractmethod
    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        pass

    def is_cp(self):
        return True

    def init(self, grids):
        """Optional pre-solve initialization hook spanning grids. Default: no-op."""


class MultiGridNodeModel(NodeModel):
    """Node that participates in multiple grid domains. Always a control point."""

    def is_cp(self):
        return True


class CompoundModel(GenericModel):
    """
    Composite component spanning multiple nodes/carriers (e.g. P2H unit).

    Subclasses implement :meth:`create` to add sub-components to the network,
    and may override :meth:`equations` for coupling constraints.

    ``create`` is a second construction phase: it derives sizing state and
    records the ids of the sub-components it added, so the unknown-attribute
    guard stays quiet for the duration of the call. Writes outside ``create``
    are guarded like any other model's.
    """

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        create = cls.__dict__.get("create")
        if create is not None and not getattr(create, "_suspends_guard", False):
            cls.create = _suspending_guard(create)

    @abstractmethod
    def create(self, network):
        """Add sub-components (nodes, branches, children) to *network*."""

    def equations(self, _network, **kwargs):
        return []

    def minimize(self, _network, **kwargs):
        return []

    def sync(self):
        """Push attributes changed after :meth:`create` onto the sub-components.

        A compound's own attributes are only build-time seeds; without this hook
        a timeseries or override written to the compound never reaches the model
        that carries the equations. Default: no-op."""


class MultiGridCompoundModel(CompoundModel):
    pass


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

    def minimize(self, _grid, _node_model, **kwargs):
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

        # Declarative only - which equations to use. Variable declaration
        # (ensure_var) is deferred to the solver's attach_formulations() pass,
        # which runs on the solve-time network copy.
        self.formulation = formulation
        # An explicitly passed formulation is pinned: it survives a
        # solver-level formulation override.
        self.formulation_pinned = formulation is not None

    @property
    def tid(self):
        return f"{self.__class__.__name__}-{self.id}".lower()

    @property
    def nid(self):
        return f"{self.model.__class__.__name__}-{self.id}".lower()

    def equations(self, *args, **kwargs):
        model_eqs = self.model.equations(*args, **kwargs)
        form_eqs = (
            []
            if self.formulation is None
            else self.formulation.equations(self.model, *args, **kwargs)
        )
        return (
            form_eqs
            + model_eqs
            + [c(self.model, *args, **kwargs) for c in self.constraints]
        )

    def minimize(self, *args, **kwargs):
        model_eqs = self.model.minimize(*args, **kwargs)
        form_eqs = (
            []
            if self.formulation is None
            else self.formulation.minimize(self.model, *args, **kwargs)
        )
        return form_eqs + model_eqs

    def __deepcopy__(self, memo):
        return _deepcopy_skip_immutables(self, memo)


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
        self.from_branch_ids = []
        self.to_branch_ids = []
        self.position = position

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

    @property
    def tid(self):
        # Include the MultiGraph key so parallel branches get distinct tids.
        if self.id[0] > self.id[1]:
            return f"branch-{self.id[0]}-{self.id[1]}-{self.id[2]}"
        else:
            return f"branch-{self.id[1]}-{self.id[0]}-{self.id[2]}"
