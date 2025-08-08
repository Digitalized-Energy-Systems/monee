from abc import ABC, abstractmethod

EL_KEY = "electricity"
GAS_KEY = "gas"
WATER_KEY = "water"

WATER = WATER_KEY
EL = EL_KEY
GAS = GAS_KEY

component_list = []


# class decorator
def model(cls):
    component_list.append(cls)
    return cls


def upper(var_or_const):
    if isinstance(var_or_const, Var):
        if var_or_const.max is None:
            return var_or_const.value
        return var_or_const.max
    return var_or_const


def lower(var_or_const):
    if isinstance(var_or_const, Var):
        if var_or_const.min is None:
            return var_or_const.value
        return var_or_const.min
    return var_or_const


def value(var_or_const):
    if isinstance(var_or_const, Const | Var):
        return var_or_const.value
    return var_or_const


class Var:
    def __init__(self, value, max=None, min=None, integer=False) -> None:
        self.value = value
        self.max = max
        self.min = min
        self.integer = integer

    def __neg__(self):
        return Var(value=-self.value, max=self.max, min=self.min)

    def __mul__(self, other):
        return Var(value=self.value * other, max=self.max, min=self.min)

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

    @property
    def values(self):
        return {k: value(v) for (k, v) in self.__dict__.items() if k[0] != "_"}

    def is_cp(self):
        return False


class NodeModel(GenericModel):
    @abstractmethod
    def equations(self, grid, in_branch_models, out_branch_models, childs, **kwargs):
        pass


class BranchModel(GenericModel):
    @abstractmethod
    def equations(self, grid, from_node_model, to_node_model, **kwargs):
        pass

    def loss_percent(self):
        return 0

    def is_cp(self):
        return False


class MultiGridBranchModel(BranchModel):
    @abstractmethod
    def equations(self, grids, from_node_model, to_node_model, **kwargs):
        pass

    def is_cp(self):
        return True


class MultiGridNodeModel(NodeModel):
    def is_cp(self):
        return True


class CompoundModel(GenericModel):
    @abstractmethod
    def create(self, network):
        pass

    def equations(self, network, **kwargs):
        return []


class MultGridCompoundModel(CompoundModel):
    def is_cp(self):
        return True


class ChildModel(GenericModel):
    def __init__(self, regulation: int = 1, **kwargs):
        super().__init__(**kwargs)

        self.regulation = regulation

    def overwrite(self, node_model):
        # optional override
        pass

    @abstractmethod
    def equations(self, grid, node_model, **kwargs):
        pass


class Component(ABC):
    def __init__(
        self,
        id,
        model,
        constraints,
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

    @property
    def tid(self):
        return f"{self.__class__.__name__}-{self.id}".lower()

    @property
    def nid(self):
        return f"{self.model.__class__.__name__}-{self.id}".lower()


class Child(Component):
    def __init__(
        self,
        child_id,
        model,
        constraints,
        grid=None,
        name=None,
        active=True,
        independent=True,
    ) -> None:
        super().__init__(child_id, model, constraints, grid, name, active, independent)

        self.node_id = None
        self.independent = independent


class Compound(Component):
    def __init__(
        self,
        compound_id,
        model: CompoundModel,
        constraints,
        connected_to,
        subcomponents,
        grid=None,
        name=None,
        active=True,
    ) -> None:
        super().__init__(compound_id, model, constraints, grid, name, active, True)

        self.connected_to = connected_to
        self.subcomponents = subcomponents

    def component_of_type(self, comp_type):
        return [
            component
            for component in self.subcomponents
            if type(component) is comp_type
        ]


class Node(Component):
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
        independent=True,
    ) -> None:
        super().__init__(node_id, model, constraints, grid, name, active, independent)

        self.child_ids = [] if child_ids is None else child_ids
        self.constraints = [] if constraints is None else constraints
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
        switched = (branch_id[1], branch_id[0], branch_id[2])
        self._remove_branch(branch_id)
        self._remove_branch(switched)


class Branch(Component):
    def __init__(
        self,
        model,
        from_node_id,
        to_node_id,
        constraints=None,
        grid=None,
        name=None,
        active=True,
        independent=True,
    ) -> None:
        super().__init__(None, model, constraints, grid, name, active, independent)

        self.from_node_id = from_node_id
        self.to_node_id = to_node_id

    @property
    def tid(self):
        if self.id[0] > self.id[1]:
            return f"branch-{self.id[0]}-{self.id[1]}"
        else:
            return f"branch-{self.id[1]}-{self.id[0]}"
