from .core import ChildModel, Const, Var, model


class NoVarChildModel(ChildModel):
    """
    No docstring provided.
    """

    def set(self, n, value):
        """
        No docstring provided.
        """
        user_attributes = [
            attr
            for attr in dir(self)
            if not attr.startswith("__") and (not callable(getattr(self, attr)))
        ]
        if n < 0 or n >= len(user_attributes):
            raise IndexError(f"No user-defined attribute at index {n}")
        attr_name = user_attributes[n]
        setattr(self, attr_name, value)

    def equations(self, grid, node, **kwargs):
        """
        No docstring provided.
        """
        return []


@model
class PowerGenerator(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_mw = -p_mw
        self.q_mvar = -q_mvar


@model
class ExtPowerGrid(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, p_mw, q_mvar, vm_pu=1, va_degree=0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_mw = Var(p_mw, name="ext_grid_p_mw")
        self.q_mvar = Var(q_mvar, name="ext_grid_q_mvar")
        self.vm_pu = vm_pu
        self.va_degree = va_degree

    def overwrite(self, node_model, grid):
        """
        No docstring provided.
        """
        node_model.vm_pu = Const(self.vm_pu)
        node_model.va_degree = Const(self.va_degree)


@model
class PowerLoad(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, p_mw, q_mvar, **kwargs) -> None:
        super().__init__(**kwargs)
        self.p_mw = p_mw
        self.q_mvar = q_mvar


@model
class Source(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, mass_flow, t_k=359, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = -mass_flow


@model
class ExtHydrGrid(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, mass_flow=-1, pressure_pu=1, t_k=356, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = Var(mass_flow, name="ext_grid_mass_flow")
        self.pressure_pu = pressure_pu
        self.t_k = t_k

    def overwrite(self, node_model, grid):
        """
        No docstring provided.
        """
        node_model.pressure_pu = Const(self.pressure_pu)
        node_model.pressure_squared_pu = Const(self.pressure_pu**2)
        node_model.t_pu = Const(self.t_k / grid.t_ref)
        node_model.t_k = Const(self.t_k)


@model
class ConsumeHydrGrid(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, mass_flow=0.1, pressure_pu=1, t_k=293, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = -mass_flow
        self.pressure_pu = pressure_pu
        self.t_k = t_k

    def overwrite(self, node_model, grid):
        """
        No docstring provided.
        """
        node_model.pressure_pu = Const(self.pressure_pu)
        node_model.pressure_squared_pu = Const(self.pressure_pu**2)


@model
class Sink(NoVarChildModel):
    """
    No docstring provided.
    """

    def __init__(self, mass_flow, **kwargs) -> None:
        super().__init__(**kwargs)
        self.mass_flow = mass_flow
