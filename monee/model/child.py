from .core import ChildModel, Var, Const, model


class NoVarChildModel(ChildModel):
    def equations(self, grid, node, **kwargs):
        return []


@model
class PowerGenerator(NoVarChildModel):
    def __init__(self, p_mw, q_mvar) -> None:
        super().__init__()
        self.p_mw = -p_mw
        self.q_mvar = -q_mvar


@model
class ExtPowerGrid(NoVarChildModel):
    def __init__(self, p_mw, q_mvar, vm_pu, va_degree) -> None:
        super().__init__()
        self.p_mw = Var(p_mw)
        self.q_mvar = Var(q_mvar)

        self.vm_pu = vm_pu
        self.va_degree = va_degree

    def overwrite(self, node_model):
        node_model.vm_pu = Const(self.vm_pu)
        node_model.va_degree = Const(self.va_degree)


@model
class PowerLoad(NoVarChildModel):
    def __init__(self, p_mw, q_mvar) -> None:
        super().__init__()
        self.p_mw = p_mw
        self.q_mvar = q_mvar


@model
class Source(NoVarChildModel):
    def __init__(self, mass_flow) -> None:
        super().__init__()
        self.mass_flow = mass_flow


@model
class ExtHydrGrid(NoVarChildModel):
    def __init__(self, mass_flow, pressure_pa, t_k) -> None:
        super().__init__()
        self.mass_flow = Var(mass_flow)

        self.pressure_pa = pressure_pa
        self.t_k = t_k

    def overwrite(self, node_model):
        node_model.pressure_pa = Const(self.pressure_pa)
        node_model.t_k = Const(self.t_k)


@model
class Sink(NoVarChildModel):
    def __init__(self, mass_flow) -> None:
        super().__init__()
        self.mass_flow = -mass_flow