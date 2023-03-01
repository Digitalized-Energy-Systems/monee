from .phys.nl.hydraulics import junction_mass_flow_balance
from .phys.nl.opf import power_balance_equation
from .core import NodeModel, Var, model


@model
class Bus(NodeModel):
    def __init__(self, base_kv) -> None:
        super().__init__()

        self.base_kv = base_kv

        self.vm_pu = Var(1)
        self.va_degree = Var(0)
        self.p_mw = Var(1)
        self.q_mvar = Var(1)

    def calc_signed_power_values(self, from_branch_models, to_branch_models, connected_node_models):
        signed_active_power = (
            [model.vars["p_from_mw"] for model in from_branch_models]
            + [model.vars["p_to_mw"] for model in to_branch_models]
            + [model.vars["p_mw"] for model in connected_node_models]
        )
        signed_reactive_power = (
            [model.vars["q_from_mvar"] for model in from_branch_models]
            + [model.vars["q_to_mvar"] for model in to_branch_models]
            + [model.vars["q_mvar"] for model in connected_node_models]
        )
        return signed_active_power, signed_reactive_power
    
    def p_mw_equation(self, from_branch_models, to_branch_models):
        return self.p_mw == sum([model.vars["p_from_mw"] for model in from_branch_models]
                + [model.vars["p_to_mw"] for model in to_branch_models])
    
    def q_mvar_equation(self, from_branch_models, to_branch_models):
        return self.q_mvar == sum([model.vars["q_from_mvar"] for model in from_branch_models]
                + [model.vars["q_to_mvar"] for model in to_branch_models])
    
    def equations(
        self, grid, from_branch_models, to_branch_models, connected_node_models, **kwargs
    ):
        signed_ap, signed_rp = self.calc_signed_power_values(from_branch_models, to_branch_models, connected_node_models)
        
        return (
            self.p_mw_equation(from_branch_models, to_branch_models),
            self.q_mvar_equation(from_branch_models, to_branch_models),
            power_balance_equation(signed_ap),
            power_balance_equation(signed_rp),
        )


@model
class Junction(NodeModel):
    def __init__(self) -> None:
        self.t_k = Var(300)
        self.pressure_pa = Var(1)

    def calc_signed_mass_flow(self, from_branch_models, to_branch_models, connected_node_models):
        return (
            [model.vars["mass_flow"] for model in from_branch_models]
            + [model.vars["mass_flow"] for model in to_branch_models]
            + [model.mass_flow for model in connected_node_models]
        )

    def equations(
        self, grid, from_branch_models, to_branch_models, connected_node_models, **kwargs
    ):
        return junction_mass_flow_balance(self.calc_signed_mass_flow(from_branch_models, to_branch_models, connected_node_models))
