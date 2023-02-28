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
        self.p_mw = Var(10000)
        self.q_mvar = Var(10000)

    def equations(
        self, grid, in_branch_models, out_branch_models, connected_node_models, **kwargs
    ):
        signed_active_power = (
            [model.vars["p_from_mw"] for model in in_branch_models]
            + [model.vars["p_to_mw"] for model in out_branch_models]
            + [model.vars["p_mw"] for model in connected_node_models]
        )
        signed_reactive_power = (
            [model.vars["q_from_mvar"] for model in in_branch_models]
            + [model.vars["q_to_mvar"] for model in out_branch_models]
            + [model.vars["q_mvar"] for model in connected_node_models]
        )

        return (
            self.p_mw == sum([model.vars["p_from_mw"] for model in in_branch_models]
                + [model.vars["p_to_mw"] for model in out_branch_models]),
            self.q_mvar == sum([model.vars["q_from_mvar"] for model in in_branch_models]
                + [model.vars["q_to_mvar"] for model in out_branch_models]),
            power_balance_equation(signed_active_power),
            power_balance_equation(signed_reactive_power),
        )


@model
class Junction(NodeModel):
    def __init__(self) -> None:
        self.t_k = Var(300)
        self.pressure_pa = Var(1)

    def equations(
        self, grid, in_branch_models, out_branch_models, connected_node_models, **kwargs
    ):
        signed_mass_flows = (
            [model.vars["mass_flow"] for model in in_branch_models]
            + [-model.vars["mass_flow"] for model in out_branch_models]
            + [model.mass_flow for model in connected_node_models]
        )
        return junction_mass_flow_balance(signed_mass_flows)
