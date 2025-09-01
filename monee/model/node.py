from .core import Intermediate, IntermediateEq, NodeModel, Var, model
from .phys.nl.hydraulics import junction_mass_flow_balance
from .phys.nl.opf import power_balance_equation


@model
class Bus(NodeModel):
    def __init__(self, base_kv) -> None:
        super().__init__()

        self.base_kv = base_kv

        self.vm_pu = Var(1)
        self.va_degree = Var(0)
        self.p_mw = Intermediate()
        self.q_mvar = Intermediate()

    def calc_signed_power_values(
        self, from_branch_models, to_branch_models, connected_node_models
    ):
        signed_active_power = (
            [
                model.vars["p_from_mw"] * model.vars["on_off"]
                for model in from_branch_models
            ]
            + [
                model.vars["p_to_mw"] * model.vars["on_off"]
                for model in to_branch_models
            ]
            + [
                model.vars["p_mw"] * model.vars["regulation"]
                for model in connected_node_models
            ]
        )
        signed_reactive_power = (
            [
                model.vars["q_from_mvar"] * model.vars["on_off"]
                for model in from_branch_models
            ]
            + [
                model.vars["q_to_mvar"] * model.vars["on_off"]
                for model in to_branch_models
            ]
            + [
                model.vars["q_mvar"] * model.vars["regulation"]
                for model in connected_node_models
            ]
        )
        return signed_active_power, signed_reactive_power

    def p_mw_equation(self, child_models):
        return IntermediateEq(
            "p_mw",
            sum(
                [
                    model.vars["p_mw"] * model.vars["regulation"]
                    for model in child_models
                ]
            ),
        )

    def q_mvar_equation(self, child_models):
        return IntermediateEq(
            "q_mvar",
            sum(
                [
                    model.vars["q_mvar"] * model.vars["regulation"]
                    for model in child_models
                ]
            ),
        )

    def equations(
        self,
        grid,
        from_branch_models,
        to_branch_models,
        connected_node_models,
        **kwargs,
    ):
        signed_ap, signed_rp = self.calc_signed_power_values(
            from_branch_models, to_branch_models, connected_node_models
        )

        return (
            self.p_mw_equation(connected_node_models),
            self.q_mvar_equation(connected_node_models),
            power_balance_equation(signed_ap),
            power_balance_equation(signed_rp),
        )


@model
class Junction(NodeModel):
    def __init__(self) -> None:
        self.t_k = Intermediate()
        self.t_pu = Var(1)
        self.pressure_pa = Intermediate()
        self.pressure_pu = Var(1)

    def calc_signed_mass_flow(
        self, from_branch_models, to_branch_models, connected_node_models
    ):
        return (
            # mass flow balance
            [
                model.vars["from_mass_flow"] * model.vars["on_off"]
                for model in from_branch_models
                if "from_mass_flow" in model.vars
            ]
            + [
                model.vars["to_mass_flow"] * model.vars["on_off"]
                for model in to_branch_models
                if "to_mass_flow" in model.vars
            ]
            + [
                -model.vars["mass_flow"] * model.vars["on_off"]
                for model in from_branch_models
                if "mass_flow" in model.vars
            ]
            + [
                model.vars["mass_flow"] * model.vars["on_off"]
                for model in to_branch_models
                if "mass_flow" in model.vars
            ]
            + [
                model.vars["mass_flow"] * model.vars["regulation"]
                for model in connected_node_models
                if "mass_flow" in model.vars
            ]
        )

    def calc_signed_heat_flow(
        self, from_branch_models, to_branch_models, connected_node_models, grid
    ):
        temp_supported = (
            len(from_branch_models) > 0
            and "t_average_k" in from_branch_models[0].vars
            or len(to_branch_models) > 0
            and "t_average_k" in to_branch_models[0].vars
        )

        if temp_supported:
            return (
                [
                    -model.vars["mass_flow"]
                    * model.vars["on_off"]
                    * (model.vars["t_from_pu"])
                    if "t_from_pu" in model.vars
                    else 0
                    for model in from_branch_models
                    if "mass_flow" in model.vars
                ]
                + [
                    model.vars["mass_flow"]
                    * model.vars["on_off"]
                    * (model.vars["t_to_pu"])
                    if "t_to_pu" in model.vars
                    else 0
                    for model in to_branch_models
                    if "mass_flow" in model.vars
                ]
                + [
                    model.vars["mass_flow"] * model.vars["regulation"] * self.t_pu
                    for model in connected_node_models
                    if "mass_flow" in model.vars
                ]
            )
        else:
            return [0]

    def equations(
        self,
        grid,
        from_branch_models,
        to_branch_models,
        connected_node_models,
        **kwargs,
    ):
        mass_flow_signed_list = self.calc_signed_mass_flow(
            from_branch_models, to_branch_models, connected_node_models
        )
        energy_flow_list = self.calc_signed_heat_flow(
            from_branch_models, to_branch_models, connected_node_models, grid
        )
        if mass_flow_signed_list:
            return (
                junction_mass_flow_balance(mass_flow_signed_list),
                junction_mass_flow_balance(energy_flow_list),
                IntermediateEq("t_k", self.t_pu * grid.t_ref),
                IntermediateEq("pressure_pa", self.pressure_pu * grid.pressure_ref),
                # self.t_k == self.t_pu * grid.t_ref,
                # self.pressure_pu * grid.pressure_ref == self.pressure_pa,
            )
        return []
