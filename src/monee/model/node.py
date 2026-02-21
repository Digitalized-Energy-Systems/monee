import math

from .core import Intermediate, IntermediateEq, NodeModel, Var, model
from .phys.core.hydraulics import junction_mass_flow_balance
from .phys.nonlinear.ac import power_balance_equation


@model
class Bus(NodeModel):
    """
    No docstring provided.
    """

    def __init__(self, base_kv) -> None:
        super().__init__()
        self.base_kv = base_kv
        self.vm_pu = Var(1, min=0, name="vm_pu")
        self.vm_pu_squared = Var(1, min=0, name="vm_pu_squared")
        self.va_radians = Var(0, name="va_radians")
        self.va_degree = Intermediate()
        self.p_mw = Intermediate()
        self.q_mvar = Intermediate()

    def calc_signed_power_values(
        self, from_branch_models, to_branch_models, connected_node_models
    ):
        """
        No docstring provided.
        """
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
        return (signed_active_power, signed_reactive_power)

    def p_mw_equation(self, child_models):
        """
        No docstring provided.
        """
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
        """
        No docstring provided.
        """
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
        """
        No docstring provided.
        """
        signed_ap, signed_rp = self.calc_signed_power_values(
            from_branch_models, to_branch_models, connected_node_models
        )
        return [
            self.p_mw_equation(connected_node_models),
            self.q_mvar_equation(connected_node_models),
            power_balance_equation(signed_ap),
            power_balance_equation(signed_rp),
            IntermediateEq("va_degree", 180 / math.pi * self.va_radians),
        ]


@model
class Junction(NodeModel):
    """
    No docstring provided.
    """

    def __init__(self) -> None:
        self.t_k = Intermediate()
        self.t_pu = Var(1, min=0, max=2, name="t_pu")
        self.pressure_squared_pu = Var(1, min=0, max=2, name="p_squared_pu")
        self.pressure_pu = Var(1, min=0, max=2, name="p_pu")
        self.mass_flow = Intermediate(1)

    def calc_signed_mass_flow(
        self, from_branch_models, to_branch_models, connected_node_models
    ):
        """
        No docstring provided.
        """
        return (
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
                -model.vars["mass_flow_pos"] * model.vars["on_off"]
                for model in from_branch_models
                if "mass_flow_pos" in model.vars
            ]
            + [
                model.vars["mass_flow_pos"] * model.vars["on_off"]
                for model in to_branch_models
                if "mass_flow_pos" in model.vars
            ]
            + [
                model.vars["mass_flow_neg"] * model.vars["on_off"]
                for model in from_branch_models
                if "mass_flow_neg" in model.vars
            ]
            + [
                -model.vars["mass_flow_neg"] * model.vars["on_off"]
                for model in to_branch_models
                if "mass_flow_neg" in model.vars
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
        """
        No docstring provided.
        """
        temp_supported = (
            len(from_branch_models) > 0
            and "t_to_pu" in from_branch_models[0].vars
            or (len(to_branch_models) > 0 and "t_to_pu" in to_branch_models[0].vars)
        )
        if temp_supported:
            Tn = self.t_pu

            terms = []

            # node is FROM-end of these branches (branch orientation: this node -> other node)
            for bm in from_branch_models:
                if "mass_flow_pos" not in bm.vars or "mass_flow_neg" not in bm.vars:
                    continue
                mpos = bm.vars["mass_flow_pos"] * bm.vars.get("on_off", 1)
                mneg = bm.vars["mass_flow_neg"] * bm.vars.get("on_off", 1)

                Tin = bm.vars["t_from_pu"]
                Tout = self.t_pu * bm.vars.get("on_off", 1)
                terms.append(mneg * Tout - mpos * Tin)

            # node is TO-end of these branches (branch orientation: other node -> this node)
            for bm in to_branch_models:
                if "mass_flow_pos" not in bm.vars or "mass_flow_neg" not in bm.vars:
                    continue
                mpos = bm.vars["mass_flow_pos"] * bm.vars.get("on_off", 1)
                mneg = bm.vars["mass_flow_neg"] * bm.vars.get("on_off", 1)

                Tin = bm.vars["t_to_pu"]  # inflow at to-end
                Tout = self.t_pu * bm.vars.get("on_off", 1)
                terms.append(-mneg * Tin + mpos * Tout)

            for nm in connected_node_models:
                if "mass_flow" not in nm.vars:
                    continue

                m_ext = nm.vars["mass_flow"] * nm.vars.get("regulation", 1)
                terms.append(m_ext * Tn)
            return terms
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
        """
        No docstring provided.
        """
        mass_flow_signed_list = self.calc_signed_mass_flow(
            from_branch_models, to_branch_models, connected_node_models
        )
        energy_flow_list = self.calc_signed_heat_flow(
            from_branch_models, to_branch_models, connected_node_models, grid
        )
        if mass_flow_signed_list:
            return [
                junction_mass_flow_balance(mass_flow_signed_list),
                junction_mass_flow_balance(energy_flow_list),
                IntermediateEq("t_k", self.t_pu * grid.t_ref),
                IntermediateEq(
                    "mass_flow",
                    sum(
                        [
                            model.vars["mass_flow"] * model.vars["regulation"]
                            for model in connected_node_models
                            if "mass_flow" in model.vars
                        ]
                    ),
                ),
            ]
        return []
