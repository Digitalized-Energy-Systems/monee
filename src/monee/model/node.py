import math

from .core import Intermediate, IntermediateEq, NodeModel, Var, model
from .phys.core.hydraulics import junction_mass_flow_balance
from .phys.nonlinear.ac import power_balance_equation
from .phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER


@model
class Bus(NodeModel):
    def __init__(self, base_kv) -> None:
        super().__init__()
        self.base_kv = base_kv
        self.vm_pu = Var(1, min=0, max=1.5, name="vm_pu")
        self.vm_pu_squared = Var(1, min=0, max=2.25, name="vm_pu_squared")
        self.va_radians = Var(0, min=-math.pi, max=math.pi, name="va_radians")
        self.va_degree = Intermediate()
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
        return (signed_active_power, signed_reactive_power)

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
        return [
            self.p_mw_equation(connected_node_models),
            self.q_mvar_equation(connected_node_models),
            power_balance_equation(signed_ap),
            power_balance_equation(signed_rp),
            IntermediateEq("va_degree", 180 / math.pi * self.va_radians),
        ]


@model
class Junction(NodeModel):
    def __init__(self) -> None:
        self.t_k = Intermediate()
        self.t_pu = Var(1, min=0, max=2, name="t_pu")
        self.pressure_squared_pu = Var(1, min=0, max=2, name="p_squared_pu")
        self.pressure_pu = Var(1, min=0, max=2, name="p_pu")
        self.mass_flow = Intermediate(1)

    def calc_signed_mass_flow(
        self, from_branch_models, to_branch_models, connected_node_models
    ):
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
                # Linepack: 0.5 splits net packing equally across both endpoints.
                # Outflow-positive: charging (>0) leaves both junctions, hence +.
                0.5 * model.vars["net_pack_kgs"] * model.vars["on_off"]
                for model in from_branch_models
                if "net_pack_kgs" in model.vars
            ]
            + [
                0.5 * model.vars["net_pack_kgs"] * model.vars["on_off"]
                for model in to_branch_models
                if "net_pack_kgs" in model.vars
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
        # LTC / Mcc-DHS replace the degenerate T_n × mass_balance with their own
        # nodal heat balance, so skip emitting it here.
        if getattr(self, "_ltc_active", False) or getattr(
            self, "_mccormick_dhs_active", False
        ):
            return [0]

        temp_supported = any("t_to_pu" in bm.vars for bm in from_branch_models) or any(
            "t_to_pu" in bm.vars for bm in to_branch_models
        )
        if temp_supported:
            Tn = self.t_pu

            terms = []

            # node is FROM-end of these branches
            for bm in from_branch_models:
                if "mass_flow_pos" not in bm.vars or "mass_flow_neg" not in bm.vars:
                    continue
                mpos = bm.vars["mass_flow_pos"] * bm.vars.get("on_off", 1)
                mneg = bm.vars["mass_flow_neg"] * bm.vars.get("on_off", 1)

                Tin = bm.vars["t_from_pu"]
                Tout = self.t_pu * bm.vars.get("on_off", 1)
                terms.append(mneg * Tout - mpos * Tin)

            # node is TO-end of these branches
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

            # Node q_mw_heat (HeatGenerator/HeatLoad) → kg/s·t_pu via c·t_ref/1e6.
            # grid may be None (compound heat balance); scale only used if needed.
            scale_mw_per_kgs = (
                SPECIFIC_HEAT_CAP_WATER * grid.t_ref / 1e6 if grid is not None else None
            )
            for nm in connected_node_models:
                if "q_mw_heat" not in nm.vars:
                    continue
                q = nm.vars["q_mw_heat"] * nm.vars.get("regulation", 1)
                terms.append(q / scale_mw_per_kgs)

            # Branch-level heat injection at the TO end (e.g. GasToHeatHG).
            for bm in to_branch_models:
                if "q_mw_heat" not in bm.vars:
                    continue
                q = bm.vars["q_mw_heat"] * bm.vars.get("on_off", 1)
                terms.append(q / scale_mw_per_kgs)

            # Conduction-style regularizer keeps ∂(heat_bal)/∂T_n non-zero
            # when Σm_out ≈ Σm_in.
            k_reg = getattr(grid, "node_heat_reg_kgs", 0.0)
            if k_reg:
                t_anchor = self.t_pu.value if hasattr(self.t_pu, "value") else 1.0
                terms.append(k_reg * (self.t_pu - t_anchor))
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
        mass_flow_signed_list = self.calc_signed_mass_flow(
            from_branch_models, to_branch_models, connected_node_models
        )
        energy_flow_list = self.calc_signed_heat_flow(
            from_branch_models, to_branch_models, connected_node_models, grid
        )
        if mass_flow_signed_list:
            eqs = [
                junction_mass_flow_balance(mass_flow_signed_list),
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
            # The nodal heat balances over a connected island are linearly
            # dependent (one is redundant). The grid-forming reference node is
            # the heat slack — drop its balance there, exactly as the slack bus
            # absorbs the power balance. Marked once per island by the solver.
            if not getattr(self, "_drop_heat_balance", False):
                eqs.insert(1, junction_mass_flow_balance(energy_flow_list))
            return eqs
        return []
