import math

from .core import Intermediate, IntermediateEq, NodeModel, PostProcess, Var, model
from .grid import PowerGrid
from .phys.core.hydraulics import junction_mass_flow_balance
from .phys.nonlinear.ac import power_balance_equation
from .phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER


@model
class Bus(NodeModel):
    def __init__(self, base_kv) -> None:
        super().__init__()
        self.base_kv = base_kv
        self.vm_pu = Var(1, min=0.5, max=1.5, name="vm_pu")
        self.vm_pu_squared = Var(1, min=0.25, max=2.25, name="vm_pu_squared")
        self.va_radians = Var(0, min=-math.pi, max=math.pi, name="va_radians")
        self.va_degree = PostProcess(lambda v: 180 / math.pi * v.va_radians)
        self.p_mw = Intermediate()
        self.q_mvar = Intermediate()

    def calc_signed_power_values(
        self, from_branch_models, to_branch_models, child_models
    ):
        # model.vars builds a dict per access - fetch once per model.
        from_vars = [model.vars for model in from_branch_models]
        to_vars = [model.vars for model in to_branch_models]
        child_vars = [model.vars for model in child_models]
        signed_active_power = (
            [vs["p_from_mw"] * vs["on_off"] for vs in from_vars]
            + [vs["p_to_mw"] * vs["on_off"] for vs in to_vars]
            + [vs["p_mw"] * vs["regulation"] for vs in child_vars]
        )
        signed_reactive_power = (
            [vs["q_from_mvar"] * vs["on_off"] for vs in from_vars]
            + [vs["q_to_mvar"] * vs["on_off"] for vs in to_vars]
            + [vs["q_mvar"] * vs["regulation"] for vs in child_vars]
        )
        return (signed_active_power, signed_reactive_power)

    def p_mw_equation(self, child_models):
        return IntermediateEq(
            "p_mw",
            sum(
                [
                    vs["p_mw"] * vs["regulation"]
                    for vs in (model.vars for model in child_models)
                ]
            ),
        )

    def q_mvar_equation(self, child_models):
        return IntermediateEq(
            "q_mvar",
            sum(
                [
                    vs["q_mvar"] * vs["regulation"]
                    for vs in (model.vars for model in child_models)
                ]
            ),
        )

    def equations(
        self,
        grid,
        from_branch_models,
        to_branch_models,
        child_models,
        **kwargs,
    ):
        signed_ap, signed_rp = self.calc_signed_power_values(
            from_branch_models, to_branch_models, child_models
        )
        self.va_degree = PostProcess(lambda v: 180 / math.pi * v.va_radians)
        sn = grid.sn_mva if isinstance(grid, PowerGrid) else 1.0
        return [
            self.p_mw_equation(child_models),
            self.q_mvar_equation(child_models),
            power_balance_equation(signed_ap, sn),
            power_balance_equation(signed_rp, sn),
        ]


@model
class Junction(NodeModel):
    def __init__(self, **kwargs) -> None:
        # Deliberately no super().__init__(): Junction sits in the cooperative
        # MRO of the coupling control nodes (e.g. CHPControlNode =
        # MultiGridNodeModel, Junction, Bus) where chaining further would hit
        # Bus.__init__ which requires base_kv. Replicate the essential
        # GenericModel state instead.
        self._ext_data = kwargs
        self.t_k = PostProcess(lambda v: float("nan"))
        self.t_pu = Var(1, min=0.3, max=2, name="t_pu")
        self.pressure_squared_pu = Var(1, min=0.5, max=2, name="pressure_squared_pu")
        self.pressure_pu = Var(1, min=0.5, max=2, name="pressure_pu")
        self.mass_flow_kgs = Intermediate(1)

    def calc_signed_mass_flow(self, from_branch_models, to_branch_models, child_models):
        # model.vars builds a dict per access - fetch once per model.
        from_vars = [model.vars for model in from_branch_models]
        to_vars = [model.vars for model in to_branch_models]
        child_vars = [model.vars for model in child_models]
        return (
            [
                vs["from_mass_flow_kgs"] * vs["on_off"]
                for vs in from_vars
                if "from_mass_flow_kgs" in vs
            ]
            + [
                vs["to_mass_flow_kgs"] * vs["on_off"]
                for vs in to_vars
                if "to_mass_flow_kgs" in vs
            ]
            + [
                -vs["mass_flow_pos_kgs"] * vs["on_off"]
                for vs in from_vars
                if "mass_flow_pos_kgs" in vs
            ]
            + [
                vs["mass_flow_pos_kgs"] * vs["on_off"]
                for vs in to_vars
                if "mass_flow_pos_kgs" in vs
            ]
            + [
                vs["mass_flow_neg_kgs"] * vs["on_off"]
                for vs in from_vars
                if "mass_flow_neg_kgs" in vs
            ]
            + [
                -vs["mass_flow_neg_kgs"] * vs["on_off"]
                for vs in to_vars
                if "mass_flow_neg_kgs" in vs
            ]
            + [
                # Linepack: 0.5 splits net packing equally across both endpoints.
                # Outflow-positive: charging (>0) leaves both junctions, hence +.
                0.5 * vs["net_pack_kgs"] * vs["on_off"]
                for vs in from_vars
                if "net_pack_kgs" in vs
            ]
            + [
                0.5 * vs["net_pack_kgs"] * vs["on_off"]
                for vs in to_vars
                if "net_pack_kgs" in vs
            ]
            + [
                vs["mass_flow_kgs"] * vs["regulation"]
                for vs in child_vars
                if "mass_flow_kgs" in vs
            ]
        )

    def calc_signed_heat_flow(  # NOSONAR
        self, from_branch_models, to_branch_models, child_models, grid
    ):
        # LTC / Mcc-DHS replace the degenerate T_n \times mass_balance with their own
        # nodal heat balance, so skip emitting it here.
        if getattr(self, "_ltc_active", False) or getattr(
            self, "_mccormick_dhs_active", False
        ):
            return [0]

        temp_supported = any("t_to_pu" in bm.vars for bm in from_branch_models) or any(
            "t_to_pu" in bm.vars for bm in to_branch_models
        )
        if temp_supported:
            t_n = self.t_pu

            terms = []

            # node is FROM-end of these branches
            for bm in from_branch_models:
                vs = bm.vars
                if "mass_flow_pos_kgs" not in vs or "mass_flow_neg_kgs" not in vs:
                    continue
                on_off = vs.get("on_off", 1)
                mpos = vs["mass_flow_pos_kgs"] * on_off
                mneg = vs["mass_flow_neg_kgs"] * on_off

                t_in = vs["t_from_pu"]
                t_out = self.t_pu * on_off
                terms.append(mneg * t_out - mpos * t_in)

            # node is TO-end of these branches
            for bm in to_branch_models:
                vs = bm.vars
                if "mass_flow_pos_kgs" not in vs or "mass_flow_neg_kgs" not in vs:
                    continue
                on_off = vs.get("on_off", 1)
                mpos = vs["mass_flow_pos_kgs"] * on_off
                mneg = vs["mass_flow_neg_kgs"] * on_off

                t_in = vs["t_to_pu"]  # inflow at to-end
                t_out = self.t_pu * on_off
                terms.append(-mneg * t_in + mpos * t_out)

            for nm in child_models:
                vs = nm.vars
                if "mass_flow_kgs" not in vs:
                    continue

                m_ext = vs["mass_flow_kgs"] * vs.get("regulation", 1)
                t_inj_k = getattr(nm, "injection_t_k", None)
                if t_inj_k is not None and grid is not None:
                    # Defined-temperature injection (Source(t_k=...)): credit
                    # the inflow enthalpy at its own temperature instead of the
                    # node's, keeping the nodal heat balance full-rank in T_n.
                    terms.append(m_ext * (t_inj_k / grid.t_ref_k))
                else:
                    terms.append(m_ext * t_n)

            # Node q_mw_heat (HeatGenerator/HeatLoad) \to kg/s \cdot t_pu via c \cdot t_ref_k/1e6.
            # grid may be None (compound heat balance); scale only used if needed.
            scale_mw_per_kgs = (
                SPECIFIC_HEAT_CAP_WATER * grid.t_ref_k / 1e6
                if grid is not None
                else None
            )
            for nm in child_models:
                vs = nm.vars
                if "q_mw_heat" not in vs:
                    continue
                q = vs["q_mw_heat"] * vs.get("regulation", 1)
                terms.append(q / scale_mw_per_kgs)

            # Branch-level heat injection at the TO end (e.g. GasToHeatHG).
            for bm in to_branch_models:
                vs = bm.vars
                if "q_mw_heat" not in vs:
                    continue
                q = vs["q_mw_heat"] * vs.get("on_off", 1)
                terms.append(q / scale_mw_per_kgs)

            # Conduction-style regularizer keeps \partial(heat_bal)/\partial T_n non-zero
            # when \sum m_out \approx \sum m_in.
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
        child_models,
        **kwargs,
    ):
        mass_flow_signed_list = self.calc_signed_mass_flow(
            from_branch_models, to_branch_models, child_models
        )
        energy_flow_list = self.calc_signed_heat_flow(
            from_branch_models, to_branch_models, child_models, grid
        )
        if mass_flow_signed_list:
            # Report-only nodal temperature, computed outside the solver.
            self.t_k = PostProcess(lambda v, tref=grid.t_ref_k: v.t_pu * tref)
            eqs = [
                junction_mass_flow_balance(mass_flow_signed_list),
                IntermediateEq(
                    "mass_flow_kgs",
                    sum(
                        [
                            vs["mass_flow_kgs"] * vs["regulation"]
                            for vs in (model.vars for model in child_models)
                            if "mass_flow_kgs" in vs
                        ]
                    ),
                ),
            ]
            # The nodal heat balances over a connected island are linearly
            # dependent (one is redundant). The grid-forming reference node is
            # the heat slack - drop its balance there, exactly as the slack bus
            # absorbs the power balance. Marked once per island by the solver.
            if not getattr(self, "_drop_heat_balance", False):
                eqs.insert(1, junction_mass_flow_balance(energy_flow_list))
            return eqs
        return []
