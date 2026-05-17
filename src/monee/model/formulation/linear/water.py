import monee.model.phys.nonlinear.hf as ohfmodel
from monee.model.core import Var

from ..core import BranchFormulation


class LinearHeatExchangerFormulation(BranchFormulation):
    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0, max=2, name="t_out_pu")
        if isinstance(model.mass_flow_design_kgs, Var):
            model.mass_flow_mag = Var(0, min=0)
        else:
            model.mass_flow_mag = Var(0, min=0, max=model.mass_flow_design_kgs)

        if model.q_mw_set <= 0 or isinstance(model.q_mw_set, Var):
            model._he_is_generator = True
            model.q_mw_delivered = Var(0, max=0, name="q_mw_delivered")
        elif model.q_mw_set > 0:
            model._he_is_generator = False
            model.q_mw_delivered = Var(0, min=0, name="q_mw_delivered")

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        if branch._he_is_generator:
            return [branch.q_mw_delivered]
        return [-branch.q_mw_delivered]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        cp_mw_per_kgs_K = ohfmodel.SPECIFIC_HEAT_CAP_WATER / 1e6
        is_dynamic_mf = isinstance(branch.mass_flow_design_kgs, Var)

        if is_dynamic_mf:
            # SubHE / dynamic: mass_flow_design_kgs is a Var set by the parent compound.
            mf_eq = branch.mass_flow_mag == branch.mass_flow_design_kgs
            mf_neg_eq = (
                branch.mass_flow_neg == branch.mass_flow_design_kgs * branch.on_off
            )
        else:
            mf_eq = branch.mass_flow_mag == branch.mass_flow_design_kgs
            mf_neg_eq = (
                branch.mass_flow_neg == branch.mass_flow_design_kgs * branch.on_off
            )

        eqs = [
            mf_eq,
            branch.direction == 0,
            branch.mass_flow_pos == 0,
            mf_neg_eq,
            branch.t_in_pu == from_node_model.vars["t_pu"],
            branch.t_from_pu == from_node_model.vars["t_pu"],
            branch.t_to_pu == branch.t_out_pu,
            # Energy balance in MW (cp/1e6 converts J/(kg·K) → MW·s/(kg·K)).
            branch.t_out_pu
            * (branch.mass_flow_design_kgs * cp_mw_per_kgs_K * grid.t_ref)
            == branch.t_in_pu
            * (branch.mass_flow_design_kgs * cp_mw_per_kgs_K * grid.t_ref)
            - branch.q_mw_delivered,
        ]
        if branch._he_is_generator:
            eqs.append(branch.q_mw_delivered >= branch.q_mw * branch.on_off)
        else:
            eqs.append(branch.q_mw_delivered <= branch.q_mw * branch.on_off)
        return eqs
