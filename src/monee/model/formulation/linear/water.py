import monee.model.phys.nonlinear.hf as ohfmodel
from monee.model.core import Var

from ..core import BranchFormulation


class LinearHeatExchangerFormulation(BranchFormulation):
    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0, max=3, name="t_in_pu")
        model.t_out_pu = Var(1, min=0, max=3, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0)
        print(model.q_w_set)
        if model.q_w_set <= 0 or isinstance(model.q_w_set, Var):
            model._he_is_generator = True
            model.q_delivered = Var(0, max=0, name="q_delivered")
        elif model.q_w_set > 0:
            model._he_is_generator = False
            model.q_delivered = Var(0, min=0, name="q_delivered")

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        if branch._he_is_generator:
            return [branch.q_delivered]
        return [-branch.q_delivered]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        cp = ohfmodel.SPECIFIC_HEAT_CAP_WATER

        eqs = [
            branch.mass_flow_mag == branch.mass_flow_design_kgs,
            branch.direction == 0,
            branch.mass_flow_pos == 0,
            branch.mass_flow_neg == branch.mass_flow_design_kgs * branch.on_off,
            # Temperature routing
            branch.t_in_pu == from_node_model.vars["t_pu"],
            branch.t_from_pu == from_node_model.vars["t_pu"],
            branch.t_to_pu == branch.t_out_pu,
            branch.t_out_pu * (branch.mass_flow_design_kgs * cp * grid.t_ref)
            == branch.t_in_pu * (branch.mass_flow_design_kgs * cp * grid.t_ref)
            - branch.q_delivered,
        ]
        if branch._he_is_generator:
            eqs.append(branch.q_delivered >= branch.q_w * branch.on_off)
        else:
            eqs.append(branch.q_delivered <= branch.q_w * branch.on_off)
        return eqs
