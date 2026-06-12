import monee.model.phys.nonlinear.hf as ohfmodel
from monee.model.core import Var

from ..core import BranchFormulation


class LinearHeatExchangerFormulation(BranchFormulation):
    def ensure_var(self, model, simulation=False):
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
        eqs = [branch.direction == 0]
        eqs += self._he_equations(branch, grid, from_node_model)
        return eqs

    def _he_equations(self, branch, grid, from_node_model):
        cp_mw_per_kgs_K = ohfmodel.SPECIFIC_HEAT_CAP_WATER / 1e6

        # equations() runs after solver-var injection, when mass_flow_design_kgs
        # is no longer a monee Var - the model's construction-time flag is the
        # only reliable dynamic/fixed discriminator here.
        is_dynamic_mf = branch._calc_mass_flow

        # Set by mark_he_flow_prescription() during solver prep; defaults to
        # the prescribing behaviour (supply/return semantics).
        prescribed = getattr(branch, "_he_flow_prescribed", True)

        if is_dynamic_mf and not prescribed:
            # SubHE whose through-flow the surrounding network already
            # determines (e.g. a fixed-mass-flow sink fed only through the
            # compound): q_mw is dictated by the control node and
            # mass_flow_design_kgs is only the sizing value (q_mw at the design
            # temperature spread). Pinning the flow to it would over-determine
            # the system, so the energy balance runs on the actual flow
            # magnitude instead.
            flow_eqs = [
                branch.mass_flow_mag == branch.mass_flow_pos + branch.mass_flow_neg
            ]
            balance_flow_kgs = branch.mass_flow_mag
        else:
            flow_eqs = [
                branch.mass_flow_mag == branch.mass_flow_design_kgs,
                branch.mass_flow_neg == branch.mass_flow_design_kgs * branch.on_off,
            ]
            balance_flow_kgs = branch.mass_flow_design_kgs

        eqs = flow_eqs + [
            branch.mass_flow_pos == 0,
            branch.t_in_pu == from_node_model.vars["t_pu"],
            branch.t_from_pu == from_node_model.vars["t_pu"],
            branch.t_to_pu == branch.t_out_pu,
            branch.t_out_pu * (balance_flow_kgs * cp_mw_per_kgs_K * grid.t_ref)
            == branch.t_in_pu * (balance_flow_kgs * cp_mw_per_kgs_K * grid.t_ref)
            - branch.q_mw_delivered,
        ]
        if branch._he_is_generator:
            eqs.append(branch.q_mw_delivered >= branch.q_mw * branch.on_off)
        else:
            eqs.append(branch.q_mw_delivered <= branch.q_mw * branch.on_off)
        return eqs
