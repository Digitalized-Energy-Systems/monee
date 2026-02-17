import math

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.hf as ohfmodel
import monee.model.phys.nonlinear.wf as owfmodel
from monee.model.core import Intermediate, IntermediateEq, Var

from ..core import BranchFormulation, NodeFormulation


class NLDarcyWeisbachNodeFormulation(NodeFormulation):
    def ensure_var(self, model):
        model.pressure_pa = Intermediate(1000000)
        model.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")
        model.pressure_squared_pu = Intermediate(1)

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        return [
            IntermediateEq("pressure_pa", lambda: node.pressure_pu * grid.pressure_ref),
        ]


class NLDarcyWeisbachBranchFormulation(BranchFormulation):
    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0, max=3, name="t_in_pu")
        model.t_out_pu = Var(1, min=0, max=3, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0, name="mass_flow_mag")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        model.t_inc = Var(1, name="temperature_increase")

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        pipe_outside_r = branch.diameter_m / 2 + branch.insulation_thickness_m
        pipe_inside_r = branch.diameter_m / 2

        UA_C = (
            2
            * math.pi
            * branch.lambda_insulation_w_per_k
            * branch.length_m
            / math.log(pipe_outside_r / pipe_inside_r)
        ) / ohfmodel.SPECIFIC_HEAT_CAP_WATER

        return [
            hydraulicsmodel.reynolds_equation(
                branch.reynolds,
                branch.mass_flow_pos + branch.mass_flow_neg,
                branch.diameter_m,
                grid.dynamic_visc,
                branch._pipe_area,
            ),
            # # assuming laminar flow (introduces error)
            branch.friction * (branch.reynolds + 0.001) == 64,
            # branch.friction == hydraulicsmodel.swamee_jain(branch.reynolds,
            #                                              branch.diameter_m,
            #                                              branch.roughness,
            #                                              kwargs["log_impl"]),
            branch.mass_flow_pos_squared == branch.mass_flow_pos * branch.mass_flow_pos,
            branch.mass_flow_neg_squared == branch.mass_flow_neg * branch.mass_flow_neg,
            branch.mass_flow_pos <= grid.f_max * branch.direction,
            branch.mass_flow_neg <= grid.f_max * (1 - branch.direction),
            branch.mass_flow_pos <= grid.f_max * branch.on_off,
            branch.mass_flow_neg <= grid.f_max * branch.on_off,
            branch.mass_flow_mag <= grid.f_max,
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pu"],
                to_node_model.vars["pressure_pu"],
                branch.mass_flow_pos_squared,
                branch.mass_flow_neg_squared,
                branch.length_m,
                branch.diameter_m,
                grid.fluid_density,
                on_off=branch.on_off,
                friction=branch.friction / grid.pressure_ref,
                **kwargs,
            ),
            # hydraulicsmodel.flow_rate_equation(
            #     mean_flow_velocity=branch.velocity,
            #     flow_rate=branch.mass_flow_pos - branch.mass_flow_neg,
            #     diameter=branch.diameter_m,
            #     fluid_density=grid.fluid_density,
            # ),
            # ohfmodel.heat_out(branch.t_out_pu,
            #                   branch.t_in_pu,
            #                   branch.temperature_ext_k,
            #                   grid.t_ref,
            #                   branch.diameter_m,
            #                   branch.insulation_thickness_m,
            #                   branch.mass_flow_pos,
            #                   branch.mass_flow_neg,
            #                   branch.lambda_insulation_w_per_k,
            #                   branch.length_m,
            #                   0),
            branch.mass_flow_mag == branch.mass_flow_pos + branch.mass_flow_neg,
            branch.alpha * (branch.mass_flow_mag + 0.001 + UA_C)
            == branch.mass_flow_mag + 0.001,
            # (branch.mass_flow_mag + 0.001) * branch.t_in == - (0/ohfmodel.SPECIFIC_HEAT_CAP_WATER * grid.t_ref),
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref
            + branch.alpha * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref)
            + 0,
            # branch.t_out_pu == branch.temperature_ext_k/grid.t_ref + (branch.t_in_pu - branch.temperature_ext_k/grid.t_ref) * alpha,
            branch.t_in_pu
            == branch.direction * to_node_model.vars["t_pu"]
            + (1 - branch.direction) * from_node_model.vars["t_pu"],
            branch.t_to_pu
            == branch.direction * to_node_model.vars["t_pu"]
            + (1 - branch.direction) * branch.t_out_pu,
            branch.t_from_pu
            == branch.direction * branch.t_out_pu
            + (1 - branch.direction) * from_node_model.vars["t_pu"],
        ]


class NLDarcyWeisbachHeatExchangerFormulation(NLDarcyWeisbachBranchFormulation):
    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0, max=3, name="t_in_pu")
        model.t_out_pu = Var(1, min=0, max=3, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0, name="mass_flow_mag")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        model.t_inc = Var(1, name="temperature_increase")

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        return [
            hydraulicsmodel.reynolds_equation(
                branch.reynolds,
                branch.mass_flow_pos + branch.mass_flow_neg,
                branch.diameter_m,
                grid.dynamic_visc,
                branch._pipe_area,
            ),
            # assuming laminar flow (introduces error)
            branch.friction * (branch.reynolds + 0.001) == 64,
            branch.mass_flow_pos <= grid.f_max * branch.direction,
            branch.mass_flow_neg <= grid.f_max * (1 - branch.direction),
            branch.mass_flow_pos <= grid.f_max * branch.on_off,
            branch.mass_flow_neg <= grid.f_max * branch.on_off,
            branch.mass_flow_mag <= grid.f_max,
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pu"],
                to_node_model.vars["pressure_pu"],
                branch.mass_flow_pos,
                branch.mass_flow_neg,
                branch.length_m,
                branch.diameter_m,
                grid.fluid_density,
                on_off=branch.on_off,
                friction=branch.friction / grid.pressure_ref,
                **kwargs,
            ),
            # hydraulicsmodel.flow_rate_equation(
            #     mean_flow_velocity=branch.velocity,
            #     flow_rate=branch.mass_flow_pos - branch.mass_flow_neg,
            #     diameter=branch.diameter_m,
            #     fluid_density=grid.fluid_density,
            # ),
            # ohfmodel.heat_out(branch.t_out_pu,
            #                   branch.t_in_pu,
            #                   branch.temperature_ext_k,
            #                   grid.t_ref,
            #                   branch.diameter_m,
            #                   0,
            #                   branch.mass_flow_pos,
            #                   branch.mass_flow_neg,
            #                   0,
            #                   branch.length_m,
            #                   branch.q_w,
            #                   no_losses=True),
            branch.mass_flow_mag == branch.mass_flow_pos + branch.mass_flow_neg,
            (branch.mass_flow_mag + 0.001) * branch.t_inc
            == -branch.q_w / (ohfmodel.SPECIFIC_HEAT_CAP_WATER * grid.t_ref),
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref
            + 1 * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref)
            + branch.t_inc,
            branch.t_in_pu
            == branch.direction * to_node_model.vars["t_pu"]
            + (1 - branch.direction) * from_node_model.vars["t_pu"],
            branch.t_to_pu
            == branch.direction * to_node_model.vars["t_pu"]
            + (1 - branch.direction) * branch.t_out_pu,
            branch.t_from_pu
            == branch.direction * branch.t_out_pu
            + (1 - branch.direction) * from_node_model.vars["t_pu"],
        ]
