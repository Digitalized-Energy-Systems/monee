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
        model.t_in_pu = Var(1, min=0.3, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0.3, max=2, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0, name="mass_flow_mag")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        model.t_inc = Var(1, min=-2, max=2, name="temperature_increase")

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

        hydraulicsmodel.piecewise_eq_friction(branch, kwargs["pwl_impl"])

        # Per-pipe big-M tightening: the physical max flow is bounded by the
        # pipe cross-section times the velocity cap, which is typically far
        # below the grid-wide ``f_max``.  Using the tighter value shrinks
        # Gurobi's LP relaxation gap and speeds up the direction/on_off B&B.
        f_max_local = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m, grid.fluid_density, grid.v_max_mps
            ),
        )

        # Unidirectional pipes pin direction = 0 (forward flow from_node →
        # to_node).  This eliminates the direction binary from the MIP tree
        # and collapses the temperature routing to the from-node as the inlet,
        # which is the common case for district-heating supply/return pipes.
        eqs = [
            # note that the dynamic visc not temperature dependent modeled
            hydraulicsmodel.reynolds_equation(
                branch.reynolds,
                branch.mass_flow_pos + branch.mass_flow_neg,
                branch.diameter_m,
                grid.dynamic_visc,
                branch._pipe_area,
            ),
            branch.mass_flow_pos_squared == branch.mass_flow_pos * branch.mass_flow_pos,
            branch.mass_flow_neg_squared == branch.mass_flow_neg * branch.mass_flow_neg,
            branch.mass_flow_pos <= f_max_local * branch.direction,
            branch.mass_flow_neg <= f_max_local * (1 - branch.direction),
            branch.mass_flow_pos <= f_max_local * branch.on_off,
            branch.mass_flow_neg <= f_max_local * branch.on_off,
            branch.mass_flow_mag <= f_max_local,
            branch.mass_flow_pos_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_squared <= f_max_local**2 * branch.on_off,
            # note that the density is not temperature dependent modeled
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
            branch.mass_flow_mag == branch.mass_flow_pos + branch.mass_flow_neg,
            branch.alpha * (branch.mass_flow_mag + UA_C) == branch.mass_flow_mag,
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref
            + branch.alpha * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref)
            + 0,
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
        if getattr(branch, "unidirectional", False):
            eqs.append(branch.direction == 0)
            eqs.append(branch.mass_flow_pos == 0)
        return eqs


class NLDarcyWeisbachHeatExchangerFormulation(NLDarcyWeisbachBranchFormulation):
    """
    Passive heat-exchanger formulation based on Darcy-Weisbach hydraulics.

    The network hydraulics determine the mass flow freely (no prescribed design
    flow).  Given the fixed heat-power ``q_w`` stored on the branch model, the
    formulation computes the resulting temperature increase (or decrease) via

        mass_flow_mag * t_inc = -q_w / (cp * t_ref)

    so that the outlet temperature rises (or falls) proportionally.  Pressure
    drop is calculated the same way as for a plain water pipe.
    """

    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0, max=2, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0, name="mass_flow_mag")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        model.t_inc = Var(1, min=-2, max=2, name="temperature_increase")

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        hydraulicsmodel.piecewise_eq_friction(branch, kwargs["pwl_impl"])

        f_max_local = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m, grid.fluid_density, grid.v_max_mps
            ),
        )

        return [
            hydraulicsmodel.reynolds_equation(
                branch.reynolds,
                branch.mass_flow_pos + branch.mass_flow_neg,
                branch.diameter_m,
                grid.dynamic_visc,
                branch._pipe_area,
            ),
            branch.mass_flow_pos_squared == branch.mass_flow_pos * branch.mass_flow_pos,
            branch.mass_flow_neg_squared == branch.mass_flow_neg * branch.mass_flow_neg,
            branch.mass_flow_pos <= f_max_local * branch.direction,
            branch.mass_flow_neg <= f_max_local * (1 - branch.direction),
            branch.mass_flow_pos <= f_max_local * branch.on_off,
            branch.mass_flow_neg <= f_max_local * branch.on_off,
            branch.mass_flow_mag <= f_max_local,
            branch.mass_flow_pos_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_squared <= f_max_local**2 * branch.on_off,
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
            branch.mass_flow_mag == branch.mass_flow_pos + branch.mass_flow_neg,
            branch.mass_flow_mag * branch.t_inc
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
