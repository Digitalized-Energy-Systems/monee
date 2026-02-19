import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
from monee.model.core import Intermediate, IntermediateEq, Var

from ..core import BranchFormulation, NodeFormulation


class NLWeymouthNodeFormulation(NodeFormulation):
    def ensure_var(self, model):
        model.pressure_pa = Intermediate(1000000)
        model.pressure_pu = Intermediate(1)
        model.pressure_squared_pu = Var(1, min=0, max=3, name="pressure_sq_pu")

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
            IntermediateEq(
                "pressure_pu", kwargs["sqrt_impl"](node.pressure_squared_pu)
            ),
            IntermediateEq("pressure_pa", lambda: node.pressure_pu * grid.pressure_ref),
        ]


class NLWeymouthBranchFormulation(BranchFormulation):
    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # linearize sqrt(p) around nominal pressure
        p0 = grid.nominal_pressure_pu
        x0 = p0**2
        p_from = p0 + (1 / (2 * p0)) * (
            from_node_model.vars["pressure_squared_pu"] - x0
        )
        p_to = p0 + (1 / (2 * p0)) * (to_node_model.vars["pressure_squared_pu"] - x0)
        p_avg = 0.5 * (p_from + p_to)

        hydraulicsmodel.piecewise_eq_friction(branch, kwargs["pwl_impl"])

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
            branch.mass_flow_pos_squared <= grid.f_max**2 * branch.direction,
            branch.mass_flow_neg_squared <= grid.f_max**2 * (1 - branch.direction),
            branch.mass_flow_pos_squared <= grid.f_max**2 * branch.on_off,
            branch.mass_flow_neg_squared <= grid.f_max**2 * branch.on_off,
            ogfmodel.pipe_weymouth(
                p_squared_i=from_node_model.vars["pressure_squared_pu"]
                * grid.pressure_ref**2,
                p_squared_j=to_node_model.vars["pressure_squared_pu"]
                * grid.pressure_ref**2,
                f_a_pos_sq=branch.mass_flow_pos_squared,
                f_a_neg_sq=branch.mass_flow_neg_squared,
                diameter_m=branch.diameter_m,
                length_m=branch.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                on_off=branch.on_off,
                friction=branch.friction,
                **kwargs,
            ),
            branch.gas_density
            == grid.pressure_ref
            * p_avg
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        ]
