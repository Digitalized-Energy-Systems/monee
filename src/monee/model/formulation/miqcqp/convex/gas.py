r"""Epigraph-relaxed Weymouth gas formulation: a convex MIQCQP (MISOCP-shaped).

Weymouth is linear in pressure-squared space; the only quadratics are the
convex epigraph relaxations :math:`m \cdot m \le m_{sq}`, kept tight by a small :math:`\varepsilon` objective
term, plus the ``direction``/``on_off`` binaries.
"""

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
from monee.model.core import Const

from ...core import BranchFormulation


class RelaxedWeymouthBranchFormulation(BranchFormulation):
    EPIGRAPH_TIGHTENING_EPS = 1e-5

    def ensure_var(self, model, simulation=False, grid=None):
        f_const = hydraulicsmodel.friction_at_high_re(
            model.diameter_m, model.roughness_m
        )

        model.friction = Const(f_const)
        model.reynolds_scaled = Const(0.0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [
            self.EPIGRAPH_TIGHTENING_EPS
            * (branch.mass_flow_pos_kgs_squared + branch.mass_flow_neg_kgs_squared)
        ]

    def _epigraph_eqs(self, branch):
        r"""Convex epigraph relaxation :math:`m^2 \le m_{sq}`; the exact sibling
        overrides this with the equality form."""
        return [
            branch.mass_flow_pos_kgs * branch.mass_flow_pos_kgs
            <= branch.mass_flow_pos_kgs_squared,
            branch.mass_flow_neg_kgs * branch.mass_flow_neg_kgs
            <= branch.mass_flow_neg_kgs_squared,
        ]

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

        p_amb = getattr(grid, "pressure_ambient_pa", 0.0)
        gas_density_kg_per_m3 = ogfmodel.reference_gas_density(grid)
        f_max_local = min(
            grid.max_mass_flow_kgs,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m,
                gas_density_kg_per_m3,
                getattr(grid, "v_max_mps", 20.0),
            ),
        )

        return self._epigraph_eqs(branch) + [
            branch.mass_flow_pos_kgs_squared <= f_max_local**2 * branch.direction,
            branch.mass_flow_neg_kgs_squared <= f_max_local**2 * (1 - branch.direction),
            branch.mass_flow_pos_kgs_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_kgs_squared <= f_max_local**2 * branch.on_off,
            ogfmodel.pipe_weymouth(
                # ABSOLUTE squared pressure (Pa^2). In gauge mode the offset uses
                # the LINEARIZED pressure p_from/p_to (affine in pressure_squared_pu,
                # the same linearization used for the density above) so the
                # constraint stays conic; a no-op when the grid's ambient is 0.
                p_squared_i=ogfmodel.abs_psq_pu(
                    from_node_model.vars["pressure_squared_pu"],
                    p_from,
                    p_amb,
                    grid.pressure_ref_pa,
                )
                * grid.pressure_ref_pa**2,
                p_squared_j=ogfmodel.abs_psq_pu(
                    to_node_model.vars["pressure_squared_pu"],
                    p_to,
                    p_amb,
                    grid.pressure_ref_pa,
                )
                * grid.pressure_ref_pa**2,
                f_a_pos_sq=branch.mass_flow_pos_kgs_squared,
                f_a_neg_sq=branch.mass_flow_neg_kgs_squared,
                diameter_m=branch.diameter_m,
                length_m=branch.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                on_off=branch.on_off,
                friction=branch.friction,
                r_specific=grid.universal_gas_constant / grid.molar_mass,
                **kwargs,
            ),
            branch.gas_density_kg_per_m3
            == (grid.pressure_ref_pa * p_avg + p_amb)
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        ]
