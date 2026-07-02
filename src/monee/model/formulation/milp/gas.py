r"""Piecewise-linear Weymouth gas formulation: a MILP (PWL of :math:`\varphi(m)` plus the
``direction`` binaries; the pressure equation is linear in :math:`p^2`-space)."""

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
from monee.model.core import Const, Var

from ..common import ensure_velocity_report
from ..core import BranchFormulation


def _pwl_m_max(model, grid) -> float:
    """Per-pipe mass-flow cap: grid max_mass_flow_kgs tightened by the velocity_mps bound at
    reference-pressure density."""
    return hydraulicsmodel.calc_local_max_mass_flow(
        grid,
        model,
        ogfmodel.reference_gas_density(grid),
        getattr(grid, "v_max_mps", 20.0),
    )


class PwlWeymouthBranchFormulation(BranchFormulation):
    r"""Variable-friction Weymouth via per-pipe PWL of :math:`\varphi(m) = friction(Re(m)) \cdot m^2`.

    Opt-in alternative to the relaxed (convex MIQCQP) Weymouth for networks
    where laminar-regime accuracy matters (``Re < 2300`` on lightly-loaded
    pipes); the constant-friction turbulent asymptote under-estimates the
    pressure drop there.
    """

    def __init__(self, n_breakpoints: int = 12):
        self.n_breakpoints = n_breakpoints

    def ensure_var(self, model, simulation=False, grid=None):
        model.phi_pwl_pos = Var(0, min=0, name="phi_pwl_pos")
        model.phi_pwl_neg = Var(0, min=0, name="phi_pwl_neg")
        model.mass_flow_pos_kgs_squared = Const(0.0)
        model.mass_flow_neg_kgs_squared = Const(0.0)
        model.reynolds_scaled = Const(0.0)
        model.friction = Const(0.0)
        ensure_velocity_report(model, grid)
        if hasattr(grid, "max_mass_flow_kgs"):
            # Pyomo Piecewise requires bounded x; 1.001x slack avoids endpoint
            # tightness. Declared on the Var abstraction so every backend gets
            # the bound natively - the previous Var.setub() in equations() was
            # pyomo-only and crashed the GEKKO path.
            m_ub = _pwl_m_max(model, grid) * 1.001
            model.mass_flow_pos_kgs.max = m_ub
            model.mass_flow_neg_kgs.max = m_ub

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # Linearise sqrt(p) around nominal pressure.
        p_from, p_to, p_avg = ogfmodel.linearized_pressure_pu(
            grid,
            from_node_model.vars["pressure_squared_pu"],
            to_node_model.vars["pressure_squared_pu"],
        )

        m_max = _pwl_m_max(branch, grid)

        # Two \varphi(m) PWLs; 0-anchor collapses the inactive side's \varphi to 0.
        # (mass_flow_pos_kgs/neg upper bounds are declared in ensure_var.)
        xs, ys = hydraulicsmodel.phi_pwl_breakpoints(
            branch.diameter_m,
            branch.roughness_m,
            grid.dynamic_visc_pas,
            branch._pipe_area,
            m_max,
            self.n_breakpoints,
        )
        kwargs["pwl_impl"].piecewise_eq(
            y=branch.phi_pwl_pos,
            x=branch.mass_flow_pos_kgs,
            xs=xs,
            ys=ys,
        )
        kwargs["pwl_impl"].piecewise_eq(
            y=branch.phi_pwl_neg,
            x=branch.mass_flow_neg_kgs,
            xs=xs,
            ys=ys,
        )

        c_sq = ogfmodel.calc_C_squared(
            branch.diameter_m,
            branch.length_m,
            grid.t_k,
            grid.compressibility,
            grid.universal_gas_constant / grid.molar_mass,
        )

        p_amb = getattr(grid, "pressure_ambient_pa", 0.0)
        # ABSOLUTE squared-pressure difference (per-unit): gauge psq difference
        # plus the gauge->absolute term. In gauge mode the offset uses the
        # LINEARIZED pressure p_from/p_to (affine in pressure_squared_pu, the same
        # linearization used for the density) so the equation stays linear; a
        # no-op when p_amb = 0.
        abs_psq_diff = ogfmodel.abs_psq_diff_pu(
            from_node_model.vars["pressure_squared_pu"],
            to_node_model.vars["pressure_squared_pu"],
            p_from,
            p_to,
            p_amb,
            grid.pressure_ref_pa,
        )

        return [
            # direction=0 - forward flow via m_neg.
            branch.mass_flow_pos_kgs <= m_max * branch.direction,
            branch.mass_flow_neg_kgs <= m_max * (1 - branch.direction),
            branch.mass_flow_pos_kgs <= m_max * branch.on_off,
            branch.mass_flow_neg_kgs <= m_max * branch.on_off,
            #
            abs_psq_diff * grid.pressure_ref_pa**2 * c_sq * branch.on_off
            == branch.phi_pwl_neg - branch.phi_pwl_pos,
            #
            ogfmodel.ideal_gas_density_eq(
                branch.gas_density_kg_per_m3, grid, p_avg, p_amb
            ),
        ]
