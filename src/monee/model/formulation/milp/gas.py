"""Piecewise-linear Weymouth gas formulation: a MILP (PWL of φ(m) plus the
``direction`` binaries; the pressure equation is linear in p²-space)."""

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
from monee.model.core import Const, Var

from ..core import BranchFormulation


def _pwl_m_max(model, grid) -> float:
    """Per-pipe mass-flow cap: grid f_max tightened by the velocity bound at
    reference-pressure density."""
    gas_density = (
        grid.pressure_ref * grid.molar_mass / (grid.universal_gas_constant * grid.t_k)
    )
    return min(
        grid.f_max,
        hydraulicsmodel.calc_max_mass_flow(
            model.diameter_m,
            gas_density,
            getattr(grid, "v_max_mps", 20.0),
        ),
    )


class PwlWeymouthBranchFormulation(BranchFormulation):
    """Variable-friction Weymouth via per-pipe PWL of ``φ(m) = friction(Re(m))·m²``.

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
        model.mass_flow_pos_squared = Const(0.0)
        model.mass_flow_neg_squared = Const(0.0)
        model.reynolds = Const(0.0)
        model.friction = Const(0.0)
        if hasattr(grid, "f_max"):
            # Pyomo Piecewise requires bounded x; 1.001x slack avoids endpoint
            # tightness. Declared on the Var abstraction so every backend gets
            # the bound natively - the previous Var.setub() in equations() was
            # pyomo-only and crashed the GEKKO path.
            m_ub = _pwl_m_max(model, grid) * 1.001
            model.mass_flow_pos.max = m_ub
            model.mass_flow_neg.max = m_ub

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        # Linearise sqrt(p) around nominal pressure.
        p0 = grid.nominal_pressure_pu
        x0 = p0**2
        p_from = p0 + (1 / (2 * p0)) * (
            from_node_model.vars["pressure_squared_pu"] - x0
        )
        p_to = p0 + (1 / (2 * p0)) * (to_node_model.vars["pressure_squared_pu"] - x0)
        p_avg = 0.5 * (p_from + p_to)

        m_max = _pwl_m_max(branch, grid)

        # Two φ(m) PWLs; 0-anchor collapses the inactive side's φ to 0.
        # (mass_flow_pos/neg upper bounds are declared in ensure_var.)
        xs, ys = hydraulicsmodel.phi_pwl_breakpoints(
            branch.diameter_m,
            branch.roughness,
            grid.dynamic_visc,
            branch._pipe_area,
            m_max,
            self.n_breakpoints,
        )
        kwargs["pwl_impl"].piecewise_eq(
            y=branch.phi_pwl_pos,
            x=branch.mass_flow_pos,
            xs=xs,
            ys=ys,
        )
        kwargs["pwl_impl"].piecewise_eq(
            y=branch.phi_pwl_neg,
            x=branch.mass_flow_neg,
            xs=xs,
            ys=ys,
        )

        C_sq = ogfmodel.calc_C_squared(
            branch.diameter_m,
            branch.length_m,
            grid.t_k,
            grid.compressibility,
        )

        return [
            # direction=0 - forward flow via m_neg.
            branch.mass_flow_pos <= m_max * branch.direction,
            branch.mass_flow_neg <= m_max * (1 - branch.direction),
            branch.mass_flow_pos <= m_max * branch.on_off,
            branch.mass_flow_neg <= m_max * branch.on_off,
            #
            (from_node_model.vars["pressure_squared_pu"] - to_node_model.vars["pressure_squared_pu"]) * grid.pressure_ref**2 * C_sq * branch.on_off
            == branch.phi_pwl_neg - branch.phi_pwl_pos,
            #
            branch.gas_density == grid.pressure_ref * p_avg * grid.molar_mass / (grid.universal_gas_constant * grid.t_k),
        ]
