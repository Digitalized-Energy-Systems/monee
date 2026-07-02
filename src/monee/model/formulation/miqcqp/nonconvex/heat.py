r"""Bilinear Darcy-Weisbach water/heat formulations: non-convex MIQCQPs.

The hydraulics mirror the gas-side epigraph relaxation, but the temperature
side carries true continuous bilinears (:math:`alpha \cdot mag`, :math:`alpha \cdot t_{in}`,
:math:`direction \cdot t`), so the model needs a global MIQCQP solver.
"""

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.hf as ohfmodel
import monee.model.phys.nonlinear.wf as owfmodel
from monee.model.core import Const, Var

from ...common import ensure_velocity_report
from ...core import BranchFormulation


def _pin_friction_const(model):
    """Pin ``friction`` to the turbulent asymptote and ``reynolds_scaled`` to 0,
    so the Reynolds eq and friction PWL can be dropped."""
    f_const = hydraulicsmodel.friction_at_high_re(model.diameter_m, model.roughness_m)
    model.friction = Const(f_const)
    model.reynolds_scaled = Const(0.0)


class BilinearDarcyWeisbachBranchFormulation(BranchFormulation):
    # See RelaxedWeymouthBranchFormulation for rationale.
    EPIGRAPH_TIGHTENING_EPS = 1e-5

    def ensure_var(self, model, simulation=False, grid=None):
        model.t_in_pu = Var(1, min=0.3, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0.3, max=2, name="t_out_pu")
        model.mass_flow_mag_kgs = Var(1, min=0, name="mass_flow_mag_kgs")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        ensure_velocity_report(model, grid)
        _pin_friction_const(model)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [
            self.EPIGRAPH_TIGHTENING_EPS
            * (branch.mass_flow_pos_kgs_squared + branch.mass_flow_neg_kgs_squared)
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        UA_C = owfmodel.pipe_insulation_ua(branch) / ohfmodel.SPECIFIC_HEAT_CAP_WATER

        # Per-pipe big-M tightening via \pi/4 \cdot D^2 \cdot \rho \cdot v_max - usually well below
        # max_mass_flow_kgs; tighter big-M shrinks the LP relaxation gap.
        f_max_local = hydraulicsmodel.calc_local_max_mass_flow(
            grid, branch, grid.fluid_density_kg_per_m3, grid.v_max_mps
        )

        # Unidirectional pipes pin direction=0 below, eliminating that binary.
        eqs = [
            # Epigraph relaxation kept tight by the \varepsilon term in minimize().
            branch.mass_flow_pos_kgs * branch.mass_flow_pos_kgs
            <= branch.mass_flow_pos_kgs_squared,
            branch.mass_flow_neg_kgs * branch.mass_flow_neg_kgs
            <= branch.mass_flow_neg_kgs_squared,
            branch.mass_flow_pos_kgs <= f_max_local * branch.direction,
            branch.mass_flow_neg_kgs <= f_max_local * (1 - branch.direction),
            branch.mass_flow_pos_kgs <= f_max_local * branch.on_off,
            branch.mass_flow_neg_kgs <= f_max_local * branch.on_off,
            branch.mass_flow_mag_kgs <= f_max_local,
            branch.mass_flow_pos_kgs_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_kgs_squared <= f_max_local**2 * branch.on_off,
            # density is not modelled as temperature-dependent
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pu"],
                to_node_model.vars["pressure_pu"],
                branch.mass_flow_pos_kgs_squared,
                branch.mass_flow_neg_kgs_squared,
                branch.length_m,
                branch.diameter_m,
                grid.fluid_density_kg_per_m3,
                on_off=branch.on_off,
                friction=branch.friction / grid.pressure_ref_pa,
                **kwargs,
            ),
            branch.mass_flow_mag_kgs
            == branch.mass_flow_pos_kgs + branch.mass_flow_neg_kgs,
            branch.alpha * (branch.mass_flow_mag_kgs + UA_C)
            == branch.mass_flow_mag_kgs,
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref_k
            + branch.alpha * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref_k),
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
            eqs.append(branch.mass_flow_pos_kgs == 0)
        return eqs


class PwlDarcyWeisbachBranchFormulation(BranchFormulation):
    r"""Variable-friction Darcy-Weisbach via a PWL of :math:`\varphi(m) = friction(Re(m)) \cdot m^2`.

    Opt-in alternative to :class:`BilinearDarcyWeisbachBranchFormulation` for
    laminar-heavy networks (Re < 2300) where the turbulent asymptote
    under-estimates pressure drop by 5–50\ :math:`\times`. Two PWLs (one per direction)
    preserve bidirectional flow gated by ``direction``. The hydraulics turn
    MILP-shaped, but the temperature bilinears keep the model a non-convex
    MIQCQP.
    """

    def __init__(self, n_breakpoints: int = 12):
        self.n_breakpoints = n_breakpoints

    def ensure_var(self, model, simulation=False, grid=None):
        model.t_in_pu = Var(1, min=0.3, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0.3, max=2, name="t_out_pu")
        model.mass_flow_mag_kgs = Var(1, min=0, name="mass_flow_mag_kgs")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        ensure_velocity_report(model, grid)
        # \varphi = friction \cdot m^2, per flow direction; replaces the squared-mf Vars.
        model.phi_pwl_pos = Var(0, min=0, name="phi_pwl_pos")
        model.phi_pwl_neg = Var(0, min=0, name="phi_pwl_neg")
        model.mass_flow_pos_kgs_squared = Const(0.0)
        model.mass_flow_neg_kgs_squared = Const(0.0)
        model.reynolds_scaled = Const(0.0)
        model.friction = Const(0.0)
        if hasattr(grid, "max_mass_flow_kgs"):
            # Pyomo Piecewise requires bounded x; 1.001x slack avoids endpoint
            # tightness. Declared on the Var abstraction so every backend gets
            # the bound natively - the previous Var.setub() in equations() was
            # pyomo-only and crashed the GEKKO path.
            m_ub = (
                hydraulicsmodel.calc_local_max_mass_flow(
                    grid, model, grid.fluid_density_kg_per_m3, grid.v_max_mps
                )
                * 1.001
            )
            model.mass_flow_pos_kgs.max = m_ub
            model.mass_flow_neg_kgs.max = m_ub

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        UA_C = owfmodel.pipe_insulation_ua(branch) / ohfmodel.SPECIFIC_HEAT_CAP_WATER

        m_max = hydraulicsmodel.calc_local_max_mass_flow(
            grid, branch, grid.fluid_density_kg_per_m3, grid.v_max_mps
        )

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

        K = branch.length_m / (
            2.0
            * grid.fluid_density_kg_per_m3
            * branch._pipe_area**2
            * branch.diameter_m
        )

        return [
            branch.mass_flow_pos_kgs <= m_max * branch.direction,
            branch.mass_flow_neg_kgs <= m_max * (1 - branch.direction),
            branch.mass_flow_pos_kgs <= m_max * branch.on_off,
            branch.mass_flow_neg_kgs <= m_max * branch.on_off,
            branch.mass_flow_mag_kgs <= m_max,
            branch.mass_flow_mag_kgs
            == branch.mass_flow_pos_kgs + branch.mass_flow_neg_kgs,
            (from_node_model.vars["pressure_pu"] - to_node_model.vars["pressure_pu"])
            * grid.pressure_ref_pa
            * branch.on_off
            == K * (branch.phi_pwl_neg - branch.phi_pwl_pos),
            branch.alpha * (branch.mass_flow_mag_kgs + UA_C)
            == branch.mass_flow_mag_kgs,
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref_k
            + branch.alpha * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref_k),
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


class BilinearPassiveHeatExchangerFormulation(BilinearDarcyWeisbachBranchFormulation):
    r"""Passive HE: free mass flow, fixed q_mw. :math:`\text{mass\_flow\_mag\_kgs} \cdot t_{inc,pu} = -q_w / (cp \cdot t_{ref,k})`
    sets the outlet temperature change. Pressure drop
    follows the plain water-pipe form."""

    def ensure_var(self, model, simulation=False, grid=None):
        model.t_in_pu = Var(1, min=0, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0, max=2, name="t_out_pu")
        model.mass_flow_mag_kgs = Var(1, min=0, name="mass_flow_mag_kgs")
        model.t_inc_pu = Var(1, min=-2, max=2, name="temperature_increase_pu")
        ensure_velocity_report(model, grid)
        _pin_friction_const(model)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        f_max_local = hydraulicsmodel.calc_local_max_mass_flow(
            grid, branch, grid.fluid_density_kg_per_m3, grid.v_max_mps
        )

        return [
            # Epigraph relaxation kept tight by the \varepsilon term in minimize().
            branch.mass_flow_pos_kgs * branch.mass_flow_pos_kgs
            <= branch.mass_flow_pos_kgs_squared,
            branch.mass_flow_neg_kgs * branch.mass_flow_neg_kgs
            <= branch.mass_flow_neg_kgs_squared,
            branch.mass_flow_pos_kgs <= f_max_local * branch.direction,
            branch.mass_flow_neg_kgs <= f_max_local * (1 - branch.direction),
            branch.mass_flow_pos_kgs <= f_max_local * branch.on_off,
            branch.mass_flow_neg_kgs <= f_max_local * branch.on_off,
            branch.mass_flow_mag_kgs <= f_max_local,
            branch.mass_flow_pos_kgs_squared <= f_max_local**2 * branch.on_off,
            branch.mass_flow_neg_kgs_squared <= f_max_local**2 * branch.on_off,
            owfmodel.darcy_weisbach_equation(
                from_node_model.vars["pressure_pu"],
                to_node_model.vars["pressure_pu"],
                branch.mass_flow_pos_kgs_squared,
                branch.mass_flow_neg_kgs_squared,
                branch.length_m,
                branch.diameter_m,
                grid.fluid_density_kg_per_m3,
                on_off=branch.on_off,
                friction=branch.friction / grid.pressure_ref_pa,
                **kwargs,
            ),
            branch.mass_flow_mag_kgs
            == branch.mass_flow_pos_kgs + branch.mass_flow_neg_kgs,
            branch.mass_flow_mag_kgs * branch.t_inc_pu  # NOSONAR
            == -branch.q_mw * 1e6 / (ohfmodel.SPECIFIC_HEAT_CAP_WATER * grid.t_ref_k),
            branch.t_out_pu == branch.t_in_pu + branch.t_inc_pu,
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
