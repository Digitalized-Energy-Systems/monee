import math

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.hf as ohfmodel
import monee.model.phys.nonlinear.wf as owfmodel
from monee.model.core import Const, Intermediate, IntermediateEq, Var

from ..core import BranchFormulation, NodeFormulation


def _pin_friction_const(model):
    """Pin ``model.friction`` to the asymptotic turbulent value.

    See ``phys.core.hydraulics.friction_at_high_re`` for the rationale and
    accuracy caveats.  Also pins ``model.reynolds`` to a Const so the
    Reynolds equation and the friction PWL can both be dropped.
    """
    f_const = hydraulicsmodel.friction_at_high_re(model.diameter_m, model.roughness)
    model.friction = Const(f_const)
    model.reynolds = Const(0.0)


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
    # See ``nonlinear/gas.py`` ``NLWeymouthBranchFormulation`` for rationale.
    EPIGRAPH_TIGHTENING_EPS = 1e-5

    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0.3, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0.3, max=2, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0, name="mass_flow_mag")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        model.t_inc = Var(1, min=-2, max=2, name="temperature_increase")
        _pin_friction_const(model)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [
            self.EPIGRAPH_TIGHTENING_EPS
            * (branch.mass_flow_pos_squared + branch.mass_flow_neg_squared)
        ]

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
            # Epigraph relaxation kept tight by the ε term in :meth:`minimize`.
            branch.mass_flow_pos * branch.mass_flow_pos <= branch.mass_flow_pos_squared,
            branch.mass_flow_neg * branch.mass_flow_neg <= branch.mass_flow_neg_squared,
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


class NLDarcyWeisbachPWLBranchFormulation(BranchFormulation):
    """Variable-friction Darcy-Weisbach via a single PWL of φ(m) = friction(Re(m))·m².

    Opt-in alternative to :class:`NLDarcyWeisbachBranchFormulation` (which
    pins friction to the high-Reynolds asymptote).  Use this when you
    need accurate pressure drops in the **laminar regime** — for water
    distribution that's a real concern: ~30 % of pipes operate at
    ``Re < 2300`` on simbench LV-rural3 at design, where the asymptotic
    friction under-estimates pressure drop by a factor of 5–50×.

    See :class:`monee.model.formulation.nonlinear.gas.NLWeymouthPWLBranchFormulation`
    for the full rationale of the PWL trick.  As with that class, two
    PWLs (one per flow direction) preserve bidirectional flow gated by
    the ``direction`` binary.

    Temperature-routing and ``mass_flow_mag``/``alpha`` constraints are
    kept identical to :class:`NLDarcyWeisbachBranchFormulation` — they
    don't involve the friction bilinear.
    """

    def __init__(self, n_breakpoints: int = 12):
        self.n_breakpoints = n_breakpoints

    def ensure_var(self, model):
        model.t_in_pu = Var(1, min=0.3, max=2, name="t_in_pu")
        model.t_out_pu = Var(1, min=0.3, max=2, name="t_out_pu")
        model.mass_flow_mag = Var(1, min=0, name="mass_flow_mag")
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        model.t_inc = Var(1, min=-2, max=2, name="temperature_increase")
        # φ aux Vars: friction · m² per flow direction.  Two PWLs let
        # the original direction binary keep its role of selecting which
        # side of the pipe is active.
        model.phi_pwl_pos = Var(0, min=0, name="phi_pwl_pos")
        model.phi_pwl_neg = Var(0, min=0, name="phi_pwl_neg")
        # Squared-mass-flow Vars unused — φ replaces them.
        model.mass_flow_pos_squared = Const(0.0)
        model.mass_flow_neg_squared = Const(0.0)
        model.reynolds = Const(0.0)
        model.friction = Const(0.0)

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

        m_max = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m, grid.fluid_density, grid.v_max_mps
            ),
        )

        # Pyomo Piecewise requires both ends of the x-Var to be bounded.
        branch.mass_flow_pos.setub(m_max * 1.001)
        branch.mass_flow_neg.setub(m_max * 1.001)

        # Build two φ(m) PWLs (one per direction).  0-anchor on the PWL
        # ensures the inactive side's φ collapses to 0.
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

        # Darcy-Weisbach pressure drop with φ replacing friction · m²:
        #   (p_i − p_j) · pressure_ref · on_off == K · -(φ_pos − φ_neg)
        # matching ``owfmodel.darcy_weisbach_equation``'s sign convention.
        K = branch.length_m / (
            2.0 * grid.fluid_density * branch._pipe_area**2 * branch.diameter_m
        )

        return [
            # Direction gating (matches the original Darcy formulation).
            branch.mass_flow_pos <= m_max * branch.direction,
            branch.mass_flow_neg <= m_max * (1 - branch.direction),
            branch.mass_flow_pos <= m_max * branch.on_off,
            branch.mass_flow_neg <= m_max * branch.on_off,
            branch.mass_flow_mag <= m_max,
            branch.mass_flow_mag == branch.mass_flow_pos + branch.mass_flow_neg,
            (from_node_model.vars["pressure_pu"] - to_node_model.vars["pressure_pu"])
            * grid.pressure_ref
            * branch.on_off
            == K * (branch.phi_pwl_neg - branch.phi_pwl_pos),
            # Temperature routing — original direction-dependent form.
            branch.alpha * (branch.mass_flow_mag + UA_C) == branch.mass_flow_mag,
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref
            + branch.alpha * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref),
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
        _pin_friction_const(model)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        branch._pipe_area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        f_max_local = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m, grid.fluid_density, grid.v_max_mps
            ),
        )

        return [
            # Epigraph relaxation kept tight by the ε term in :meth:`minimize`.
            branch.mass_flow_pos * branch.mass_flow_pos <= branch.mass_flow_pos_squared,
            branch.mass_flow_neg * branch.mass_flow_neg <= branch.mass_flow_neg_squared,
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
            == -branch.q_mw * 1e6 / (ohfmodel.SPECIFIC_HEAT_CAP_WATER * grid.t_ref),
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
