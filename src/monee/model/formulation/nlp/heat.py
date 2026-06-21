"""Smooth Darcy-Weisbach water/heat formulations: non-convex NLPs, binary-free."""

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.hf as ohfmodel
import monee.model.phys.nonlinear.smooth as smoothmodel
import monee.model.phys.nonlinear.wf as owfmodel
from monee.model.core import Const, Var

from ..core import BranchFormulation
from ..milp.heat import FixedFlowHeatExchangerFormulation
from .gas import FRICTION_MODELS, _ensure_friction_vars, _pin, _seed_mag


def _ensure_smooth_flow_vars(model, friction_model, simulation=False):
    model.mass_flow_kgs = Var(0.0, name="mass_flow_kgs")
    mag0 = _seed_mag(model) if simulation else 0.1
    model.mass_flow_mag_kgs = Var(mag0, min=0, name="mass_flow_mag_kgs")
    model.t_in_pu = Var(1, min=0.3, max=2, name="t_in_pu")
    model.t_out_pu = Var(1, min=0.3, max=2, name="t_out_pu")
    model.direction = Const(1)
    model.mass_flow_pos_kgs_squared = Const(0.0)
    model.mass_flow_neg_kgs_squared = Const(0.0)
    if simulation:
        _pin(model, "velocity_mps")
    _ensure_friction_vars(model, friction_model)


def _flow_and_pressure_eqs(
    formulation, branch, grid, from_node_model, to_node_model, **kwargs
):
    """Smooth signed split, flow bounds, and pressure drop.

    Pressure drop is Darcy-Weisbach friction over ``length_m`` unless the branch
    carries a ``loss_coefficient`` (pandapipes-style minor loss), in which case a
    zero-length :math:`\\zeta \\cdot \\tfrac{\\rho}{2} v^2` drop is used instead.

    Returns ``(eqs, signed, mag)`` for the caller to append temperature physics.
    """
    sqrt_impl = kwargs["sqrt_impl"]
    f_max_local = hydraulicsmodel.calc_local_max_mass_flow(
        grid, branch, grid.fluid_density_kg_per_m3, grid.v_max_mps
    )
    signed = branch.mass_flow_kgs
    mag = branch.mass_flow_mag_kgs
    eqs = [
        mag == smoothmodel.smooth_abs(signed, formulation.smoothing_eps, sqrt_impl),
        branch.mass_flow_pos_kgs == 0.5 * (mag + signed),  # NOSONAR
        branch.mass_flow_neg_kgs == 0.5 * (mag - signed),  # NOSONAR
    ]
    if not kwargs.get("simulation", False):
        eqs += [
            signed <= f_max_local * branch.on_off,
            -signed <= f_max_local * branch.on_off,
        ]

    loss_coefficient = getattr(branch, "loss_coefficient", None)
    if loss_coefficient is not None:
        # pandapipes heat_exchanger: zero-length minor loss, no friction factor.
        eqs.append(
            smoothmodel.minor_loss_pressure(
                from_node_model.vars["pressure_pu"],
                to_node_model.vars["pressure_pu"],
                loss_coefficient,
                signed,
                mag,
                branch.diameter_m,
                grid.fluid_density_kg_per_m3,
                grid.pressure_ref_pa,
            )
        )
        return eqs, signed, mag

    area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)
    drop_term, friction_eqs = smoothmodel.drop_term_and_eqs(
        formulation.friction_model,
        branch,
        grid.dynamic_visc_pas,
        area,
        signed,
        mag,
        f_max_local,
        **kwargs,
    )
    eqs.append(
        smoothmodel.darcy_pressure(
            from_node_model.vars["pressure_pu"],
            to_node_model.vars["pressure_pu"],
            drop_term / grid.pressure_ref_pa,
            branch.length_m,
            branch.diameter_m,
            grid.fluid_density_kg_per_m3,
        )
    )
    eqs += friction_eqs
    return eqs, signed, mag


def _temperature_transport_eqs(branch, from_node_model, to_node_model):
    """Smooth upwinding: the ``direction`` binary of the MIQCQP model is replaced
    by the ``mass_flow_pos_kgs/neg`` weights (multiplied form avoids dividing by mag)."""
    mag = branch.mass_flow_mag_kgs
    mpos = branch.mass_flow_pos_kgs
    mneg = branch.mass_flow_neg_kgs
    t_from = from_node_model.vars["t_pu"]
    t_to = to_node_model.vars["t_pu"]
    return [
        mag * branch.t_in_pu == mpos * t_to + mneg * t_from,
        mag * branch.t_to_pu == mpos * t_to + mneg * branch.t_out_pu,
        mag * branch.t_from_pu == mpos * branch.t_out_pu + mneg * t_from,
    ]


def _ua_per_cp(branch):
    return owfmodel.pipe_insulation_ua(branch) / ohfmodel.SPECIFIC_HEAT_CAP_WATER


class SmoothDarcyWeisbachBranchFormulation(BranchFormulation):
    """Pure-NLP Darcy-Weisbach water/heat pipe for GEKKO IPOPT/APOPT.

    Smooth signed flow + smooth temperature upwinding; insulation losses via the
    ``alpha`` attenuation factor. No ``direction`` binary. See
    :class:`monee.model.formulation.nlp.gas.SmoothWeymouthBranchFormulation`
    for the ``friction_model`` options.
    """

    def __init__(
        self,
        friction_model="constant",
        smoothing_eps=1e-3,
    ):
        assert friction_model in FRICTION_MODELS, friction_model
        self.friction_model = friction_model
        self.smoothing_eps = smoothing_eps

    def ensure_var(self, model, simulation=False, grid=None):
        _ensure_smooth_flow_vars(model, self.friction_model, simulation)
        model.alpha = Var(0.01, min=0, max=1, name="alpha")
        if simulation:
            _pin(model, "q_mw", "t_inc_pu")

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        eqs, signed, mag = _flow_and_pressure_eqs(
            self, branch, grid, from_node_model, to_node_model, **kwargs
        )
        UA_C = _ua_per_cp(branch)
        eqs += [
            branch.alpha * (mag + UA_C) == mag,
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref_k
            + branch.alpha * (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref_k),
        ]
        eqs += _temperature_transport_eqs(branch, from_node_model, to_node_model)
        if getattr(branch, "unidirectional", False) and not kwargs.get(
            "simulation", False
        ):
            eqs.append(signed <= 0)
        return eqs


class SmoothPassiveHeatExchangerFormulation(BranchFormulation):
    """Passive HE: free mass flow, fixed ``q_mw`` sets the outlet temperature
    change. Smooth Darcy pressure drop; no insulation losses (alpha = 1)."""

    def __init__(
        self,
        friction_model="constant",
        smoothing_eps=1e-3,
    ):
        assert friction_model in FRICTION_MODELS, friction_model
        self.friction_model = friction_model
        self.smoothing_eps = smoothing_eps

    def ensure_var(self, model, simulation=False, grid=None):
        _ensure_smooth_flow_vars(model, self.friction_model, simulation)
        model.t_inc_pu = Var(1, min=-2, max=2, name="temperature_increase_pu")

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        eqs, _, mag = _flow_and_pressure_eqs(
            self, branch, grid, from_node_model, to_node_model, **kwargs
        )
        eqs += [
            mag * branch.t_inc_pu  # NOSONAR
            == -branch.q_mw * 1e6 / (ohfmodel.SPECIFIC_HEAT_CAP_WATER * grid.t_ref_k),
            branch.t_out_pu
            == branch.temperature_ext_k / grid.t_ref_k
            + (branch.t_in_pu - branch.temperature_ext_k / grid.t_ref_k)
            + branch.t_inc_pu,
        ]
        eqs += _temperature_transport_eqs(branch, from_node_model, to_node_model)
        return eqs


class SmoothHeatExchangerFormulation(FixedFlowHeatExchangerFormulation):
    """Active HE driven by a fixed design mass flow. Same balance as the
    fixed-flow formulation but with the ``direction`` binary pinned to a
    constant (and its ``direction == 0`` equation dropped) so the model stays
    a pure NLP."""

    def ensure_var(self, model, simulation=False, grid=None):
        super().ensure_var(model, simulation, grid=grid)
        model.direction = Const(0)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return self._he_equations(branch, grid, from_node_model)
