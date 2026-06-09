import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.smooth as smoothmodel
from monee.model.core import Const, Var

from ..core import BranchFormulation
from .gas import NLWeymouthNodeFormulation

FRICTION_MODELS = ("constant", "pwl", "nonlinear")


def _pin(model, *names):
    """Pin model Vars the active formulation never constrains to a Const so no
    phantom degrees of freedom are injected — required for a square IMODE=1
    simulation solve. Only used in ``simulation`` mode; the default (IMODE=3)
    path leaves them free, where IPOPT parks them harmlessly."""
    for name in names:
        v = getattr(model, name, None)
        if isinstance(v, Var):
            setattr(model, name, Const(v.value))


def _ensure_friction_vars(model, friction_model):
    """Set up the friction state for the chosen model, independent of any
    formulation applied earlier (which may have pinned these to ``Const``)."""
    if friction_model == "constant":
        model.friction = Const(
            hydraulicsmodel.friction_at_high_re(model.diameter_m, model.roughness)
        )
        model.reynolds = Const(0.0)
    elif friction_model == "nonlinear":
        model.friction = Var(0.02, min=0, max=7, name="friction")
        model.reynolds = Var(1e-3, min=0, max=10, name="reynolds")
    else:  # pwl: one odd spline ψ(m) replaces friction·m·|m|.
        model.friction = Const(0.0)
        model.reynolds = Const(0.0)
        model.psi_pwl = Var(0.0, name="psi_pwl")


class SmoothWeymouthBranchFormulation(BranchFormulation):
    """Pure-NLP Weymouth gas pipe for GEKKO IPOPT/APOPT.

    One signed mass-flow var drives a smooth pressure drop
    ``(p_i²-p_j²)·C²·on_off == -friction · m · |m|``. ``mass_flow_pos/neg`` are
    kept as the public interface (consumed by the nodal balance) but bound to the
    smooth split of ``m``, so no ``direction`` binary and no epigraph relaxation
    are needed. ``on_off`` stays an optional switch.

    ``friction_model``: ``"constant"`` (turbulent asymptote per pipe),
    ``"pwl"`` (odd spline of the drop term) or ``"nonlinear"`` (smooth laminar↔
    turbulent friction blend).
    """

    def __init__(
        self,
        friction_model="constant",
        smoothing_eps=1e-3,
        n_breakpoints=12,
        simulation=False,
    ):
        assert friction_model in FRICTION_MODELS, friction_model
        self.friction_model = friction_model
        self.smoothing_eps = smoothing_eps
        self.n_breakpoints = n_breakpoints
        # simulation=True squares the model for an IMODE=1 steady-state solve:
        # pins phantom vars and drops the operational flow-limit inequalities.
        self.simulation = simulation

    def ensure_var(self, model):
        # mass_flow is already the signed flow (model defines it as pos − neg);
        # promote it to the decision var instead of adding a redundant one.
        model.mass_flow = Var(0.0, name="mass_flow")
        model.mass_flow_mag = Var(0.1, min=0, name="mass_flow_mag")
        # Neutralise the MISOCP-only vars so no integer/aux vars get injected.
        model.direction = Const(1)
        model.mass_flow_pos_squared = Const(0.0)
        model.mass_flow_neg_squared = Const(0.0)
        if self.simulation:
            _pin(model, "velocity")
        _ensure_friction_vars(model, self.friction_model)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        sqrt_impl = kwargs["sqrt_impl"]
        area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        gas_density = (
            grid.pressure_ref
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k)
        )
        f_max_local = min(
            grid.f_max,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m, gas_density, getattr(grid, "v_max_mps", 20.0)
            ),
        )

        # Linearise √p around nominal pressure for the density estimate.
        p0 = grid.nominal_pressure_pu
        x0 = p0**2
        p_from = p0 + (1 / (2 * p0)) * (
            from_node_model.vars["pressure_squared_pu"] - x0
        )
        p_to = p0 + (1 / (2 * p0)) * (to_node_model.vars["pressure_squared_pu"] - x0)
        p_avg = 0.5 * (p_from + p_to)

        signed = branch.mass_flow
        mag = branch.mass_flow_mag
        drop_term, friction_eqs = smoothmodel.drop_term_and_eqs(
            self.friction_model, branch, grid.dynamic_visc, area, signed, mag,
            f_max_local, **kwargs,
        )

        eqs = [
            mag == smoothmodel.smooth_abs(signed, self.smoothing_eps, sqrt_impl),
            branch.mass_flow_pos == 0.5 * (mag + signed),
            branch.mass_flow_neg == 0.5 * (mag - signed),
        ]
        if not self.simulation:
            # Operational flow limits — dropped in simulation mode (their slacks
            # would break a square IMODE=1 solve; limits are checked post-hoc).
            eqs += [
                signed <= f_max_local * branch.on_off,
                -signed <= f_max_local * branch.on_off,
            ]
        eqs += [
            smoothmodel.weymouth_pressure(
                p_sq_i=from_node_model.vars["pressure_squared_pu"]
                * grid.pressure_ref**2,
                p_sq_j=to_node_model.vars["pressure_squared_pu"]
                * grid.pressure_ref**2,
                drop_term=drop_term,
                diameter_m=branch.diameter_m,
                length_m=branch.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                on_off=branch.on_off,
            ),
            branch.gas_density
            == grid.pressure_ref
            * p_avg
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        ]
        return eqs + friction_eqs


class SmoothWeymouthSimNodeFormulation(NLWeymouthNodeFormulation):
    """Gas junction for a square IMODE=1 simulation: pins the unused ``t_pu``
    (gas carries no temperature, so it would otherwise float as a phantom)."""

    def ensure_var(self, model):
        super().ensure_var(model)
        _pin(model, "t_pu")
