"""Smooth Weymouth gas formulation: a non-convex NLP, binary-free."""

import monee.model.phys.core.hydraulics as hydraulicsmodel
import monee.model.phys.nonlinear.gf as ogfmodel
import monee.model.phys.nonlinear.smooth as smoothmodel
from monee.model.core import Const, Var

from ..core import BranchFormulation

FRICTION_MODELS = ("constant", "pwl", "nonlinear", "hybrid")


def _seed_mag(model):
    r"""Warm-start magnitude for ``mass_flow_mag_kgs``: the design flow the network
    generator stored on the pipe (``mass_flow_nominal_kgs``), else the flat 0.1.
    A good :math:`|m|` seed alone cuts the smooth-NLP iteration count by ~20\ :math:`\times` - the sign
    of the signed flow is recovered cheaply, the magnitude is what conditions the
    Darcy/Weymouth and temperature-transport rows."""
    seed = getattr(model, "mass_flow_nominal_kgs", None)
    return seed if seed and seed > 0 else 0.1


def _pin(model, *names):
    """Pin model Vars the active formulation never constrains to a Const so no
    phantom degrees of freedom are injected - required for a square IMODE=1
    simulation solve. Only used in ``simulation`` mode; the default (IMODE=3)
    path leaves them free, where IPOPT parks them harmlessly."""
    for name in names:
        v = getattr(model, name, None)
        if isinstance(v, Var):
            setattr(model, name, Const(v.value))


def _ensure_friction_vars(model, friction_model):
    """Set up the friction state for the chosen model, independent of any
    formulation applied earlier (which may have pinned these to ``Const``)."""
    if friction_model in ("constant", "hybrid"):
        # Both carry the fully-rough factor as the turbulent coefficient. "hybrid"
        # additionally adds a linear laminar term in the drop (see
        # smooth.hybrid_drop_term); no Reynolds/friction variable is needed.
        model.friction = Const(
            hydraulicsmodel.friction_at_high_re(model.diameter_m, model.roughness_m)
        )
        model.reynolds_scaled = Const(0.0)
    elif friction_model == "nonlinear":
        model.friction = Var(0.02, min=0, max=7, name="friction")
        model.reynolds_scaled = Var(1e-3, min=0, max=10, name="reynolds_scaled")
    else:  # pwl: one odd spline \psi(m) replaces friction \cdot m \cdot |m|.
        model.friction = Const(0.0)
        model.reynolds_scaled = Const(0.0)
        model.psi_pwl = Var(0.0, name="psi_pwl")


class SmoothWeymouthBranchFormulation(BranchFormulation):
    r"""Pure-NLP Weymouth gas pipe for GEKKO IPOPT/APOPT.

    One signed mass-flow var drives a smooth pressure drop
    :math:`(p_i^2-p_j^2) \cdot C^2 \cdot \text{on\_off} = -friction \cdot m \cdot |m|`. ``mass_flow_pos_kgs/neg`` are
    kept as the public interface (consumed by the nodal balance) but bound to the
    smooth split of ``m``, so no ``direction`` binary and no epigraph relaxation
    are needed. ``on_off`` stays an optional switch.

    ``friction_model``:

    * ``"constant"`` - fully-rough turbulent asymptote per pipe (Reynolds-independent).
    * ``"hybrid"`` - laminar Hagen-Poiseuille term (:math:`64/Re`, linear in :math:`m`)
      plus the fully-rough turbulent term; equals pandapipes' default ``nikuradse``
      law and is QCQP-representable (see :func:`smooth.hybrid_drop_term`).
    * ``"nonlinear"`` - smooth laminar\ :math:`\leftrightarrow` turbulent Swamee-Jain
      blend (:math:`\approx` Colebrook), Reynolds-coupled.
    * ``"pwl"`` - piecewise-linear (odd cubic spline) of the full drop term.
    """

    def __init__(
        self,
        friction_model="constant",
        smoothing_eps=1e-3,
        n_breakpoints=12,
    ):
        assert friction_model in FRICTION_MODELS, friction_model
        self.friction_model = friction_model
        self.smoothing_eps = smoothing_eps
        self.n_breakpoints = n_breakpoints

    def ensure_var(self, model, simulation=False, grid=None):
        # mass_flow_kgs is already the signed flow (model defines it as pos − neg);
        # promote it to the decision var instead of adding a redundant one.
        model.mass_flow_kgs = Var(0.0, name="mass_flow_kgs")
        # Seed |m| only for the square simulation solve (see nlp.heat).
        # simulation=True also squares the model for an IMODE=1 steady-state
        # solve: pins phantom vars (and equations() drops the flow limits).
        mag0 = _seed_mag(model) if simulation else 0.1
        model.mass_flow_mag_kgs = Var(mag0, min=0, name="mass_flow_mag_kgs")
        # Neutralise the MIQCQP-only vars so no integer/aux vars get injected.
        model.direction = Const(1)
        model.mass_flow_pos_kgs_squared = Const(0.0)
        model.mass_flow_neg_kgs_squared = Const(0.0)
        if simulation:
            _pin(model, "velocity_mps")
        _ensure_friction_vars(model, self.friction_model)

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        sqrt_impl = kwargs["sqrt_impl"]
        area = hydraulicsmodel.calc_pipe_area(branch.diameter_m)

        gas_density_kg_per_m3 = ogfmodel.reference_gas_density(grid)
        f_max_local = min(
            grid.max_mass_flow_kgs,
            hydraulicsmodel.calc_max_mass_flow(
                branch.diameter_m,
                gas_density_kg_per_m3,
                getattr(grid, "v_max_mps", 20.0),
            ),
        )

        # Linearise \sqrt{p} around nominal pressure for the density estimate.
        p0 = grid.nominal_pressure_pu
        x0 = p0**2
        p_from = p0 + (1 / (2 * p0)) * (
            from_node_model.vars["pressure_squared_pu"] - x0
        )
        p_to = p0 + (1 / (2 * p0)) * (to_node_model.vars["pressure_squared_pu"] - x0)
        p_avg = 0.5 * (p_from + p_to)

        signed = branch.mass_flow_kgs
        mag = branch.mass_flow_mag_kgs
        drop_term, friction_eqs = smoothmodel.drop_term_and_eqs(
            self.friction_model,
            branch,
            grid.dynamic_visc_pas,
            area,
            signed,
            mag,
            f_max_local,
            **kwargs,
        )

        eqs = [
            mag == smoothmodel.smooth_abs(signed, self.smoothing_eps, sqrt_impl),
            branch.mass_flow_pos_kgs == 0.5 * (mag + signed),
            branch.mass_flow_neg_kgs == 0.5 * (mag - signed),
        ]
        if not kwargs.get("simulation", False):
            # Operational flow limits - dropped in simulation mode (their slacks
            # would break a square IMODE=1 solve; limits are checked post-hoc).
            eqs += [
                signed <= f_max_local * branch.on_off,
                -signed <= f_max_local * branch.on_off,
            ]
        p_amb = getattr(grid, "pressure_ambient_pa", 0.0)
        eqs += [
            smoothmodel.weymouth_pressure(
                psq_pu_i=from_node_model.vars["pressure_squared_pu"],
                psq_pu_j=to_node_model.vars["pressure_squared_pu"],
                p_pu_i=from_node_model.vars["pressure_pu"],
                p_pu_j=to_node_model.vars["pressure_pu"],
                drop_term=drop_term,
                diameter_m=branch.diameter_m,
                length_m=branch.length_m,
                t_k=grid.t_k,
                compressibility=grid.compressibility,
                pressure_ref_pa=grid.pressure_ref_pa,
                on_off=branch.on_off,
                r_specific=grid.universal_gas_constant / grid.molar_mass,
                pressure_ambient_pa=p_amb,
            ),
            # density uses ABSOLUTE pressure = (gauge p_avg)*p_ref + p_ambient
            branch.gas_density_kg_per_m3
            == (grid.pressure_ref_pa * p_avg + p_amb)
            * grid.molar_mass
            / (grid.universal_gas_constant * grid.t_k),
        ]
        return eqs + friction_eqs
