r"""Smooth, binary-free hydraulic primitives for pure-NLP (IPOPT/APOPT) solves.

The MISOCP-shaped models in :mod:`gf`/:mod:`wf` rely on a ``direction`` binary,
a ``mass_flow_pos_kgs``/``mass_flow_neg_kgs`` split and an epigraph relaxation kept tight
by an objective term - none of which survive an NLP relaxation. The helpers here
express the same physics with a single signed mass flow and the smooth identity
:math:`f_{pos}^2 - f_{neg}^2 = m \cdot |m|` (with :math:`|m| \approx \sqrt{m^2+\varepsilon^2}`), so the pressure drop is :math:`C^\infty`,
monotonic and odd in the flow.

A pipe's pressure drop is written as :math:`\Delta p = -\text{drop\_term}` (in the carrier's units),
where ``drop_term`` is :math:`friction \cdot m \cdot |m|` for the analytic friction models and
a single odd cubic spline :math:`\psi(m)` for the PWL model.
"""

import math

import monee.model.phys.core.hydraulics as hyd

from .gf import calc_C_squared


def smooth_abs(signed, eps, sqrt_impl=math.sqrt):
    r"""Smooth absolute value :math:`\sqrt{m^2 + \varepsilon^2}`. :math:`\varepsilon` is a small mass-flow scale [kg/s]."""
    return sqrt_impl(signed * signed + eps * eps)


def weymouth_pressure(
    psq_pu_i,
    psq_pu_j,
    drop_term,
    diameter_m,
    length_m,
    t_k,
    compressibility,
    pressure_ref_pa,
    on_off,
):
    r"""Signed Weymouth, row-normalised by the per-pipe pressure coefficient.

    The raw form :math:`\Delta(p^2) \cdot C^2 \cdot \text{on\_off} = -\text{drop\_term}` carries a coefficient
    :math:`C^2 \cdot p_{ref}^2 \propto D^5` on the (dimensionless) squared-pressure difference, which
    spans ~6 orders across a wide-diameter network - skewing IPOPT's column
    scaling and conditioning. Dividing through by that coefficient gives every
    pipe a unit coefficient on the pressure term (the solution is unchanged):

    .. math::

        (p_i^2 - p_j^2)_{pu} \cdot \text{on\_off} = -\text{drop\_term} / (C^2 \cdot p_{ref}^2)
    """
    coeff = calc_C_squared(diameter_m, length_m, t_k, compressibility) * pressure_ref_pa**2
    return (psq_pu_i - psq_pu_j) * on_off == -drop_term / coeff


def darcy_pressure(p_i, p_j, drop_term, length_m, diameter_m, fluid_density_kg_per_m3):
    r"""Signed Darcy-Weisbach: :math:`(p_i-p_j) = -K \cdot \text{drop\_term}` with
    :math:`K = L / (2 \cdot \rho \cdot A^2 \cdot D)`."""
    area = hyd.calc_pipe_area(diameter_m)
    K = length_m / (2.0 * fluid_density_kg_per_m3 * area**2 * diameter_m)
    return (p_i - p_j) == -K * drop_term


def smooth_friction_blend(
    reynolds_scaled,
    diameter_m,
    roughness_m,
    log_impl,
    exp_impl,
    re_crit=2300.0,
    sharpness=400.0,
):
    """Darcy friction factor as a single smooth function of Reynolds.

    Sigmoid blend of the laminar law ``64/Re`` and the turbulent Swamee-Jain
    correlation around ``re_crit`` - replaces the non-smooth ``if Re < 2300``
    switch so IPOPT sees a continuously differentiable friction. ``reynolds_scaled``
    is ``Re / REYNOLDS_SCALE`` (see :data:`hyd.REYNOLDS_SCALE`).
    """
    Re = reynolds_scaled * hyd.REYNOLDS_SCALE
    f_lam = 64.0 / (Re + 1.0)
    f_turb = hyd.swamee_jain(Re, diameter_m, roughness_m, log_impl)
    w = 1.0 / (1.0 + exp_impl(-(Re - re_crit) / sharpness))
    return (1.0 - w) * f_lam + w * f_turb


def signed_psi_breakpoints(
    diameter_m, roughness_m, dynamic_visc_pas, area, m_max, n_breakpoints=12
):
    r"""Odd breakpoints for :math:`\psi(m) = friction(Re(|m|)) \cdot m \cdot |m|` on
    :math:`m \in [-m_{max}, m_{max}]` - the full Darcy/Weymouth drop term as one spline."""
    if m_max <= 0:
        return [-1e-6, 0.0, 1e-6], [0.0, 0.0, 0.0]
    mags = hyd.logspace(max(m_max * 1e-4, 1e-9), m_max, max(2, n_breakpoints - 1))

    def psi(m):
        Re = m * diameter_m / (dynamic_visc_pas * area)
        return hyd.friction_value(Re, diameter_m, roughness_m) * m * m

    xs = [-m for m in reversed(mags)] + [0.0] + list(mags)
    ys = [-psi(m) for m in reversed(mags)] + [0.0] + [psi(m) for m in mags]
    return xs, ys


def drop_term_and_eqs(
    friction_model, model, dynamic_visc_pas, area, signed, mag, m_max, **kwargs
):
    r"""Return ``(drop_term, eqs)`` for the pressure-drop equation.

    ``constant``/``nonlinear`` :math:`\to` :math:`friction \cdot m \cdot |m|` (plus the Reynolds/blend
    closure for ``nonlinear``); ``pwl`` :math:`\to` an odd cubic spline :math:`\psi(\text{signed})`.
    """
    if friction_model == "pwl":
        xs, ys = signed_psi_breakpoints(
            model.diameter_m, model.roughness_m, dynamic_visc_pas, area, m_max
        )
        kwargs["pwl_impl"].piecewise_eq(y=model.psi_pwl, x=signed, xs=xs, ys=ys)
        return model.psi_pwl, []

    drop_term = model.friction * signed * mag
    if friction_model == "constant":
        return drop_term, []
    return drop_term, [
        hyd.reynolds_equation(
            model.reynolds_scaled, mag, model.diameter_m, dynamic_visc_pas, area
        ),
        model.friction
        == smooth_friction_blend(
            model.reynolds_scaled,
            model.diameter_m,
            model.roughness_m,
            kwargs["log_impl"],
            kwargs["exp_impl"],
        ),
    ]
