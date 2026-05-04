# Branch-Flow / second-order-cone formulation of the AC power flow with an
# off-nominal-tap transformer in series with the branch impedance.
#
# Model: ideal transformer (a:1) at the from-end followed by series impedance
# Z = r + j x.  Define V_i' = V_i / a (the voltage on the secondary of the
# ideal transformer), so V_j = V_i' - Z * I_ij and S_ij = V_i' * conj(I_ij).
# Squaring: |V_j|^2 = |V_i|^2 / a^2 - 2 * Re(conj(V_i') * Z * I_ij) + |Z|^2 * |I|^2,
# which simplifies to
#     W_j = W_i / a^2 - 2 (r*P + x*Q) + |Z|^2 * ell
# and the SOC relaxation becomes
#     P^2 + Q^2 <= (W_i / a^2) * ell.
# Active/reactive losses are unaffected by the ideal transformer (lossless),
# so the per-branch loss equations stay tap-free.
#
# For lines (tap = 1) all expressions reduce to the standard BFM/SOCP form.


def active_power_loss(
    var_active_power_from, var_active_power_to, var_im_ij_pu, resistance_r
):
    return var_active_power_from == var_im_ij_pu * resistance_r - var_active_power_to


def reactive_power_loss(
    var_reactive_power_from, var_reactive_power_to, var_im_ij_pu, reactance_x
):
    return var_reactive_power_from == var_im_ij_pu * reactance_x - var_reactive_power_to


def voltage_drop(
    var_voltage_pu_i,
    var_voltage_pu_j,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_im_ij_pu,
    resistance_r,
    reactance_x,
    tap=1.0,
):
    return var_voltage_pu_j - (
        var_voltage_pu_i / (tap * tap)
        - 2
        * (
            resistance_r * var_active_power_ij_pu
            + reactance_x * var_reactive_power_ij_pu
        )
        + (resistance_r**2 + reactance_x**2) * var_im_ij_pu
    )


def soc_rel(
    var_voltage_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_im_ij_pu,
    tap=1.0,
):
    """Branch-flow rotated SOC: ``P² + Q² ≤ (W/tap²) · ell``.

    Pyomo writes this as a single quadratic constraint with the bilinear
    ``W·ell``.  Gurobi *can* auto-detect the rotated cone (W, ell ≥ 0), but
    in our multi-energy model the presence of unrelated quadratic
    equalities (Weymouth ``m² == m_squared``) flips the whole problem to
    non-convex MIQCP, and once in that mode Gurobi spatial-branches on
    every bilinear — including this one.

    See :func:`soc_rel_lorentz` for an explicit Lorentz-cone reformulation
    that avoids the W·ell bilinear via the auxiliary ``s = (W/tap² + ell)/2``
    and ``d = (W/tap² − ell)/2``: ``P² + Q² + d² ≤ s²``, which Gurobi
    handles as a convex cone in barrier/SOCP even alongside non-convex
    bilinears in the rest of the model.
    """
    return (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
        <= (var_voltage_pu_i / (tap * tap)) * var_im_ij_pu
    )


def soc_rel_lorentz(
    var_voltage_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_im_ij_pu,
    var_s,
    var_d,
    tap=1.0,
):
    """Lorentz-cone form of :func:`soc_rel`.

    Returns three constraints:

    * ``s ==  (W/tap² + ell)/2``  (linear)
    * ``d ==  (W/tap² − ell)/2``  (linear)
    * ``P² + Q² + d² ≤ s²``       (convex Lorentz cone)

    The original rotated cone ``P² + Q² ≤ (W/tap²) · ell`` is recovered
    because ``s² − d² = (W/tap²) · ell``.
    """
    Wp = var_voltage_pu_i / (tap * tap)
    return [
        var_s == 0.5 * (Wp + var_im_ij_pu),
        var_d == 0.5 * (Wp - var_im_ij_pu),
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2 + var_d**2 <= var_s**2,
    ]


def gap_expr(
    var_voltage_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_im_ij_pu,
    tap=1.0,
):
    return (var_voltage_pu_i / (tap * tap)) * var_im_ij_pu - (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
    )
