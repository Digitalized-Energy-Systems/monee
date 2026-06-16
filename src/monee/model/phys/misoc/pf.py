# Branch-Flow SOCP with an ideal a:1 transformer in series with Z = r + jx.
#   W_j = W_i / a^2 - 2(r \cdot P + x \cdot Q) + |Z|^2 \cdot ell
#   P^2 + Q^2 \le (W_i / a^2) \cdot ell
# Losses are tap-free (ideal transformer is lossless). tap=1 reduces to BFM.


def active_power_loss(
    var_active_power_from, var_active_power_to, var_ell_pu, resistance_r
):
    return var_active_power_from == var_ell_pu * resistance_r - var_active_power_to


def reactive_power_loss(
    var_reactive_power_from, var_reactive_power_to, var_ell_pu, reactance_x
):
    return var_reactive_power_from == var_ell_pu * reactance_x - var_reactive_power_to


def voltage_drop(
    var_w_pu_i,
    var_w_pu_j,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_ell_pu,
    resistance_r,
    reactance_x,
    tap=1.0,
):
    return var_w_pu_j - (
        var_w_pu_i / (tap * tap)
        - 2
        * (
            resistance_r * var_active_power_ij_pu
            + reactance_x * var_reactive_power_ij_pu
        )
        + (resistance_r**2 + reactance_x**2) * var_ell_pu
    )


def soc_rel(
    var_w_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_ell_pu,
    tap=1.0,
):
    r"""Rotated SOC :math:`P^2 + Q^2 \le (W/tap^2) \cdot ell`. See :func:`soc_rel_lorentz`
    for the Lorentz reformulation that survives non-convex MIQCP siblings."""
    return (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
        <= (var_w_pu_i / (tap * tap)) * var_ell_pu
    )


def soc_eq(
    var_w_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_ell_pu,
    tap=1.0,
):
    r"""Exact (non-convex) form of :func:`soc_rel`: :math:`P^2 + Q^2 = (W/tap^2) \cdot ell`.
    Pins the branch-flow model to the physical surface; requires a global
    MIQCQP solver (SCIP, Gurobi)."""
    return (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
        == (var_w_pu_i / (tap * tap)) * var_ell_pu
    )


def soc_rel_lorentz(
    var_w_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_ell_pu,
    var_soc_sum,
    var_soc_diff,
    tap=1.0,
):
    r"""Lorentz form of :func:`soc_rel`: :math:`s = (W/tap^2+ell)/2`, :math:`d = (W/tap^2-ell)/2`,
    :math:`P^2 + Q^2 + d^2 \le s^2` (since :math:`s^2 - d^2 = (W/tap^2) \cdot ell`)."""
    Wp = var_w_pu_i / (tap * tap)
    return [
        var_soc_sum == 0.5 * (Wp + var_ell_pu),
        var_soc_diff == 0.5 * (Wp - var_ell_pu),
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2 + var_soc_diff**2
        <= var_soc_sum**2,
    ]


def gap_expr(
    var_w_pu_i,
    var_active_power_ij_pu,
    var_reactive_power_ij_pu,
    var_ell_pu,
    tap=1.0,
):
    return (var_w_pu_i / (tap * tap)) * var_ell_pu - (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
    )
