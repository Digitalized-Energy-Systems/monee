# Branch-Flow SOCP with an ideal a:1 transformer in series with Z = r + jx.
#   W_j = W_i / a² - 2(r·P + x·Q) + |Z|²·ell
#   P² + Q² ≤ (W_i / a²) · ell
# Losses are tap-free (ideal transformer is lossless). tap=1 reduces to BFM.


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
    """Rotated SOC ``P² + Q² ≤ (W/tap²) · ell``. See :func:`soc_rel_lorentz`
    for the Lorentz reformulation that survives non-convex MIQCP siblings."""
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
    """Lorentz form of :func:`soc_rel`: ``s = (W/tap²+ell)/2``, ``d = (W/tap²-ell)/2``,
    ``P² + Q² + d² ≤ s²`` (since ``s² − d² = (W/tap²)·ell``)."""
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
