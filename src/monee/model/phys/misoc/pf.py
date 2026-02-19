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
):
    return var_voltage_pu_j - (
        var_voltage_pu_i
        - 2
        * (
            resistance_r * var_active_power_ij_pu
            + reactance_x * var_reactive_power_ij_pu
        )
        + (resistance_r**2 + reactance_x**2) * var_im_ij_pu
    )


def soc_rel(
    var_voltage_pu_i, var_active_power_ij_pu, var_reactive_power_ij_pu, var_im_ij_pu
):
    return (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
        <= var_voltage_pu_i * var_im_ij_pu
    )


def gap_expr(
    var_voltage_pu_i, var_active_power_ij_pu, var_reactive_power_ij_pu, var_im_ij_pu
):
    return var_voltage_pu_i * var_im_ij_pu - (
        var_active_power_ij_pu**2 + var_reactive_power_ij_pu**2
    )
