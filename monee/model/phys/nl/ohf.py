import math

SPECIFIC_HEAT_CAP_WATER = 4184


def to_celsius(t_k):
    """
    No docstring provided.
    """
    return t_k - 273.15


def calc_prandtl(t_k):
    """
    No docstring provided.
    """
    return 50000 / (to_celsius(t_k) ** 2 + 155 * to_celsius(t_k) + 3700)


def heat_transfer_loss(
    heat_transfer_flow_loss_var,
    t_var,
    k_insulation_w_per_k,
    ext_t,
    pipe_length,
    pipe_inside_radius,
    pipe_outside_radius,
):
    """
    No docstring provided.
    """
    return (
        heat_transfer_flow_loss_var
        == 2
        * math.pi
        * k_insulation_w_per_k
        * pipe_length
        * (t_var - ext_t)
        / math.log(pipe_outside_radius / pipe_inside_radius)
    )


def temp_flow(t_in_scaled, t_out_scaled, heat_loss, mass_flow, sign_impl):
    """
    No docstring provided.
    """
    return heat_loss == -mass_flow * SPECIFIC_HEAT_CAP_WATER * (
        t_in_scaled - t_out_scaled
    )


def heat_transfer_coefficient_inside_pipe_db(
    heat_transfer_coefficient_var,
    reynolds_var,
    t_1_var,
    t_2_var,
    pipe_inside_diameter,
    thermal_conductivity=0.598,
):
    """
    No docstring provided.
    """
    return (
        heat_transfer_coefficient_var
        == 0.023
        * (thermal_conductivity / pipe_inside_diameter)
        * abs(reynolds_var) ** 0.8
        * calc_prandtl((t_1_var + t_2_var) / 2) ** 0.3
    )


def heat_transfer_coefficient_inside_pipe(
    heat_transfer_coefficient_var,
    reynolds_var,
    t_1_var,
    t_2_var,
    pipe_inside_diameter,
    thermal_conductivity=0.598,
):
    """
    No docstring provided.
    """
    prandtl = calc_prandtl((t_1_var + t_2_var) / 2)
    friction_f = abs(reynolds_var) / 64
    return (
        heat_transfer_coefficient_var
        == thermal_conductivity
        / pipe_inside_diameter
        * (friction_f / 8 * (abs(reynolds_var) - 1000) * prandtl)
        / (1 + 12.7 * (friction_f / 8) ** 0.5 * (prandtl ** (2 / 3) - 1))
    )
