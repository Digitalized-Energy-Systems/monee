import math

# J/(kg*K)
SPECIFIC_HEAT_CAP_WATER = 4184


def to_celsius(t_k):
    return t_k - 273.15


# water, 1bar
def calc_prandtl(t_k):
    return 50000 / (to_celsius(t_k) ** 2 + 155 * to_celsius(t_k) + 3700)


# https://engineeringlibrary.org/reference/conduction-heat-transfer-doe-handbook
def heat_transfer_loss(
    heat_transfer_flow_loss_var,
    t_var,
    t_var2,
    k_insulation_w_per_k,
    ext_t,
    pipe_length,
    pipe_inside_diameter,
    pipe_outside_diameter
):
    return (
        heat_transfer_flow_loss_var
        == 2
        * math.pi
        * k_insulation_w_per_k
        * pipe_length
        * (((t_var - ext_t) + (t_var2 - ext_t)) / 2)
        / math.log(pipe_outside_diameter / pipe_inside_diameter)
    )


def heat_transfer_pipe(
    heat_transfer_flow_loss_var,
    t_1_var,
    t_2_var,
):
    return (t_1_var - t_2_var) * SPECIFIC_HEAT_CAP_WATER == heat_transfer_flow_loss_var


def heat_exchange_pipe(
    heat_transfer_flow_loss_var,
    t_1_var,
    t_2_var,
    mass_flow_var,
):
    return (t_1_var - t_2_var) * SPECIFIC_HEAT_CAP_WATER * (
        -mass_flow_var
    ) == heat_transfer_flow_loss_var


# Dittus-Bölter correlation
def heat_transfer_coefficient_inside_pipe_db(
    heat_transfer_coefficient_var,
    reynolds_var,
    t_1_var,
    t_2_var,
    pipe_inside_diameter,
    thermal_conductivity=0.598,
):
    return heat_transfer_coefficient_var == (
        0.023
        * (thermal_conductivity / pipe_inside_diameter)
        * abs(reynolds_var) ** 0.8
        * calc_prandtl((t_1_var + t_2_var) / 2) ** 0.3
    )


# Gnielinski correlation
def heat_transfer_coefficient_inside_pipe(
    heat_transfer_coefficient_var,
    reynolds_var,
    t_1_var,
    t_2_var,
    pipe_inside_diameter,
    thermal_conductivity=0.598,
):
    prandtl = calc_prandtl((t_1_var + t_2_var) / 2)
    friction_f = abs(reynolds_var) / 64
    return heat_transfer_coefficient_var == (
        (thermal_conductivity / pipe_inside_diameter)
        * ((friction_f / 8) * (abs(reynolds_var) - 1000) * prandtl)
        / (1 + 12.7 * (friction_f / 8) ** 0.5 * (prandtl ** (2 / 3) - 1))
    )
