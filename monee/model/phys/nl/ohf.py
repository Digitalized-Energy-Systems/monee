import math
from . import hydraulics


def to_celsius(t_k):
    return t_k - 273.15


# water, 1bar
def calc_prandtl(t_k):
    return 50000 / (to_celsius(t_k) ** 2 + 155 * to_celsius(t_k) + 3700)


# https://engineeringlibrary.org/reference/conduction-heat-transfer-doe-handbook
def heat_transfer_loss(
    heat_transfer_flow_loss_var,
    t_var,
    k_insulation_w_per_k,
    ext_t,
    pipe_length,
    pipe_inside_diameter,
    pipe_outside_diameter,
):
    return (
        heat_transfer_flow_loss_var
        == 2
        * math.pi
        * k_insulation_w_per_k
        * pipe_length
        * (t_var - ext_t)
        / math.log(pipe_outside_diameter / pipe_inside_diameter)
    )


def heat_transfer_pipe(
    heat_transfer_var,
    heat_transfer_flow_loss_var,
    t_1_var,
    t_2_var,
    heat_transfer_coefficient_var,
    diameter,
):
    return (
        heat_transfer_var - heat_transfer_flow_loss_var
        == heat_transfer_coefficient_var
        * hydraulics.calc_pipe_area(diameter)
        * (t_1_var - t_2_var)
    )


def heat_transfer_coefficient_inside_pipe(
    heat_transfer_coefficient_var,
    reynolds_var,
    t_1_var,
    t_2_var,
    pipe_inside_diameter,
    thermal_conductivity=0.598,
):
    return heat_transfer_coefficient_var == (
        0.023
        * thermal_conductivity
        / pipe_inside_diameter
        * reynolds_var**0.8
        * calc_prandtl((t_1_var + t_2_var) / 2) ** 0.3
    )
