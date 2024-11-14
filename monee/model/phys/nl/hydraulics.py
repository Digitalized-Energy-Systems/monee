import math

import numpy as np


def calc_pipe_area(diameter_m):
    return math.pi * diameter_m**2 / 4


# prandtl nikurdse formula
# https://core.ac.uk/download/pdf/38640864.pdf
def calc_nikurdse(internal_diameter_m, roughness):
    return 1 / (2 * np.log10(internal_diameter_m / roughness) + 1.14) ** 2


def reynolds_equation(rey_var, flow_var, diameter_m, dynamic_visc, pipe_area):
    return rey_var == abs(flow_var) * diameter_m / (dynamic_visc * pipe_area)


def junction_mass_flow_balance(flows):
    return sum(flows) == 0


def pipe_pressure(pd_min_pa, pd_max_pa, from_pressure_pa, to_pressure_pa):
    return (
        pd_min_pa <= abs(from_pressure_pa - to_pressure_pa)
        and abs(from_pressure_pa - to_pressure_pa) <= pd_max_pa
    )


def pipe_mass_flow(max_v, min_v, v):
    return min_v <= v <= max_v


def friction_model(rey, nikurdse):
    return (64 / rey) + nikurdse


def flow_rate_equation(mean_flow_velocity, flow_rate, diameter):
    return flow_rate == mean_flow_velocity * diameter**2 * math.pi / 4
