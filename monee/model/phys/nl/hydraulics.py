import numpy as np
import math


def calc_pipe_area(diameter):
    return math.pi * diameter**2 / 4


# prandtl nikurdse formula
# https://core.ac.uk/download/pdf/38640864.pdf
def calc_nikurdse(d, k):
    return 1 / (2 * np.log10(d / k) + 1.14) ** 2


def reynolds_equation(rey_var, flow_var, d, dynamic_visc, pipe_area):
    return rey_var == flow_var * d / (dynamic_visc * pipe_area)


def junction_mass_flow_balance(flows):
    return sum(flows) == 0


def pipe_pressure(pd_min, pd_max, p_i, p_j):
    return pd_min <= p_i - p_j and p_i - p_j <= pd_max


def pipe_mass_flow(max_v, min_v, v):
    return min_v <= v <= max_v


def friction_model(rey, nikurdse):
    return (64 / rey) + nikurdse


def flow_rate_equation(mean_flow_velocity, flow_rate, diameter):
    return flow_rate == mean_flow_velocity * diameter**2 * math.pi / 4
