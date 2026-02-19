import math

import numpy as np


def calc_pipe_area(diameter_m):
    """
    No docstring provided.
    """
    return math.pi * diameter_m**2 / 4


def calc_nikurdse(internal_diameter_m, roughness):
    """
    No docstring provided.
    """
    return 1 / (2 * np.log10(3.71 * internal_diameter_m / roughness)) ** 2


def reynolds_equation(rey_var, mass_flow, diameter_m, dynamic_visc, pipe_area):
    """
    No docstring provided.
    """
    return rey_var == mass_flow * diameter_m / (dynamic_visc * pipe_area)


def junction_mass_flow_balance(flows):
    """
    No docstring provided.
    """
    return sum(flows) == 0


def pipe_mass_flow(max_v, min_v, v):
    """
    No docstring provided.
    """
    return min_v <= v <= max_v


def flow_rate_equation(mean_flow_velocity, flow_rate, diameter, fluid_density):
    """
    No docstring provided.
    """
    return mean_flow_velocity == flow_rate / (
        fluid_density * (diameter**2 * math.pi / 4)
    )


def swamee_jain(reynolds_var, diameter_m, roughness, log_func):
    """
    No docstring provided.
    """
    term1 = roughness / diameter_m / 3.7
    term2 = 5.74 / (reynolds_var + 1) ** 0.9  # avoid infeasaiblity at Re=0
    denominator = log_func(term1 + term2) ** 2
    f = 0.25 / denominator
    return f


def piecewise_eq_friction(model, pwl):
    re_pts = [
        20,
        200,
        500,
        1000,
        1500,
        2000,
        4000,
        8000,
        15000,
        30000,
        50000,
        80000,
        200000,
        400000,
        1000000,
    ]

    f_pts = []
    for r in re_pts:
        if r < 2000:
            f_pts.append(64.0 / r)
        else:
            f_pts.append(swamee_jain(r, model.diameter_m, model.roughness, math.log10))

    pwl.piecewise_eq(
        y=model.friction,
        x=model.reynolds,
        xs=re_pts,
        ys=f_pts,
    )
