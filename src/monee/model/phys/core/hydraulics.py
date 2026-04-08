import math

import numpy as np

REY_BINS = [
    50,
    100,
    200,
    400,
    800,
    1200,
    1600,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
    3200,
    3500,
    3800,
    4200,
    4600,
    5000,
    6000,
    7000,
    8000,
    1e4,
    1.5e4,
    2e4,
    3e4,
    5e4,
    1e5,
    2e5,
    5e5,
    1e6,
    2e6,
    5e6,
    1e7,
]


def calc_pipe_area(diameter_m):
    return math.pi * diameter_m**2 / 4


def calc_nikurdse(internal_diameter_m, roughness):
    return 1 / (2 * np.log10(3.71 * internal_diameter_m / roughness)) ** 2


def reynolds_equation(rey_var, mass_flow, diameter_m, dynamic_visc, pipe_area):
    return rey_var == mass_flow * diameter_m / (dynamic_visc * pipe_area)


def junction_mass_flow_balance(flows):
    return sum(flows) == 0


def pipe_mass_flow(max_v, min_v, v):
    return min_v <= v <= max_v


def flow_rate_equation(mean_flow_velocity, flow_rate, diameter, fluid_density):
    return mean_flow_velocity == flow_rate / (
        fluid_density * (diameter**2 * math.pi / 4)
    )


def swamee_jain(reynolds_var, diameter_m, roughness, log_func):
    term1 = roughness / diameter_m / 3.7
    term2 = 5.74 / (reynolds_var + 1) ** 0.9  # avoid infeasaiblity at Re=0
    denominator = log_func(term1 + term2) ** 2
    f = 0.25 / denominator
    return f


def churchill_friction(Re, D, eps):
    Re = max(Re, 1.0)
    A = (2.457 * math.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * eps / D))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)


def filter_near_linear(xs, ys, rtol=1e-6):
    if len(xs) <= 2:
        return xs, ys

    keep_x = [xs[0]]
    keep_y = [ys[0]]

    prev_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])

    for i in range(1, len(xs) - 1):
        slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])

        if abs(slope - prev_slope) > rtol * max(1.0, abs(prev_slope)):
            keep_x.append(xs[i])
            keep_y.append(ys[i])
            prev_slope = slope

    keep_x.append(xs[-1])
    keep_y.append(ys[-1])

    return keep_x, keep_y


def friction_value(Re, D, eps):
    if Re < 2300:
        return 64.0 / Re
    return swamee_jain(Re, D, eps, math.log10)


def logspace(a, b, n):
    la = math.log10(a)
    lb = math.log10(b)
    return [10 ** (la + i * (lb - la) / (n - 1)) for i in range(n)]


def piecewise_eq_friction(model, pwl):
    D = model.diameter_m
    eps = model.roughness

    xs = []

    # intentionally coarse below 2000
    xs += logspace(10.0, 2000.0, 8)

    # modest resolution in transition
    xs += logspace(2000.0, 4000.0, 8)[1:]

    # more detail in turbulent regime
    xs += logspace(4000.0, 1e7, 8)[1:]

    ys = [friction_value(x, D, eps) for x in xs]

    xs, ys = filter_near_linear(xs, ys, rtol=1e-7)

    pwl.piecewise_eq(
        y=model.friction,
        x=model.reynolds,
        xs=xs,
        ys=ys,
    )
