from . import hydraulics


def calc_a(z, r, t, m):
    return z * r * t / m


def calc_w(pipe_length, diameter, mass_flow_zero, pressure_zero, a, pipe_area):
    return (pipe_length / diameter) * (
        (mass_flow_zero**2 * a**2) / (pipe_area**2 * pressure_zero**2)
    )


def junction_pressure(p, p_nom):
    return p == p_nom**2


import math
R_specific = 504.5      # J/(kg·K) for natural gas

def calc_C_squared(diameter_m, friction, length_m, t_k, compressability):
    numerator = math.pi**2 * diameter_m**5
    denominator = 128 * friction * length_m * R_specific * t_k * compressability

    C_squared = numerator / denominator
    return C_squared


def pipe_weymouth(p_i, p_j, f_a, rey, diameter_m, roughness, length_m, t_k, compressibility, on_off=1, **kwargs):
    return ((p_i)**2 - (p_j)**2) * calc_C_squared(diameter_m, hydraulics.swamee_jain(rey, diameter_m, roughness, kwargs["log_impl"]), length_m, t_k, compressibility) * on_off  == (-abs(f_a) * f_a)


def normal_pressure(p, p_squared):
    return p**2 == p_squared


def compressor_boost(comp_ratio, p_i, p_j):
    return comp_ratio * p_i == p_j


def compressor_ratio_one(comp_ratio, v):
    return v * (1 - comp_ratio) <= 0


def compressor_limits(comp_up_limit, comp_ratio, comp_lower_limit):
    return comp_lower_limit <= comp_ratio <= comp_up_limit
