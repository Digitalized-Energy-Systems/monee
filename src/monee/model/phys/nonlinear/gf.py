import math

from ..core import hydraulics


def calc_a(z, r, t, m):
    """
    No docstring provided.
    """
    return z * r * t / m


def calc_w(pipe_length, diameter, mass_flow_zero, pressure_zero, a, pipe_area):
    """
    No docstring provided.
    """
    return (
        pipe_length
        / diameter
        * (mass_flow_zero**2 * a**2 / (pipe_area**2 * pressure_zero**2))
    )


def junction_pressure(p, p_nom):
    """
    No docstring provided.
    """
    return p == p_nom**2


R_specific = 504.5


def calc_C_squared(diameter_m, friction, length_m, t_k, compressability):
    """
    No docstring provided.
    """
    numerator = math.pi**2 * diameter_m**5
    denominator = 128 * friction * length_m * R_specific * t_k * compressability
    C_squared = numerator / denominator
    return C_squared


def pipe_weymouth(
    p_i,
    p_j,
    f_a,
    rey,
    diameter_m,
    roughness,
    length_m,
    t_k,
    compressibility,
    on_off=1,
    **kwargs,
):
    """
    No docstring provided.
    """
    return (p_i**2 - p_j**2) * calc_C_squared(
        diameter_m,
        hydraulics.swamee_jain(rey, diameter_m, roughness, kwargs["log_impl"]),
        length_m,
        t_k,
        compressibility,
    ) * on_off == -abs(f_a) * f_a


def normal_pressure(p, p_squared):
    """
    No docstring provided.
    """
    return p**2 == p_squared


def compressor_boost(comp_ratio, p_i, p_j):
    """
    No docstring provided.
    """
    return comp_ratio * p_i == p_j


def compressor_ratio_one(comp_ratio, v):
    """
    No docstring provided.
    """
    return v * (1 - comp_ratio) <= 0


def compressor_limits(comp_up_limit, comp_ratio, comp_lower_limit):
    """
    No docstring provided.
    """
    return comp_lower_limit <= comp_ratio <= comp_up_limit
