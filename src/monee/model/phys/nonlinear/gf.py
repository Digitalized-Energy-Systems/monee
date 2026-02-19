import math


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


def calc_C_squared(diameter_m, length_m, t_k, compressability):
    """
    No docstring provided.
    """
    numerator = math.pi**2 * diameter_m**5
    denominator = 128 * length_m * R_specific * t_k * compressability
    C_squared = numerator / denominator
    return C_squared


def pipe_weymouth(
    p_squared_i,
    p_squared_j,
    f_a_pos_sq,
    f_a_neg_sq,
    diameter_m,
    length_m,
    t_k,
    compressibility,
    on_off=1,
    friction=None,
    **kwargs,
):
    return (p_squared_i - p_squared_j) * calc_C_squared(
        diameter_m,
        length_m,
        t_k,
        compressibility,
    ) * on_off == friction * -(f_a_pos_sq - f_a_neg_sq)


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
