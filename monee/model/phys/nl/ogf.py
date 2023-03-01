from . import hydraulics


def calc_a(z, r, t, m):
    return z * r * t / m


def calc_w(pipe_length, diameter, mass_flow_zero, pressure_zero, a, pipe_area):
    return (pipe_length / diameter) * (
        (mass_flow_zero**2 * a**2) / (pipe_area**2 * pressure_zero**2)
    )


def junction_pressure(p, p_nom):
    return p == p_nom**2


def pipe_weymouth(p_i, p_j, w, f_a, rey, nikurdse):
    return p_i - p_j == hydraulics.friction_model(rey, nikurdse) * w * abs(f_a) * f_a


def normal_pressure(p, p_squared):
    return p**2 == p_squared


def compressor_boost(comp_ratio, p_i, p_j):
    return comp_ratio * p_i == p_j


def compressor_ratio_one(comp_ratio, v):
    return v * (1 - comp_ratio) <= 0


def compressor_limits(comp_up_limit, comp_ratio, comp_lower_limit):
    return comp_lower_limit <= comp_ratio <= comp_up_limit
