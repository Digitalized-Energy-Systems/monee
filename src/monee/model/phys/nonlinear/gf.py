import math

R_specific = 504.5


def calc_C_squared(diameter_m, length_m, t_k, compressibility):
    numerator = math.pi**2 * diameter_m**5
    denominator = 128 * length_m * R_specific * t_k * compressibility
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
