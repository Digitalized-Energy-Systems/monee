import math


def darcy_friction(reynolds_var):
    return 64 / (reynolds_var + 1)


def darcy_weisbach_equation(
    p_i,
    p_j,
    m_pos_sq,
    m_neg_sq,
    pipe_length,
    diameter_m,
    fluid_density_kg_per_m3,
    on_off=1,
    friction=None,
    **kwargs,
):
    A = math.pi * diameter_m**2 / 4  # pipe cross-section [m^2]

    resistance = (
        friction
        * (pipe_length / diameter_m)
        * (1.0 / (2.0 * fluid_density_kg_per_m3 * A**2))
    )

    return (p_i - p_j) == resistance * -(m_pos_sq - m_neg_sq)
