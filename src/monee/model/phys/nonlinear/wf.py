import math


def darcy_friction(reynolds_var):
    """
    No docstring provided.
    """
    return 64 / (reynolds_var + 1)


def darcy_weisbach_equation(
    p_i,
    p_j,
    m_pos_sq,
    m_neg_sq,
    pipe_length,
    diameter_m,
    fluid_density,
    on_off=1,
    friction=None,
    **kwargs,
):
    # friction = hydraulics.swamee_jain(
    #     reynolds_var, diameter_m, roughness, kwargs["log_impl"]
    # )
    # if use_darcy_friction:
    #     friction = darcy_friction(reynolds_var)
    # return -velocity_var * abs(velocity_var) * friction == on_off * (
    #     2 * (p_i - p_j) / (pipe_length / diameter_m * fluid_density)
    # )
    A = math.pi * diameter_m**2 / 4  # pipe cross-sectional area [m²]

    Rm = friction * (pipe_length / diameter_m) * (1.0 / (2.0 * fluid_density * A**2))

    return (p_i - p_j) == Rm * -(m_pos_sq - m_neg_sq)
