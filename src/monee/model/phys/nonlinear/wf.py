from ..core import hydraulics


def darcy_friction(reynolds_var):
    """
    No docstring provided.
    """
    return 64 / (reynolds_var + 1)


def darcy_weisbach_equation(
    p_start_var,
    p_end_var,
    reynolds_var,
    velocity_var,
    pipe_length,
    diameter_m,
    fluid_density,
    roughness,
    on_off=1,
    use_darcy_friction=False,
    **kwargs,
):
    """
    No docstring provided.
    """
    friction = hydraulics.swamee_jain(
        reynolds_var, diameter_m, roughness, kwargs["log_impl"]
    )
    if use_darcy_friction:
        friction = darcy_friction(reynolds_var)
    return -velocity_var * abs(velocity_var) * friction == on_off * (
        2 * (p_start_var - p_end_var) / (pipe_length / diameter_m * fluid_density)
    )
