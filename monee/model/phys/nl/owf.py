from . import hydraulics


# laminar flow Dary friction
def darcy_friction(reynolds_var):
    return 64 / (reynolds_var + 1)


# https://apps.dtic.mil/sti/citations/AD0874542
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
    Original
    return (p_start_var - p_end_var) * on_off ==(
        swamee_jain(reynolds_var, diameter_m, roughness, kwargs["log_impl"]) *
        (pipe_length / diameter_m) *
        (fluid_density * -velocity_var*abs(velocity_var) / 2)
    )
    """
    # reformulated to optimize numerical stability
    # note swamee-jain decreases stability due to the log
    friction = hydraulics.swamee_jain(
        reynolds_var, diameter_m, roughness, kwargs["log_impl"]
    )
    if use_darcy_friction:
        friction = darcy_friction(reynolds_var)

    return -velocity_var * abs(velocity_var) * friction == on_off * (
        2 * (p_start_var - p_end_var) / ((pipe_length / diameter_m) * fluid_density)
    )
