import math


# https://apps.dtic.mil/sti/citations/AD0874542
def darcy_weisbach_equation(
    p_start_var,
    p_end_var,
    reynolds_var,
    mean_flow_velocity_var,
    nikurdse,
    pipe_length,
    diameter,
    fluid_density,
):
    return p_start_var - p_end_var == ((64 / reynolds_var) + nikurdse) * pipe_length * (
        fluid_density / 2
    ) * (mean_flow_velocity_var**2 / diameter)
