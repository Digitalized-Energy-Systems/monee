import math

from monee.model.phys.core.hydraulics import calc_pipe_area


def pipe_insulation_ua(branch):
    r"""Cylindrical-conduction insulation conductance :math:`UA = 2\pi\lambda L / \ln(r_{out}/r_{in})`
    [W/K] of a buried/insulated pipe, with :math:`r_{in} = D/2` and
    :math:`r_{out} = D/2 + \text{insulation\_thickness}`."""
    pipe_outside_r = branch.diameter_m / 2 + branch.insulation_thickness_m
    pipe_inside_r = branch.diameter_m / 2
    return (
        2
        * math.pi
        * branch.lambda_insulation_w_per_m_k
        * branch.length_m
        / math.log(pipe_outside_r / pipe_inside_r)
    )


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
    A = calc_pipe_area(diameter_m)  # pipe cross-section [m^2]

    resistance = (
        friction
        * (pipe_length / diameter_m)
        * (1.0 / (2.0 * fluid_density_kg_per_m3 * A**2))
    )

    # on_off gates the pressure coupling exactly like gf.pipe_weymouth: an off
    # pipe (flows big-M'd to 0) must not enforce p_i == p_j.
    return (p_i - p_j) * on_off == resistance * -(m_pos_sq - m_neg_sq)
