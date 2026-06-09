import math


def power_balance_equation(signed_flows):
    """
    No docstring provided.
    """
    return sum(signed_flows) == 0


def calc_branch_t(tap, shift):
    """
    real and imaginary part of transformer tap ration.
    """
    return (tap * math.cos(shift), tap * math.sin(shift))


def square_relax(v_sq_var, v_var, v_min, v_max): #quadratic terms v² relaxed into convex envelopes (according paper Fig 3, equation in 4.1 power flow relaxation)
    """
    Convex bounds of v^2 on [v_min, v_max].
    """
    return [
        v_sq_var >= v_var**2,
        v_sq_var <= (v_max + v_min) * v_var - v_max * v_min,
    ]


def cosine_relax(
    cs_var,
    delta_var,
    delta_max,
    on_off=1,
    delta_big_m=None,
):
    """
    Quadratic relaxation of cosine from the QC paper.
    If on_off is binary, uses the switched version from the paper.
    convex outer approximation of cosine
    """
    if delta_big_m is None:
        delta_big_m = delta_max

    alpha = (1 - math.cos(delta_max)) / (delta_max**2) #alpha as helper variable for whole equation for figure 1a

    return [#cosine variable bounded by constraints for convex outer approx.
        cs_var <= on_off - alpha * delta_var**2 + (1 - on_off) * alpha * delta_big_m**2,
        cs_var >=  on_off * math.cos(delta_max),
        cs_var <= on_off,
        delta_var >= -on_off * delta_max - (1 - on_off) * delta_big_m,
        delta_var <= on_off * delta_max + (1 - on_off) * delta_big_m,
    ]


def sine_relax(
    s_var,
    delta_var,
    delta_max,
    on_off=1,
    delta_big_m=None,
):
    """
    Polyhedral relaxation of sine from the QC paper.
    If on_off is binary, uses the switched version from the paper.
    """
    if delta_big_m is None:
        delta_big_m = delta_max

    c = math.cos(delta_max / 2)
    rhs = math.sin(delta_max / 2) - c * delta_max / 2
    s_bound = math.sin(delta_max / 2) + c * delta_max / 2

    return [
        s_var - c * delta_var <= on_off * rhs + (1 - on_off) * c * delta_big_m,
        -s_var + c * delta_var <= on_off * rhs + (1 - on_off) * c * delta_big_m,
        s_var >= on_off * math.sin(-delta_max),
        s_var <= on_off * math.sin(delta_max),
        s_var >= -on_off * s_bound,
        s_var <= on_off * s_bound,
        delta_var >= -on_off * delta_max - (1 - on_off) * delta_big_m,
        delta_var <= on_off * delta_max + (1 - on_off) * delta_big_m,
    ]


def mccormick_relax(
    product_var,
    x_var,
    y_var,
    x_lb,
    x_ub,
    y_lb,
    y_ub,
):
    """
    McCormick envelope for product_var = x_var * y_var. (according to Fig 2 in paper)
    Convex envelope for bilinear expressions introduced by McCormick -> relaxed using sequential bilinear approache (in paper sequential McCormick relaxation used as it offers comparably tight bounds)
    In chapter 4.1 Power flow relaxation in paper
    """
    return [
        product_var >= x_lb * y_var + y_lb * x_var - x_lb * y_lb,
        product_var >= x_ub * y_var + y_ub * x_var - x_ub * y_ub,
        product_var <= x_lb * y_var + y_ub * x_var - x_lb * y_ub,
        product_var <= x_ub * y_var + y_lb * x_var - x_ub * y_lb,
    ]


def int_flow_from_p(
    p_from_var,
    v_sq_from_var, #squared voltage
    wc_var, #v_i*v_j*cos(theta_i-theta_j)
    ws_var, #v_i*v_j*sin(theta_i-theta_j)
    g_branch,
    b_branch,
    tap,
    shift,
    g_from=0,
    on_off=1,
):
    """
    QC relaxation of active power flow from-side.
    """
    tr, ti = calc_branch_t(tap, shift)
    return p_from_var == ( #Ohm's law linearized , paper equation 1
        (g_branch + g_from) / tap**2 * v_sq_from_var
        + (-g_branch * tr + b_branch * ti) / tap**2 * wc_var
        + (-b_branch * tr - g_branch * ti) / tap**2 * ws_var
    )


def int_flow_from_q(
    q_from_var,
    v_sq_from_var,
    wc_var,
    ws_var,
    g_branch,
    b_branch,
    tap,
    shift,
    b_from=0,
    on_off=1,
):
    """
    QC relaxation of reactive power flow from-side.
    """
    tr, ti = calc_branch_t(tap, shift)
    return q_from_var == ( #Ohm's law linearized , paper equation 2
        -(b_branch + b_from) / tap**2 * v_sq_from_var
        - (-b_branch * tr - g_branch * ti) / tap**2 * wc_var
        + (-g_branch * tr + b_branch * ti) / tap**2 * ws_var
    )


def int_flow_to_p(
    p_to_var,
    v_sq_to_var,
    wc_var,
    ws_var,
    g_branch,
    b_branch,
    tap,
    shift,
    g_to=0,
    on_off=1,
):
    """
    QC relaxation of active power flow to-side.
    """
    tr, ti = calc_branch_t(tap, shift)
    return p_to_var == (
        (g_branch + g_to) * v_sq_to_var
        + (-g_branch * tr - b_branch * ti) / tap**2 * wc_var
        - (-b_branch * tr + g_branch * ti) / tap**2 * ws_var
    )


def int_flow_to_q(
    q_to_var,
    v_sq_to_var,
    wc_var,
    ws_var,
    g_branch,
    b_branch,
    tap,
    shift,
    b_to=0,
    on_off=1,
):
    """
    QC relaxation of reactive power flow to-side.
    """
    tr, ti = calc_branch_t(tap, shift)
    return q_to_var == (
        -(b_branch + b_to) * v_sq_to_var
        - (-b_branch * tr + g_branch * ti) / tap**2 * wc_var
        - (-g_branch * tr - b_branch * ti) / tap**2 * ws_var
    )


def current_flow_equation(
    i_var,
    v_sq_from_var,
    v_sq_to_var,
    wc_var,
    g_branch,
    b_branch,
):
    """
    Strengthening equation l_ij = (g^2 + b^2)(v_i^2 + v_j^2 - 2 w_c)
    Equation 23
    """
    return i_var == (g_branch**2 + b_branch**2) * (v_sq_from_var + v_sq_to_var - 2 * wc_var)


def current_soc_relax(
    p_var,
    q_var,
    v_sq_var,
    i_var, #QC current-squared-like variable l_ij power line
    v_sq_ub,
    on_off=1,
):
    """
    Switched current-magnitude relaxation.
    Uses the stronger perspective-style bound z * l * v_u^2 >= p^2 + q^2, Equation 21
    """
    return on_off * i_var * v_sq_ub >= p_var**2 + q_var**2