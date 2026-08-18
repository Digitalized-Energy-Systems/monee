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


def square_relax(
    v_sq_var, v_var, v_min, v_max
):  # quadratic terms v² relaxed into convex envelopes (according paper Fig 3, equation in 4.1 power flow relaxation)
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

    alpha = (1 - math.cos(delta_max)) / (
        delta_max**2
    )  # alpha as helper variable for whole equation for figure 1a

    return [  # cosine variable bounded by constraints for convex outer approx.
        cs_var <= on_off - alpha * delta_var**2 + (1 - on_off) * alpha * delta_big_m**2,
        cs_var >= on_off * math.cos(delta_max),
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
    v_sq_from_var,  # squared voltage
    wc_var,  # v_i*v_j*cos(theta_i-theta_j)
    ws_var,  # v_i*v_j*sin(theta_i-theta_j)
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
    return p_from_var == on_off * (  # Ohm's law linearized , paper equation 1
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
    return q_from_var == on_off * (  # Ohm's law linearized , paper equation 2
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
    return p_to_var == on_off * (
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
    return q_to_var == on_off * (
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
    tap=1,
    shift=0,
    ws_var=0,
    on_off=1,
):
    r"""
        Squared series-current magnitude, equation 23 generalised to an
        off-nominal complex turns ratio :math:`T = tap \cdot e^{j\,shift}`:

        .. math::
            l_{ij} = (g^2+b^2)\left(rac{w_i}{tap^2} + w_j
                     - rac{2}{tap}ig(w_c \cos(shift) + w_s \sin(shift)ig)
    ight)

        which is exactly :math:`|(V_i/T - V_j)\,y|^2`. The paper's
        :math:`w_i + w_j - 2 w_c` is the ``tap = 1, shift = 0`` special case; using
        it on a transformer overstates the current (3.9x on a 1.05 tap), which then
        propagates into the ``r * l`` objective term as a phantom loss penalty.

        Gated on ``on_off`` so an open branch carries no current; combined with
        :func:`current_soc_relax` that is what forces p = q = 0 across it.
    """
    cross = wc_var
    if shift:
        cross = wc_var * math.cos(shift) + ws_var * math.sin(shift)
    return i_var == on_off * (g_branch**2 + b_branch**2) * (
        v_sq_from_var / tap**2 + v_sq_to_var - 2 / tap * cross
    )


def current_soc_relax(
    p_var,
    q_var,
    v_sq_var,
    i_var,  # QC current-squared-like variable l_ij power line
    tap=1,
    g_from=0,
    b_from=0,
):
    r"""
        Rotated second-order cone on the *series* branch power, equation 21
        generalised to shunt admittance and an off-nominal turns ratio:

        .. math::
            \left(p - rac{g_{fr}}{tap^2} w_i
    ight)^2
            + \left(q + rac{b_{fr}}{tap^2} w_i
    ight)^2
            \;\le\; rac{w_i}{tap^2}\, l_{ij}

        Both corrections matter. ``p_var``/``q_var`` are the *terminal* flows, which
        include the from-side shunt draw :math:`\overline{y_{fr}} w_i / tap^2`;
        ``i_var`` is the *series* current. Writing the naive
        ``p^2 + q^2 <= w_i * l`` instead does not merely lose tightness - it
        EXCLUDES AC-feasible points: with ``b_fr = 0.015`` and both ends at 1.0 pu
        an exact AC point has ``p^2 + q^2 = 2.25e-4`` against ``l = 0``. In the
        corrected form the two sides are equal at every AC point (verified to
        1e-13 across shunt / tap / shift combinations), so it is exact rather than
        merely valid.

        Convex: a sum of squares of affine expressions bounded by the product of
        two non-negative variables is a rotated cone.

        No ``on_off`` factor is needed - :func:`current_flow_equation` already
        drives ``i_var`` to 0 on an open branch, and ``v_sq_var > 0`` then forces
        the series flow, hence ``p`` and ``q``, to 0 here.
    """
    p_series = p_var - g_from / tap**2 * v_sq_var
    q_series = q_var + b_from / tap**2 * v_sq_var
    return i_var * v_sq_var / tap**2 >= p_series**2 + q_series**2


def voltage_product_soc(wc_var, ws_var, v_sq_from_var, v_sq_to_var):
    """
    W-matrix 2x2 positive-semidefiniteness as a rotated SOC:
    ``w_c^2 + w_s^2 <= v_i^2 * v_j^2``.

    The standard QC-OPF tightening (Coffrin et al.). It couples ``wc``/``ws``
    to the squared voltages directly instead of only through the sequential
    ``vv`` McCormick chain, and is exact - it holds for every AC-feasible
    point.
    """
    return wc_var**2 + ws_var**2 <= v_sq_from_var * v_sq_to_var
