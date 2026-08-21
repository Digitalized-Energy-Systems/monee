import math


def power_balance_equation(signed_flows):
    return sum(signed_flows) == 0


def calc_branch_t(tap, shift):
    """
    Real and imaginary parts of the transformer tap ratio.

    shift must be in radians.
    """
    return (tap * math.cos(shift), tap * math.sin(shift))


def _is_fixed_one(value):
    """
    True only for an actual numeric constant equal to one.

    Do not evaluate symbolic Monee variables in Python boolean expressions.
    """
    return isinstance(value, (int, float, bool)) and float(value) == 1.0


def _require_sn_mva(sn_mva):
    if sn_mva is None or sn_mva <= 0:
        raise ValueError(
            "sn_mva must be provided and > 0 because the paper uses p.u. "
            "power while Monee provides MW/MVAr variables."
        )


def _power_pu(power_mw_or_mvar, sn_mva):
    _require_sn_mva(sn_mva)
    return power_mw_or_mvar / sn_mva


def _validate_angle_bounds(delta_max, delta_big_m=None):
    if delta_max <= 0:
        raise ValueError("delta_max must be > 0.")
    if delta_max > math.pi / 2:
        raise ValueError(
            "The QC trigonometric relaxations in the paper require "
            "delta_max <= pi/2."
        )
    if delta_big_m is not None and delta_big_m < delta_max:
        raise ValueError("delta_big_m must be >= delta_max.")


def _resolve_delta_big_m(on_off, delta_max, delta_big_m):
    """
    theta^M is required for a switchable line. For a permanently active line,
    theta^M drops out and delta_max can be used.
    """
    if delta_big_m is None:
        if _is_fixed_one(on_off):
            delta_big_m = delta_max
        else:
            raise ValueError(
                "delta_big_m must be provided for switched QC constraints."
            )

    _validate_angle_bounds(delta_max, delta_big_m)
    return delta_big_m


def square_relax(v_sq_var, v_var, v_min, v_max):
    """
    Convex hull of v_sq = v^2 on [v_min, v_max].
    Paper Section 4.1 / square-function envelope.
    """
    if not (0 < v_min <= v_max):
        raise ValueError("Require 0 < v_min <= v_max.")

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
    Quadratic cosine relaxation.

    With on_off = 1: continuous QC cosine relaxation.
    With on_off = z: switched cosine relaxation, paper Eq. (28).
    """
    delta_big_m = _resolve_delta_big_m(
        on_off, delta_max, delta_big_m
    )

    alpha = (1.0 - math.cos(delta_max)) / (delta_max**2)

    return [
        cs_var
        <= on_off
        - alpha * delta_var**2
        + (1 - on_off) * alpha * delta_big_m**2,

        cs_var >= on_off * math.cos(delta_max),
        cs_var <= on_off,

        delta_var
        >= -on_off * delta_max - (1 - on_off) * delta_big_m,
        delta_var
        <= on_off * delta_max + (1 - on_off) * delta_big_m,
    ]


def sine_relax(
    s_var,
    delta_var,
    delta_max,
    on_off=1,
    delta_big_m=None,
):
    """
    Full polyhedral switched sine relaxation, paper Eq. (27).

    Includes the |delta| strengthening inequality that was missing from the
    original implementation.
    """
    delta_big_m = _resolve_delta_big_m(
        on_off, delta_max, delta_big_m
    )

    c = math.cos(delta_max / 2.0)
    sin_half = math.sin(delta_max / 2.0)

    tangent_rhs = sin_half - c * delta_max / 2.0
    s_abs_bound = sin_half + c * delta_max / 2.0

    theta_abs_rhs = (
        on_off
        * (
            sin_half
            - c * delta_max / 2.0
            + math.sin(delta_max)
        )
        + (1 - on_off) * c * delta_big_m
    )

    return [
        # Tangent inequalities.
        s_var - c * delta_var
        <= on_off * tangent_rhs
        + (1 - on_off) * c * delta_big_m,

        -s_var + c * delta_var
        <= on_off * tangent_rhs
        + (1 - on_off) * c * delta_big_m,

        # |s| <= z * (...)
        s_var <= on_off * s_abs_bound,
        -s_var <= on_off * s_abs_bound,

        # c * |delta| <= ...
        c * delta_var <= theta_abs_rhs,
        -c * delta_var <= theta_abs_rhs,

        # z sin(-theta_u) <= s <= z sin(theta_u)
        s_var >= on_off * math.sin(-delta_max),
        s_var <= on_off * math.sin(delta_max),

        # Switched phase-angle bounds.
        delta_var
        >= -on_off * delta_max - (1 - on_off) * delta_big_m,
        delta_var
        <= on_off * delta_max + (1 - on_off) * delta_big_m,
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
    Ordinary McCormick envelope for product_var = x_var * y_var.
    This is the continuous QC envelope used in Section 4.
    """
    if x_lb > x_ub or y_lb > y_ub:
        raise ValueError("Invalid McCormick bounds.")

    return [
        product_var >= x_lb * y_var + y_lb * x_var - x_lb * y_lb,
        product_var >= x_ub * y_var + y_ub * x_var - x_ub * y_ub,
        product_var <= x_lb * y_var + y_ub * x_var - x_lb * y_ub,
        product_var <= x_ub * y_var + y_lb * x_var - x_ub * y_lb,
    ]


def mccormick_on_off_relax(
    product_var,
    x_var,
    y_var,
    on_off,
    x_lb,
    x_ub,
    y_lb,
    y_ub,
    product_lb,
    product_ub,
):
    """
    On/off McCormick formulation from paper Eq. (35).

    IMPORTANT: Eq. (35) assumes x, y, and product_var are zero in the off
    state. Therefore x_var and y_var must be the appropriate switched /
    perspective variables, not ordinary shared bus variables that remain
    nonzero when the line is open.
    """
    if x_lb > x_ub or y_lb > y_ub:
        raise ValueError("Invalid McCormick bounds.")
    if product_lb > product_ub:
        raise ValueError("Invalid product bounds.")

    z = on_off

    return [
        product_var >= x_lb * y_var + y_lb * x_var - z * x_lb * y_lb,
        product_var >= x_ub * y_var + y_ub * x_var - z * x_ub * y_ub,
        product_var <= x_lb * y_var + y_ub * x_var - z * x_lb * y_ub,
        product_var <= x_ub * y_var + y_lb * x_var - z * x_ub * y_lb,

        z * product_lb <= product_var,
        product_var <= z * product_ub,

        z * x_lb <= x_var,
        x_var <= z * x_ub,

        z * y_lb <= y_var,
        y_var <= z * y_ub,
    ]


def perspective_voltage_relax(
    v_sq_p_var,
    v_sq_var,
    v_min,
    v_max,
    on_off,
):
    """
    Paper Eq. (34), perspective squared-voltage variable for switched flows.
    """
    if not (0 < v_min <= v_max):
        raise ValueError("Require 0 < v_min <= v_max.")

    return [
        v_sq_p_var
        >= v_sq_var - (1 - on_off) * v_max**2,

        v_sq_p_var
        <= v_sq_var - (1 - on_off) * v_min**2,
    ]


def int_flow_from_p(
    p_from_var,
    v_sq_from_var,
    wc_var,
    ws_var,
    g_branch,
    b_branch,
    tap,
    shift,
    sn_mva,
    g_from=0,
    on_off=1,
    v_sq_p_var=None,
):
    """
    QC active-power flow, from side.

    Paper equations use p.u.; Monee gives p_from_var in MW.
    Hence p_from_var / sn_mva is used on the left-hand side.

    For a switched branch, v_sq_p_var should be the perspective squared-voltage
    variable from paper Eq. (34).
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var from Eq. (34)."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_from_var if v_sq_p_var is None else v_sq_p_var

    p_from_pu = _power_pu(p_from_var, sn_mva)

    return p_from_pu == (
        (g_branch + g_from) / tap**2 * v_used
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
    sn_mva,
    b_from=0,
    on_off=1,
    v_sq_p_var=None,
):
    """
    QC reactive-power flow, from side.

    Paper equations use p.u.; Monee gives q_from_var in MVAr.
    Hence q_from_var / sn_mva is used on the left-hand side.
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var from Eq. (34)."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_from_var if v_sq_p_var is None else v_sq_p_var

    q_from_pu = _power_pu(q_from_var, sn_mva)

    return q_from_pu == (
        -(b_branch + b_from) / tap**2 * v_used
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
    sn_mva,
    g_to=0,
    on_off=1,
    v_sq_p_var=None,
):
    """
    QC active-power flow, to side.

    Paper equations use p.u.; Monee gives p_to_var in MW.
    Hence p_to_var / sn_mva is used on the left-hand side.
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var from Eq. (34)."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_to_var if v_sq_p_var is None else v_sq_p_var

    p_to_pu = _power_pu(p_to_var, sn_mva)

    return p_to_pu == (
        (g_branch + g_to) * v_used
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
    sn_mva,
    b_to=0,
    on_off=1,
    v_sq_p_var=None,
):
    """
    QC reactive-power flow, to side.

    Paper equations use p.u.; Monee gives q_to_var in MVAr.
    Hence q_to_var / sn_mva is used on the left-hand side.
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var from Eq. (34)."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_to_var if v_sq_p_var is None else v_sq_p_var

    q_to_pu = _power_pu(q_to_var, sn_mva)

    return q_to_pu == (
        -(b_branch + b_to) * v_used
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
    Paper Eq. (23), simplified-network current relation.

        l_ij = (g^2+b^2)(v_i + v_j - 2 wc_ij)

    i_var represents l_ij, i.e. a squared-current-like p.u. auxiliary variable.
    """
    return i_var == (
        (g_branch**2 + b_branch**2)
        * (v_sq_from_var + v_sq_to_var - 2 * wc_var)
    )


def current_flow_equation_extended(
    i_var,
    v_sq_from_var,
    v_sq_to_var,
    wc_var,
    ws_var,
    q_from_var,
    g_branch,
    b_branch,
    tap,
    shift,
    b_charge,
    sn_mva,
):
    """
    Paper Eq. (49), transformer / line-charging current relation.

    Paper equations use p.u.; Monee gives q_from_var in MVAr, so q_from_var is
    divided by sn_mva before it appears in Eq. (49).
    """
    tr, ti = calc_branch_t(tap, shift)
    q_from_pu = _power_pu(q_from_var, sn_mva)

    return i_var == (
        (g_branch**2 + b_branch**2)
        * (
            v_sq_from_var / tap**2
            + v_sq_to_var
            - 2.0 * (tr * wc_var + ti * ws_var) / tap**2
        )
        - b_charge * q_from_pu
        - (b_charge / (2.0 * tap))**2 * v_sq_from_var
    )


def current_soc_relax(
    p_var,
    q_var,
    v_sq_var,
    i_var,
    sn_mva,
    tap=1.0,
):
    """
    Paper Eq. (21), or Eq. (48) when tap != 1.

        (v_sq / tap^2) * l >= p_pu^2 + q_pu^2

    Paper equations use p.u.; Monee gives P in MW and Q in MVAr.
    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)

    return (
        (v_sq_var / tap**2) * i_var
        >= p_pu**2 + q_pu**2
    )


def current_upper_bound(thermal_limit_sq_pu, v_min):
    """
    Proposition 5:
        l^u = t_ij / (v_i^l)^2

    thermal_limit_sq_pu is the paper's p.u. t_ij.
    """
    if thermal_limit_sq_pu < 0:
        raise ValueError("thermal_limit_sq_pu must be nonnegative.")
    if v_min <= 0:
        raise ValueError("v_min must be > 0.")

    return thermal_limit_sq_pu / (v_min**2)


def thermal_limit_sq_pu_from_mva(max_s_mva, sn_mva):
    """
    Convert a Monee apparent-power magnitude limit to the paper's p.u. t_ij:

        t_ij = (S_max_MVA / S_base_MVA)^2
    """
    _require_sn_mva(sn_mva)
    if max_s_mva < 0:
        raise ValueError("max_s_mva must be nonnegative.")

    return (max_s_mva / sn_mva)**2


def thermal_limit_on_off_relax(
    p_var,
    q_var,
    thermal_limit_sq_pu,
    on_off,
    sn_mva,
):
    """
    Paper Eq. (29):

        p_pu^2 + q_pu^2 <= z^2 * t_ij

    Paper equations use p.u.; Monee gives P in MW and Q in MVAr.
    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)

    return (
        p_pu**2 + q_pu**2
        <= on_off**2 * thermal_limit_sq_pu
    )


def current_switch_relax(
    p_var,
    q_var,
    v_sq_var,
    i_var,
    v_sq_ub,
    i_ub,
    on_off,
    sn_mva,
):
    """
    Paper Eqs. (30)-(31), simplified switched-current strengthening.

        z*l*(v_u)^2 >= p_pu^2 + q_pu^2
        0 <= l <= z*l^u
        z*v_sq*l^u >= p_pu^2 + q_pu^2

    Paper equations use p.u.; Monee gives P in MW and Q in MVAr.

    NOTE: this strengthening is *not* part of the paper's actual QC-OTS
    formulation (Appendix A.2.1 lists (3)-(7), (16), (21), (23), (32)-(34) as
    the constraint set) -- it is presented as an additional valid, optional
    tightening. It is provided here for callers who want the extra strength
    on switching-heavy problems; it is intentionally not wired into every
    switchable branch by default. Using it requires a thermal limit t_ij per
    branch (thermal_limit_sq_pu_from_mva) and an upper bound on i_var
    (current_upper_bound).
    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)
    s_sq_pu = p_pu**2 + q_pu**2

    return [
        on_off * i_var * v_sq_ub >= s_sq_pu,
        i_var >= 0,
        i_var <= on_off * i_ub,
        on_off * v_sq_var * i_ub >= s_sq_pu,
    ]


def generation_cost(p_g, c0=0.0, c1=1.0, c2=0.0):
    """
    Paper Eq. (11), generator fuel-cost objective term for a single generator:

        c2*(p_g)^2 + c1*p_g + c0

    Paper Eq. (10) (pure active-power / loss minimization) is the special
    case c2=0, c1=1, c0=0 -- the paper's own default when no cost curve is
    given (Section 2.3).

    p_g is the *generated* active-power magnitude (positive for generation),
    matching the paper's p^g_i convention -- NOT Monee's internal signed
    p_mw (which is negative for generation); callers pass -model.p_mw.

    Convexity requirement: the paper's relaxations (and this whole QC model)
    are only meaningful as a convex program if the objective stays convex
    too, which requires c2 >= 0 -- true for any physical fuel-cost curve.
    A negative c2 is rejected here rather than silently producing a
    nonconvex objective.
    """
    if isinstance(c2, (int, float)) and c2 < 0:
        raise ValueError(
            "Paper Eq. (11) requires c2 >= 0 for the OPF objective to "
            f"remain convex; got c2={c2!r}."
        )

    return c2 * p_g**2 + c1 * p_g + c0