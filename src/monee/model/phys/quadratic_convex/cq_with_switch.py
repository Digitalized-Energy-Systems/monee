import math


def calc_branch_t(tap, shift):
    """
    Real and imaginary parts of the transformer tap ratio.
    shift must be in radians.
    Returns (t_R, t_I)
    """
    return (tap * math.cos(shift), tap * math.sin(shift))


def _is_fixed_one(value):
    """
    True only for an actual numeric constant equal to 1 (e.g. line permanently on).
    """
    return isinstance(value, (int, float, bool)) and float(value) == 1.0


def _require_sn_mva(sn_mva):
    """
    Check for sn_mva with value error raising unless given and >0
    """
    if sn_mva is None or sn_mva <= 0:
        raise ValueError(
            "sn_mva must be provided and > 0 because the paper uses p.u. "
            "power while Monee provides MW/MVAr variables."
        )


def _power_pu(power_mw_or_mvar, sn_mva):
    """
    concert MW/ MVAr to p.u.
    Returns power in p.u.

    """
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
    Return theta^M (largest possible angle difference across open line, its two bus angles are then unconstrained).
    For a permanently-on line every (1 - z)*theta^M term is zero, so delta_max is used as a placeholder;
    a switchable line must provide delta_big_m. Also validates the bounds.
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
    Convex hull of square function, v_sq = v^2 on [v_min, v_max], set <v^2>^R.
    v_sq_var is lifted variable v-tilde that replaces v^2 in power-flow equations.
    It is bounded, not set equal to v^2:
    v_sq <= (v_max + v_min)*v - v_max*v_min
    tighter voltage bounds give a tighter relaxation.
    """
    if not (0 < v_min <= v_max):
        raise ValueError("Require 0 < v_min <= v_max.")

    return [
        v_sq_var >= v_var**2,
        v_sq_var <= (v_max + v_min) * v_var - v_max * v_min,
    ]


def cosine_relax(
    cs_var, delta_var, delta_max, on_off=1, delta_big_m=None):
    """
    Quadratic cos(delta) relaxation on [-delta_max, delta_max].
    cos_var is bounded, not set equal to cos(delta):
    cs <= 1 - alpha*delta^2   alpha = (1 - cos(delta_max)) / delta_max^2
    cs >= cos(delta_max)
    With on_off = 1: continuous QC cosine relaxation.
    With on_off = z: disjunctive cosine relaxation
    """
    delta_big_m = _resolve_delta_big_m(on_off, delta_max, delta_big_m)
    alpha = (1.0 - math.cos(delta_max)) / (delta_max**2)

    return [
        cs_var <= on_off- alpha * delta_var**2+ (1 - on_off) * alpha * delta_big_m**2,
        cs_var >= on_off * math.cos(delta_max),
        cs_var <= on_off,

        delta_var>= -on_off * delta_max - (1 - on_off) * delta_big_m,
        delta_var<= on_off * delta_max + (1 - on_off) * delta_big_m,
    ]


def sine_relax( s_var,delta_var, delta_max, on_off=1, delta_big_m=None):
    """
    Polyhedral relaxation of sin(delta) on [-delta_max, delta_max].
    s_var is lifted variable for sin(theta_i-theta_j).
    s_var is bounded by the tangent lines to sin at +-delta_max/2 and the box |s| <= sin(delta_max)
    over-estiamtion by polytope slop compared to exact convex enelope (irrational coefficients)
    With on_off = z: s = 0 when z = 0, and big-M term allows any |delta| <= delta_big_m for an open line.
    """
    delta_big_m = _resolve_delta_big_m(on_off, delta_max, delta_big_m)

    cos_half = math.cos(delta_max / 2.0)
    sin_half = math.sin(delta_max / 2.0)
    tangent_rhs = sin_half - cos_half * delta_max / 2.0 #tangent on right hand side (rhs)
    s_abs_bound = sin_half + cos_half * delta_max / 2.0
    abs_delta_rhs = ( on_off * (sin_half - cos_half * delta_max / 2.0+ math.sin(delta_max) ) + (1 - on_off) * cos_half * delta_big_m)

    return [
        s_var - cos_half * delta_var <= on_off * tangent_rhs+ (1 - on_off) * cos_half * delta_big_m,
        -s_var + cos_half * delta_var <= on_off * tangent_rhs+ (1 - on_off) * cos_half * delta_big_m,
        s_var <= on_off * s_abs_bound,
        -s_var <= on_off * s_abs_bound,
        cos_half * delta_var <= abs_delta_rhs,
        -cos_half * delta_var <= abs_delta_rhs,
        s_var >= on_off * math.sin(-delta_max),
        s_var <= on_off * math.sin(delta_max),
        delta_var>= -on_off * delta_max - (1 - on_off) * delta_big_m,
        delta_var <= on_off * delta_max + (1 - on_off) * delta_big_m,
    ]


def mccormick_on_off_relax( product_var, x_var, y_var,on_off, x_lb, x_ub, y_lb,y_ub, product_lb,product_ub):
    """
    On/off McCormick envelope of w = x*y, the convex hull of
    off state (w = x = y = 0 at z = 0) and the McCormick set at z = 1.

    The constant terms of the four McCormick inequalities and all bounds are
    scaled by z, so the off state stays feasible although lb > 0. Linear in (w, x, y, z).

    x_var, y_var and product_var must be zero when z = 0: pass switched copies, not bus
    variables, which stay nonzero when a line is open.
    product_lb/ub must be valid bounds of x*y.
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


def perspective_voltage_relax( v_sq_p_var, v_sq_var, v_min, v_max,on_off):
    """
    perspective squared-voltage variable for switched flows: v_sq - (1-z)*v_max^2  <=  v_sq_p  <=  v_sq - (1-z)*v_min^2.
    v_sq_p is v_sq when z=1, together with z*v_min^2 <= v_sq_p <= z*v_max^2, the inequalities give the convex hull.
    """
    if not (0 < v_min <= v_max):
        raise ValueError("Require 0 < v_min <= v_max.")

    return [
        v_sq_p_var>= v_sq_var - (1 - on_off) * v_max**2,
        v_sq_p_var <= v_sq_var - (1 - on_off) * v_min**2,
    ]


def int_flow_from_p( p_from_var, v_sq_from_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, g_from=0, on_off=1, v_sq_p_var=None):
    """
    QC active-power flow, from side, linear in lifted variables
    Enforces that switchable line must pass v_sq_p_var.
    p_from_var in MVAr ( p_from_var / sn_mva is used on the left-hand side)
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_from_var if v_sq_p_var is None else v_sq_p_var
    p_from_pu = _power_pu(p_from_var, sn_mva)
    return p_from_pu == (
        (g_branch + g_from) / tap**2 * v_used
        + (-g_branch * tr + b_branch * ti) / tap**2 * wc_var
        + (-b_branch * tr - g_branch * ti) / tap**2 * ws_var
    )


def int_flow_from_q( q_from_var, v_sq_from_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, b_from=0, on_off=1,v_sq_p_var=None):
    """
    QC reactive-power flow, from side, linear in lifted variables.
    v_sq_p_var replaces v_sq_from_var and is required for switchable lines.
    q_from_var in MVAr ( q_from_var / sn_mva is used on the left-hand side)
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_from_var if v_sq_p_var is None else v_sq_p_var

    q_from_pu = _power_pu(q_from_var, sn_mva)

    return q_from_pu == (
        -(b_branch + b_from) / tap**2 * v_used
        - (-b_branch * tr - g_branch * ti) / tap**2 * wc_var
        + (-g_branch * tr + b_branch * ti) / tap**2 * ws_var
    )


def int_flow_to_p( p_to_var, v_sq_to_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, g_to=0, on_off=1, v_sq_p_var=None):
    """
    QC active-power flow, to side.
    tap² sits at from-side voltage term, hence no tap in to side
    ws and wc are defined with angle difference θi − θj (from bus minus to bus), and to-side equation uses θj − θi, leading to wc term keeps its sign and ws term flips it
    p_to_var in MVAr ( p_to_var / sn_mva is used on the left-hand side)

    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_to_var if v_sq_p_var is None else v_sq_p_var

    p_to_pu = _power_pu(p_to_var, sn_mva)

    return p_to_pu == (
        (g_branch + g_to) * v_used
        + (-g_branch * tr - b_branch * ti) / tap**2 * wc_var
        - (-b_branch * tr + g_branch * ti) / tap**2 * ws_var
    )


def int_flow_to_q( q_to_var, v_sq_to_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, b_to=0, on_off=1, v_sq_p_var=None):
    """
    QC reactive-power flow, into the line at the to bus,
    q_to_var in MVAr ( p_to_var / sn_mva is used on the left-hand side)
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var."
        )

    tr, ti = calc_branch_t(tap, shift)
    v_used = v_sq_to_var if v_sq_p_var is None else v_sq_p_var

    q_to_pu = _power_pu(q_to_var, sn_mva)

    return q_to_pu == (
        -(b_branch + b_to) * v_used
        - (-b_branch * tr + g_branch * ti) / tap**2 * wc_var
        - (-g_branch * tr - b_branch * ti) / tap**2 * ws_var
    )


def current_flow_equation(i_var, v_sq_from_var, v_sq_to_var, wc_var, g_branch, b_branch):
    """
    simplified-network current relation, for a plain line (no tap, shift or shunt)
    """
    return i_var == ((g_branch**2 + b_branch**2)* (v_sq_from_var + v_sq_to_var - 2 * wc_var))


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
    transformer / line-charging current relation.

    l (i_var): squared magnitude, in p.u., of the current leaving the tapped side of the branch (series plus charging current),
    q: from-end reactive power (MVAr, divided by sn_mva)
    b_charge : total line charging b^c, split equally between the ends.
    """
    tr, ti = calc_branch_t(tap, shift)
    q_from_pu = _power_pu(q_from_var, sn_mva)

    return i_var == (
        (g_branch**2 + b_branch**2)
        * ( v_sq_from_var / tap**2 + v_sq_to_var- 2.0 * (tr * wc_var + ti * ws_var) / tap**2)
        - b_charge * q_from_pu- (b_charge / (2.0 * tap))**2 * v_sq_from_var)


def current_soc_relax( p_var, q_var, v_sq_var, i_var, sn_mva, tap=1.0):
    """
    (v_sq / tap^2) * l >= p^2 + q^2        (p, q in p.u.)
    relaxation of nonconvex equ 1=(p² + q²)/v² to inequality equ.
    convex because v_sq >= 0 and l>= 0.
    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)

    return (
        (v_sq_var / tap**2) * i_var >= p_pu**2 + q_pu**2)

###The following functions can be used to strengthen the implementation by adding thermal limits  per line. Only relevant for switchable lines
def current_upper_bound(thermal_limit_sq_pu, v_min):
    """
    Upper bound on the current variable l
    l^u = t_ij / (v_i^l)^2
    v_min is lower bound of current
    Valid for a line without tap ratio
    thermal_limit_sq_pu is the paper's p.u. t_ij.
    """
    if thermal_limit_sq_pu < 0:
        raise ValueError("thermal_limit_sq_pu must be non-negative.")
    if v_min <= 0:
        raise ValueError("v_min must be > 0.")
    return thermal_limit_sq_pu / (v_min**2)


def thermal_limit_sq_pu_from_mva(max_s_mva, sn_mva):
    """
    Thermal limit t = (S_max / S_base)^2 in p.u.
    t is the squared limit, constraint stays quadratic
    """
    _require_sn_mva(sn_mva)
    if max_s_mva < 0:
        raise ValueError("max_s_mva must be non-negative.")
    return (max_s_mva / sn_mva)**2


def thermal_limit_on_off_relax( p_var, q_var, thermal_limit_sq_pu, on_off, sn_mva):
    """
    On/off thermal limit,
    p_pu^2 + q_pu^2 <= z^2 * t_ij (p.u.)
    t is thermal_limit_sq_pu
    z is squared, for fractional z this is convex hull of on/off set and tighter than z*t.
    convex for z>= 0, forces p=q=0 for open line
    Apply to flow at each end of line.
    Monee gives P in MW and Q in MVAr.
    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)

    return (
        p_pu**2 + q_pu**2
        <= on_off**2 * thermal_limit_sq_pu
    )


def current_switch_relax( p_var, q_var, v_sq_var, i_var, v_sq_ub, i_ub, on_off, sn_mva):
    """
    On/off strengthening of the current cone p.u.
    (z = on_off, l = i_var, l^u = i_ub, v_sq_ub = (v^u)^2):

        z * l * v_sq_ub >= p^2 + q^2    , convex hull of the on/off set
        0 <= l <= z * l^u
        z * v_sq * l^u  >= p^2 + q^2    , a further valid inequality
    cones are rotated second-order cones
    four constraints are returned in order above, part of strong QC-OTS (currently not implemented, it needs a thermal limit t_ig per branch)
    p_var, q_var in MW/ MVAr and divided by sn_mva
    v_sq_var is squared voltage of from bus

    Monee gives P in MW and Q in MVAr.
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
    Cost of one generator p = c2*p_g^2 + c1*p_g + c0
    Goal: minimization of generated power
    p_g: generated power as positive number (in unit coefficients refer to)
    p_mw is negative for generation
    c2 >= 0 required for objective to stay convex
    Not part of QC formulation itself, which minimizes branch losses
    """
    if isinstance(c2, (int, float)) and c2 < 0:
        raise ValueError(
            "Paper Eq. (11) requires c2 >= 0 for the OPF objective to "
            f"remain convex; got c2={c2!r}."
        )

    return c2 * p_g**2 + c1 * p_g + c0