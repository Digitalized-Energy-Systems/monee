"""Physics of the quadratic convex (QC) relaxation of the AC power flow.

Implements the relaxations of Hijazi, Coffrin and Van Hentenryck, "Convex
quadratic relaxations for mixed-integer nonlinear programs in power systems",
Math. Prog. Comp. 9 (2017) 321-367. Equation numbers refer to this paper.

Notation:
z = on_off,
delta_max = theta^u,
delta_big_m = theta^M,
v_sq = v-tilde (lifted v^2),
l = i_var (squared current),
wc and ws = lifted v_i * v_j * cos / sin(theta_i - theta_j).
Powers are passed in MW / MVAr (by monee) and converted to p.u. with sn_mva.
"""
import math


def calc_branch_t(tap, shift):
    """
    Return the rectangular tap ratio (t_R, t_I), Appendix C.
    tap is the tap ratio, shift the phase shift in radians.
    """
    return (tap * math.cos(shift), tap * math.sin(shift))


def _is_fixed_one(value):
    """
    Return True if value is a plain number equal to 1 (line permanently on).
    """
    return isinstance(value, (int, float, bool)) and float(value) == 1.0


def _require_sn_mva(sn_mva):
    """
    Raise ValueError unless the base power sn_mva (MVA) is given and > 0.
    """
    if sn_mva is None or sn_mva <= 0:
        raise ValueError(
            "sn_mva must be provided and > 0 because the paper uses p.u. "
            "power while Monee provides MW/MVAr variables."
        )


def _power_pu(power_mw_or_mvar, sn_mva):
    """
    Convert power in MW or MVAr to p.u. by dividing by sn_mva.
    """
    _require_sn_mva(sn_mva)
    return power_mw_or_mvar / sn_mva


def _validate_angle_bounds(delta_max, delta_big_m=None):
    """
    Check 0 < delta_max <= pi/2 (validity range of the trigonometric relaxations) and delta_big_m >= delta_max.
    """
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
    Return theta^M, the largest angle difference across an open line (Paper Sect. 5.2).
    For a permanently-on line all (1 - z) * theta^M terms vanish, so delta_max is used as placeholder. A switchable line must provide delta_big_m.
    """
    if delta_big_m is None:
        if _is_fixed_one(on_off):
            delta_big_m = delta_max
        else:
            raise ValueError("delta_big_m must be provided for switched QC constraints.")
    _validate_angle_bounds(delta_max, delta_big_m)
    return delta_big_m


def square_relax(v_sq_var, v_var, v_min, v_max):
    """
    Convex envelope of v^2 on [v_min, v_max], Eq. (16).
        v_sq >= v^2
        v_sq <= (v_max + v_min) * v - v_max * v_min
    v_sq_var is lifted variable v-tilde that replaces v^2 in power-flow equations (bounded, not set equal to v^2)
    It equals v^2 only at v_min and v_max, so tighter voltage bounds give a tighter relaxation.
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
    Quadratic relaxation of cos(delta) on [-delta_max, delta_max].
    For on_off = 1 the continuous set of Eq. (14):
        cs <= 1 - alpha * delta^2,  alpha = (1 - cos(delta_max)) / delta_max^2
        cs >= cos(delta_max)
    For a switchable line (on_off = z) the on/off form of Eq. (28):
        cs = 0 for z = 0, and a big-M term allows any |delta| <= delta_big_m.
    This is a relaxation of the convex hull, not necessarily the hull itself (Prop. 6).
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
    For on_off = 1 the continuous set of Eq. (15):
        s lies between the tangent lines to sin at +-delta_max/2 and within |s| <= sin(delta_max). The
        polytope slightly overestimates the exact convex envelope, whose coefficients are irrational.
    For a switchable line (on_off = z) the on/off form of Eq. (27):
        s = 0 for z = 0, and a big-M term allows any |delta| <= delta_big_m.
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
    On/off McCormick envelope of w = x * y, Eq. (35).
    Convex hull of the off state (w = x = y = 0 at z = 0) and the McCormick set at z = 1.
    The constant terms and all bounds are scaled by z, so the off state is feasible although lb > 0. Linear in (w, x, y, z).

    x_var, y_var and product_var must be zero when z = 0: pass switched copies, not bus variables, which stay nonzero when a line is open.
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
        x_var <= z * x_ub, # Eq. (35) prints  x * x_ub in the paper, interpreted as typo
        z * y_lb <= y_var,
        y_var <= z * y_ub,
    ]


def perspective_voltage_relax( v_sq_p_var, v_sq_var, v_min, v_max,on_off):
    """
    Perspective copy v_sq_p of the squared voltage, Eq. (34).
        v_sq - (1 - z) * v_max^2 <= v_sq_p <= v_sq - (1 - z) * v_min^2
    v_sq_p is v_sq when z=1. (Inequalities alone do not force v_sq_p = 0 for z = 0)
    Together with z*v_min^2 <= v_sq_p <= z*v_max^2, the inequalities give the convex hull.
    """
    if not (0 < v_min <= v_max):
        raise ValueError("Require 0 < v_min <= v_max.")

    return [
        v_sq_p_var>= v_sq_var - (1 - on_off) * v_max**2,
        v_sq_p_var <= v_sq_var - (1 - on_off) * v_min**2,
    ]


def int_flow_from_p( p_from_var, v_sq_from_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, g_from=0, on_off=1, v_sq_p_var=None):
    """
    Active power at the from bus, Eq. (12); with tap and shift Eq. (42).
    Linear in the lifted variables v_sq, wc and ws. p_from_var is in MW. For switchable lines v_sq_p_var (Eq. 34) is required and replaces
    v_sq_from_var, an open line then carries no power. g_from (end conductance) is not part of equation.
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
    Reactive power at the from bus, Eq. (13); with tap and shift Eq. (44).
    b_from is half the line charging (b^c / 2). q_from_var is in MVAr.
    Otherwise as int_flow_from_p.
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


def int_flow_to_p( p_to_var, v_sq_to_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, g_to=0, on_off=1, v_sq_p_var=None,*, v_sq_from_var):
    """
    Active power at the to bus, Equ(12); with tap and shift Equ (43).
    p_to_var in MW,
    tap² sits at from-side voltage term, hence no tap in to side
    ws and wc are defined with angle difference θi − θj (from bus minus to bus), and to-side equation uses θj − θi, so the
    sin(theta_to - theta_from) term appears as -ws.
    Otherwise as int_flow_from_p.
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var."
        )

    t_real, t_imag = calc_branch_t(tap, shift)
    v_used = v_sq_to_var if v_sq_p_var is None else v_sq_p_var

    p_to_pu = _power_pu(p_to_var, sn_mva)

    return p_to_pu == (
            #(g_branch + g_to) * v_sq_from_var  # Eq. (43) as printed: v_i^2
            (g_branch + g_to) * v_used  # corrected: v_j^2, Eq. (43) prints v_i^2, a typo
            + (-g_branch * t_real - b_branch * t_imag) / tap ** 2 * wc_var
            - (-b_branch * t_real + g_branch * t_imag) / tap ** 2 * ws_var
    )


def int_flow_to_q( q_to_var, v_sq_to_var, wc_var, ws_var, g_branch, b_branch, tap, shift, sn_mva, b_to=0, on_off=1, v_sq_p_var=None, *, v_sq_from_var):
    """
    Reactive power at the to bus, Equ (13); with tap and shift Equ (45).
    q_to_var in MVAr, b_to is half the line charging b^c/2
    Sign handling of ws as in int_flow_to_p
    """
    if not _is_fixed_one(on_off) and v_sq_p_var is None:
        raise ValueError(
            "Switched power flow requires v_sq_p_var."
        )

    t_real, t_imag = calc_branch_t(tap, shift)
    v_used = v_sq_to_var if v_sq_p_var is None else v_sq_p_var

    q_to_pu = _power_pu(q_to_var, sn_mva)

    return q_to_pu == (
            #-(b_branch + b_to) * v_sq_from_var  # Eq. (45) as printed: v_i^2
            -(b_branch + b_to) * v_used  # corrected: v_j^2, Eq. (45) prints v_i^2, a typo
            - (-b_branch * t_real + g_branch * t_imag) / tap ** 2 * wc_var
            - (-g_branch * t_real - b_branch * t_imag) / tap ** 2 * ws_var
    )


def current_flow_equation(i_var, v_sq_from_var, v_sq_to_var, wc_var, g_branch, b_branch):
    """
    Current relation for a plain line (no tap, shift or shunt) Equ(23)
    i_var is l, squared series-current magnitudes in p.u.
    For switchable lines pass perspective copies v_sq_p (Equ 34), so that l = 0 for open line.
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
    Current relation with tap ratio, shift, and line charging Equ (49)
    transformer / line-charging current relation.

    l (i_var): squared magnitude, in p.u., of the current leaving the tapped side of the branch (series plus charging current),
    q: from-end reactive power (MVAr, divided by sn_mva)
    b_charge : total line charging b^c, split equally between the ends.
    Only for permanently active branches, no on/off form in paper
    """
    tr, ti = calc_branch_t(tap, shift)
    q_from_pu = _power_pu(q_from_var, sn_mva)

    return i_var == (
        (g_branch**2 + b_branch**2)
        * ( v_sq_from_var / tap**2 + v_sq_to_var- 2.0 * (tr * wc_var + ti * ws_var) / tap**2)
        - b_charge * q_from_pu- (b_charge / (2.0 * tap))**2 * v_sq_from_var)


def current_soc_relax( p_var, q_var, v_sq_var, i_var, sn_mva, tap=1.0):
    """
    Current-magnitude cone, Equ (21); with tap ratio Equ (48)
    (v_sq / tap^2) * l >= p^2 + q^2        (p, q in p.u.)
    relaxation of nonconvex equ l=(p² + q²)/v² (Equ20) to inequality equ.
    convex because v_sq >= 0 and l>= 0. p_var is in MW, q_var in MVAr.

    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)

    return (
        (v_sq_var / tap**2) * i_var >= p_pu**2 + q_pu**2)

###The following functions can be used to strengthen the implementation by adding thermal limits  per line (Proposition 5, Equs 29-31). Only relevant for switchable lines
def current_upper_bound(thermal_limit_sq_pu, v_min):
    """
    Upper bound l^u = t / v_min^2 on the current variable l, Prop. 5.
    t in p.u. (thermal_limit_sq_pu_form_mva; thermal_limit_sq_pu is the paper's p.u. t_ij)
    v_min is lower bound of voltage magnitude bound of from bus, valid for a line without tap ratio
    """
    if thermal_limit_sq_pu < 0:
        raise ValueError("thermal_limit_sq_pu must be non-negative.")
    if v_min <= 0:
        raise ValueError("v_min must be > 0.")
    return thermal_limit_sq_pu / (v_min**2)


def thermal_limit_sq_pu_from_mva(max_s_mva, sn_mva):
    """
    Return thermal limit t = (S_max / S_base)^2 in p.u., right hand side of Equ (8)
    t is the squared limit, constraint stays quadratic
    max_s_mva in MVA
    """
    _require_sn_mva(sn_mva)
    if max_s_mva < 0:
        raise ValueError("max_s_mva must be non-negative.")
    return (max_s_mva / sn_mva)**2


def thermal_limit_on_off_relax( p_var, q_var, thermal_limit_sq_pu, on_off, sn_mva):
    """
    On/off thermal limit, Equ (29)
    p_pu^2 + q_pu^2 <= z^2 * t_ij (p.u.)
    t is thermal_limit_sq_pu
    z is squared, for fractional z this is convex hull of on/off set and tighter than z*t.
    convex for z>= 0, forces p=q=0 for open line
    Apply to flow at each end of line.
    p_var in MW, q_var in MVAr
    """
    p_pu = _power_pu(p_var, sn_mva)
    q_pu = _power_pu(q_var, sn_mva)

    return (
        p_pu**2 + q_pu**2
        <= on_off**2 * thermal_limit_sq_pu
    )


def current_switch_relax( p_var, q_var, v_sq_var, i_var, v_sq_ub, i_ub, on_off, sn_mva):
    """
    On/off strengthening of the current cone Equs (30-31) in p.u.
    (z = on_off, l = i_var, l^u = i_ub, v_sq_ub = (v^u)^2):

        z * l * v_sq_ub >= p^2 + q^2    , convex hull of the on/off set
        0 <= l <= z * l^u
        z * v_sq * l^u  >= p^2 + q^2    , a further valid inequality
    cones are rotated second-order cones
    four constraints are returned in order above, part of strong QC-OTS (currently not implemented, it needs a thermal limit t_ij per branch for i_ub = l^u)
    v_sq_ub is (v^u)^2 and v_sq_var is squared voltage of from bus
    p_var is in MW, q_var in MVAr.
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
    Cost of one generator p = c2*p_g^2 + c1*p_g + c0 Equ (11), defaults (c1=1) give equ (10)
    Goal: minimization of generated power
    p_g: generated power as positive number (in unit coefficients refer to)
    p_mw is negative for generation, pass -p_mw
    c2 >= 0 required for objective to stay convex
    Not part of QC formulation itself, which minimizes branch losses
    """
    if isinstance(c2, (int, float)) and c2 < 0:
        raise ValueError(
            "Paper Eq. (11) requires c2 >= 0 for the OPF objective to "
            f"remain convex; got c2={c2!r}."
        )

    return c2 * p_g**2 + c1 * p_g + c0