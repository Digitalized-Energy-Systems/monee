import math


def power_balance_equation(signed_flows):
    """Kirchhoff active/reactive power balance at one bus."""
    return sum(signed_flows) == 0
    # Paper Eq. (3) for active power or Eq. (4) for reactive power,
    # depending on the supplied signed-flow terms.


def validate_voltage_bounds(v_min, v_max, name="voltage"):
    """Validate the finite positive voltage interval needed by QC envelopes."""
    if v_min is None or v_max is None:
        raise ValueError(
            f"QC requires finite lower and upper bounds for {name}."
        )
    if not 0.0 < v_min <= v_max:
        raise ValueError(
            f"QC requires 0 < {name}_min <= {name}_max; "
            f"got [{v_min}, {v_max}]."
        )


def validate_angle_limit(theta_max):
    """Validate the symmetric angle interval assumed in the paper."""
    if theta_max is None:
        raise ValueError("QC requires a finite branch angle-difference limit.")
    if not 0.0 < theta_max <= math.pi / 2.0:
        raise ValueError(
            "The paper's QC trigonometric relaxations require "
            f"0 < theta_max <= pi/2; got {theta_max}."
        )

def voltage_square_bounds(v_min, v_max):
    """
    Return bounds for the lifted variable representing v**2.
    """
    validate_voltage_bounds(v_min, v_max)

    return v_min**2, v_max**2


def voltage_product_bounds(
    v_from_min,
    v_from_max,
    v_to_min,
    v_to_max,
):
    """
    Return bounds for vv = v_from * v_to.
    """
    validate_voltage_bounds(
        v_from_min,
        v_from_max,
        name="from_voltage",
    )
    validate_voltage_bounds(
        v_to_min,
        v_to_max,
        name="to_voltage",
    )

    return (
        v_from_min * v_to_min,
        v_from_max * v_to_max,
    )


def trigonometric_bounds(theta_max):
    """
    Return valid bounds for cos(theta) and sin(theta) over the symmetric
    interval [-theta_max, theta_max].
    """
    validate_angle_limit(theta_max)

    cos_min = math.cos(theta_max)
    cos_max = 1.0

    sin_min = -math.sin(theta_max)
    sin_max = math.sin(theta_max)

    return cos_min, cos_max, sin_min, sin_max
def node_variable_bounds(v_min, v_max):
    """Return bounds for the lifted squared-voltage variable."""
    validate_voltage_bounds(v_min, v_max)
    return v_min**2, v_max**2
    # Paper Eq. (16): valid bounds for v_i_tilde ~= v_i^2.


def branch_variable_bounds(
    vm_from_min,
    vm_from_max,
    vm_to_min,
    vm_to_max,
    theta_max,
):
    """Derive all lifted-variable bounds required by paper Eqs. (14)--(19).

    The physical voltage and angle limits are inputs. This helper derives the
    bounds of the QC auxiliary variables; it does not invent operational limits.
    """
    validate_voltage_bounds(vm_from_min, vm_from_max, "from_voltage")
    validate_voltage_bounds(vm_to_min, vm_to_max, "to_voltage")
    validate_angle_limit(theta_max)

    vv_min = vm_from_min * vm_to_min
    vv_max = vm_from_max * vm_to_max

    cos_min = math.cos(theta_max)
    cos_max = 1.0
    sin_min = -math.sin(theta_max)
    sin_max = math.sin(theta_max)

    # Since voltage magnitudes and cos(theta) are nonnegative on
    # [-theta_max, theta_max] with theta_max <= pi/2, these bounds are valid.
    wc_min = vv_min * cos_min
    wc_max = vv_max

    # vv is positive and sine has symmetric bounds, so the largest absolute
    # product uses vv_max.
    ws_min = vv_max * sin_min
    ws_max = vv_max * sin_max

    return {
        "angle_min": -theta_max,
        "angle_max": theta_max,
        "cos_min": cos_min,
        "cos_max": cos_max,
        "sin_min": sin_min,
        "sin_max": sin_max,
        "vv_min": vv_min,
        "vv_max": vv_max,
        "wc_min": wc_min,
        "wc_max": wc_max,
        "ws_min": ws_min,
        "ws_max": ws_max,
    }


def _validate_interval(lower, upper, name):
    if lower is None or upper is None or lower > upper:
        raise ValueError(f"Invalid {name} bounds [{lower}, {upper}].")


def _require_always_on(on_off):
    """Reject switching variables that require paper Eqs. (27)--(35)."""
    if not isinstance(on_off, (int, float)):
        raise NotImplementedError(
            "This file implements the continuous always-on QC formulation. "
            "A symbolic/binary on_off requires paper Eqs. (27)--(35)."
        )
    if not math.isclose(float(on_off), 1.0):
        raise NotImplementedError(
            "This file implements the continuous always-on QC formulation. "
            "A non-unit on_off requires paper Eqs. (27)--(35)."
        )


def square_relaxation(x_square, x, x_min, x_max):
    """Convex envelope of ``x_square = x**2`` on ``[x_min, x_max]``."""
    _validate_interval(x_min, x_max, "square-relaxation")

    return [
        x_square >= x**2,
        # Paper Eq. (16): convex lower surface of <v_i^2>^R.

        x_square <= (x_min + x_max) * x - x_min * x_max,
        # Paper Eq. (16): secant upper surface of <v_i^2>^R.
    ]


def mccormick_relaxation(product, x, y, x_min, x_max, y_min, y_max):
    """McCormick convex envelope of ``product = x*y``."""
    _validate_interval(x_min, x_max, "McCormick x")
    _validate_interval(y_min, y_max, "McCormick y")

    return [
        product >= x_min * y + y_min * x - x_min * y_min,
        # Paper definition of <x,y>^M below Eq. (19): lower plane 1.

        product >= x_max * y + y_max * x - x_max * y_max,
        # Paper definition of <x,y>^M below Eq. (19): lower plane 2.

        product <= x_min * y + y_max * x - x_min * y_max,
        # Paper definition of <x,y>^M below Eq. (19): upper plane 1.

        product <= x_max * y + y_min * x - x_max * y_min,
        # Paper definition of <x,y>^M below Eq. (19): upper plane 2.
    ]


def angle_difference_constraints(
    angle_difference,
    va_from_rad,
    va_to_rad,
    theta_max,
):
    """Define and bound theta_ij = theta_i - theta_j."""
    validate_angle_limit(theta_max)

    return [
        angle_difference == va_from_rad - va_to_rad,
        # Implementation auxiliary used by paper Eqs. (14) and (15):
        # theta_ij = theta_i - theta_j. The paper writes it inline.

        angle_difference >= -theta_max,
        # Paper Eq. (9): lower phase-angle-difference limit.

        angle_difference <= theta_max,
        # Paper Eq. (9): upper phase-angle-difference limit.
    ]


def cosine_relaxation(cos_value, angle_difference, theta_max):
    """Quadratic outer relaxation of cosine on [-theta_max, theta_max]."""
    validate_angle_limit(theta_max)
    coefficient = (1.0 - math.cos(theta_max)) / theta_max**2

    return [
        cos_value <= 1.0 - coefficient * angle_difference**2,
        # Paper Eq. (14): quadratic upper bound for cos(theta_ij).

        cos_value >= math.cos(theta_max),
        # Paper Eq. (14): constant lower bound for cos(theta_ij).
    ]


def sine_relaxation(sin_value, angle_difference, theta_max):
    """Polyhedral outer relaxation of sine on [-theta_max, theta_max]."""
    validate_angle_limit(theta_max)
    half_angle = theta_max / 2.0
    slope = math.cos(half_angle)
    intercept = math.sin(half_angle)

    return [
        sin_value <= slope * (angle_difference - half_angle) + intercept,
        # Paper Eq. (15): upper tangent inequality for sin(theta_ij).

        sin_value >= slope * (angle_difference + half_angle) - intercept,
        # Paper Eq. (15): lower tangent inequality for sin(theta_ij).
    ]


def calc_branch_t(tap, shift):
    """Return rectangular transformer-tap components."""
    if tap <= 0.0:
        raise ValueError(f"Transformer tap must be positive; got {tap}.")

    return (
        tap * math.cos(shift),
        # Paper Appendix C before Eq. (42): t_R = tap*cos(shift).

        tap * math.sin(shift),
        # Paper Appendix C before Eq. (42): t_I = tap*sin(shift).
    )


def int_flows(
    p_from_var,
    q_from_var,
    p_to_var,
    q_to_var,
    vm_from_pu_squared,
    vm_to_pu_squared,
    w_cos_pu_squared,
    w_sin_pu_squared,
    g_branch,
    b_branch,
    tap,
    shift,
    g_from=0.0,
    b_from=0.0,
    g_to_pu=0.0,
    b_to_pu=0.0,
    on_off=1,
    s_base=1.0,
):
    """Return the four lifted QC branch-flow equations.

    ``g_from``, ``b_from``, ``g_to_pu`` and ``b_to_pu`` retain monee's
    independent terminal-shunt representation. Appendix C presents symmetric
    line charging, so independent terminal shunts are an explicitly marked
    implementation extension.
    """
    _require_always_on(on_off)
    if s_base <= 0.0:
        raise ValueError(f"s_base must be positive; got {s_base}.")

    tr, ti = calc_branch_t(tap, shift)
    tap_squared = tap**2

    p_from_pu = (
        (g_branch + g_from) / tap_squared * vm_from_pu_squared
        + (-g_branch * tr + b_branch * ti)
        / tap_squared
        * w_cos_pu_squared
        + (-b_branch * tr - g_branch * ti)
        / tap_squared
        * w_sin_pu_squared
    )

    q_from_pu = (
        -(b_branch + b_from) / tap_squared * vm_from_pu_squared
        - (-b_branch * tr - g_branch * ti)
        / tap_squared
        * w_cos_pu_squared
        + (-g_branch * tr + b_branch * ti)
        / tap_squared
        * w_sin_pu_squared
    )

    p_to_pu = (
        (g_branch + g_to_pu) * vm_to_pu_squared
        + (-g_branch * tr - b_branch * ti)
        / tap_squared
        * w_cos_pu_squared
        - (-b_branch * tr + g_branch * ti)
        / tap_squared
        * w_sin_pu_squared
    )

    q_to_pu = (
        -(b_branch + b_to_pu) * vm_to_pu_squared
        - (-b_branch * tr + g_branch * ti)
        / tap_squared
        * w_cos_pu_squared
        - (-g_branch * tr - b_branch * ti)
        / tap_squared
        * w_sin_pu_squared
    )

    return [
        p_from_var == s_base * p_from_pu,
        # Paper Eq. (42); reduces to paper Eq. (12) without tap/shift/shunts.
        # UNIT ADAPTATION: paper pu -> monee MW via S_base [MVA].
        # DEVIATION: independent from-terminal shunt conductance is retained.

        q_from_var == s_base * q_from_pu,
        # Paper Eq. (44); reduces to paper Eq. (13) without tap/shift/shunts.
        # UNIT ADAPTATION: paper pu -> monee MVAr via S_base [MVA].
        # DEVIATION: independent from-terminal shunt parameters are retained.

        p_to_var == s_base * p_to_pu,
        # Paper Eq. (43), reverse-terminal lifted flow equation.
        # UNIT ADAPTATION: paper pu -> monee MW via S_base [MVA].
        # CORRECTION: the reverse diagonal uses v_j^2 (to-terminal voltage).
        # DEVIATION: independent to-terminal shunt conductance is retained.

        q_to_var == s_base * q_to_pu,
        # Paper Eq. (45), reverse-terminal lifted flow equation.
        # UNIT ADAPTATION: paper pu -> monee MVAr via S_base [MVA].
        # CORRECTION: the reverse diagonal uses v_j^2 (to-terminal voltage).
        # DEVIATION: independent to-terminal shunt parameters are retained.
    ]


def terminal_admittances(
    g_branch,
    b_branch,
    tap,
    shift,
    g_from=0.0,
    b_from=0.0,
    g_to_pu=0.0,
    b_to_pu=0.0,
):
    """Return Yff, Yft, Ytf and Ytt for the two-terminal branch model."""
    if tap <= 0.0:
        raise ValueError(f"Transformer tap must be positive; got {tap}.")

    y_series = complex(g_branch, b_branch)
    complex_tap = complex(
        tap * math.cos(shift),
        tap * math.sin(shift),
    )

    y_ff = (y_series + complex(g_from, b_from)) / tap**2
    y_ft = -y_series / complex_tap.conjugate()
    y_tf = -y_series / complex_tap
    y_tt = y_series + complex(g_to_pu, b_to_pu)

    return y_ff, y_ft, y_tf, y_tt


def terminal_current_squared_links(
    current_from_pu_squared,
    current_to_pu_squared,
    vm_from_pu_squared,
    vm_to_pu_squared,
    w_cos_pu_squared,
    w_sin_pu_squared,
    g_branch,
    b_branch,
    tap,
    shift,
    g_from=0.0,
    b_from=0.0,
    g_to_pu=0.0,
    b_to_pu=0.0,
):
    """Link terminal squared currents to the lifted voltage variables."""
    y_ff, y_ft, y_tf, y_tt = terminal_admittances(
        g_branch=g_branch,
        b_branch=b_branch,
        tap=tap,
        shift=shift,
        g_from=g_from,
        b_from=b_from,
        g_to_pu=g_to_pu,
        b_to_pu=b_to_pu,
    )

    from_cross = y_ff * y_ft.conjugate()
    to_cross = y_tf * y_tt.conjugate()

    current_from_expression = (
        abs(y_ff) ** 2 * vm_from_pu_squared
        + abs(y_ft) ** 2 * vm_to_pu_squared
        + 2.0
        * (
            from_cross.real * w_cos_pu_squared
            - from_cross.imag * w_sin_pu_squared
        )
    )

    current_to_expression = (
        abs(y_tf) ** 2 * vm_from_pu_squared
        + abs(y_tt) ** 2 * vm_to_pu_squared
        + 2.0
        * (
            to_cross.real * w_cos_pu_squared
            - to_cross.imag * w_sin_pu_squared
        )
    )

    return [
        current_from_pu_squared == current_from_expression,
        # Paper Eq. (23) when tap=1, shift=0 and terminal shunts are zero.
        # DEVIATION: affine Appendix-C/two-terminal generalisation for monee.

        current_to_pu_squared == current_to_expression,
        # Reverse-terminal counterpart of paper Eq. (23).
        # DEVIATION: the paper states one oriented-current equation; monee
        # models both terminal currents, so the reverse link is added.
    ]


def current_soc_relaxations(
    p_from_pu,
    q_from_pu,
    p_to_pu,
    q_to_pu,
    vm_from_pu_squared,
    vm_to_pu_squared,
    current_from_pu_squared,
    current_to_pu_squared,
):
    """Return the strengthened rotated-SOC current constraints."""
    return [
        p_from_pu**2 + q_from_pu**2
        <= vm_from_pu_squared * current_from_pu_squared,
        # Paper Eq. (21): from-oriented current-magnitude relaxation.

        p_to_pu**2 + q_to_pu**2
        <= vm_to_pu_squared * current_to_pu_squared,
        # Reverse-terminal counterpart of paper Eq. (21).
        # DEVIATION: added because monee models both branch ends explicitly.
    ]


def branch_equations(
    p_from_var,
    q_from_var,
    p_to_var,
    q_to_var,
    vm_from_pu,
    vm_to_pu,
    vm_from_pu_squared,
    vm_to_pu_squared,
    va_from_rad,
    va_to_rad,
    angle_difference,
    cos_angle_difference,
    sin_angle_difference,
    vm_product_pu_squared,
    w_cos_pu_squared,
    w_sin_pu_squared,
    current_from_pu_squared,
    current_to_pu_squared,
    vm_from_min,
    vm_from_max,
    vm_to_min,
    vm_to_max,
    theta_max,
    g_branch,
    b_branch,
    tap,
    shift,
    g_from=0.0,
    b_from=0.0,
    g_to_pu=0.0,
    b_to_pu=0.0,
    on_off=1,
    s_base=1.0,
):
    """Build the complete strengthened continuous QC branch formulation.

    This single helper keeps all QC-specific bound derivation and equations in
    this physical-model module. The formulation integration file only needs to
    read physical limits, create bounded variables, and pass them here.
    """
    _require_always_on(on_off)
    bounds = branch_variable_bounds(
        vm_from_min=vm_from_min,
        vm_from_max=vm_from_max,
        vm_to_min=vm_to_min,
        vm_to_max=vm_to_max,
        theta_max=theta_max,
    )

    if s_base <= 0.0:
        raise ValueError(f"s_base must be positive; got {s_base}.")

    p_from_pu = p_from_var / s_base
    q_from_pu = q_from_var / s_base
    p_to_pu = p_to_var / s_base
    q_to_pu = q_to_var / s_base

    return [
        *angle_difference_constraints(
            angle_difference=angle_difference,
            va_from_rad=va_from_rad,
            va_to_rad=va_to_rad,
            theta_max=theta_max,
        ),
        # Definition and paper Eq. (9) bounds for theta_ij.

        *cosine_relaxation(
            cos_value=cos_angle_difference,
            angle_difference=angle_difference,
            theta_max=theta_max,
        ),
        # Paper Eq. (14): relaxed cosine variable.

        *sine_relaxation(
            sin_value=sin_angle_difference,
            angle_difference=angle_difference,
            theta_max=theta_max,
        ),
        # Paper Eq. (15): relaxed sine variable.

        *mccormick_relaxation(
            product=vm_product_pu_squared,
            x=vm_from_pu,
            y=vm_to_pu,
            x_min=vm_from_min,
            x_max=vm_from_max,
            y_min=vm_to_min,
            y_max=vm_to_max,
        ),
        # Paper Eq. (17): vv_ij_tilde in <v_i,v_j>^M.

        *mccormick_relaxation(
            product=w_cos_pu_squared,
            x=vm_product_pu_squared,
            y=cos_angle_difference,
            x_min=bounds["vv_min"],
            x_max=bounds["vv_max"],
            y_min=bounds["cos_min"],
            y_max=bounds["cos_max"],
        ),
        # Paper Eq. (18): wc_ij_tilde in <vv_ij_tilde,cs_ij_tilde>^M.

        *mccormick_relaxation(
            product=w_sin_pu_squared,
            x=vm_product_pu_squared,
            y=sin_angle_difference,
            x_min=bounds["vv_min"],
            x_max=bounds["vv_max"],
            y_min=bounds["sin_min"],
            y_max=bounds["sin_max"],
        ),
        # Paper Eq. (19): ws_ij_tilde in <vv_ij_tilde,s_ij_tilde>^M.

        *int_flows(
            p_from_var=p_from_var,
            q_from_var=q_from_var,
            p_to_var=p_to_var,
            q_to_var=q_to_var,
            vm_from_pu_squared=vm_from_pu_squared,
            vm_to_pu_squared=vm_to_pu_squared,
            w_cos_pu_squared=w_cos_pu_squared,
            w_sin_pu_squared=w_sin_pu_squared,
            g_branch=g_branch,
            b_branch=b_branch,
            tap=tap,
            shift=shift,
            g_from=g_from,
            b_from=b_from,
            g_to_pu=g_to_pu,
            b_to_pu=b_to_pu,
            on_off=on_off,
            s_base=s_base,
        ),
        # Paper Eqs. (42)--(45), reducing to paper Eqs. (12),(13).

        *terminal_current_squared_links(
            current_from_pu_squared=current_from_pu_squared,
            current_to_pu_squared=current_to_pu_squared,
            vm_from_pu_squared=vm_from_pu_squared,
            vm_to_pu_squared=vm_to_pu_squared,
            w_cos_pu_squared=w_cos_pu_squared,
            w_sin_pu_squared=w_sin_pu_squared,
            g_branch=g_branch,
            b_branch=b_branch,
            tap=tap,
            shift=shift,
            g_from=g_from,
            b_from=b_from,
            g_to_pu=g_to_pu,
            b_to_pu=b_to_pu,
        ),
        # Paper Eq. (23), with marked two-terminal/tap/shunt generalisation.

        *current_soc_relaxations(
            p_from_pu=p_from_pu,
            q_from_pu=q_from_pu,
            p_to_pu=p_to_pu,
            q_to_pu=q_to_pu,
            vm_from_pu_squared=vm_from_pu_squared,
            vm_to_pu_squared=vm_to_pu_squared,
            current_from_pu_squared=current_from_pu_squared,
            current_to_pu_squared=current_to_pu_squared,
        ),
        # Paper Eq. (21), applied at both monee branch terminals.
    ]

'''import math


def power_balance_equation(signed_flows):
    """
    Kirchhoff current / power balance.
    Paper Eqs. (3) and (4)
    """
    return sum(signed_flows) == 0


def calc_branch_t(tap, shift):
    """
    Transformer tap helper.
    in Monee, not directly from paper.
    paper line model without transformer tap/phase-shift.
    monee's AC formulation includes transformer tap magnitude and phase shift,
    coefficient structure from existing polar AC NLP.
    """
    return (tap * math.cos(shift), tap * math.sin(shift))


def is_plain_one(value):
    return isinstance(value, (int, float)) and value == 1


def is_plain_zero(value):
    return isinstance(value, (int, float)) and value == 0


def square_relaxation(x_square, x, x_min, x_max):
    """
    Convex envelope for x_square ~= x^2 over [x_min, x_max].

    Paper Eq. (16): v_i_tilde in <v_i^2>^R

    with Eq. (19):
        v_tilde >= v^2
        v_tilde <= (v^u + v^l) * v - v^u * v^l

    Meaning:
    exact AC equations use v_i^2.
    QC relaxation, v_i^2 replaced by new variable x_square (constrained to lie inside the convex envelope of x^2).
    from nonlinear and nonconvex to convex relaxation
    """
    return [
        # Lower convex bound: x_square must be above x^2.
        x_square >= x * x,

        # Upper secant bound over [x_min, x_max].
        x_square <= (x_min + x_max) * x - x_min * x_max,
    ]


def mccormick_relaxation(w, x, y, x_min, x_max, y_min, y_max):
    """
    McCormick envelope exact values are replaced by inequalities within bound
    """
    return [
        # McCormick lower bound 1.
        w >= x_min * y + y_min * x - x_min * y_min,

        # McCormick lower bound 2.
        w >= x_max * y + y_max * x - x_max * y_max,

        # McCormick upper bound 1.
        w <= x_min * y + y_max * x - x_min * y_max,

        # McCormick upper bound 2.
        w <= x_max * y + y_min * x - x_max * y_min,
    ]


def binary_product_relaxation(w, z, x, x_min, x_max):
    """
    Convex hull for w ~= z * x, with z in {0, 1}.
    Paper Eq. (24): Perspective / convex hull reformulation for on/off constraints.
     If z = 1, then w = x.
    If z = 0, then w = 0.

    No direct writing of z * x inside the flow equations, to no mixed integer (binary*nonbinary)

    Note:
        This helper is a practical compact binary-continuous product relaxation.
        It is consistent with the perspective/on-off idea in Eq. (24), although
        it is not the full extended disjunctive formulation from the paper.
    """
    if is_plain_one(z):
        return [w == x]

    if is_plain_zero(z):
        return [w == 0]

    return [
        w >= x_min * z,
        w <= x_max * z,
        w >= x - x_max * (1 - z),
        w <= x - x_min * (1 - z),
    ]


def cosine_relaxation(cs, theta, theta_max, on_off=1, theta_big_m=None):
    """
    Quadratic relaxation of cos(theta).

    Paper reference:
        Eq. (14):
            cs_ij_tilde in <cos(theta_i - theta_j)>^R

        Definition below Eq. (19):
            cs_tilde <= 1 - ((1 - cos(theta^u)) / (theta^u)^2) * theta^2
            cs_tilde >= cos(theta^u)

        Switchable-line version:
            Eq. (28), cosine function disjunction.

    Meaning:
        cs is a relaxed approximation of cos(theta).
        It is not forced to equal cos(theta), but it is bounded by a convex
        outer approximation.

    Why:
        The exact cosine term makes the AC model nonconvex. The paper replaces
        it with a quadratic convex relaxation.
    """
    k = (1.0 - math.cos(theta_max)) / (theta_max**2)

    if is_plain_one(on_off):
        return [
            # Operational angle-difference bound.
            # Paper reference: Eq. (9)
            theta >= -theta_max,
            theta <= theta_max,

            # Cosine QC upper bound.
            # Paper reference: Eq. (14), definition of <cos(theta)>^R.
            cs <= 1.0 - k * theta * theta,

            # Cosine lower bound.
            # Paper reference: Eq. (14), definition of <cos(theta)>^R.
            cs >= math.cos(theta_max),
        ]

    if theta_big_m is None:
        raise ValueError("theta_big_m is required for switchable QC cosine relaxation.")

    return [
        # Angle is tightly bounded when the line is on.
        # When the line is off, theta may relax to the larger big-M interval.
        # Paper reference: Eq. (28).
        theta >= -on_off * theta_max - (1 - on_off) * theta_big_m,
        theta <= on_off * theta_max + (1 - on_off) * theta_big_m,

        # On/off cosine relaxation.
        # If on_off = 1, this reduces to the normal QC cosine inequality.
        # If on_off = 0, cs is forced toward 0 and the angle is relaxed.
        # Paper reference: Eq. (28).
        cs <= on_off - k * theta * theta + (1 - on_off) * k * theta_big_m**2,

        # Bounds on switched cosine variable.
        # Paper reference: Eq. (28).
        cs >= on_off * math.cos(theta_max),
        cs <= on_off,
    ]


def sine_relaxation(s, theta, theta_max, on_off=1, theta_big_m=None):
    """
    Polyhedral relaxation of sin(theta).

    Paper reference:
        Eq. (15):
            s_ij_tilde in <sin(theta_i - theta_j)>^R

        Definition below Eq. (19):
            s_tilde <= cos(theta^u / 2) * (theta - theta^u / 2)
                       + sin(theta^u / 2)

            s_tilde >= cos(theta^u / 2) * (theta + theta^u / 2)
                       - sin(theta^u / 2)

        Switchable-line version:
            Eq. (27), sine function disjunction.

    Meaning:
        s is a relaxed approximation of sin(theta), bounded by two linear
        inequalities.

    Why:
        The sine term is nonlinear and nonconvex. The paper replaces it with a
        polyhedral outer approximation.
    """
    a = math.cos(theta_max / 2.0)
    b = math.sin(theta_max / 2.0)
    rhs_on = b - a * theta_max / 2.0

    if is_plain_one(on_off):
        return [
            # Operational angle-difference bound.
            # Paper reference: Eq. (9)
            theta >= -theta_max,
            theta <= theta_max,

            # Upper sine envelope.
            # Paper reference: Eq. (15), definition of <sin(theta)>^R.
            s <= a * (theta - theta_max / 2.0) + b,

            # Lower sine envelope.
            # Paper reference: Eq. (15), definition of <sin(theta)>^R.
            s >= a * (theta + theta_max / 2.0) - b,

            # Explicit sine bounds over [-theta_max, theta_max].
            s >= -math.sin(theta_max),
            s <= math.sin(theta_max),
        ]

    if theta_big_m is None:
        raise ValueError("theta_big_m is required for switchable QC sine relaxation.")

    abs_s_rhs = on_off * (b + a * theta_max / 2.0)
    abs_theta_rhs = (
        on_off * (b - a * theta_max / 2.0 + math.sin(theta_max))
        + (1 - on_off) * a * theta_big_m
    )

    return [
        # Angle bound with on/off disjunction.
        # Paper reference: Eq. (27).
        theta >= -on_off * theta_max - (1 - on_off) * theta_big_m,
        theta <= on_off * theta_max + (1 - on_off) * theta_big_m,

        # First two linear inequalities of the on/off sine envelope.
        # Paper reference: Eq. (27).
        s - a * theta <= on_off * rhs_on + (1 - on_off) * a * theta_big_m,
        -s + a * theta <= on_off * rhs_on + (1 - on_off) * a * theta_big_m,

        # Absolute-value style bounds on s.
        # Implemented as two linear inequalities to avoid abs().
        # Paper reference: Eq. (27), |s| bound.
        s <= abs_s_rhs,
        -s <= abs_s_rhs,

        # Absolute-value style bounds on theta.
        # Implemented as two linear inequalities to avoid abs().
        # Paper reference: Eq. (27), cos(theta^u / 2) * |theta| bound.
        a * theta <= abs_theta_rhs,
        -a * theta <= abs_theta_rhs,

        # Switched sine bounds.
        # If on_off = 0, s is forced to 0.
        # Paper reference: Eq. (27).
        s >= on_off * (-math.sin(theta_max)),
        s <= on_off * math.sin(theta_max),
    ]


def qc_int_flows(  # NOSONAR
    p_from_var,
    q_from_var,
    p_to_var,
    q_to_var,
    vm_from_pu,
    vm_to_pu,
    vm_from_pu_square,
    vm_to_pu_square,
    va_from_rad,
    va_to_rad,
    theta_var,
    cs_var,
    s_var,
    vv_var,
    vvc_var,
    vvs_var,
    g_branch,
    b_branch,
    tap,
    shift,
    vm_from_min,
    vm_from_max,
    vm_to_min,
    vm_to_max,
    theta_max,
    g_from=0,
    b_from=0,
    g_to_pu=0,
    b_to_pu=0,
    on_off=1,
    theta_big_m=None,
    vm_from_pu_square_on=None,
    vm_to_pu_square_on=None,
):
    """
    Quadratic-convex relaxed AC branch flow equations.

    Paper reference:
        Original nonlinear AC equations:
            Eq. (1): active branch power p_ij
            Eq. (2): reactive branch power q_ij

        QC-core replacement:
            Eq. (12): p_ij = g_ij * v_i_tilde
                              - g_ij * wc_ij_tilde
                              - b_ij * ws_ij_tilde

            Eq. (13): q_ij = -b_ij * v_i_tilde
                               + b_ij * wc_ij_tilde
                               - g_ij * ws_ij_tilde

            Eq. (14): relaxed cosine variable cs_ij_tilde
            Eq. (15): relaxed sine variable s_ij_tilde
            Eq. (16): relaxed squared voltage v_i_tilde
            Eq. (17): relaxed voltage product vv_ij_tilde
            Eq. (18): relaxed product wc_ij_tilde = vv_ij_tilde * cs_ij_tilde
            Eq. (19): relaxed product ws_ij_tilde = vv_ij_tilde * s_ij_tilde

    Meaning:
        This function replaces exact nonlinear AC flow expressions with the
        relaxed QC-core variables from the paper.

    Note:
        The paper writes the base equations without tap, phase shift and shunt
        terms. Here we keep monee's existing tap/shift/shunt coefficient structure
        and substitute the nonlinear terms with their QC relaxed variables.
    """
    tr, ti = calc_branch_t(tap, shift)

    vv_min = vm_from_min * vm_to_min
    vv_max = vm_from_max * vm_to_max

    cs_min = 0.0 if not is_plain_one(on_off) else math.cos(theta_max)
    cs_max = 1.0

    s_min = -math.sin(theta_max)
    s_max = math.sin(theta_max)

    constraints = []

    # Defines theta_ij = theta_i - theta_j.
    #
    # Paper reference:
    #   Eqs. (1), (2), (14), (15)
    #
    # Meaning:
    #   The AC equations and the sine/cosine relaxations are all functions of
    #   the voltage angle difference theta_i - theta_j.
    constraints.append(theta_var == va_from_rad - va_to_rad)

    # Relaxed cosine variable cs_ij_tilde.
    #
    # Paper reference:
    #   Eq. (14), or Eq. (28) if on_off is a switching variable.
    constraints += cosine_relaxation(
        cs=cs_var,
        theta=theta_var,
        theta_max=theta_max,
        on_off=on_off,
        theta_big_m=theta_big_m,
    )

    # Relaxed sine variable s_ij_tilde.
    #
    # Paper reference:
    #   Eq. (15), or Eq. (27) if on_off is a switching variable.
    constraints += sine_relaxation(
        s=s_var,
        theta=theta_var,
        theta_max=theta_max,
        on_off=on_off,
        theta_big_m=theta_big_m,
    )

    # Relaxed voltage product:
    #   vv_ij_tilde ~= v_i * v_j
    #
    # Paper reference:
    #   Eq. (17)
    constraints += mccormick_relaxation(
        w=vv_var,
        x=vm_from_pu,
        y=vm_to_pu,
        x_min=vm_from_min,
        x_max=vm_from_max,
        y_min=vm_to_min,
        y_max=vm_to_max,
    )

    # Relaxed cosine power-flow product:
    #   wc_ij_tilde ~= vv_ij_tilde * cs_ij_tilde
    #
    # Paper reference:
    #   Eq. (18)
    constraints += mccormick_relaxation(
        w=vvc_var,
        x=vv_var,
        y=cs_var,
        x_min=vv_min,
        x_max=vv_max,
        y_min=cs_min,
        y_max=cs_max,
    )

    # Relaxed sine power-flow product:
    #   ws_ij_tilde ~= vv_ij_tilde * s_ij_tilde
    #
    # Paper reference:
    #   Eq. (19)
    constraints += mccormick_relaxation(
        w=vvs_var,
        x=vv_var,
        y=s_var,
        x_min=vv_min,
        x_max=vv_max,
        y_min=s_min,
        y_max=s_max,
    )

    if is_plain_one(on_off):
        # Always-on branch:
        #
        # Paper reference:
        #   Eqs. (12) and (13)
        #
        # Meaning:
        #   Use the relaxed voltage-square variables directly.
        vm_from_flow_square = vm_from_pu_square
        vm_to_flow_square = vm_to_pu_square
    else:
        if vm_from_pu_square_on is None or vm_to_pu_square_on is None:
            raise ValueError(
                "vm_from_pu_square_on and vm_to_pu_square_on are required "
                "for switchable QC branch flows."
            )

        # Switched voltage-square product:
        #   vm_from_pu_square_on ~= on_off * vm_from_pu_square
        #
        # Paper reference:
        #   Eq. (24), general on/off convex-hull idea.
        #
        # Meaning:
        #   If the line is off, the v_i^2 contribution in the flow equation
        #   must disappear.
        constraints += binary_product_relaxation(
            w=vm_from_pu_square_on,
            z=on_off,
            x=vm_from_pu_square,
            x_min=vm_from_min**2,
            x_max=vm_from_max**2,
        )

        # Same switched voltage-square product for the to-side.
        #
        # Paper reference:
        #   Eq. (24), general on/off convex-hull idea.
        constraints += binary_product_relaxation(
            w=vm_to_pu_square_on,
            z=on_off,
            x=vm_to_pu_square,
            x_min=vm_to_min**2,
            x_max=vm_to_max**2,
        )

        # Additional bounds forcing relaxed product terms to zero when the
        # branch is switched off.
        #
        # Paper reference:
        #   Related to Eqs. (27) and (28), where sine/cosine variables are
        #   switched by z.
        constraints += [
            vvc_var >= 0,
            vvc_var <= on_off * vv_max,
            vvs_var >= -on_off * vv_max * math.sin(theta_max),
            vvs_var <= on_off * vv_max * math.sin(theta_max),
        ]

        vm_from_flow_square = vm_from_pu_square_on
        vm_to_flow_square = vm_to_pu_square_on

    # Active power from-side flow.
    #
    # Paper reference:
    #   Eq. (12), generalized with monee's tap, shift and shunt coefficients.
    #
    # Meaning:
    #   Replaces:
    #       v_i^2,
    #       v_i * v_j * cos(theta_i - theta_j),
    #       v_i * v_j * sin(theta_i - theta_j)
    #
    #   with:
    #       vm_from_flow_square,
    #       vvc_var,
    #       vvs_var.
    p_from = (
        (g_branch + g_from) / tap**2 * vm_from_flow_square
        + (-g_branch * tr + b_branch * ti) / tap**2 * vvc_var
        + (-b_branch * tr - g_branch * ti) / tap**2 * vvs_var
    )

    # Reactive power from-side flow.
    #
    # Paper reference:
    #   Eq. (13), generalized with monee's tap, shift and shunt coefficients.
    q_from = (
        -(b_branch + b_from) / tap**2 * vm_from_flow_square
        - (-b_branch * tr - g_branch * ti) / tap**2 * vvc_var
        + (-g_branch * tr + b_branch * ti) / tap**2 * vvs_var
    )

    # Active power to-side flow.
    #
    # Paper reference:
    #   Eq. (12), applied to the reverse direction.
    #
    # Meaning:
    #   The paper notes AC flow is asymmetric. The reverse side gets its own
    #   p_to expression. cos(theta) is even and sin(theta) is odd, which is why
    #   the signs differ from p_from.
    p_to = (
        (g_branch + g_to_pu) * vm_to_flow_square
        + (-g_branch * tr - b_branch * ti) / tap**2 * vvc_var
        - (-b_branch * tr + g_branch * ti) / tap**2 * vvs_var
    )

    # Reactive power to-side flow.
    #
    # Paper reference:
    #   Eq. (13), applied to the reverse direction.
    q_to = (
        -(b_branch + b_to_pu) * vm_to_flow_square
        - (-b_branch * tr + g_branch * ti) / tap**2 * vvc_var
        - (-g_branch * tr - b_branch * ti) / tap**2 * vvs_var
    )

    constraints += [
        p_from_var == p_from,
        q_from_var == q_from,
        p_to_var == p_to,
        q_to_var == q_to,
    ]

    return constraints'''