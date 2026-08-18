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
