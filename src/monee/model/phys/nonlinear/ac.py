import math


def power_balance_equation(signed_flows, scale=1.0):
    # The nodal balance residual is MW-magnitude; dividing by the grid's
    # apparent-power base expresses it per-unit (O(1)) so IPOPT's feasibility
    # measure isn't dominated by the MW scale. Solution-invariant (scaling an
    # equality by a nonzero constant). ``scale == 1`` is the inert default.
    if scale != 1.0:
        return sum(signed_flows) / scale == 0
    return sum(signed_flows) == 0


def calc_branch_t(tap, shift):
    return (tap * math.cos(shift), tap * math.sin(shift))


def int_flow_from_p(
    p_from_var,
    vm_from_pu,
    vm_to_pu,
    va_from_rad,
    va_to_rad,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    g_from=0,
    on_off=1,
):
    tr, ti = calc_branch_t(tap, shift)
    return p_from_var == on_off * (
        (g_branch + g_from) / tap**2 * vm_from_pu**2
        + (-g_branch * tr + b_branch * ti)
        / tap**2
        * (vm_from_pu * vm_to_pu * cos_impl(va_from_rad - va_to_rad))
        + (-b_branch * tr - g_branch * ti)
        / tap**2
        * (vm_from_pu * vm_to_pu * sin_impl(va_from_rad - va_to_rad))
    )


def int_flow_from_q(
    q_from_var,
    vm_from_pu,
    vm_to_pu,
    va_from_rad,
    va_to_rad,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    b_from=0,
    on_off=1,
):
    tr, ti = calc_branch_t(tap, shift)
    return q_from_var == on_off * (
        -(b_branch + b_from) / tap**2 * vm_from_pu**2
        - (-b_branch * tr - g_branch * ti)
        / tap**2
        * (vm_from_pu * vm_to_pu * cos_impl(va_from_rad - va_to_rad))
        + (-g_branch * tr + b_branch * ti)
        / tap**2
        * (vm_from_pu * vm_to_pu * sin_impl(va_from_rad - va_to_rad))
    )


def int_flow_to_p(
    p_to_var,
    vm_from_pu,
    vm_to_pu,
    va_from_rad,
    va_to_rad,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    g_to_pu=0,
    on_off=1,
):
    tr, ti = calc_branch_t(tap, shift)
    return p_to_var == on_off * (
        (g_branch + g_to_pu) * vm_to_pu**2
        + (-g_branch * tr - b_branch * ti)
        / tap**2
        * (vm_to_pu * vm_from_pu * cos_impl(va_to_rad - va_from_rad))
        + (-b_branch * tr + g_branch * ti)
        / tap**2
        * (vm_to_pu * vm_from_pu * sin_impl(va_to_rad - va_from_rad))
    )


def int_flow_to_q(
    q_to_var,
    vm_from_pu,
    vm_to_pu,
    va_from_rad,
    va_to_rad,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    b_to_pu=0,
    on_off=1,
):
    tr, ti = calc_branch_t(tap, shift)
    return q_to_var == on_off * (
        -(b_branch + b_to_pu) * vm_to_pu**2
        - (-b_branch * tr + g_branch * ti)
        / tap**2
        * (vm_to_pu * vm_from_pu * cos_impl(va_to_rad - va_from_rad))
        + (-g_branch * tr - b_branch * ti)
        / tap**2
        * (vm_to_pu * vm_from_pu * sin_impl(va_to_rad - va_from_rad))
    )


def int_flows(  # NOSONAR
    p_from_var,
    q_from_var,
    p_to_var,
    q_to_var,
    vm_from_pu,
    vm_to_pu,
    va_from_rad,
    va_to_rad,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    g_from=0,
    b_from=0,
    g_to_pu=0,
    b_to_pu=0,
    on_off=1,
    s_base=1,
):
    """All four AC branch flow equations (P/Q at both ends) in one pass.

    Mathematically identical to calling :func:`int_flow_from_p`,
    :func:`int_flow_from_q`, :func:`int_flow_to_p` and :func:`int_flow_to_q`
    separately, but the sub-terms shared across the four (``vm_from*vm_to``, the
    sine/cosine of the bus-angle difference, ``vm_from**2``, ``vm_to**2``) are
    built **once** and reused. The to-direction reuses the from-direction's
    ``cos``/``sin`` via the even/odd identities ``cos(-x) = cos(x)`` and
    ``sin(-x) = -sin(x)``. The numeric float coefficients are formed exactly as
    in the per-equation functions, so the only change is node sharing - the
    symbolic graph these equations contribute is ~halved, which speeds model
    assembly on every backend (the equation bodies are backend-agnostic).

    The bracketed expressions are per-unit power on the grid's apparent-power
    base; ``s_base`` (= ``grid.sn_mva``) scales them to the MW/MVAr that
    ``p_*_mw``/``q_*_mvar`` and the nodal balance are expressed in. With the
    default ``sn_mva = 1`` (per-unit == MW) this is a no-op.

    Returns ``[p_from_eq, q_from_eq, p_to_eq, q_to_eq]``.
    """
    tr, ti = calc_branch_t(tap, shift)
    vm_from_sq = vm_from_pu**2
    vm_to_sq = vm_to_pu**2
    vv = vm_from_pu * vm_to_pu
    # cos is even and sin is odd, so the to-direction (angle va_to - va_from)
    # reuses these with cos unchanged and sin negated (see the ``-`` signs on the
    # vvs terms in p_to / q_to below).
    vvc = vv * cos_impl(va_from_rad - va_to_rad)
    vvs = vv * sin_impl(va_from_rad - va_to_rad)

    p_from = (
        (g_branch + g_from) / tap**2 * vm_from_sq
        + (-g_branch * tr + b_branch * ti) / tap**2 * vvc
        + (-b_branch * tr - g_branch * ti) / tap**2 * vvs
    )
    q_from = (
        -(b_branch + b_from) / tap**2 * vm_from_sq
        - (-b_branch * tr - g_branch * ti) / tap**2 * vvc
        + (-g_branch * tr + b_branch * ti) / tap**2 * vvs
    )
    p_to = (
        (g_branch + g_to_pu) * vm_to_sq
        + (-g_branch * tr - b_branch * ti) / tap**2 * vvc
        - (-b_branch * tr + g_branch * ti) / tap**2 * vvs
    )
    q_to = (
        -(b_branch + b_to_pu) * vm_to_sq
        - (-b_branch * tr + g_branch * ti) / tap**2 * vvc
        - (-g_branch * tr - b_branch * ti) / tap**2 * vvs
    )

    if s_base != 1:
        p_from, q_from, p_to, q_to = (
            s_base * p_from,
            s_base * q_from,
            s_base * p_to,
            s_base * q_to,
        )

    # Skip the ``on_off *`` factor only when it is the plain scalar 1 (the common
    # always-on case). A switchable line passes ``on_off`` as a decision variable
    # (e.g. ``Var(1, integer=True)``); comparing that to 1 would evaluate a
    # backend expression's truthiness, so fall through to the multiply for any
    # non-scalar (and for a scalar != 1, e.g. a deactivated 0).
    if isinstance(on_off, (int, float)) and on_off == 1:
        return [
            p_from_var == p_from,
            q_from_var == q_from,
            p_to_var == p_to,
            q_to_var == q_to,
        ]
    return [
        p_from_var == on_off * p_from,
        q_from_var == on_off * q_from,
        p_to_var == on_off * p_to,
        q_to_var == on_off * q_to,
    ]
