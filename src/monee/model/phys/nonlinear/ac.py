import math


def power_balance_equation(signed_flows):
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
