import math


def power_balance_equation(signed_flows):
    """
    No docstring provided.
    """
    return sum(signed_flows) == 0


def calc_branch_t(tap, shift):
    """
    No docstring provided.
    """
    return (tap * math.cos(shift), tap * math.sin(shift))


def int_flow_from_p(
    p_from_var,
    vm_from_var,
    vm_to_var,
    va_from_var,
    va_to_var,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    g_from=0,
    on_off=1,
):
    """
    No docstring provided.
    """
    tr, ti = calc_branch_t(tap, shift)
    return p_from_var == on_off * (
        (g_branch + g_from) / tap**2 * vm_from_var**2
        + (-g_branch * tr + b_branch * ti)
        / tap**2
        * (vm_from_var * vm_to_var * cos_impl(va_from_var - va_to_var))
        + (-b_branch * tr - g_branch * ti)
        / tap**2
        * (vm_from_var * vm_to_var * sin_impl(va_from_var - va_to_var))
    )


def int_flow_from_q(
    q_from_var,
    vm_from_var,
    vm_to_var,
    va_from_var,
    va_to_var,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    b_from=0,
    on_off=1,
):
    """
    No docstring provided.
    """
    tr, ti = calc_branch_t(tap, shift)
    return q_from_var == on_off * (
        -(b_branch + b_from) / tap**2 * vm_from_var**2
        - (-b_branch * tr - g_branch * ti)
        / tap**2
        * (vm_from_var * vm_to_var * cos_impl(va_from_var - va_to_var))
        + (-g_branch * tr + b_branch * ti)
        / tap**2
        * (vm_from_var * vm_to_var * sin_impl(va_from_var - va_to_var))
    )


def int_flow_to_p(
    p_to_var,
    vm_from_var,
    vm_to_var,
    va_from_var,
    va_to_var,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    g_to=0,
    on_off=1,
):
    """
    No docstring provided.
    """
    tr, ti = calc_branch_t(tap, shift)
    return p_to_var == on_off * (
        (g_branch + g_to) * vm_to_var**2
        + (-g_branch * tr - b_branch * ti)
        / tap**2
        * (vm_to_var * vm_from_var * cos_impl(va_to_var - va_from_var))
        + (-b_branch * tr + g_branch * ti)
        / tap**2
        * (vm_to_var * vm_from_var * sin_impl(va_to_var - va_from_var))
    )


def int_flow_to_q(
    q_to_var,
    vm_from_var,
    vm_to_var,
    va_from_var,
    va_to_var,
    g_branch,
    b_branch,
    tap,
    shift,
    cos_impl=math.cos,
    sin_impl=math.sin,
    b_to=0,
    on_off=1,
):
    """
    No docstring provided.
    """
    tr, ti = calc_branch_t(tap, shift)
    return q_to_var == on_off * (
        -(b_branch + b_to) * vm_to_var**2
        - (-b_branch * tr + g_branch * ti)
        / tap**2
        * (vm_to_var * vm_from_var * cos_impl(va_to_var - va_from_var))
        + (-g_branch * tr - b_branch * ti)
        / tap**2
        * (vm_to_var * vm_from_var * sin_impl(va_to_var - va_from_var))
    )
