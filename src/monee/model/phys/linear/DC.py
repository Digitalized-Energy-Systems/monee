"""DC power-flow equations.

Source (equation numbers refer to it): MATPOWER User's Manual v8.1 (2025),
Sect. 3.7 "DC Modeling", https://matpower.org/docs/MATPOWER-manual.pdf

DC power flow simplifies AC power flow with three assumptions:
- no losses: resistance and line charging are ignored (3.18),
- all voltage magnitudes are 1 p.u. (3.19),
- angle differences are small, so sin(x) ~ x (3.20).

Only active power and voltage angles remain; reactive power is ignored.
The manual works in p.u.; we multiply by ``s_base`` to get MW.
"""


def branch_susceptance(br_x_pu, tap=1.0):
    """b = 1 / (x * tap), see below Eq. (3.24). A tap of 0 means a plain line (tap = 1)."""
    if br_x_pu == 0:
        raise ValueError(
            "DC power flow needs a nonzero branch reactance (b = 1 / (x * tap))."
        )
    if tap == 0:
        tap = 1.0
    if tap < 0:
        raise ValueError("Transformer tap ratio must be positive.")
    return 1.0 / (br_x_pu * tap)


def int_flows(
    p_from_var,
    p_to_var,
    va_from_rad,
    va_to_rad,
    br_x_pu,
    tap=1.0,
    shift=0.0,
    on_off=1,
    s_base=1.0,
):
    """Branch flow equations, Eq. (3.23)/(3.24):
    p_from = b * (angle_from - angle_to - shift)
    p_to   = -p_from   (no losses)
    ``on_off = 0`` makes an open branch carry no power.
    """
    b = branch_susceptance(br_x_pu, tap)
    p_from = s_base * b * (va_from_rad - va_to_rad - shift)
    return [
        p_from_var == on_off * p_from,
        p_to_var == -on_off * p_from,
    ]


def shunt_p(gs_mw):
    """Shunt active power, Eq. (3.26): with |V| = 1 it is just G_sh."""
    return gs_mw
