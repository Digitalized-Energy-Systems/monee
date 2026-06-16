import math

# Legacy fallback specific gas constant R/M [J/(kg*K)] for methane (M=0.0165
# kg/mol, i.e. 8.314/0.0165 ~ 504.5 - NOT the lgas default M=0.0181). Callers
# should pass the grid-derived ``r_specific`` (= universal_gas_constant /
# molar_mass) so the gas's actual molar mass is honoured; this module-level
# default only keeps direct/legacy calls working.
R_specific = 504.5


def abs_psq_diff_pu(psq_pu_i, psq_pu_j, p_pu_i, p_pu_j, pressure_ambient_pa, pressure_ref_pa):
    r"""Absolute squared-pressure difference (per-unit of :math:`p_{ref}^2`) when the
    node pressures are GAUGE.

    The Weymouth law needs absolute pressures. With gauge node pressures
    :math:`p_g` and ambient :math:`p_a`, the absolute is :math:`p = p_g + p_a`, so

    .. math::

        p_i^2 - p_j^2 = (p_{g,i}+p_a)^2 - (p_{g,j}+p_a)^2
                      = (p_{sq,i}-p_{sq,j}) + 2\,a\,(p_{pu,i}-p_{pu,j})

    with :math:`a = p_a/p_{ref}` (the constant :math:`a^2` cancels in the difference).
    With ``pressure_ambient_pa = 0`` this is exactly the historical absolute form
    :math:`p_{sq,i} - p_{sq,j}`.

    ``p_pu`` is whatever pressure the caller supplies: the NLP passes the exact
    :math:`\sqrt{p_{sq}}` intermediate (it tolerates the non-linearity); the MILP
    and convex MIQCQP pass the LINEARIZED pressure
    :math:`p_0 + (p_{sq}-p_0^2)/(2 p_0)` (affine in the decision variable, the same
    linearization they use for density), so the offset stays linear / conic.
    """
    base = psq_pu_i - psq_pu_j
    if pressure_ambient_pa == 0.0:
        # Default absolute convention: no offset, and no reference to p_pu at all
        # (keeps the equation byte-identical to the pre-gauge form).
        return base
    amb_pu = pressure_ambient_pa / pressure_ref_pa
    return base + 2.0 * amb_pu * (p_pu_i - p_pu_j)


def abs_psq_pu(psq_pu, p_pu, pressure_ambient_pa, pressure_ref_pa):
    r"""Single-node absolute squared pressure (per-unit), dropping the constant
    :math:`a^2` that cancels in any i-j difference. Short-circuits to the gauge
    ``psq_pu`` when ambient is 0 (no ``p_pu`` reference). See
    :func:`abs_psq_diff_pu` for the exact-vs-linearized ``p_pu`` convention."""
    if pressure_ambient_pa == 0.0:
        return psq_pu
    return psq_pu + 2.0 * (pressure_ambient_pa / pressure_ref_pa) * p_pu


def abs_pressure_pu(pressure_pu, pressure_ambient_pa, pressure_ref_pa):
    """Absolute pressure (per-unit of ``pressure_ref_pa``) from a gauge ``pressure_pu``."""
    return pressure_pu + pressure_ambient_pa / pressure_ref_pa


def reference_gas_density(grid):
    r"""Ideal-gas density [kg/m^3] at the grid's reference ABSOLUTE pressure.

    Absolute = ``pressure_ref_pa`` + ``pressure_ambient_pa`` (the latter is 0 in
    the default absolute convention). Used to size per-pipe velocity / mass-flow
    caps, so the cap reflects the true operating density under a gauge grid."""
    p_abs = grid.pressure_ref_pa + getattr(grid, "pressure_ambient_pa", 0.0)
    return p_abs * grid.molar_mass / (grid.universal_gas_constant * grid.t_k)


def calc_C_squared(diameter_m, length_m, t_k, compressibility, r_specific=R_specific):
    numerator = math.pi**2 * diameter_m**5
    denominator = 16 * length_m * r_specific * t_k * compressibility
    C_squared = numerator / denominator
    return C_squared


def pipe_weymouth(
    p_squared_i,
    p_squared_j,
    f_a_pos_sq,
    f_a_neg_sq,
    diameter_m,
    length_m,
    t_k,
    compressibility,
    on_off=1,
    friction=None,
    r_specific=R_specific,
    **kwargs,
):
    return (p_squared_i - p_squared_j) * calc_C_squared(
        diameter_m,
        length_m,
        t_k,
        compressibility,
        r_specific,
    ) * on_off == friction * -(f_a_pos_sq - f_a_neg_sq)
