import math


def calc_pipe_area(diameter_m):
    return math.pi * diameter_m**2 / 4


def calc_max_mass_flow(diameter_m, fluid_density_kg_per_m3, v_max_mps):
    r"""Per-pipe mass-flow upper bound [kg/s] from a velocity cap (:math:`\pi/4 \cdot D^2 \cdot \rho \cdot v_{max}`).
    Used to tighten per-pipe big-M relaxations."""
    return calc_pipe_area(diameter_m) * fluid_density_kg_per_m3 * v_max_mps


def calc_local_max_mass_flow(grid, branch, density, v_max):
    r"""Per-pipe mass-flow upper bound [kg/s]: the smaller of ``grid.max_mass_flow_kgs``
    and the velocity cap :math:`\pi/4 \cdot D^2 \cdot \rho \cdot v_{max}`. ``density`` is the
    fluid density and ``v_max`` the velocity cap used for the velocity bound."""
    return min(
        grid.max_mass_flow_kgs,
        calc_max_mass_flow(branch.diameter_m, density, v_max),
    )


def calc_min_diameter_for_mass_flow(
    mass_flow_kgs, fluid_density_kg_per_m3, v_design_mps
):
    r"""Smallest pipe diameter [m] carrying ``mass_flow_kgs`` at ``v_design_mps``.

    Inverse of :func:`calc_max_mass_flow`: :math:`D = \sqrt{4 \cdot m / (\pi \cdot \rho \cdot v)}`. Used to
    size DHS pipes to their required capacity so the velocity cap does not bind.
    Returns 0 for degenerate inputs (:math:`m \le 0` or non-positive :math:`\rho/v`)."""
    if mass_flow_kgs <= 0 or fluid_density_kg_per_m3 <= 0 or v_design_mps <= 0:
        return 0.0
    return math.sqrt(
        4.0 * mass_flow_kgs / (math.pi * fluid_density_kg_per_m3 * v_design_mps)
    )


# model.reynolds_scaled is stored as Re/1e6, so friction PWL breakpoints sit in [0,10]
# rather than [0,1e7] - keeps the matrix coefficient range manageable.
# Friction y-values are still computed at unscaled Re.
REYNOLDS_SCALE = 1e6


def reynolds_equation(rey_var, mass_flow_kgs, diameter_m, dynamic_visc_pas, pipe_area):
    return rey_var == mass_flow_kgs * diameter_m / (
        dynamic_visc_pas * pipe_area * REYNOLDS_SCALE
    )


def junction_mass_flow_balance(flows):
    return sum(flows) == 0


def pipe_mass_flow(max_v, min_v, v):
    return min_v <= v <= max_v


def swamee_jain(reynolds_var, diameter_m, roughness_m, log_func):
    term1 = roughness_m / diameter_m / 3.7
    term2 = 5.74 / (reynolds_var + 1) ** 0.9  # +1 avoids infeasibility at Re=0
    denominator = log_func(term1 + term2) ** 2
    f = 0.25 / denominator
    return f


def friction_at_high_re(diameter_m: float, roughness_m: float) -> float:
    r"""Turbulent asymptote :math:`f = 0.25 / \log_{10}(\varepsilon / (3.7 \cdot D))^2` (Swamee-Jain at :math:`Re \to \infty`).
    Within a few % of Colebrook for :math:`Re \gtrsim 10^4`; under-estimates pressure drop in
    laminar (:math:`Re < 2300`). Returns 0 for degenerate inputs (:math:`D \le 0`, :math:`\varepsilon \ge 3.7 \cdot D`)."""
    if diameter_m <= 0 or roughness_m <= 0:
        return 0.0
    term1 = roughness_m / diameter_m / 3.7
    if term1 >= 1.0:
        return 0.0
    return 0.25 / math.log10(term1) ** 2


def filter_near_linear(xs, ys, rtol=1e-6):
    if len(xs) <= 2:
        return xs, ys

    keep_x = [xs[0]]
    keep_y = [ys[0]]

    prev_slope = (ys[1] - ys[0]) / (xs[1] - xs[0])

    for i in range(1, len(xs) - 1):
        slope = (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])

        if abs(slope - prev_slope) > rtol * max(1.0, abs(prev_slope)):
            keep_x.append(xs[i])
            keep_y.append(ys[i])
            prev_slope = slope

    keep_x.append(xs[-1])
    keep_y.append(ys[-1])

    return keep_x, keep_y


def friction_value(reynolds, diameter, eps):
    if reynolds < 2300:
        return 64.0 / reynolds
    return swamee_jain(reynolds, diameter, eps, math.log10)


def logspace(a, b, n):
    la = math.log10(a)
    lb = math.log10(b)
    return [10 ** (la + i * (lb - la) / (n - 1)) for i in range(n)]


def phi_pwl_breakpoints(
    diameter_m: float,
    roughness_m: float,
    dynamic_visc_pas: float,
    pipe_area: float,
    m_max: float,
    n_breakpoints: int = 12,
):
    r"""Breakpoints for :math:`\varphi(m) = friction(Re(m)) \cdot m^2` on :math:`m \in [0, m_{max}]`.

    Log-spaced :math:`[m_{max} \cdot 10^{-4}, m_{max}]` plus a 0-anchor, so both the laminar
    tail (:math:`\varphi \propto m`) and turbulent regime (:math:`\varphi \propto m^2`) resolve well in a single SOS2.
    """
    if m_max <= 0:
        return [0.0, 1e-6], [0.0, 0.0]

    n_log = max(2, n_breakpoints - 1)
    m_lo = max(m_max * 1e-4, 1e-9)  # avoid Re = 0 in friction_value
    log_xs = logspace(m_lo, m_max, n_log)

    xs = [0.0] + list(log_xs)
    ys = [0.0]
    for m in log_xs:
        reynolds = m * diameter_m / (dynamic_visc_pas * pipe_area)
        f = friction_value(reynolds, diameter_m, roughness_m)
        ys.append(f * m * m)

    xs, ys = filter_near_linear(xs, ys, rtol=1e-3)
    return xs, ys
