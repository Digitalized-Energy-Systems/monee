import math

import numpy as np

REY_BINS = [
    50,
    100,
    200,
    400,
    800,
    1200,
    1600,
    2000,
    2200,
    2400,
    2600,
    2800,
    3000,
    3200,
    3500,
    3800,
    4200,
    4600,
    5000,
    6000,
    7000,
    8000,
    1e4,
    1.5e4,
    2e4,
    3e4,
    5e4,
    1e5,
    2e5,
    5e5,
    1e6,
    2e6,
    5e6,
    1e7,
]


def calc_pipe_area(diameter_m):
    return math.pi * diameter_m**2 / 4


def calc_max_mass_flow(diameter_m, fluid_density, v_max_mps):
    """Physical upper bound on pipe mass flow [kg/s] from a velocity cap.

    ``v_max_mps`` is the maximum fluid velocity allowed in the pipe; for
    district-heating water 3–5 m/s is typical (noise/erosion limit).  The
    resulting bound is used for per-pipe big-M tightening so Gurobi's LP
    relaxation and presolve see the actual physical capacity of each pipe
    rather than a grid-wide worst-case.
    """
    return calc_pipe_area(diameter_m) * fluid_density * v_max_mps


def calc_nikurdse(internal_diameter_m, roughness):
    return 1 / (2 * np.log10(3.71 * internal_diameter_m / roughness)) ** 2


# ``model.reynolds`` is stored in "millions" — i.e. Re / 1e6 — so the
# friction PWL breakpoints land in [0, 10] instead of [0, 1e7].  Without
# this scaling the SOS2 lambda formulation puts coefficients of ~1e7 in the
# constraint matrix (the rightmost breakpoint), which combines with the
# 1e-6 pressure_pa coefficients to span 13 orders of magnitude — far
# enough that Gurobi flags ``Model contains large matrix coefficient
# range``.  Friction values themselves are still computed at the physical
# Reynolds (xs are scaled at PWL build, ys are computed at unscaled Re).
REYNOLDS_SCALE = 1e6


def reynolds_equation(rey_var, mass_flow, diameter_m, dynamic_visc, pipe_area):
    return rey_var == mass_flow * diameter_m / (
        dynamic_visc * pipe_area * REYNOLDS_SCALE
    )


def junction_mass_flow_balance(flows):
    return sum(flows) == 0


def pipe_mass_flow(max_v, min_v, v):
    return min_v <= v <= max_v


def flow_rate_equation(mean_flow_velocity, flow_rate, diameter, fluid_density):
    return mean_flow_velocity == flow_rate / (
        fluid_density * (diameter**2 * math.pi / 4)
    )


def swamee_jain(reynolds_var, diameter_m, roughness, log_func):
    term1 = roughness / diameter_m / 3.7
    term2 = 5.74 / (reynolds_var + 1) ** 0.9  # avoid infeasaiblity at Re=0
    denominator = log_func(term1 + term2) ** 2
    f = 0.25 / denominator
    return f


def friction_at_high_re(diameter_m: float, roughness: float) -> float:
    """Asymptotic turbulent friction factor — Swamee-Jain at ``Re → ∞``.

    Drops the ``5.74 / Re^0.9`` term, leaving the roughness-only Colebrook
    limit ``f = 0.25 / log₁₀(ε / (3.7·D))²``.  Useful for formulations that
    pin friction to a constant (rather than carrying a friction Var + a
    Reynolds Var + a friction PWL): in turbulent operation friction is
    weakly Reynolds-dependent and the asymptotic value is within a few %
    of the true Colebrook root for ``Re ≳ 10⁴``.

    Off-design caveat: at low flow (laminar regime, ``Re < 2300``) the
    physical friction factor goes to ``64/Re`` and is much larger than
    this asymptote.  Models that pin friction here will under-estimate
    pressure drop on lightly-loaded pipes.

    Degenerate inputs: ``D == 0`` (placeholder pipes inserted during
    compound deactivation, see ``solver/core.py``) or ``ε >= 3.7·D``
    (very rough pipe) make the formula undefined / divergent — return
    ``0`` so the caller can still pin a Const without crashing.
    """
    if diameter_m <= 0 or roughness <= 0:
        return 0.0
    term1 = roughness / diameter_m / 3.7
    if term1 >= 1.0:
        return 0.0
    return 0.25 / math.log10(term1) ** 2


def churchill_friction(Re, D, eps):
    Re = max(Re, 1.0)
    A = (2.457 * math.log(1.0 / ((7.0 / Re) ** 0.9 + 0.27 * eps / D))) ** 16
    B = (37530.0 / Re) ** 16
    return 8.0 * ((8.0 / Re) ** 12 + 1.0 / (A + B) ** 1.5) ** (1.0 / 12.0)


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


def friction_value(Re, D, eps):
    if Re < 2300:
        return 64.0 / Re
    return swamee_jain(Re, D, eps, math.log10)


def logspace(a, b, n):
    la = math.log10(a)
    lb = math.log10(b)
    return [10 ** (la + i * (lb - la) / (n - 1)) for i in range(n)]


def piecewise_eq_friction(model, pwl):
    D = model.diameter_m
    eps = model.roughness

    xs = []

    # intentionally coarse below 2000
    xs += logspace(10.0, 2000.0, 4)

    # modest resolution in transition
    xs += logspace(2000.0, 4000.0, 4)[1:]

    # more detail in turbulent regime
    xs += logspace(4000.0, 1e7, 4)[1:]

    # Friction is computed at the *physical* Reynolds (the y-values stay
    # the same), but the PWL x-axis is the rescaled Var ``model.reynolds``
    # (Re / REYNOLDS_SCALE), so the SOS2 breakpoints land in [0, 10].
    ys = [friction_value(x, D, eps) for x in xs]
    xs = [x / REYNOLDS_SCALE for x in xs]

    # Anchor a breakpoint at Re = 0 so a branch with no flow remains feasible.
    # The Weymouth term is friction * mass_flow^2 ≡ 0 when mass_flow = 0, so the
    # interpolated friction at Re = 0 has no physical effect — we just need the
    # PWL domain to include 0.  Re-use the smallest tabulated friction to keep
    # the slope between (0, friction(10)) finite and well-behaved.
    xs = [0.0] + xs
    ys = [ys[0]] + ys

    xs, ys = filter_near_linear(xs, ys, rtol=1e-7)

    pwl.piecewise_eq(
        y=model.friction,
        x=model.reynolds,
        xs=xs,
        ys=ys,
    )


def phi_pwl_breakpoints(
    diameter_m: float,
    roughness: float,
    dynamic_visc: float,
    pipe_area: float,
    m_max: float,
    n_breakpoints: int = 12,
):
    """Breakpoints for ``φ(m) = friction(Re(m)) · m²`` on ``m ∈ [0, m_max]``.

    ``φ`` is the per-pipe pressure-drop kernel that appears in Weymouth /
    Darcy-Weisbach.  Replacing the bilinear ``friction · m²`` with a PWL
    of ``φ`` against ``m`` collapses the two-variable non-convexity into a
    single SOS2 piecewise-linear constraint while preserving the full
    Reynolds dependence of friction (laminar 64/Re ⇒ φ ∝ m, turbulent
    asymptote ⇒ φ ∝ m²).

    Breakpoint placement: log-spaced from ``m_max·1e-4`` to ``m_max`` so
    both the laminar tail (small ``m``) and the turbulent regime resolve
    well, with a 0-anchor so the PWL domain includes zero flow.  The
    log spacing is essential because the laminar-to-turbulent regime
    transition spans 3+ orders of magnitude in ``Re``; uniform spacing
    in ``m`` would put almost all breakpoints in the turbulent regime
    and miss the laminar curvature.
    """
    if m_max <= 0:
        return [0.0, 1e-6], [0.0, 0.0]

    n_log = max(2, n_breakpoints - 1)
    m_lo = max(m_max * 1e-4, 1e-9)  # avoid Re = 0 in friction_value
    log_xs = logspace(m_lo, m_max, n_log)

    xs = [0.0] + list(log_xs)
    ys = [0.0]
    for m in log_xs:
        Re = m * diameter_m / (dynamic_visc * pipe_area)
        f = friction_value(Re, diameter_m, roughness)
        ys.append(f * m * m)

    xs, ys = filter_near_linear(xs, ys, rtol=1e-3)
    return xs, ys
