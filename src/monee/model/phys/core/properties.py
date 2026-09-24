"""Fluid-property correlations evaluated at a grid's reference operating point.

A grid's transport properties are not free constants: water density and
viscosity depend on temperature, and a real gas's compressibility depends on
pressure.
"""

import math

# Molar mass of dry air [kg/mol], for the gas specific-gravity (gamma_g = M/M_air).
_MOLAR_MASS_AIR = 0.0289647


def water_density_kg_per_m3(t_k: float) -> float:
    r"""Liquid-water density [kg/m^3] at temperature ``t_k`` (Kell, 1975).

    Fifth-order fit at 1 atm, valid 0-150 deg C to ~1e-3 % - reproduces 998.2
    at 20 deg C, 970.0 at 83 deg C, 943.1 at 120 deg C. Pressure dependence over
    DH operating ranges (a few bar to tens of bar) is negligible vs temperature.

    Reference:
        G. S. Kell, "Density, thermal expansivity, and compressibility of liquid
        water from 0 to 150 deg C ...", J. Chem. Eng. Data 20(1), 97-105 (1975).
        DOI 10.1021/je60064a005 - https://pubs.acs.org/doi/10.1021/je60064a005
    """
    t_c = t_k - 273.15
    return (
        999.83952
        + 16.945176 * t_c
        - 7.9870401e-3 * t_c**2
        - 46.170461e-6 * t_c**3
        + 105.56302e-9 * t_c**4
        - 280.54253e-12 * t_c**5
    ) / (1.0 + 16.879850e-3 * t_c)


def water_dynamic_viscosity_pas(t_k: float) -> float:
    r"""Liquid-water dynamic viscosity [Pa s] at ``t_k`` (Vogel-Tammann-Fulcher form).

    :math:`\mu = 2.414\cdot10^{-5}\,\cdot\,10^{247.8/(T-140)}` with ``T`` in K -
    the textbook two-parameter fit, ~1 % over 0-100 deg C (1.00e-3 at 20 deg C,
    3.39e-4 at 83 deg C). Viscosity only enters the Reynolds-dependent friction
    models (``nonlinear``/``pwl``); the ``constant`` model does not use it.

    Reference:
        Vogel-Tammann-Fulcher (VTF) form; constants A=2.414e-5 Pa s, B=247.8 K,
        C=140 K per T. Al-Shemmeri, "Engineering Fluid Mechanics", Ventus
        Publishing (2012), ISBN 978-87-403-0114-4 (also Engineering ToolBox:
        https://www.engineeringtoolbox.com/water-dynamic-kinematic-viscosity-d_596.html).
    """
    if t_k <= 140.0:
        # The VTF form has a pole at 140 K; liquid water is far above this
        # (>= 273 K), so an input at/below it is a programming error, not physics.
        raise ValueError(
            f"water viscosity correlation undefined at t_k={t_k} K "
            "(Vogel-Tammann-Fulcher pole at 140 K)"
        )
    return 2.414e-5 * 10.0 ** (247.8 / (t_k - 140.0))


def gas_compressibility(pressure_pa: float, t_k: float, molar_mass: float) -> float:
    r"""Real-gas compressibility factor Z at ``pressure_pa`` / ``t_k`` (Papay, 1968).

    Explicit natural-gas correlation
    :math:`Z = 1 - 3.52\,p_r e^{-2.260 T_r} + 0.274\,p_r^2 e^{-1.878 T_r}`,
    where the reduced pressure/temperature use pseudo-critical properties
    estimated from the gas specific gravity (Standing, 1977). Z -> 1 as
    p -> 0 (ideal gas) and falls with pressure (0.97-0.98 at ~10 bar, lower at
    transmission pressures). Accurate for natural gas at p_r < ~3, the
    distribution/low-transmission regime monee targets.

    References:
        J. Papay, OGIL Musz. Tud. Kozl., Budapest (1968), pp. 267-273 - explicit
        fit to the Standing-Katz Z chart (e.g. Ahmed, "Reservoir Engineering
        Handbook", 4th ed., Gulf Professional, 2010, ch. 3).
        Pseudo-criticals: M. B. Standing, "Volumetric and Phase Behavior of Oil
        Field Hydrocarbon Systems", SPE, Dallas (1977):
        Tpc = 168 + 325*gamma - 12.5*gamma^2 [R], Ppc = 677 + 15*gamma - 37.5*gamma^2 [psia].
    """
    gamma_g = molar_mass / _MOLAR_MASS_AIR
    # Standing (1977) dry-gas pseudo-criticals: Rankine / psia -> SI.
    t_pc_k = (168.0 + 325.0 * gamma_g - 12.5 * gamma_g**2) / 1.8
    p_pc_pa = (677.0 + 15.0 * gamma_g - 37.5 * gamma_g**2) * 6894.757
    t_r = t_k / t_pc_k
    p_r = pressure_pa / p_pc_pa
    return (
        1.0
        - 3.52 * p_r * math.exp(-2.260 * t_r)
        + 0.274 * p_r**2 * math.exp(-1.878 * t_r)
    )
