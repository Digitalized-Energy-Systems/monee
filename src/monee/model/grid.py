from dataclasses import dataclass

from .core import model
from .phys.core.properties import (
    gas_compressibility,
    water_density_kg_per_m3,
    water_dynamic_viscosity_pas,
)

# Gas/heat energy-rate conversion: power[MW] = mass_flow[kg/s] \cdot HHV[kWh/kg] \cdot 3.6,
# since 1 kWh/s = 3600 kW = 3.6 MW (3600 s/h \div 1000 kW/MW). Used wherever a gas
# mass flow is turned into a power, so the bare 3.6 never appears inline.
KGPS_KWHPERKG_TO_MW = 3.6

# Standard atmosphere [Pa] (sea level). Set a gas grid's ``pressure_ambient_pa``
# to this to make its node pressures GAUGE (absolute = gauge + 1 atm), matching
# pandapipes and the conventional way operating/MAOP pressures are stated.
STANDARD_ATMOSPHERE_PA = 101325.0


@model
@dataclass(unsafe_hash=True)
class Grid:
    """Marker/base for a resource domain (electrical, gas, heat)."""

    name: str


@model
@dataclass(unsafe_hash=True)
class PowerGrid(Grid):
    """Electrical grid domain."""

    sn_mva: float = 1
    vm_pu_max: float = 1.5
    vm_pu_min: float = 0.5


@model
@dataclass(unsafe_hash=True)
class WaterGrid(Grid):
    """Water/heat grid domain.

    ``fluid_density_kg_per_m3`` and ``dynamic_visc_pas`` default to ``None`` and
    are then derived from ``t_ref_k`` via the standard water correlations (see
    :mod:`monee.model.phys.core.properties`), so the transport properties always
    match the grid's declared operating temperature. Pass explicit values to
    override (e.g. to model a non-water fluid or a different reference)."""

    t_ref_k: float = 356
    pressure_ref_pa: float = 1000000
    max_mass_flow_kgs: float = 200
    v_max_mps: float = 5.0
    fluid_density_kg_per_m3: float | None = None
    dynamic_visc_pas: float | None = None

    def __post_init__(self):
        if self.fluid_density_kg_per_m3 is None:
            self.fluid_density_kg_per_m3 = water_density_kg_per_m3(self.t_ref_k)
        if self.dynamic_visc_pas is None:
            self.dynamic_visc_pas = water_dynamic_viscosity_pas(self.t_ref_k)


_LGAS_COMMON = {
    "universal_gas_constant": 8.314,
    "t_k": 300,
    "t_ref_k": 356,
    "pressure_ref_pa": 1000000,
    "nominal_pressure_pu": 1,
    "max_mass_flow_kgs": 20,
    "pressure_squared_pu_max": 1.3,
    "pressure_squared_pu_min": 0.7,
}

GAS_GRID_ATTRS = {
    # Realistic L-gas (N2/CO2-diluted natural gas): heavier and lower-calorific
    # than pure methane. Values match pandapipes' 'lgas' fluid library.
    "lgas": {
        **_LGAS_COMMON,
        "molar_mass": 0.0181138902,
        "dynamic_visc_pas": 1.2086239755175486e-05,
        "higher_heating_value_kwh_per_kg": 11.79011,
    },
    # Methane / H-gas (pure natural gas): lighter and higher-calorific than
    # L-gas. The molar mass / HHV match monee's pre-2026 defaults (M=0.0165
    # kg/mol, HHV=15.3 kWh/kg ~ methane).
    "methane": {
        **_LGAS_COMMON,
        "molar_mass": 0.0165,
        "dynamic_visc_pas": 1.2190162697374919e-05,
        "higher_heating_value_kwh_per_kg": 15.3,
    },
}

# Single source of truth for the default gas heating value (the lgas grid's HHV)
# and its MJ/kg form. Network-sizing heuristics and HHV fallbacks reference these
# instead of re-hardcoding the constant, so they always match the gas physics
# (which reads ``grid.higher_heating_value_kwh_per_kg``). 1 kWh = 3.6 MJ.
DEFAULT_GAS_HHV_KWH_PER_KG = GAS_GRID_ATTRS["lgas"]["higher_heating_value_kwh_per_kg"]
DEFAULT_GAS_HHV_MJ_PER_KG = DEFAULT_GAS_HHV_KWH_PER_KG * 3.6


@model
@dataclass(unsafe_hash=True)
class GasGrid(Grid):
    """Gas grid domain. Construct via :func:`create_gas_grid` for defaults.

    ``compressibility`` defaults to ``None`` and is then derived from the
    reference pressure (``pressure_ref_pa``), temperature (``t_k``) and
    ``molar_mass`` via the Papay real-gas correlation (see
    :mod:`monee.model.phys.core.properties`), so Z reflects the grid's declared
    operating pressure rather than a fixed ideal-gas 1. Pass an explicit value to
    override (e.g. to force ideal gas with ``compressibility=1``)."""

    molar_mass: float
    dynamic_visc_pas: float
    higher_heating_value_kwh_per_kg: float
    universal_gas_constant: float
    t_k: float
    t_ref_k: float
    pressure_ref_pa: float
    nominal_pressure_pu: float
    max_mass_flow_kgs: float
    pressure_squared_pu_max: float
    pressure_squared_pu_min: float
    compressibility: float | None = None
    pressure_ambient_pa: float = 0.0

    def __post_init__(self):
        if self.compressibility is None:
            self.compressibility = gas_compressibility(
                self.pressure_ref_pa, self.t_k, self.molar_mass
            )


@model
@dataclass(unsafe_hash=True)
class NoGrid(Grid):
    """Marker for components not bound to any grid."""


NO_GRID = NoGrid("None")


def create_gas_grid(
    name, type="lgas", t_ref_k=None, pressure_ref_pa=None, pressure_ambient_pa=None
):
    """Return a :class:`GasGrid` populated from ``GAS_GRID_ATTRS[type]``.

    ``t_ref_k`` / ``pressure_ref_pa`` override the reference condition so the
    derived compressibility tracks the grid's actual operating pressure. Pass
    them here rather than mutating the grid afterwards, so ``__post_init__``
    derives Z from the final values (a post-construction assignment leaves the
    already-derived Z stale).

    ``pressure_ambient_pa`` selects the pressure convention: 0 (default) keeps
    node pressures ABSOLUTE; :data:`STANDARD_ATMOSPHERE_PA` makes them GAUGE
    (absolute = gauge + 1 atm), the standard way distribution operating pressures
    are stated. It does not affect the derived Z (which depends on
    ``pressure_ref_pa`` only)."""
    attrs = dict(GAS_GRID_ATTRS[type])
    if t_ref_k is not None:
        attrs["t_ref_k"] = t_ref_k
    if pressure_ref_pa is not None:
        attrs["pressure_ref_pa"] = pressure_ref_pa
    if pressure_ambient_pa is not None:
        attrs["pressure_ambient_pa"] = pressure_ambient_pa
    return GasGrid(name, **attrs)


def create_water_grid(
    name, t_ref_k=WaterGrid.t_ref_k, pressure_ref_pa=WaterGrid.pressure_ref_pa
):
    """Return a :class:`WaterGrid` whose density/viscosity are derived from
    ``t_ref_k`` (see :class:`WaterGrid`). Pass ``t_ref_k`` here rather than
    mutating the grid afterwards, so the derived properties match the final
    reference temperature."""
    return WaterGrid(name, t_ref_k=t_ref_k, pressure_ref_pa=pressure_ref_pa)


def create_power_grid(name, sn_mva=1):
    return PowerGrid(name, sn_mva=sn_mva)
