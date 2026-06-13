from dataclasses import dataclass

from .core import model

# Gas/heat energy-rate conversion: power[MW] = mass_flow[kg/s] \cdot HHV[kWh/kg] \cdot 3.6,
# since 1 kWh/s = 3600 kW = 3.6 MW (3600 s/h \div 1000 kW/MW). Used wherever a gas
# mass flow is turned into a power, so the bare 3.6 never appears inline.
KGPS_KWHPERKG_TO_MW = 3.6


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
    # Lower bound applied to bus ``vm_pu`` at node creation. 0 is unphysical and
    # lets an NLP solver (IPOPT) drive the 1/vm current-equation Jacobian to
    # infinity during restoration; a small positive floor keeps it bounded. 0.5
    # is low enough to preserve realistic stressed / load-shedding operating
    # points (e.g. ~0.68 pu) yet still bounds the Jacobian.
    vm_pu_min: float = 0.5


@model
@dataclass(unsafe_hash=True)
class WaterGrid(Grid):
    """Water/heat grid domain."""

    fluid_density_kg_per_m3: float = 998
    dynamic_visc_pas: float = 0.000596
    t_ref_k: float = 356
    pressure_ref_pa: float = 1000000
    max_mass_flow_kgs: float = 200
    # v_max_mps caps per-pipe mass-flow via \pi/4 \cdot D^2 \cdot \rho \cdot v_max; combined with max_mass_flow_kgs
    # by min(...), so it can only tighten. 5 m/s is generous for DH water.
    v_max_mps: float = 5.0


GAS_GRID_ATTRS = {
    "lgas": {
        "compressibility": 1,
        "molar_mass": 0.0165,
        "dynamic_visc_pas": 1.2190162697374919e-05,
        "higher_heating_value_kwh_per_kg": 15.3,
        "universal_gas_constant": 8.314,
        "t_k": 300,
        "t_ref_k": 356,
        "pressure_ref_pa": 1000000,
        "nominal_pressure_pu": 1,
        "max_mass_flow_kgs": 20,
        "pressure_squared_pu_max": 1.3,
        "pressure_squared_pu_min": 0.7,
    }
}


@model
@dataclass(unsafe_hash=True)
class GasGrid(Grid):
    """Gas grid domain. Construct via :func:`create_gas_grid` for defaults."""

    compressibility: float
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


@model
@dataclass(unsafe_hash=True)
class NoGrid(Grid):
    """Marker for components not bound to any grid."""


NO_GRID = NoGrid("None")


def create_gas_grid(name, type="lgas"):
    """Return a :class:`GasGrid` populated from ``GAS_GRID_ATTRS[type]``."""
    return GasGrid(name, **GAS_GRID_ATTRS[type])


def create_water_grid(name):
    return WaterGrid(name)


def create_power_grid(name, sn_mva=1):
    return PowerGrid(name, sn_mva=sn_mva)
