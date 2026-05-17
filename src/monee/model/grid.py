from dataclasses import dataclass

from .core import model


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


@model
@dataclass(unsafe_hash=True)
class WaterGrid(Grid):
    """Water/heat grid domain."""

    fluid_density: float = 998
    dynamic_visc: float = 0.000596
    t_ref: float = 356
    pressure_ref: float = 1000000
    f_max: float = 200
    # v_max_mps caps per-pipe mass-flow via π/4·D²·ρ·v_max; combined with f_max
    # by min(...), so it can only tighten. 5 m/s is generous for DH water.
    v_max_mps: float = 5.0


GAS_GRID_ATTRS = {
    "lgas": {
        "compressibility": 1,
        "molar_mass": 0.0165,
        "gas_temperature": 300,
        "dynamic_visc": 1.2190162697374919e-05,
        "higher_heating_value": 15.3,
        "universal_gas_constant": 8.314,
        "t_k": 300,
        "t_ref": 356,
        "pressure_ref": 1000000,
        "nominal_pressure_pu": 1,
        "f_max": 20,
        "p_squared_pu_max": 1.3,
        "p_squared_pu_min": 0.7,
    }
}


@model
@dataclass(unsafe_hash=True)
class GasGrid(Grid):
    """Gas grid domain. Construct via :func:`create_gas_grid` for defaults."""

    compressibility: float
    molar_mass: float
    gas_temperature: float
    dynamic_visc: float
    higher_heating_value: float
    universal_gas_constant: float
    t_k: float
    t_ref: float
    pressure_ref: float
    nominal_pressure_pu: float
    f_max: float
    p_squared_pu_max: float
    p_squared_pu_min: float


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
