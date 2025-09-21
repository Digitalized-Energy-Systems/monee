from dataclasses import dataclass

from .core import model


@model
@dataclass(unsafe_hash=True)
class Grid:
    name: str


@model
@dataclass(unsafe_hash=True)
class PowerGrid(Grid):
    sn_mva: float = 1


@model
@dataclass(unsafe_hash=True)
class WaterGrid(Grid):
    fluid_density: float = (
        998  # use better approximation for the accordings temperatures
    )
    dynamic_visc: float = (
        0.000596  # use better approximation for the according temperatures
    )
    t_ref: float = 356  # slight lower value than the typical 359 as (mostly)
    pressure_ref: float = 1000000


GAS_GRID_ATTRS = {
    "lgas": {
        "compressibility": 1,
        "molar_mass": 0.0165,
        "gas_temperature": 300,
        "dynamic_visc": 1.2190162697374919e-05,
        "higher_heating_value": 15.3,  # kWh/kg,
        "universal_gas_constant": 8.314,
        "t_k": 300,
        "t_ref": 356,
        "pressure_ref": 1000000,
    }
}


@model
@dataclass(unsafe_hash=True)
class GasGrid(Grid):
    compressibility: float
    molar_mass: float
    gas_temperature: float
    dynamic_visc: float
    higher_heating_value: float
    universal_gas_constant: float
    t_k: float
    t_ref: float
    pressure_ref: float


@model
@dataclass(unsafe_hash=True)
class NoGrid(Grid):
    pass


NO_GRID = NoGrid("None")


def create_gas_grid(name, type="lgas"):
    return GasGrid(name, **GAS_GRID_ATTRS[type])


def create_water_grid(name):
    return WaterGrid(name)


def create_power_grid(name, sn_mva=1):
    return PowerGrid(name, sn_mva=sn_mva)
