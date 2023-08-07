from .core import model
from dataclasses import dataclass


@model
@dataclass
class Grid:
    name: str


@model
@dataclass
class PowerGrid(Grid):
    sn_mva: float = 1


@model
@dataclass
class WaterGrid(Grid):
    fluid_density: float = 1
    dynamic_visc: float = 0.000596


GAS_GRID_ATTRS = {
    "lgas": {
        "compressibility": 1,
        "molar_mass": 18.1138902,
        "gas_temperature": 300,
        "dynamic_visc": 1.2190162697374919e-05,
        "higher_heating_value": 0.0116,
    }
}


@model
@dataclass
class GasGrid(Grid):
    compressibility: float
    molar_mass: float
    gas_temperature: float
    dynamic_visc: float
    higher_heating_value: float


@model
@dataclass
class NoGrid(Grid):
    pass


NO_GRID = NoGrid("None")


def create_gas_grid(name, type):
    return GasGrid(name, **GAS_GRID_ATTRS[type])


def create_water_grid(name):
    return WaterGrid(name)


def create_power_grid(name, sn_mva=1):
    return PowerGrid(name, sn_mva=sn_mva)
