from .core import model
from dataclasses import dataclass


@model
@dataclass
class Grid:
    name: str


@model
@dataclass
class PowerGrid(Grid):
    sn_mva: float


@model
@dataclass
class WaterGrid(Grid):
    fluid_density: float
    dynamic_visc: float


@model
@dataclass
class GasGrid(Grid):
    compressibility: float
    molar_mass: float
    gas_temperature: float
    dynamic_visc: float
