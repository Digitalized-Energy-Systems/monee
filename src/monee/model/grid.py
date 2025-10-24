from dataclasses import dataclass

from .core import model


@model
@dataclass(unsafe_hash=True)
class Grid:
    """
    Represents a resource grid or domain within a multi-energy network model.

    The Grid class serves as an abstract base or marker for specific grid types, such as electrical, gas, or heat grids. It provides a common interface for identifying and managing different resource domains in network simulations or optimization frameworks. Use this class as a foundation for implementing grid-specific logic, attributes, or methods.

    Attributes:
        name (str): The unique name or identifier for the grid domain (e.g., "el", "gas", "heat").

    Example:
        class ElectricalGrid(Grid):
            def __init__(self):
                self.name = "el"

        grid = ElectricalGrid()
        print(grid.name)  # Output: "el"
    """

    name: str


@model
@dataclass(unsafe_hash=True)
class PowerGrid(Grid):
    """
    Represents an electrical power grid domain within a multi-energy network model.

    PowerGrid extends the base Grid class to encapsulate properties specific to electrical networks, such as the system base power (sn_mva). Use this class to model electrical infrastructure, perform power flow calculations, or integrate electrical dynamics into multi-domain simulations. It is typically instantiated as part of a larger network model where electrical transmission, distribution, or sector coupling is required.

    Attributes:
        sn_mva (float): System base power in megavolt-amperes (MVA). Used for per-unit normalization and power system calculations. Defaults to 1.

    Example:
        power_grid = PowerGrid(name="el", sn_mva=100)
        print(power_grid.name)    # Output: "el"
        print(power_grid.sn_mva)  # Output: 100
    """

    sn_mva: float = 1


@model
@dataclass(unsafe_hash=True)
class WaterGrid(Grid):
    """
    Represents a water grid domain within a multi-energy network model, encapsulating key physical properties for hydraulic simulations.

    WaterGrid extends the base Grid class to provide attributes specific to water networks, such as fluid density, dynamic viscosity, reference temperature, and reference pressure. Use this class to model water infrastructure, perform hydraulic calculations, or integrate water system dynamics into multi-domain simulations. It is typically instantiated as part of a larger network model where water transport, distribution, or sector coupling is required.

    Attributes:
        fluid_density (float): Density of water in kg/m³. Defaults to 998.
        dynamic_visc (float): Dynamic viscosity of water in Pa·s. Defaults to 0.000596.
        t_ref (float): Reference temperature in Kelvin. Defaults to 356.
        pressure_ref (float): Reference pressure in Pascals. Defaults to 1,000,000.

    Example:
        water_grid = WaterGrid()
        print(water_grid.fluid_density)   # Output: 998
        print(water_grid.dynamic_visc)    # Output: 0.000596
    """

    fluid_density: float = 998
    dynamic_visc: float = 0.000596
    t_ref: float = 356
    pressure_ref: float = 1000000


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
    }
}


@model
@dataclass(unsafe_hash=True)
class GasGrid(Grid):
    """
    Represents a gas grid domain within a multi-energy network model, encapsulating physical and thermodynamic properties of the gas system.

    GasGrid extends the base Grid class to provide attributes and parameters specific to gas networks, such as compressibility, molar mass, temperature, and reference pressures. Use this class to model gas infrastructure, perform flow calculations, or integrate gas dynamics into multi-domain simulations. It is typically instantiated as part of a larger network model where gas transport, conversion, or coupling with other energy domains is required.

    Attributes:
        compressibility (float): Compressibility factor of the gas.
        molar_mass (float): Molar mass of the gas (kg/mol).
        gas_temperature (float): Operating temperature of the gas (K).
        dynamic_visc (float): Dynamic viscosity of the gas (Pa·s).
        higher_heating_value (float): Higher heating value of the gas (J/kg).
        universal_gas_constant (float): Universal gas constant (J/(mol·K)).
        t_k (float): Absolute temperature in Kelvin for calculations.
        t_ref (float): Reference temperature in Kelvin.
        pressure_ref (float): Reference pressure in Pascals.

    Example:
        gas_grid = GasGrid()
        gas_grid.compressibility = 0.98
        gas_grid.molar_mass = 0.016
        gas_grid.gas_temperature = 293.15
        # ... set other properties as needed
    """

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
    """
    Represents a placeholder or null grid domain within a multi-energy network model.

    NoGrid is a special subclass of Grid used to indicate the absence of a specific resource domain or to serve as a default when no grid assignment is required. Use this class in scenarios where a component, node, or branch does not belong to any physical grid, or when you need to explicitly represent "no grid" in network configuration or logic.

    Example:
        grid = NoGrid()
        # Use grid as a marker for components not associated with any resource domain
    """


NO_GRID = NoGrid("None")


def create_gas_grid(name, type="lgas"):
    """
    Creates and returns a GasGrid instance with predefined attributes for a specified gas grid type.

    This function streamlines the instantiation of a GasGrid by automatically applying a set of standard physical and thermodynamic properties based on the selected grid type (e.g., 'lgas'). Use this function when you need to quickly set up a gas grid domain for simulation, analysis, or integration into a multi-energy network model. The function retrieves the appropriate attribute set from GAS_GRID_ATTRS and passes them to the GasGrid constructor.

    Args:
        name (str): The unique name or identifier for the gas grid.
        type (str, optional): The type of gas grid to create (e.g., 'lgas'). Must be a key in GAS_GRID_ATTRS. Defaults to 'lgas'.

    Returns:
        GasGrid: An instance of GasGrid initialized with the specified name and standard attributes for the chosen type.

    Raises:
        KeyError: If the specified type is not found in GAS_GRID_ATTRS.

    Examples:
        # Create a default low-pressure gas grid
        gas_grid = create_gas_grid('city_gas')

        # Create a high-pressure gas grid if defined in GAS_GRID_ATTRS
        gas_grid = create_gas_grid('transmission_gas', type='hgas')
    """
    return GasGrid(name, **GAS_GRID_ATTRS[type])


def create_water_grid(name):
    """
    Creates and returns a WaterGrid instance with the specified name.

    Use this function to instantiate a water grid domain for use in network modeling, hydraulic simulation, or multi-domain energy system analysis. The function initializes a WaterGrid object with default physical properties and assigns the provided name for identification within the network.

    Args:
        name (str): The unique name or identifier for the water grid.

    Returns:
        WaterGrid: An instance of WaterGrid initialized with the given name and default attributes.

    Examples:
        # Create a water grid named "district_water"
        grid = create_water_grid('district_water')
    """
    return WaterGrid(name)


def create_power_grid(name, sn_mva=1):
    """
    Creates and returns a PowerGrid instance with the specified name and system base power.

    Use this function to instantiate an electrical power grid domain for use in network modeling, simulation, or optimization. The function allows you to specify the grid's unique name and the system base power (sn_mva), which is essential for per-unit calculations and power system analysis.

    Args:
        name (str): The unique name or identifier for the power grid.
        sn_mva (float, optional): System base power in megavolt-amperes (MVA). Defaults to 1.

    Returns:
        PowerGrid: An instance of PowerGrid initialized with the given name and base power.

    Examples:
        # Create a default power grid with 1 MVA base
        grid = create_power_grid('el')

        # Create a power grid with a custom base power
        grid = create_power_grid('transmission', sn_mva=100)
    """
    return PowerGrid(name, sn_mva=sn_mva)
