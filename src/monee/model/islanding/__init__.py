"""
Islanding system for monee — multi-carrier grid restoration.

Public API
----------
Core abstractions::

    from monee.model.islanding import (
        GridFormingMixin,
        IslandingMode,
        NetworkIslandingConfig,
    )

Per-carrier modes and grid-forming child models::

    from monee.model.islanding import (
        ElectricityIslandingMode,
        GridFormingGenerator,     # electricity
        GasIslandingMode,
        GridFormingSource,        # gas and water
        WaterIslandingMode,
    )

Convenience::

    from monee import enable_islanding          # top-level helper
    enable_islanding(network, electricity=True)

"""

from .core import GridFormingMixin, IslandingMode, NetworkIslandingConfig
from .el import ElectricityIslandingMode, GridFormingGenerator
from .gas import GasIslandingMode, GridFormingSource
from .water import WaterIslandingMode
