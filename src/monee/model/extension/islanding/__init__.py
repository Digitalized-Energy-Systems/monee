"""
Islanding system for monee — multi-carrier grid restoration.

All public names are also re-exported at the top level (``import monee``)
and from :mod:`monee.model.extension`, so user code typically does::

    import monee

    monee.enable_islanding(network, electricity=True)
    gen = monee.GridFormingGenerator(p_mw_max=5.0, q_mvar_max=2.0)

The full API also remains importable from this submodule::

    from monee.model.extension.islanding import (
        GridFormingMixin,
        IslandingMode,
        NetworkIslandingConfig,
        ElectricityIslandingMode,
        GridFormingGenerator,     # electricity
        GasIslandingMode,
        GridFormingSource,        # gas and water
        WaterIslandingMode,
    )
"""

from .core import GridFormingMixin, IslandingMode, NetworkIslandingConfig
from .el import ElectricityIslandingMode, GridFormingGenerator
from .gas import GasIslandingMode, GridFormingSource
from .water import WaterIslandingMode
