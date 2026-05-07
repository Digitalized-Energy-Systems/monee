"""
Network-level extensions for monee.

Extensions implement :class:`NetworkAspect`: a solver-agnostic interface that
participates in variable injection (``prepare``) and equation registration
(``equations`` / ``inter_step_equations`` / ``inter_temporal_equations``).
Attach to a network with ``network.add_extension(...)``.

Public API
----------
All public names are re-exported at the top level (``import monee``), so
user code typically does::

    import monee

    net.add_extension(monee.LumpedThermalCapacitance())
    net.add_extension(monee.GasLinepack())
    monee.enable_islanding(net, electricity=True)

For custom extensions, subclass :class:`NetworkAspect` (also at the top
level) and ``net.add_extension(MyExtension())``.

The full surface — built-in extensions, base class, and the islanding
system — is also reachable as ``monee.model.extension.*`` if you prefer
the qualified path.
"""

from .core import NetworkAspect
from .ltc import LumpedThermalCapacitance
from .linepack import GasLinepack
from .islanding import (
    ElectricityIslandingMode,
    GasIslandingMode,
    GridFormingGenerator,
    GridFormingMixin,
    GridFormingSource,
    IslandingMode,
    NetworkIslandingConfig,
    WaterIslandingMode,
)
