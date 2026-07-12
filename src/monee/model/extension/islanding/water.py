"""Water-carrier islanding mode (structurally identical to :class:`GasIslandingMode`)."""

from __future__ import annotations

from monee.model.grid import WaterGrid

from .core import PressureGatedIslandingMode


class WaterIslandingMode(PressureGatedIslandingMode):
    """Water/heat islanding. Use :class:`GridFormingSource` from the gas module."""

    carrier_grid_type = WaterGrid
    var_prefix = "water"

    def __init__(self, big_m_conn: float | None = None) -> None:
        self.big_m_conn = big_m_conn
