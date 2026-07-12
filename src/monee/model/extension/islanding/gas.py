"""Gas-carrier islanding mode and grid-forming source model."""

from __future__ import annotations

from monee.model.core import ChildModel, Const, Var, model
from monee.model.grid import GasGrid

from .core import GridFormingMixin, PressureGatedIslandingMode


@model
class GridFormingSource(ChildModel, GridFormingMixin):
    """Grid-forming source: pins pressure (and t_pu/t_k) on the junction, leaves
    mass_flow_kgs as a Var to absorb island imbalance."""

    def __init__(
        self,
        pressure_pu: float = 1.0,
        t_k: float = 356.0,
        mass_flow_max_kgs: float = 1e6,
    ) -> None:
        super().__init__()
        self.mass_flow_kgs = Var(
            0, min=-mass_flow_max_kgs, max=mass_flow_max_kgs, name="gf_mass_flow"
        )
        self._pressure_pu = pressure_pu
        self._t_k = t_k

    def overwrite(self, node_model, grid) -> None:
        # Pin only when leading an island without an ext grid (stamped by
        # IslandingMode.stamp_gf_leadership); a second absolute pressure pin
        # in the same hydraulic island over-determines the drop equations.
        if not getattr(self, "_gf_leading", True):
            return
        node_model.pressure_pu = Const(self._pressure_pu)
        node_model.pressure_squared_pu = Const(self._pressure_pu**2)
        node_model.t_pu = Const(self._t_k / grid.t_ref_k)
        node_model.t_k = Const(self._t_k)

    def equations(self, grid, node_model, **kwargs):
        return []


class GasIslandingMode(PressureGatedIslandingMode):
    r"""Gas islanding: connectivity flow plus :math:`pressure_{pu} \le 2 \cdot e` on regular
    junctions (GF junctions already pin pressure via overwrite())."""

    carrier_grid_type = GasGrid
    var_prefix = "gas"

    def __init__(self, big_m_conn: float | None = None) -> None:
        self.big_m_conn = big_m_conn
