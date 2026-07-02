"""Water-carrier islanding mode (structurally identical to :class:`GasIslandingMode`)."""

from __future__ import annotations

from monee.model.core import Var
from monee.model.grid import WaterGrid
from monee.model.network import Network

from .core import GridFormingMixin, IslandingMode


class WaterIslandingMode(IslandingMode):
    """Water/heat islanding. Use :class:`GridFormingSource` from the gas module."""

    carrier_grid_type = WaterGrid
    var_prefix = "water"

    def __init__(self, big_m_conn: float | None = None) -> None:
        self.big_m_conn = big_m_conn

    def prepare(self, network: Network) -> None:
        for node in network.nodes:
            if isinstance(node.grid, WaterGrid) and node.active:
                node.model.e_water = Var(1, min=0, max=1, integer=True, name="e_water")
                is_gf = any(
                    isinstance(c.model, GridFormingMixin) and c.active
                    for c in network.childs_by_ids(node.child_ids)
                )
                if is_gf:
                    node.model.c_src_water = Var(1, min=0, name="c_src_water")
        for branch in network.branches:
            if isinstance(branch.grid, WaterGrid) and branch.active:
                branch.model.c_water_fwd = Var(0, min=0, name="c_water_fwd")
                branch.model.c_water_rev = Var(0, min=0, name="c_water_rev")

    def add_physical_constraints(
        self, network, gf_nodes, regular_nodes, e_vars
    ) -> list:
        eqs = []
        for node in regular_nodes:
            e = e_vars[node.id]
            eqs.append(node.model.pressure_pu <= 2.0 * e)
        return eqs
