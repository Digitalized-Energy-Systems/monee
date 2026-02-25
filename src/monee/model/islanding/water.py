"""
Water-carrier islanding mode.

Water networks share the same junction model (``Junction`` with pressure_pu) and
the same grid-forming child (``GridFormingSource``) as gas networks.  The only
difference is the ``carrier_grid_type`` class attribute.
"""

from __future__ import annotations

from monee.model.core import Var
from monee.model.grid import WaterGrid
from monee.model.network import Network

from .core import GridFormingMixin, IslandingMode


class WaterIslandingMode(IslandingMode):
    """
    Islanding configuration for the water/heat carrier.

    Structurally identical to ``GasIslandingMode``; pressure bounds are applied to
    de-energised junctions so that ``pressure_pu = 0`` when ``e_i = 0``.

    Use ``GridFormingSource`` (from ``monee.model.islanding.gas``) as the grid-forming
    child for water junctions.

    Parameters
    ----------
    big_m_conn : int
        Big-M for connectivity-flow arc capacity.  Must be ≥ number of carrier nodes.
    """

    carrier_grid_type = WaterGrid
    var_prefix = "water"

    def __init__(self, big_m_conn: int = 200) -> None:
        self.big_m_conn = big_m_conn

    def prepare(self, network: Network) -> None:
        """Phase 1 — add water islanding Var placeholders."""
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
        """Force ``pressure_pu = 0`` for de-energised water junctions (returned as list)."""
        eqs = []
        for node in regular_nodes:
            e = e_vars[node.id]
            # Junction.pressure_pu has max=2 in per-unit; use 2.0 as the big-M.
            eqs.append(node.model.pressure_pu <= 2.0 * e)
        return eqs
