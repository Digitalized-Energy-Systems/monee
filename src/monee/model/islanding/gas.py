"""
Gas-carrier islanding mode and grid-forming source model.
"""

from __future__ import annotations

from monee.model.core import ChildModel, Const, Var, model
from monee.model.grid import GasGrid
from monee.model.network import Network

from .core import GridFormingMixin, IslandingMode

# ---------------------------------------------------------------------------
# GridFormingSource — grid-forming child model for gas (and water)
# ---------------------------------------------------------------------------


@model
class GridFormingSource(ChildModel, GridFormingMixin):
    """
    Grid-forming source for islanded gas or water networks.

    Acts as the pressure reference of its island: the node's pressure is pinned to
    ``pressure_pu`` via ``overwrite()``, and the source's ``mass_flow`` is a *variable*
    so it can absorb any supply–demand imbalance in the island.

    Parameters
    ----------
    pressure_pu : float
        Pressure setpoint in per-unit.  Pinned on the junction model by
        ``overwrite()``.
    t_k : float
        Temperature in Kelvin (used to pin ``t_pu`` and ``t_k`` on the junction).
    mass_flow_max : float
        Maximum absolute mass flow (kg/s).  Set to a large value if unconstrained.
    """

    def __init__(
        self,
        pressure_pu: float = 1.0,
        t_k: float = 356.0,
        mass_flow_max: float = 1e6,
    ) -> None:
        super().__init__()
        self.mass_flow = Var(
            0, min=-mass_flow_max, max=mass_flow_max, name="gf_mass_flow"
        )
        self._pressure_pu = pressure_pu
        self._t_k = t_k

    def overwrite(self, node_model, grid) -> None:
        """Pin the junction pressure (like ``ExtHydrGrid.overwrite``)."""
        node_model.pressure_pu = Const(self._pressure_pu)
        node_model.pressure_squared_pu = Const(self._pressure_pu**2)
        node_model.t_pu = Const(self._t_k / grid.t_ref)
        node_model.t_k = Const(self._t_k)

    def equations(self, grid, node_model, **kwargs):
        return []


# ---------------------------------------------------------------------------
# GasIslandingMode
# ---------------------------------------------------------------------------


class GasIslandingMode(IslandingMode):
    """
    Islanding configuration for the gas carrier.

    Adds:
    * Single-commodity connectivity-flow constraints (via the base class).
    * Pressure bounds conditional on energisation for regular junctions:
      ``pressure_pu ≤ p_max · e_i`` so that de-energised junctions have pressure = 0.

    The pressure at grid-forming junctions is already pinned by
    ``GridFormingSource.overwrite()`` (or ``ExtHydrGrid.overwrite()``), so no
    additional pinning is needed for GF nodes in this method.

    Parameters
    ----------
    big_m_conn : int
        Big-M for connectivity-flow arc capacity.  Must be ≥ number of carrier nodes.
    """

    carrier_grid_type = GasGrid
    var_prefix = "gas"

    def __init__(self, big_m_conn: int = 200) -> None:
        self.big_m_conn = big_m_conn

    def prepare(self, network: Network) -> None:
        """Phase 1 — add gas islanding Var placeholders."""
        for node in network.nodes:
            if isinstance(node.grid, GasGrid) and node.active:
                node.model.e_gas = Var(1, min=0, max=1, integer=True, name="e_gas")
                is_gf = any(
                    isinstance(c.model, GridFormingMixin) and c.active
                    for c in network.childs_by_ids(node.child_ids)
                )
                if is_gf:
                    node.model.c_src_gas = Var(1, min=0, name="c_src_gas")
        for branch in network.branches:
            if isinstance(branch.grid, GasGrid) and branch.active:
                branch.model.c_gas_fwd = Var(0, min=0, name="c_gas_fwd")
                branch.model.c_gas_rev = Var(0, min=0, name="c_gas_rev")

    def add_physical_constraints(
        self, network, gf_nodes, regular_nodes, e_vars
    ) -> list:
        """
        Gas-specific physical constraint equations (returned as a list).

        Force ``pressure_pu = 0`` for de-energised junctions via an upper bound
        that is conditional on ``e_i``.
        """
        eqs = []
        for node in regular_nodes:
            e = e_vars[node.id]
            # Junction.pressure_pu has max=2 in per-unit; use 2.0 as the big-M so
            # that when e_i=0 pressure is forced to 0, and when e_i=1 the bound is
            # non-binding (pressure is free up to its existing model bounds).
            eqs.append(node.model.pressure_pu <= 2.0 * e)
        return eqs
