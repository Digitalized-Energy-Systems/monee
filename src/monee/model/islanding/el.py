"""
Electricity-carrier islanding mode and grid-forming generator model.
"""

from __future__ import annotations

from monee.model.child import NoVarChildModel
from monee.model.core import Const, Var, model
from monee.model.grid import PowerGrid
from monee.model.network import Network
from monee.model.phys.islanding import (
    angle_lower_bound_energized,
    angle_upper_bound_energized,
    source_reference_angle,
)

from .core import GridFormingMixin, IslandingMode

# ---------------------------------------------------------------------------
# GridFormingGenerator — grid-forming child model for electricity
# ---------------------------------------------------------------------------


@model
class GridFormingGenerator(NoVarChildModel, GridFormingMixin):
    """
    Grid-forming generator for islanded electricity networks.

    Acts as the slack bus of its island: variable active/reactive power output
    (absorbs the island's power imbalance) with a fixed voltage magnitude setpoint.

    Unlike ``PowerGenerator`` (which has *fixed* p/q), this component has *variable*
    ``p_mw`` and ``q_mvar`` so that it can balance any generation–load mismatch in its
    island.  The voltage angle is **not** pinned here; instead, the
    ``ElectricityIslandingMode`` formulation pins the angle to 0 via
    ``source_reference_angle`` from ``dc.py``.

    Parameters
    ----------
    p_mw_max : float
        Maximum active power injection (and absorption) in MW.
    q_mvar_max : float
        Maximum reactive power injection (and absorption) in Mvar.
    vm_pu : float
        Voltage magnitude setpoint in per-unit.  The ``overwrite()`` method pins
        ``node_model.vm_pu`` to this constant, making this bus a PV/slack bus.
    """

    def __init__(
        self, p_mw_max: float, q_mvar_max: float, vm_pu: float = 1.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.p_mw = Var(0, min=-p_mw_max, max=p_mw_max, name="gf_p_mw")
        self.q_mvar = Var(0, min=-q_mvar_max, max=q_mvar_max, name="gf_q_mvar")
        self._vm_pu_setpoint = vm_pu

    def overwrite(self, node_model, grid) -> None:
        """Pin the bus voltage magnitude (like ``ExtPowerGrid.overwrite``)."""
        node_model.vm_pu = Const(self._vm_pu_setpoint)
        node_model.vm_pu_squared = Const(self._vm_pu_setpoint**2)
        # Angle is NOT pinned here — the islanding formulation does it.


# ---------------------------------------------------------------------------
# ElectricityIslandingMode
# ---------------------------------------------------------------------------


class ElectricityIslandingMode(IslandingMode):
    """
    Islanding configuration for the electricity carrier.

    Adds:
    * Single-commodity connectivity-flow constraints (via the base class).
    * ``source_reference_angle`` at every grid-forming bus → θ = 0.
    * ``angle_{upper,lower}_bound_energized`` at every regular bus →
      forces θ = 0 when ``e_i = 0`` (de-energised).

    Parameters
    ----------
    angle_bound : float
        Maximum absolute angle (radians) for energised buses.  Default ≈ π.
    big_m_conn : int
        Big-M for connectivity-flow arc capacity.  Must be ≥ number of carrier nodes.
    """

    carrier_grid_type = PowerGrid
    var_prefix = "el"

    def __init__(self, angle_bound: float = 3.15, big_m_conn: int = 200) -> None:
        self.angle_bound = angle_bound
        self.big_m_conn = big_m_conn

    def prepare(self, network: Network) -> None:
        """Phase 1 — add electricity islanding Var placeholders."""
        for node in network.nodes:
            if isinstance(node.grid, PowerGrid) and node.active:
                node.model.e_el = Var(1, min=0, max=1, integer=True, name="e_el")
                is_gf = any(
                    isinstance(c.model, GridFormingMixin) and c.active
                    for c in network.childs_by_ids(node.child_ids)
                )
                if is_gf:
                    node.model.c_src_el = Var(1, min=0, name="c_src_el")
        for branch in network.branches:
            if isinstance(branch.grid, PowerGrid) and branch.active:
                branch.model.c_el_fwd = Var(0, min=0, name="c_el_fwd")
                branch.model.c_el_rev = Var(0, min=0, name="c_el_rev")

    def add_physical_constraints(
        self, network, gf_nodes, regular_nodes, e_vars
    ) -> list:
        """
        Electricity-specific physical constraint equations (returned as a list).

        * Grid-forming nodes: angle reference = 0 (``source_reference_angle`` from dc.py).
        * Regular nodes: angle bounds conditional on energisation.
        """
        eqs = []
        for node in gf_nodes:
            eqs.append(source_reference_angle(node.model.va_radians))
        for node in regular_nodes:
            e = e_vars[node.id]
            eqs.append(
                angle_upper_bound_energized(node.model.va_radians, self.angle_bound, e)
            )
            eqs.append(
                angle_lower_bound_energized(node.model.va_radians, self.angle_bound, e)
            )
        return eqs
