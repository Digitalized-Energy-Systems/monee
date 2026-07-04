"""Electricity-carrier islanding mode and grid-forming generator model."""

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

from .core import GridFormingMixin, IslandingMode, node_leads_island


@model
class GridFormingGenerator(NoVarChildModel, GridFormingMixin):
    """Grid-forming generator: variable p_mw/q_mvar (absorbs island imbalance)
    with a pinned vm_pu. Angle is pinned by :class:`ElectricityIslandingMode`.

    The vm pin applies only when this child leads its island
    (``_gf_leading``, stamped by :meth:`IslandingMode.stamp_gf_leadership`):
    on an ext-grid-led component an unconditional pin would over-constrain
    the power flow (no voltage gradient, so no transport)."""

    def __init__(
        self, p_mw_max: float, q_mvar_max: float, vm_pu: float = 1.0, **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.p_mw = Var(0, min=-p_mw_max, max=p_mw_max, name="gf_p_mw")
        self.q_mvar = Var(0, min=-q_mvar_max, max=q_mvar_max, name="gf_q_mvar")
        self._vm_pu_setpoint = vm_pu

    def overwrite(self, node_model, grid) -> None:
        if not getattr(self, "_gf_leading", True):
            return
        node_model.vm_pu = Const(self._vm_pu_setpoint)
        node_model.vm_pu_squared = Const(self._vm_pu_setpoint**2)


class ElectricityIslandingMode(IslandingMode):
    r"""Electricity islanding: connectivity flow plus :math:`\theta=0` at GF buses and
    energisation-gated angle bounds elsewhere."""

    carrier_grid_type = PowerGrid
    var_prefix = "el"
    gated_child_attrs = ("p_mw", "q_mvar")

    def __init__(
        self, angle_bound: float = 3.15, big_m_conn: float | None = None
    ) -> None:
        self.angle_bound = angle_bound
        self.big_m_conn = big_m_conn

    def prepare(self, network: Network) -> None:
        self.prepare_common(network)
        for node in network.nodes:
            if isinstance(node.grid, PowerGrid) and node.active:
                # Claim bus-angle management: this mode pins \theta=0 at GF buses and
                # applies energisation-gated bounds elsewhere, so the grid-former
                # must NOT statically pin va_radians in its overwrite().
                node.model._islanding_angle_managed = True
                node.model.e_el = Var(1, min=0, max=1, integer=True, name="e_el")
                is_gf = any(
                    self.is_grid_forming(c)
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
        eqs = []
        bounded_nodes = list(regular_nodes)
        for node in gf_nodes:
            # theta=0 only at the island reference; a second pin in the same
            # island over-constrains the flow.
            if node_leads_island(network, node, self):
                eqs.append(source_reference_angle(node.model.va_radians))
            else:
                bounded_nodes.append(node)
        for node in bounded_nodes:
            e = e_vars[node.id]
            eqs.append(
                angle_upper_bound_energized(node.model.va_radians, self.angle_bound, e)
            )
            eqs.append(
                angle_lower_bound_energized(node.model.va_radians, self.angle_bound, e)
            )
        return eqs
