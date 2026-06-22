"""Islanding system for multi-carrier grid restoration.

``IslandingMode`` is the per-carrier base (implements :class:`NetworkAspect`);
``NetworkIslandingConfig`` bundles modes for registration via
``network.add_extension()``."""

from __future__ import annotations

from abc import ABC, abstractmethod

from monee.model.child import GridFormingMixin
from monee.model.extension.core import NetworkAspect
from monee.model.network import Network
from monee.model.phys.islanding import (
    connectivity_arc_capacity_line,
    connectivity_arc_capacity_source,
    connectivity_demand_balance,
    connectivity_super_source_supply,
)


def _collect_islanding_state(network: Network, mode: IslandingMode, ignored_nodes: set):
    """Partition carrier nodes into GF/regular and collect injected vars."""
    prefix = mode.var_prefix
    grid_type = mode.carrier_grid_type
    e_attr = f"e_{prefix}"
    cf_attr = f"c_{prefix}_fwd"
    cr_attr = f"c_{prefix}_rev"
    cs_attr = f"c_src_{prefix}"

    gf_nodes, regular_nodes = [], []
    e_vars: dict = {}
    c_src_vars: dict = {}

    for node in network.nodes:
        if (
            not isinstance(node.grid, grid_type)
            or not node.active
            or node.id in ignored_nodes
        ):
            continue
        childs = network.childs_by_ids(node.child_ids)
        is_gf = any(mode.is_grid_forming(c) for c in childs)
        (gf_nodes if is_gf else regular_nodes).append(node)
        e_vars[node.id] = getattr(node.model, e_attr)
        if is_gf:
            c_src_vars[node.id] = getattr(node.model, cs_attr)

    c_fwd_vars: dict = {}
    c_rev_vars: dict = {}
    for branch in network.branches:
        if not isinstance(branch.grid, grid_type) or not branch.active:
            continue
        if branch.from_node_id in ignored_nodes or branch.to_node_id in ignored_nodes:
            continue
        c_fwd_vars[branch.id] = getattr(branch.model, cf_attr)
        c_rev_vars[branch.id] = getattr(branch.model, cr_attr)

    return gf_nodes, regular_nodes, e_vars, c_fwd_vars, c_rev_vars, c_src_vars


def _branch_inflow_outflow(node, c_fwd_vars, c_rev_vars, network):
    """Return (inflow, outflow) connectivity-flow terms for *node*.
    c_fwd flows from→to; c_rev flows to→from."""
    inflow, outflow = [], []
    for branch_id, c_fwd in c_fwd_vars.items():
        branch = network.branch_by_id(branch_id)
        c_rev = c_rev_vars[branch_id]
        if branch.from_node_id == node.id:
            outflow.append(c_fwd)
            inflow.append(c_rev)
        elif branch.to_node_id == node.id:
            inflow.append(c_fwd)
            outflow.append(c_rev)
    return inflow, outflow


def _build_connectivity_equations(
    network, gf_nodes, regular_nodes, e_vars, c_fwd_vars, c_rev_vars, c_src_vars, big_m
) -> list:
    r"""Single-commodity connectivity flow: GF energised (e=1), arc caps via
    :math:`\text{big\_m} \cdot \text{on\_off}`, per-node balance :math:`\sum_{in} - \sum_{out} = e`, super-source supply
    :math:`\sum c_{src} = \sum e`."""
    eqs = []
    all_nodes = gf_nodes + regular_nodes

    for node in gf_nodes:
        eqs.append(e_vars[node.id] == 1)

    for branch_id, c_fwd in c_fwd_vars.items():
        on_off = network.branch_by_id(branch_id).model.on_off
        eqs.append(connectivity_arc_capacity_line(c_fwd, on_off, big_m))
        eqs.append(connectivity_arc_capacity_line(c_rev_vars[branch_id], on_off, big_m))

    for node in gf_nodes:
        eqs.append(connectivity_arc_capacity_source(c_src_vars[node.id], 1, big_m))

    for node in all_nodes:
        inflow, outflow = _branch_inflow_outflow(node, c_fwd_vars, c_rev_vars, network)
        in_sum = sum(inflow) if inflow else 0
        out_sum = sum(outflow) if outflow else 0
        e = e_vars[node.id]
        if node in gf_nodes:
            eqs.append(
                connectivity_demand_balance(in_sum + c_src_vars[node.id], out_sum, e)
            )
        else:
            eqs.append(connectivity_demand_balance(in_sum, out_sum, e))

    if c_src_vars:
        eqs.append(
            connectivity_super_source_supply(
                sum(c_src_vars.values()), sum(e_vars.values())
            )
        )

    return eqs


class IslandingMode(NetworkAspect, ABC):
    """Per-carrier islanding base. Subclasses set ``carrier_grid_type`` and
    ``var_prefix``, and may override :meth:`add_physical_constraints` to add
    e.g. angle pinning / pressure bounds."""

    carrier_grid_type: type
    var_prefix: str

    def is_grid_forming(self, child) -> bool:
        return isinstance(child.model, GridFormingMixin) and child.active

    @abstractmethod
    def prepare(self, network: Network) -> None:
        """Add Var placeholders before solver variable injection."""

    def equations(self, network: Network, ignored_nodes: set) -> list:
        gf_nodes, regular_nodes, e_vars, c_fwd_vars, c_rev_vars, c_src_vars = (
            _collect_islanding_state(network, self, ignored_nodes)
        )
        if not e_vars:
            return []
        eqs = _build_connectivity_equations(
            network,
            gf_nodes,
            regular_nodes,
            e_vars,
            c_fwd_vars,
            c_rev_vars,
            c_src_vars,
            len(network.nodes) * 10,
        )
        eqs += self.add_physical_constraints(network, gf_nodes, regular_nodes, e_vars)
        return eqs

    def add_physical_constraints(self, *_) -> list:
        """Carrier-specific physics (override in subclasses). Empty by default."""
        return []


class NetworkIslandingConfig(NetworkAspect):
    """Bundle per-carrier :class:`IslandingMode` instances; register via
    ``network.add_extension`` or :func:`enable_islanding`."""

    def __init__(
        self,
        electricity: IslandingMode | None = None,
        gas: IslandingMode | None = None,
        water: IslandingMode | None = None,
    ) -> None:
        self.electricity = electricity
        self.gas = gas
        self.water = water

    def modes(self) -> list[IslandingMode]:
        return [m for m in [self.electricity, self.gas, self.water] if m is not None]

    def prepare(self, network: Network) -> None:
        for mode in self.modes():
            mode.prepare(network)

    def equations(self, network: Network, ignored_nodes: set) -> list:
        eqs = []
        for mode in self.modes():
            eqs += mode.equations(network, ignored_nodes)
        return eqs
