"""Islanding system for multi-carrier grid restoration.

``IslandingMode`` is the per-carrier base (implements :class:`NetworkAspect`);
``NetworkIslandingConfig`` bundles modes for registration via
``network.add_extension()``."""

from __future__ import annotations

from abc import ABC, abstractmethod

from monee.model.child import ExtHydrGrid, ExtPowerGrid, GridFormingMixin
from monee.model.core import Var
from monee.model.extension.core import NetworkAspect
from monee.model.network import Network
from monee.model.phys.islanding import (
    connectivity_arc_capacity_line,
    connectivity_arc_capacity_source,
    connectivity_demand_balance,
    connectivity_super_source_supply,
)


def _real_carrier_components(network: Network, grid_type) -> list[set]:
    """Connected components of the carrier subgraph over the *real* topology
    (active branches with on_off not fixed to 0), matching
    ``generate_real_topology`` semantics."""
    parent: dict = {}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for node in network.nodes:
        if isinstance(node.grid, grid_type) and node.active:
            parent[node.id] = node.id
    for branch in network.branches:
        if not isinstance(branch.grid, grid_type) or not branch.active:
            continue
        on_off = getattr(branch.model, "on_off", None)
        if on_off is not None and type(on_off) is not Var and on_off == 0:
            continue
        if branch.from_node_id in parent and branch.to_node_id in parent:
            parent[find(branch.from_node_id)] = find(branch.to_node_id)

    components: dict = {}
    for nid in parent:
        components.setdefault(find(nid), set()).add(nid)
    return list(components.values())


def node_leads_island(network: Network, node, mode: IslandingMode) -> bool:
    """True if *node* carries the reference child of its island: an ext grid, or
    a grid-forming child stamped as leading by :meth:`IslandingMode.prepare`."""
    for child in network.childs_by_ids(node.child_ids):
        if not mode.is_grid_forming(child):
            continue
        if isinstance(child.model, ExtPowerGrid | ExtHydrGrid):
            return True
        if getattr(child.model, "_gf_leading", True):
            return True
    return False


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
    e.g. angle pinning / pressure bounds. ``big_m_conn`` (set by subclass
    constructors) overrides the connectivity-arc big-M; ``None`` uses the
    network-sized default ``len(nodes) * 10``."""

    carrier_grid_type: type
    var_prefix: str
    big_m_conn: float | None = None
    gated_child_attrs: tuple = ()

    def is_grid_forming(self, child) -> bool:
        return isinstance(child.model, GridFormingMixin) and child.active

    def prepare_common(self, network: Network) -> None:
        """Shared prepare steps: leadership stamping always; injection gating
        and the energisation objective only for plain flow solves (an
        optimization problem sheds via its own vars and applies after
        prepare, so gating there would fight it)."""
        self.stamp_gf_leadership(network)
        if not getattr(network, "_solve_has_optimization_problem", False):
            self.gate_fixed_injections(network)
            self.add_energisation_objective(network)

    def gate_fixed_injections(self, network: Network) -> None:
        """Replace fixed numeric child injections with Vars tied to the host
        node's energisation binary (``var == setpoint * e``, added in the
        equations phase), so islands that cannot balance can de-energise
        nodes instead of rendering the whole solve infeasible."""
        if not self.gated_child_attrs:
            return
        for node in network.nodes:
            if not isinstance(node.grid, self.carrier_grid_type) or not node.active:
                continue
            for child in network.childs_by_ids(node.child_ids):
                if not child.active or self.is_grid_forming(child):
                    continue
                gated = {}
                for attr in self.gated_child_attrs:
                    val = getattr(child.model, attr, None)
                    if not isinstance(val, int | float) or isinstance(val, bool):
                        continue
                    if val == 0:
                        continue
                    gated[attr] = val
                    setattr(
                        child.model,
                        attr,
                        Var(
                            val,
                            min=min(0.0, val),
                            max=max(0.0, val),
                            name=f"islanding_gated_{attr}",
                        ),
                    )
                if gated:
                    child.model._islanding_gated_attrs = gated

    def add_energisation_objective(self, network: Network) -> None:
        """Minimize the number of de-energised nodes so blackout is a last
        resort, never a free alternative to serving reachable load."""
        e_attr = f"e_{self.var_prefix}"
        grid_type = self.carrier_grid_type

        def energisation_penalty(net):
            total = 0
            for n in net.nodes:
                if (
                    not isinstance(n.grid, grid_type)
                    or not n.active
                    or getattr(n, "ignored", False)
                ):
                    continue
                e = getattr(n.model, e_attr, None)
                if e is not None:
                    total = total + (1 - e)
            return total

        network.objectives.append(energisation_penalty)

    def _injection_gate_equations(self, network: Network, nodes, e_vars) -> list:
        eqs = []
        for node in nodes:
            e = e_vars[node.id]
            for child in network.childs_by_ids(node.child_ids):
                if not child.active or getattr(child, "ignored", False):
                    continue
                gated = getattr(child.model, "_islanding_gated_attrs", None)
                if not gated:
                    continue
                for attr, setpoint in gated.items():
                    eqs.append(getattr(child.model, attr) == setpoint * e)
        return eqs

    def stamp_gf_leadership(self, network: Network) -> None:
        """Stamp ``_gf_leading`` on every grid-forming (non-ext) child: an ext
        grid always leads its component, so GF children there must not pin
        voltage/pressure references; a component without an ext grid gets
        exactly one deterministic GF leader."""
        for component in _real_carrier_components(network, self.carrier_grid_type):
            ext_led = False
            gf_children: list = []
            for nid in component:
                node = network.node_by_id(nid)
                for child in network.childs_by_ids(node.child_ids):
                    if not child.active:
                        continue
                    if isinstance(child.model, ExtPowerGrid | ExtHydrGrid):
                        ext_led = True
                    elif isinstance(child.model, GridFormingMixin):
                        gf_children.append((nid, child))
            leader = (
                None if ext_led else min((nid for nid, _ in gf_children), default=None)
            )
            for nid, child in gf_children:
                child.model._gf_leading = nid == leader

    @abstractmethod
    def prepare(self, network: Network) -> None:
        """Add Var placeholders before solver variable injection."""

    def equations(self, network: Network, ignored_nodes: set) -> list:
        gf_nodes, regular_nodes, e_vars, c_fwd_vars, c_rev_vars, c_src_vars = (
            _collect_islanding_state(network, self, ignored_nodes)
        )
        if not e_vars:
            return []
        big_m = (
            self.big_m_conn if self.big_m_conn is not None else len(network.nodes) * 10
        )
        eqs = _build_connectivity_equations(
            network,
            gf_nodes,
            regular_nodes,
            e_vars,
            c_fwd_vars,
            c_rev_vars,
            c_src_vars,
            big_m,
        )
        eqs += self.add_physical_constraints(network, gf_nodes, regular_nodes, e_vars)
        eqs += self._injection_gate_equations(
            network, gf_nodes + regular_nodes, e_vars
        )
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
