"""
Islanding system for multi-carrier grid restoration.

Carrier-independent backbone:

* ``IslandingMode``          – base class per carrier; implements ``NetworkConstraint``.
* ``NetworkIslandingConfig`` – bundles per-carrier modes; registered via
                               ``network.add_extension()``.

Internal helpers (not part of the public API):

* ``_collect_islanding_state``      – partitions nodes into GF / regular and collects vars.
* ``_build_connectivity_equations`` – carrier-agnostic connectivity-flow constraint builder.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from monee.model.child import GridFormingMixin
from monee.model.formulation.core import NetworkConstraint
from monee.model.network import Network
from monee.model.phys.islanding import (
    connectivity_arc_capacity_line,
    connectivity_arc_capacity_source,
    connectivity_demand_balance,
    connectivity_super_source_supply,
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _collect_islanding_state(network: Network, mode: IslandingMode, ignored_nodes: set):
    """
    Partition carrier nodes into grid-forming (GF) and regular, then collect the
    already-injected solver variable references.

    Returns
    -------
    gf_nodes      : list[Node]
    regular_nodes : list[Node]
    e_vars        : dict[node_id → injected_var]
    c_fwd_vars    : dict[branch_id → injected_var]
    c_rev_vars    : dict[branch_id → injected_var]
    c_src_vars    : dict[node_id → injected_var]   (GF nodes only)
    """
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
    """
    Return (inflow_terms, outflow_terms) of connectivity-flow variables for *node*.

    Convention for a branch (from, to):
      c_fwd flows from → to  (outflow from ``from``, inflow to ``to``)
      c_rev flows to → from  (outflow from ``to``,   inflow to ``from``)
    """
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
    """
    Carrier-independent single-commodity connectivity-flow equations.

    Returns a plain list of relational expressions (no solver-specific calls).

    Constraints:
    1. GF nodes always energized: e_k = 1
    2. Arc capacity:              c_fwd/rev ≤ big_m · on_off
    3. Super-source arc capacity: c_src ≤ big_m  (GF nodes, always enabled)
    4. Per-node balance:          Σ_in c – Σ_out c = e_i
    5. Super-source supply:       Σ c_src = Σ e_i
    """
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


# ---------------------------------------------------------------------------
# IslandingMode — implements NetworkConstraint
# ---------------------------------------------------------------------------


class IslandingMode(NetworkConstraint, ABC):
    """
    Per-carrier islanding configuration.

    Subclasses must set the class attributes ``carrier_grid_type`` and ``var_prefix``,
    and may override ``add_physical_constraints`` to add carrier-specific constraints
    (e.g. angle pinning for DC electricity, pressure bounds for gas/water).

    Implements ``NetworkConstraint``:
    - ``prepare(network)``               → Phase 1: adds Var placeholders.
    - ``equations(network, ignored)``    → Phase 2: returns constraint list.
    """

    carrier_grid_type: type  # e.g. PowerGrid — set in subclass
    var_prefix: str  # e.g. "el"       — set in subclass

    def is_grid_forming(self, child) -> bool:
        """Return True if *child* anchors an island for this carrier."""
        return isinstance(child.model, GridFormingMixin) and child.active

    @abstractmethod
    def prepare(self, network: Network) -> None:
        """
        Phase 1 — add ``Var`` placeholders to node and branch models before
        solver variable injection.

        The normal ``inject_gekko_vars`` / ``inject_pyomo_vars`` loops pick up
        these ``Var`` objects automatically.  Each subclass sets its attributes
        directly, e.g. ``node.model.e_el = Var(...)``.
        """

    def equations(self, network: Network, ignored_nodes: set) -> list:
        """
        Phase 2 — return all islanding constraint equations as a plain list.

        Combines connectivity-flow constraints with carrier-specific physical
        constraints from ``add_physical_constraints``.
        """
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
        """
        Carrier-specific physical constraints (returned as a plain list).

        Override in subclasses to add e.g. angle pinning for electricity or
        pressure bounds for gas/water.  The default returns an empty list.
        """
        return []


# ---------------------------------------------------------------------------
# NetworkIslandingConfig — implements NetworkConstraint
# ---------------------------------------------------------------------------


class NetworkIslandingConfig(NetworkConstraint):
    """
    Container bundling per-carrier ``IslandingMode`` instances.

    Register on a network via ``network.add_extension(config)``, or use the
    top-level ``enable_islanding()`` helper which also sets
    ``network.islanding_config`` for ``find_ignored_nodes`` compatibility.
    """

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
        """Return the list of active (non-None) carrier modes."""
        return [m for m in [self.electricity, self.gas, self.water] if m is not None]

    def prepare(self, network: Network) -> None:
        """Phase 1: add Var placeholders for all active carrier modes."""
        for mode in self.modes():
            mode.prepare(network)

    def equations(self, network: Network, ignored_nodes: set) -> list:
        """Phase 2: return equations for all active carrier modes."""
        eqs = []
        for mode in self.modes():
            eqs += mode.equations(network, ignored_nodes)
        return eqs
