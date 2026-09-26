"""
Cross-cutting helpers used by load-shedding / dispatch problems.
"""

from __future__ import annotations

import networkx as nx

from monee.model.child import ExtHydrGrid
from monee.model.core import Var
from monee.model.grid import (
    DEFAULT_GAS_HHV_KWH_PER_KG,
    KGPS_KWHPERKG_TO_MW,
    GasGrid,
    WaterGrid,
)
from monee.model.multi import (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
    _heat_branches,
)
from monee.model.node import Bus
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER

# Fallback HHV (kWh/kg) when a gas grid is missing ``higher_heating_value_kwh_per_kg``.
_HHV_DEFAULT = DEFAULT_GAS_HHV_KWH_PER_KG

_UNBOUND_MAX_I_KA = 999.0


def line_loading_limit(branch_model, side: str, max_loading: float, basis: str = "mva"):
    """Cap branch loading at *max_loading* (per unit of the branch rating).

    ``basis="mva"`` (default) caps apparent power as the per-unit loading-squared
    form ``(p**2 + q**2) / (max_loading * max_s_mva)**2 <= 1`` - smooth, and the
    same solution as MATPOWER's |S|^2 <= RATE_A^2. The normalization is what makes
    it O(1): the bare ``p**2 + q**2 <= RATE_A**2`` form has residual/curvature of
    order RATE_A^2 (~1e6-1e8 in MW), which ill-conditions the OPF and balloons the
    IPOPT iteration count. When the branch carries no ``max_s_mva`` (None), it
    falls back to the current basis below, so networks that only have ``max_i_ka``
    keep their existing current-based behaviour.

    ``basis="current"`` caps current as |I| <= max_loading * max_i_ka (the
    original behaviour). Unbounded ratings drop the constraint.
    """
    if side not in ("from", "to"):
        raise ValueError(f"side must be 'from' or 'to', got {side!r}")
    if basis not in ("mva", "apparent", "current"):
        raise ValueError(f"basis must be 'mva' or 'current', got {basis!r}")

    if basis in ("mva", "apparent"):
        max_s_mva = getattr(branch_model, "max_s_mva", None)
        if max_s_mva is not None:
            p = getattr(branch_model, f"p_{side}_mw")
            q = getattr(branch_model, f"q_{side}_mvar")
            limit = max_loading * max_s_mva
            return (p**2 + q**2) / (limit * limit) <= 1.0
        # No MVA rating on this branch: fall through to the current basis.

    max_i_ka = getattr(branch_model, "max_i_ka", None)
    if max_i_ka is None or max_i_ka >= _UNBOUND_MAX_I_KA:
        return True
    # Under the branch-flow formulations current_pu_squared is the SERIES current,
    # so this caps the terminal current only up to the pi-shunt charging current.
    # Branches that carry charging (MATPOWER, CIM) reach this code only when they
    # have no max_s_mva, and both importers leave those unrated.
    scale_attr = f"_misocp_loading_{side}_scale_squared"
    if hasattr(branch_model, scale_attr):
        scale_sq = getattr(branch_model, scale_attr)
        if scale_sq <= 0:
            return True
        return branch_model.current_pu_squared <= (max_loading * max_loading) / scale_sq
    return getattr(branch_model, f"loading_{side}_pu") <= max_loading


def make_node_var_bounds_hook(node_type, attr, squared_attr, bounds):
    """``_controllable_appliables`` hook bounding *attr* (and ``lo^2..hi^2`` on
    *squared_attr*) on independent nodes of exactly *node_type*. Only Var-typed
    attributes are touched: whichever of the pair the formulation uses as its
    actual decision variable gets the bound, while reporting Intermediates are
    left alone (a static bound there would be a no-op)."""
    lo, hi = bounds

    def _apply_bounds(network):
        for component in network.all_components():
            model = component.model
            if type(model) is not node_type or not component.independent:
                continue
            v = getattr(model, attr, None)
            vsq = getattr(model, squared_attr, None)
            if type(v) is Var:
                v.min, v.max = lo, hi
            if type(vsq) is Var:
                vsq.min, vsq.max = lo * lo, hi * hi

    return _apply_bounds


def make_vm_bounds_hook(bounds_vm):
    """Bound bus voltage magnitudes on the formulation's actual decision
    variable: ``vm_pu_squared`` under the MISOCP relaxation, ``vm_pu`` under
    the AC NLP."""
    return make_node_var_bounds_hook(Bus, "vm_pu", "vm_pu_squared", bounds_vm)


_WATER_TEMPERATURE_ATTRS = ("t_pu", "t_from_pu", "t_to_pu", "t_in_pu", "t_out_pu")


def make_water_temperature_bounds_hook(bounds_t_k):
    """``_controllable_appliables`` hook narrowing every temperature Var of a
    water-grid component to ``bounds_t_k`` (K, converted with the component's
    grid ``t_ref_k``). Existing bounds are only tightened; a Var pinned outside
    the band (e.g. by a timeseries) keeps its pin. Multi-grid coupler control
    nodes are left alone."""
    lo_k, hi_k = bounds_t_k

    def _apply_bounds(network):
        for component in network.all_components():
            grid = component.grid
            if not isinstance(grid, WaterGrid) or not _live(component):
                continue
            for attr in _WATER_TEMPERATURE_ATTRS:
                var = getattr(component.model, attr, None)
                if type(var) is Var:
                    _tighten(var, lo_k / grid.t_ref_k, hi_k / grid.t_ref_k)

    return _apply_bounds


def _tighten(var, lo, hi):
    new_lo = lo if var.min is None else max(var.min, lo)
    new_hi = hi if var.max is None else min(var.max, hi)
    if new_lo <= new_hi:
        var.min, var.max = new_lo, new_hi


def gas_mw_per_kgs(grid):
    """MW of higher heating value carried by 1 kg/s of the gas on *grid*."""
    return KGPS_KWHPERKG_TO_MW * getattr(
        grid, "higher_heating_value_kwh_per_kg", _HHV_DEFAULT
    )


def _live(component):
    return component.active and not component.ignored


def _carries_water(grid):
    if isinstance(grid, dict):
        grid = list(grid.values())
    if isinstance(grid, (list, tuple)):
        return any(isinstance(g, WaterGrid) for g in grid)
    return isinstance(grid, WaterGrid)


def _switched_off(branch):
    on_off = getattr(branch.model, "on_off", 1)
    return isinstance(on_off, (int, float)) and on_off == 0


def _water_islands(network):
    water_ids = {
        node.id for node in network.nodes if _live(node) and _carries_water(node.grid)
    }
    graph = nx.Graph()
    graph.add_nodes_from(water_ids)
    for branch in network.branches:
        if (
            _live(branch)
            and not _switched_off(branch)
            and branch.from_node_id in water_ids
            and branch.to_node_id in water_ids
        ):
            graph.add_edge(branch.from_node_id, branch.to_node_id)
    return list(nx.connected_components(graph))


def _island_slack(network, island):
    slacks = [
        child
        for node_id in island
        for child in network.childs_by_ids(network.node_by_id(node_id).child_ids)
        if _live(child) and type(child.model) is ExtHydrGrid
    ]
    if len(slacks) > 1:
        raise ValueError(
            f"A heat island holds {len(slacks)} active water ExtHydrGrid slacks "
            f"(child ids {[c.id for c in slacks]}); the economic dispatch prices "
            "the heat of one slack per island. Networks built with "
            "heat_plant_mode='two_port' are not supported."
        )
    return slacks[0] if slacks else None


def _node_heat_terms(network, node):
    if getattr(node.model, "_mccormick_dhs_active", False) or getattr(
        node.model, "_ltc_active", False
    ):
        raise ValueError(
            "Heat pricing in the economic dispatch needs the nodal heat balance "
            "of the smooth or MISOCP heat formulations; the McCormick "
            "formulation (formulation='heat_convex_milp') and LTC replace it and "
            "are not supported."
        )
    from_models = [
        network.branch_by_id(b).model
        for b in node.from_branch_ids
        if _live(network.branch_by_id(b))
    ]
    to_models = [
        network.branch_by_id(b).model
        for b in node.to_branch_ids
        if _live(network.branch_by_id(b))
    ]
    if isinstance(node.grid, WaterGrid):
        heat_children = [
            child.model
            for child in network.childs_by_ids(node.child_ids)
            if _live(child) and "mass_flow_kgs" not in child.model.vars
        ]
        return node.model.calc_signed_heat_flow(
            from_models, to_models, heat_children, node.grid
        )
    return node.model.calc_signed_heat_flow(
        _heat_branches(from_models), _heat_branches(to_models), [], None
    )


def water_slack_heat_terms(network):
    """``[(slack_child, q_mw)]``: the heat each water :class:`ExtHydrGrid`
    supplies to its island, as an expression over the solver variables.

    The slack's heat is implicit (its node's heat balance is the dropped,
    redundant one), so it is recovered as the island sum of every nodal heat
    balance without the mass-flow children: heat loads, exchanger duties and
    pipe losses minus all other heat injection. Positive means the slack
    supplies heat, negative that it absorbs a surplus. Water leaving through a
    Sink or ConsumeHydrGrid counts as returned to the plant at its own
    temperature, and the enthalpy of a hot water Source counts as slack heat."""
    terms = []
    for island in _water_islands(network):
        slack = _island_slack(network, island)
        if slack is None:
            continue
        total = sum(
            sum(_node_heat_terms(network, network.node_by_id(node_id)))
            for node_id in island
        )
        scale_mw_per_kgs = SPECIFIC_HEAT_CAP_WATER * slack.grid.t_ref_k / 1e6
        terms.append((slack, scale_mw_per_kgs * total))
    return terms


def cp_input_rated_mw(component):  # NOSONAR

    model = component.model
    grid = getattr(component, "grid", None)

    def _hhv():
        gg = None
        if isinstance(grid, dict):
            gg = grid.get(GasGrid)
        elif isinstance(grid, list):
            gg = next((g for g in grid if isinstance(g, GasGrid)), None)
        elif isinstance(grid, GasGrid):
            gg = grid
        return (
            getattr(gg, "higher_heating_value_kwh_per_kg", _HHV_DEFAULT)
            if gg
            else _HHV_DEFAULT
        )

    def _scalar(x):
        # Read the bound (or the value) off a Var; pass scalars through.
        if isinstance(x, Var):
            fallback = x.value if x.value is not None else 0.0
            return x.max if x.max is not None else fallback
        return x if x is not None else 0.0

    # ---- Gas-input CPs ----------------------------------------------------
    if isinstance(model, (CHPControlNode, CHPHGControlNode, GasToHeatControlNode)):
        return (
            "gas",
            abs(_scalar(model.gas_mass_flow_kgs)) * KGPS_KWHPERKG_TO_MW * _hhv(),
        )
    if isinstance(model, GasToPower):
        eff = max(getattr(model, "efficiency", 1.0), 1e-6)
        # el_mw is stored as -p_mw_setpoint (scalar).
        return ("gas", abs(_scalar(model.el_mw)) / eff)
    if isinstance(model, GasToHeatHG):
        eff = max(getattr(model, "efficiency", 1.0), 1e-6)
        return ("gas", abs(_scalar(model.heat_mw)) / eff)

    # ---- Power-input CPs --------------------------------------------------
    if isinstance(model, PowerToHeatControlNode):
        # el_mw is the solved draw; load_p_mw carries the nameplate setpoint.
        return ("power", abs(_scalar(model.load_p_mw)))
    if isinstance(model, PowerToHeatHG):
        return ("power", abs(_scalar(model.load_p_mw)))
    if isinstance(model, PowerToGas):
        eff = max(getattr(model, "efficiency", 1.0), 1e-6)
        # gas_mass_flow_kgs stored as -mass_flow_setpoint_kgs; rated input power = output / eta.
        return (
            "power",
            abs(_scalar(model.gas_mass_flow_kgs)) * KGPS_KWHPERKG_TO_MW * _hhv() / eff,
        )

    return None
