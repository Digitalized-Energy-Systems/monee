"""
Cross-cutting helpers used by load-shedding / dispatch problems.
"""

from __future__ import annotations

import numbers

import networkx as nx

from monee.model.branch import HeatExchanger, WaterPipe
from monee.model.child import ExtHydrGrid
from monee.model.core import Var, is_plain_number
from monee.model.formulation.miqcqp.convex.el import terminal_current_pu_squared
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
    GenericTransferBranch,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
    SubHE,
    _heat_branches,
)
from monee.model.node import Bus
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER
from monee.model.phys.nonlinear.wf import pipe_insulation_ua

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
    original behaviour). Unbounded ratings drop the constraint. A rating that
    is a decision variable (a controllable ``max_i_ka``) is kept in the
    constraint.
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
    rating_fixed = isinstance(max_i_ka, numbers.Real)
    if max_i_ka is None or (rating_fixed and max_i_ka >= _UNBOUND_MAX_I_KA):
        return True
    i_base_attr = f"_misocp_loading_{side}_i_base_ka"
    if hasattr(branch_model, i_base_attr):
        i_base = getattr(branch_model, i_base_attr)
        i_sq = terminal_current_pu_squared(
            branch_model.current_pu_squared,
            getattr(branch_model, f"p_{side}_mw"),
            getattr(branch_model, f"q_{side}_mvar"),
            getattr(branch_model, f"p_series_{side}_mw"),
            getattr(branch_model, f"q_series_{side}_mvar"),
            getattr(branch_model, f"_misocp_shunt_{side}_per_mw"),
        )
        if not rating_fixed:
            # Multiplied out, so the decision rating is not a divisor.
            return i_sq * i_base**2 <= (max_loading * max_i_ka) ** 2
        scale_sq = (i_base / max_i_ka) ** 2
        return i_sq <= (max_loading * max_loading) / scale_sq
    return getattr(branch_model, f"loading_{side}_pu") <= max_loading


def make_node_var_bounds_hook(node_type, attr, squared_attr, bounds, grid_types=None):
    """Problem hook (``OptimizationProblem.add_hook``) bounding *attr* (and
    ``lo^2..hi^2`` on *squared_attr*) on independent nodes of exactly
    *node_type*, restricted to
    nodes on a grid of one of *grid_types* when given. Only Var-typed
    attributes are touched: whichever of the pair the formulation uses as its
    actual decision variable gets the bound, while reporting Intermediates are
    left alone (a static bound there would be a no-op)."""
    lo, hi = bounds

    def _apply_bounds(network):
        for component in network.all_components():
            model = component.model
            if type(model) is not node_type or not component.independent:
                continue
            if grid_types is not None and not isinstance(component.grid, grid_types):
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
    """Problem hook (``OptimizationProblem.add_hook``) narrowing every
    temperature Var of a water-grid component to ``bounds_t_k`` (K, converted
    with the component's
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


def water_slack_heat_terms(network, exact=False):
    """``[(slack_child, q_mw)]``: the heat each water :class:`ExtHydrGrid`
    supplies to its island, as an expression over the solver variables.

    The slack's heat is implicit (its node's heat balance is the dropped,
    redundant one), so it is recovered as the island sum of every nodal heat
    balance without the mass-flow children: heat loads, exchanger duties and
    pipe losses minus all other heat injection. Positive means the slack
    supplies heat, negative that it absorbs a surplus. Water leaving through a
    Sink or ConsumeHydrGrid counts as returned to the plant at its own
    temperature, and the enthalpy of a hot water Source counts as slack heat.
    ``exact=True`` writes the sum with every branch's exact heat where each
    branch of the island has one (:func:`_island_heat_mw_exact`)."""
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
        exact_mw = _island_heat_mw_exact(network, island) if exact else None
        terms.append(
            (slack, scale_mw_per_kgs * total if exact_mw is None else exact_mw)
        )
    return terms


def _exact_branch_heat_mw(branch):
    """Heat a water branch takes out of the water [MW], ``c m (T_in - T_out)``,
    in exact closed form: an exchanger's duty, zero for a compound's loss-free
    transfer link, or a MIQCQP pipe's loss ``UA alpha (T_in - T_ext)`` (equal
    through ``alpha (m + UA/c) = m``). None for branches without one (smooth
    NLP pipes, passive exchangers)."""
    model = branch.model
    vs = model.vars
    if isinstance(model, GenericTransferBranch):
        return 0
    if isinstance(model, HeatExchanger) and "q_mw_delivered" in vs:
        return vs["q_mw_delivered"]
    if (
        isinstance(model, WaterPipe)
        and "alpha" in vs
        and not is_plain_number(vs.get("direction", 1))
    ):
        return (
            pipe_insulation_ua(model)
            * vs["alpha"]
            * (vs["t_in_pu"] * branch.grid.t_ref_k - model.temperature_ext_k)
            / 1e6
        )
    return None


def _island_heat_mw_exact(network, island):  # NOSONAR
    """The island sum of :func:`water_slack_heat_terms` with each branch's
    enthalpy flows replaced by its exact heat (:func:`_exact_branch_heat_mw`).
    Equal on every solution, but the relaxation of the bilinear ``m T``
    balances does not conserve energy, so the summed form lets a relaxed
    network deliver heat without producing it or losing any in its pipes.
    None when a branch has no exact form or a node a heat regulariser."""
    total = 0
    for branch in network.branches:
        if not (
            _live(branch)
            and not _switched_off(branch)
            and branch.from_node_id in island
            and branch.to_node_id in island
        ):
            continue
        if "t_from_pu" not in branch.model.vars and not isinstance(branch.model, SubHE):
            continue
        heat = _exact_branch_heat_mw(branch)
        if heat is None:
            return None
        total = total + heat
    for node_id in island:
        node = network.node_by_id(node_id)
        water = isinstance(node.grid, WaterGrid)
        if water and getattr(node.grid, "node_heat_reg_kgs", 0.0):
            return None
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
        if not water:
            from_models, to_models = (
                _heat_branches(from_models),
                _heat_branches(to_models),
            )
        if not any("t_to_pu" in m.vars for m in from_models + to_models):
            continue
        heat_children = (
            [
                child.model.vars
                for child in network.childs_by_ids(node.child_ids)
                if _live(child) and "mass_flow_kgs" not in child.model.vars
            ]
            if water
            else []
        )
        for vs in heat_children:
            if "q_mw_heat" in vs:
                total = total + vs["q_mw_heat"] * vs.get("regulation", 1)
        for m in to_models:
            if "q_mw_heat" in m.vars:
                total = total + m.vars["q_mw_heat"] * m.vars.get("on_off", 1)
    return total


def bound_water_slack_heat(constraints, bounds_mw, exact=False):
    """Bound the heat every water :class:`ExtHydrGrid` supplies to its island
    (:func:`water_slack_heat_terms`) to ``bounds_mw = (lo, hi)`` in MW, load
    convention (supply negative); a ``None`` side is left open. ``exact`` uses
    the branch form, which keeps a relaxation with regulated exchanger duties
    energy-conserving but adds a bilinear loss term per pipe."""
    lo, hi = bounds_mw
    exchange = constraints.with_models(
        lambda network: [
            q_mw for _, q_mw in water_slack_heat_terms(network, exact=exact)
        ]
    )
    if lo is not None:
        exchange.equation(lambda q_mw: -q_mw >= lo)
    if hi is not None:
        exchange.equation(lambda q_mw: -q_mw <= hi)


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
