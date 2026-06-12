"""Minimal load shedding optimisation for multi-energy grids.

Minimises total unserved energy across electrical, gas, and thermal carriers
(single- or multi-period).
"""

import logging
import math

from monee.model.branch import (
    GenericPowerBranch,
    HeatExchanger,
    HeatExchangerGenerator,
    HeatExchangerLoad,
    PassiveHeatExchanger,
    PassiveHeatExchangerGenerator,
    PassiveHeatExchangerLoad,
)
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    HeatGenerator,
    HeatLoad,
    PowerGenerator,
    PowerLoad,
    Sink,
    Source,
)
from monee.model.grid import GasGrid, WaterGrid
from monee.model.multi import (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
)
from monee.model.node import Bus, Junction
from monee.problem.core import (
    REGULATION_ATTR,
    Constraints,
    Objectives,
    OptimizationProblem,
    nan_to_zero,
)
from monee.problem.utils import cp_input_rated_mw, line_loading_limit

WEIGHT_DEMAND = 1e3
WEIGHT_GENERATOR = 0.1
# 10× weaker than the demand-shed cost so ties prefer self-sufficiency.
EXT_SLACK_WEIGHT_RATIO = 0.1

# Fallback HHV (kWh/kg) for Sink/Source on a grid without higher_heating_value.
_HHV_DEFAULT = 15.3

# Types that participate in the load-shedding objective.
_DEMAND_TYPES = (
    PowerLoad,
    HeatLoad,
    HeatExchangerLoad,
    PassiveHeatExchangerLoad,
    Sink,
)
_GENERATOR_TYPES = (
    PowerGenerator,
    HeatGenerator,
    HeatExchangerGenerator,
    PassiveHeatExchangerGenerator,
    Source,
)
# HE types for the objective; SubHE is excluded via exact type() checks.
_HE_OBJECTIVE_TYPES = (
    HeatExchangerLoad,
    HeatExchangerGenerator,
    PassiveHeatExchanger,
    PassiveHeatExchangerLoad,
    PassiveHeatExchangerGenerator,
)
# Coupling-point models (control nodes + branch HG variants).  When the
# load-shedding problem is constructed with ``include_coupling_points=True``
# every CP regulation cut is penalised at the demand weight, scaled by the
# CP's nameplate input MW.
_COUPLING_POINT_TYPES = (
    CHPControlNode,
    CHPHGControlNode,
    GasToHeatControlNode,
    PowerToHeatControlNode,
    GasToPower,
    PowerToGas,
    PowerToHeatHG,
    GasToHeatHG,
)


def _gas_mw_factor(grid):
    """Return ``3.6 · HHV`` (kg/s → MW); falls back to :data:`_HHV_DEFAULT`."""
    hhv = getattr(grid, "higher_heating_value", _HHV_DEFAULT)
    return 3.6 * hhv


_SQRT_3 = math.sqrt(3.0)


def _aux_objective_upper_bound(
    network,
    *,
    vm_pu_fallback: float = 1.1,
    max_line_loading: float | None = None,
) -> float:
    """Upper bound on ``|Σ aux_obj|`` over the formulation-level minimize hooks.

    Contributors:
      * MISOCP electricity: ``current_pu·br_r`` capped by ``min(SOC_bound,
        line_loading_bound)``;
      * Linear HE: ``|q_mw_set|``;
      * NL gas/water epigraph (ε=1e-5): ``1e-5·2·M²`` per branch;
      * McCormick-DHS t_pu pull (ε=1e-6): ``1e-6·2`` per junction.
    Triangle inequality absorbs signs.
    """
    total = 0.0
    for component in network.all_components():
        m = component.model
        # br_r/br_x on PowerLine/Trafo are computed in equations() - at
        # _apply time they are still 0, so call calc_r_x to read them now.
        br_r = getattr(m, "br_r", None)
        br_x = getattr(m, "br_x", None)
        from_model = to_model = None
        if hasattr(component, "from_node_id"):
            try:
                from_model = network.node_by_id(component.from_node_id).model
                to_model = network.node_by_id(component.to_node_id).model
                if hasattr(m, "calc_r_x"):
                    br_r, br_x = m.calc_r_x(component.grid, from_model, to_model)
            except Exception:
                from_model = to_model = None
        if isinstance(br_r, (int, float)) and isinstance(br_x, (int, float)):
            denom = br_r * br_r + br_x * br_x
            if denom > 0 and br_r > 0:
                vm_max = getattr(component.grid, "vm_pu_max", vm_pu_fallback)
                w_max = vm_max * vm_max
                current_pu_max = 4.0 * w_max / denom  # SOC physics bound
                # Tighten via line-loading constraint:
                # current_pu ≤ max_loading² · (max_i_ka / I_base)²
                max_i_ka = getattr(m, "max_i_ka", None)
                grid = component.grid
                if (
                    max_line_loading is not None
                    and isinstance(max_i_ka, (int, float))
                    and from_model is not None
                    and to_model is not None
                    and getattr(grid, "sn_mva", None)
                    and getattr(from_model, "base_kv", None)
                    and getattr(to_model, "base_kv", None)
                ):
                    tap = getattr(m, "tap", 1.0) or 1.0
                    i_base_from = grid.sn_mva / (_SQRT_3 * from_model.base_kv) / tap
                    i_base_to = grid.sn_mva / (_SQRT_3 * to_model.base_kv)
                    i_base_min = min(i_base_from, i_base_to)
                    loading_bound = max_line_loading**2 * (max_i_ka / i_base_min) ** 2
                    current_pu_max = min(current_pu_max, loading_bound)
                total += current_pu_max * br_r
        # Linear HX: |q_mw_delivered| ≤ |q_mw_set·regulation| ⇒ bound |q_mw_set|.
        q_mw_set = getattr(m, "q_mw_set", None)
        if isinstance(q_mw_set, (int, float)):
            total += abs(q_mw_set)
        # NL gas/water epigraph ε·(m_pos²+m_neg²) (ε≈1e-5; use 1e-4 for safety).
        m_pos_sq = getattr(m, "mass_flow_pos_squared", None)
        m_neg_sq = getattr(m, "mass_flow_neg_squared", None)
        if m_pos_sq is not None and m_neg_sq is not None:
            ub_pos = getattr(m_pos_sq, "max", None) or 100.0**2
            ub_neg = getattr(m_neg_sq, "max", None) or 100.0**2
            total += 1e-4 * (ub_pos + ub_neg)
        # McCormick-DHS t_pu pull: ε·(1-t_pu), t_pu∈[0,2].
        t_pu = getattr(m, "t_pu", None)
        if t_pu is not None:
            total += 1e-6 * 2.0
    return total


def _make_auto_priority_floor_hook(
    weights: dict,
    *,
    alpha: float,
    debug: bool,
    max_line_loading: float | None,
    weight_for_load=None,
):
    """Return a ``_controllable_appliables`` hook that retunes *weights*.

    Sets ``weights['demand'] = max(weights['demand'], α·A_max)`` and scales
    ``weights['generator']`` by the same factor so the user-set ratio is kept.

    When ``weight_for_load`` is supplied, per-load weights returned by the
    callback may be smaller than ``weights['demand']``.  To preserve the
    auto-floor contract (every load's shed cost dominates aux), the hook
    samples the minimum effective weight across all demand-class models
    and, if it is below the floor, scales every weight up by
    ``floor / min_w``.  Generator weights and the per-load callback's
    returns are both lifted by the same scale via the ``weights["scale"]``
    multiplier consulted in ``weight_fn`` - the user-set load-priority
    *ratios* are preserved.
    """
    _log = logging.getLogger(__name__)

    def _hook(network):
        a_max = _aux_objective_upper_bound(network, max_line_loading=max_line_loading)
        floor = alpha * a_max

        if weight_for_load is not None:
            # Sample the minimum effective per-load weight.  We consider
            # every model that ``weight_fn`` would route through
            # ``weight_for_load`` (demand + HX-objective types).
            min_w: float | None = None
            for model in network.all_models():
                if isinstance(model, _DEMAND_TYPES + _HE_OBJECTIVE_TYPES):
                    try:
                        w = weight_for_load(model)
                    except Exception:
                        w = None
                    if w is None:
                        # Model falls back to ``weights['demand']``.
                        w = weights["demand"]
                    if min_w is None or float(w) < min_w:
                        min_w = float(w)
            if min_w is not None and min_w < floor:
                scale = floor / min_w if min_w > 0 else 1.0
                # ``weights['scale']`` multiplies BOTH the callback return
                # AND the default ``weights['demand']`` - see ``weight_fn``.
                weights["scale"] = weights.get("scale", 1.0) * scale
                weights["generator"] = weights["generator"] * scale
                if debug:
                    _log.warning(
                        "Auto priority floor (per-load): A_max=%.3g, α=%.3g, "
                        "min_w=%.3g → scale ×%.3g (generator weight scaled too)",
                        a_max,
                        alpha,
                        min_w,
                        scale,
                    )
            elif debug:
                _log.warning(
                    "Auto priority floor (per-load): A_max=%.3g, α·A_max=%.3g "
                    "already covered by per-load min_w=%.3g - no change.",
                    a_max,
                    floor,
                    min_w or 0.0,
                )
            return

        # Legacy path: no per-load callback, scale the single demand weight.
        old_demand = weights["demand"]
        if floor > old_demand:
            scale = floor / old_demand if old_demand > 0 else 1.0
            weights["demand"] = floor
            weights["generator"] = weights["generator"] * scale
            if debug:
                _log.warning(
                    "Auto priority floor: A_max=%.3g, α=%.3g → "
                    "demand_weight %.3g → %.3g (generator scaled ×%.3g)",
                    a_max,
                    alpha,
                    old_demand,
                    weights["demand"],
                    scale,
                )
        elif debug:
            _log.warning(
                "Auto priority floor: A_max=%.3g, α·A_max=%.3g already "
                "covered by user-supplied demand_weight=%.3g - no change.",
                a_max,
                floor,
                old_demand,
            )

    return _hook


def _shedding_mw(model, gas_mw_factor=None, cp_rated_mw=None):
    """Unserved-energy expression for *model* in MW-equivalent.

    For LinearHX branches the under-delivery gap ``|q_mw_set - q_mw_delivered|``
    captures both regulation cuts and physical shortfall from a narrow ΔT,
    which a pure ``(1-reg)`` proxy would miss. External grids return 0.
    CP types (when ``include_coupling_points=True``) are penalised by
    ``cp_rated_mw · (1 - regulation)``.
    """
    reg = getattr(model, "regulation", 1)

    if isinstance(model, _COUPLING_POINT_TYPES):
        rated = cp_rated_mw if cp_rated_mw is not None else 0
        return rated * (1 - reg)

    if isinstance(model, (PowerLoad, PowerGenerator)):
        p = nan_to_zero(model.p_mw)
        # Sign-known by construction: PowerLoad.p_mw ≥ 0, PowerGenerator ≤ 0.
        if isinstance(model, PowerLoad):
            return p * (1 - reg)
        return (-p) * (1 - reg)

    if isinstance(model, (HeatLoad, HeatGenerator)):
        q = nan_to_zero(model.q_mw_heat)
        if isinstance(model, HeatLoad):
            return q * (1 - reg)
        return (-q) * (1 - reg)

    if isinstance(model, _HE_OBJECTIVE_TYPES) or type(model) is HeatExchanger:
        q_mw_set = getattr(model, "q_mw_set", 0)
        q_mw_delivered = getattr(model, "q_mw_delivered", None)
        if q_mw_delivered is not None and isinstance(q_mw_set, (int, float)):
            if q_mw_set > 0:
                return q_mw_set - q_mw_delivered
            if q_mw_set < 0:
                return q_mw_delivered - q_mw_set
        if isinstance(q_mw_set, (int, float)):
            return abs(q_mw_set) * (1 - reg)
        return q_mw_set * (1 - reg)

    if isinstance(model, (Sink, Source)):
        mf = nan_to_zero(model.mass_flow)
        factor = gas_mw_factor if gas_mw_factor is not None else 3.6 * _HHV_DEFAULT
        if isinstance(model, Sink):
            return mf * factor * (1 - reg)
        return (-mf) * factor * (1 - reg)

    return 0


def _calc_objective(model_to_data):
    """Sum weighted unserved energy.

    ``model_to_data: {model → (weight, gas_factor|None, cp_rated_mw|None)}``.
    ``cp_rated_mw`` is only populated for coupling-point models when
    ``include_coupling_points=True``; ``gas_factor`` only for Sink/Source.
    """
    return sum(
        _shedding_mw(model, gas_mw_factor=g, cp_rated_mw=cp) * weight
        for model, (weight, g, cp) in model_to_data.items()
    )


def create_min_load_shedding_problem(
    *,
    demand_weight=WEIGHT_DEMAND,
    generator_weight=WEIGHT_GENERATOR,
    weight_for_load=None,
    bounds_el=(0.9, 1.1),
    bounds_gas=(0.9, 1.1),
    bounds_heat=(0.9, 1.1),
    max_line_loading=1.5,
    ext_grid_el_bounds=(-3, 3),
    ext_grid_gas_bounds=(-10, 10),
    ext_grid_heat_bounds=(-10, 10),
    regulation_ramp_limit=None,
    include_storages=False,
    include_ext_grids=True,
    include_coupling_points=False,
    check_vm=True,
    check_pressure=True,
    check_temperature=True,
    check_line_loading=True,
    lex_objectives=False,
    auto_priority_floor=True,
    priority_safety_factor=10.0,
    debug=False,
):
    """Create a minimal load-shedding optimisation problem for multi-energy grids.

    Each demand/generator/coupling gets a regulation Var ∈ [0,1]; the objective
    penalises ``(1-regulation)`` weighted per category. Gas Sink/Source shed is
    energy-converted via the enclosing grid's HHV. External grids contribute
    only via ``ext_grid_*_bounds`` constraints (and an optional quadratic slack
    nudge to zero exchange when ``include_ext_grids=True``).

    ``lex_objectives`` (Pyomo only) solves in two phases: shed first, then
    formulation-tightening aux terms with the phase-1 optimum pinned.
    ``auto_priority_floor`` (default True) scales ``demand_weight`` to
    ``α · A_max`` so the shed term dominates aux terms regardless of network size.

    ``include_coupling_points`` (default False) extends the demand-side of the
    objective to coupling-point components (CHP / CHPHG / G2H / P2H control
    nodes and the HG branch variants P2G / G2P / P2H_HG / G2H_HG). Each CP is
    penalised at ``demand_weight · cp_input_rated_mw · (1 - regulation)`` -
    i.e. treated like a load on its input carrier (gas or power).
    """
    problem = OptimizationProblem(debug=debug, lex_objectives=lex_objectives)

    # Mutable so the auto-priority-floor hook can retune at _apply time.
    # ``scale`` is a multiplicative factor applied on top of both the
    # default ``demand`` weight AND any per-load weight returned by
    # ``weight_for_load`` - the auto-floor hook uses it to lift every
    # demand-side weight uniformly when the minimum effective per-load
    # weight would otherwise fall below ``α·A_max``.  Default 1.0 makes
    # it a no-op for legacy (no callback) usage.
    _weights = {
        "demand": float(demand_weight),
        "generator": float(generator_weight),
        "scale": 1.0,
    }

    problem.controllable_demands(REGULATION_ATTR)
    problem.controllable_generators(REGULATION_ATTR)
    problem.controllable_cps(REGULATION_ATTR)
    problem.controllable_backup_lines()
    if include_ext_grids:
        problem.controllable_ext()
    if include_storages:
        problem.controllable_storages()

    if auto_priority_floor:
        problem._controllable_appliables.append(
            _make_auto_priority_floor_hook(
                _weights,
                alpha=priority_safety_factor,
                debug=debug,
                max_line_loading=max_line_loading if check_line_loading else None,
                weight_for_load=weight_for_load,
            )
        )

    if check_vm:
        problem.bounds(bounds_el, lambda m, _: type(m) is Bus, ["vm_pu"])
    if check_pressure:
        problem.bounds(bounds_gas, lambda m, _: type(m) is Junction, ["pressure_pu"])
    if check_temperature:
        problem.bounds(
            bounds_heat,
            lambda m, g: type(m) is Junction and type(g) is WaterGrid,
            ["t_pu"],
        )

    # Lookups go through _weights so the auto-priority-floor can retune.
    # When ``weight_for_load`` is set and applicable, the per-load
    # weight it returns takes precedence - multiplied by the auto-
    # floor scale so per-load weights and the legacy demand weight
    # share the same aux-dominance lift.
    def weight_fn(model):
        scale = _weights.get("scale", 1.0)
        if weight_for_load is not None and isinstance(
            model, _DEMAND_TYPES + _HE_OBJECTIVE_TYPES
        ):
            try:
                w = weight_for_load(model)
            except Exception:
                w = None
            if w is not None:
                return float(w) * scale
        if isinstance(model, _DEMAND_TYPES):
            return _weights["demand"] * scale
        # CPs are penalised at the demand weight when include_coupling_points
        # is on - the input draw is treated as a load on its input carrier.
        if isinstance(model, _COUPLING_POINT_TYPES):
            return _weights["demand"] * scale
        # Bare HeatExchanger / PassiveHeatExchanger: route by sign of q_mw_set.
        if isinstance(model, (HeatExchanger, PassiveHeatExchanger)):
            q = getattr(model, "q_mw_set", 0)
            if isinstance(q, (int, float)):
                return (_weights["demand"] if q > 0 else _weights["generator"]) * scale
            return _weights["demand"] * scale
        if isinstance(model, _GENERATOR_TYPES):
            return _weights["generator"]
        return 1

    objective_types = _DEMAND_TYPES + _GENERATOR_TYPES + _HE_OBJECTIVE_TYPES
    if include_coupling_points:
        objective_types = objective_types + _COUPLING_POINT_TYPES

    def _is_objective_model(m):
        # type() is needed for HeatExchanger to exclude SubHE (compound-internal).
        return isinstance(m, objective_types) or type(m) is HeatExchanger

    def _is_gas_grid(g):
        # Water-grid Sinks/Sources are heating-loop mass flow, not gas demand;
        # applying HHV would yield phantom MW-scale shed penalty.
        if g is None:
            return False
        grids = g if isinstance(g, list) else [g]
        return any(
            gg is not None and hasattr(gg, "higher_heating_value") for gg in grids
        )

    # Populated by _objective_models, read by _data_attacher.
    #   model_to_grid  - Sink/Source HHV lookup (gas grid only).
    #   model_to_cp_mw - CP nameplate input MW (when include_coupling_points).
    model_to_grid: dict = {}
    model_to_cp_mw: dict = {}

    def _objective_models(network):
        """Like Objectives.select but filters water-grid Sink/Source (heating
        loop ≠ gas demand) and (opt-in) folds in coupling-point components."""
        model_to_grid.clear()
        model_to_cp_mw.clear()
        out = []
        # Standard child/branch loads + generators + HX. Inactive/ignored
        # components are excluded (mirroring Constraints.select and the CP
        # loop below): their Vars are never registered with the backend, so
        # e.g. an ignored HeatExchangerLoad's q_mw_delivered would otherwise
        # leak a raw monee Var into the objective expression.
        for component in network.all_components():
            if not component.active or component.ignored:
                continue
            model = component.model
            grid = getattr(component, "grid", None)
            if not _is_objective_model(model):
                continue
            if isinstance(model, (Sink, Source)) and not _is_gas_grid(grid):
                continue
            # CP control nodes also surface via all_models_with_grid, but we
            # need ``component`` (not just ``model``) to compute the rated MW
            # via cp_input_rated_mw - handled in the dedicated loop below.
            if isinstance(model, _COUPLING_POINT_TYPES):
                continue
            out.append(model)
            model_to_grid[model] = grid
        # Opt-in CP coverage: control nodes (in network.nodes) + branch CPs.
        if include_coupling_points:
            for component in list(network.nodes) + list(network.branches):
                if not isinstance(component.model, _COUPLING_POINT_TYPES):
                    continue
                if not component.active or component.ignored:
                    continue
                cp = cp_input_rated_mw(component)
                if cp is None:
                    continue
                model = component.model
                out.append(model)
                model_to_grid[model] = getattr(component, "grid", None)
                model_to_cp_mw[model] = cp[1]
        return out

    def _data_attacher(model):
        weight = weight_fn(model)
        gas_factor = None
        cp_rated = None
        if isinstance(model, (Sink, Source)):
            gas_factor = _gas_mw_factor(model_to_grid.get(model))
        if isinstance(model, _COUPLING_POINT_TYPES):
            cp_rated = model_to_cp_mw.get(model)
        return (weight, gas_factor, cp_rated)

    objectives = Objectives()
    objectives.with_models(_objective_models).data(_data_attacher).calculate(
        _calc_objective
    )

    # Quadratic ext-grid slack toward zero exchange, weighted at
    # demand_weight · EXT_SLACK_WEIGHT_RATIO. Off when include_ext_grids=False.
    if include_ext_grids:

        def _ext_slack_models(network):
            out = []
            for model in network.all_models():
                if isinstance(model, (ExtPowerGrid, ExtHydrGrid)):
                    out.append(model)
            return out

        def _ext_slack_data(_model):
            return _weights["demand"] * EXT_SLACK_WEIGHT_RATIO

        def _ext_slack_calc(model_to_data):
            total = 0
            for model, weight in model_to_data.items():
                if isinstance(model, ExtPowerGrid):
                    total = total + model.p_mw * model.p_mw * weight
                elif isinstance(model, ExtHydrGrid):
                    total = total + model.mass_flow * model.mass_flow * weight
            return total

        objectives.with_models(_ext_slack_models).data(_ext_slack_data).calculate(
            _ext_slack_calc
        )

    problem.objectives = objectives

    constraints = Constraints()

    if check_line_loading:
        constraints.select_types(GenericPowerBranch).equation(
            lambda m: line_loading_limit(m, "from", max_line_loading)
        ).equation(lambda m: line_loading_limit(m, "to", max_line_loading))

    if include_ext_grids:
        constraints.select_types(ExtPowerGrid).equation(
            lambda m: m.p_mw >= ext_grid_el_bounds[0]
        ).equation(lambda m: m.p_mw <= ext_grid_el_bounds[1])

        constraints.select(
            lambda c: type(c.grid) is GasGrid and type(c.model) is ExtHydrGrid
        ).equation(lambda m: m.mass_flow >= ext_grid_gas_bounds[0]).equation(
            lambda m: m.mass_flow <= ext_grid_gas_bounds[1]
        )

        constraints.select(
            lambda c: type(c.grid) is WaterGrid and type(c.model) is ExtHydrGrid
        ).equation(lambda m: m.mass_flow >= ext_grid_heat_bounds[0]).equation(
            lambda m: m.mass_flow <= ext_grid_heat_bounds[1]
        )

    if regulation_ramp_limit is not None:
        constraints.regulation_ramp(regulation_ramp_limit)

    problem.constraints = constraints
    return problem
