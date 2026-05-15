"""Minimal load shedding optimisation for multi-energy grids.

Provides a formulation that minimises total unserved energy across
electrical, gas, and thermal carriers.  Works for both single-period
and multi-period solves.

Typical usage::

    import monee

    prob = monee.create_min_load_shedding_problem()
    result = monee.run_energy_flow_optimization(net, optimization_problem=prob)
"""

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
from monee.model.node import Bus, Junction
from monee.problem.core import (
    REGULATION_ATTR,
    Constraints,
    Objectives,
    OptimizationProblem,
    nan_to_zero,
)

WEIGHT_DEMAND = 1e3
WEIGHT_GENERATOR = 0.1
# Ratio of ext-grid slack penalty to demand weight.  Picked so the
# nudge-to-zero force on the slack bus is ~10× weaker than the cost
# of shedding 1 MW of demand, i.e. the optimiser will gladly accept
# a few MW of ext exchange to avoid any shedding, but in a tie will
# pick the solution closer to self-sufficiency.
EXT_SLACK_WEIGHT_RATIO = 0.1

# Fallback higher heating value (kWh/kg) for gas Sink/Source when the
# enclosing :class:`~monee.model.grid.GasGrid` does not expose
# ``higher_heating_value``.  The conversion factor ``3.6 · HHV`` yields
# MW per (kg/s) — see :func:`_gas_mw_factor`.
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
# Heat exchanger types for the objective — excludes SubHE (internal to
# compound models like CHP/P2H/G2H) by using exact type checks for the
# base classes.
_HE_OBJECTIVE_TYPES = (
    HeatExchangerLoad,
    HeatExchangerGenerator,
    PassiveHeatExchanger,
    PassiveHeatExchangerLoad,
    PassiveHeatExchangerGenerator,
)


def _gas_mw_factor(grid):
    """Return the (kg/s → MW) conversion factor for a gas *grid*.

    ``3.6 · HHV`` where the HHV is in kWh/kg.  Falls back to
    :data:`_HHV_DEFAULT` when the grid is ``None`` or does not expose
    ``higher_heating_value`` (e.g. a Sink mistakenly placed on a water
    grid — filtered out before reaching the objective, but defended
    against here).
    """
    hhv = getattr(grid, "higher_heating_value", _HHV_DEFAULT)
    return 3.6 * hhv


_SQRT_3 = math.sqrt(3.0)


def _aux_objective_upper_bound(
    network,
    *,
    vm_pu_fallback: float = 1.1,
    max_line_loading: float | None = None,
) -> float:
    """Provable over-estimate of ``|Σ pm.aux_obj_exprs|`` for a network.

    Sums per-formulation contributions to the formulation-level
    tightening objective (the ``minimize`` hooks on each Formulation
    class):

    * **MISOCP electricity** (``MISOCPElectricityBranchFormulation``):
      contributes ``current_pu · br_r`` per power branch.  Two upper
      bounds apply: the physics-only SOC bound
      ``ell_phys = 4·w_max / (br_r²+br_x²)`` (see
      :func:`monee.model.formulation.misoc.el._ell_physics_max`), and
      — when line-loading limits are enforced — the operational bound
      ``max_loading² · (max_i_ka / I_base)²`` from
      :func:`monee.problem.utils.line_loading_limit`.  We use the
      smaller of the two; for typical distribution lines the loading
      bound is ~10²–10⁴× tighter and is what makes the bound usable.
    * **Linear heat exchanger** (``LinearHeatExchangerFormulation``):
      contributes ``±q_mw_delivered`` per HE branch.  The formulation
      caps ``|q_mw_delivered| ≤ |q_mw_set·regulation|`` ⇒ bound term
      ``= |q_mw_set|`` (regulation ≤ 1).
    * **NL gas/water epigraph tightener** (``mass_flow_*_squared``
      with ε = 1e-5): bounded by ``1e-5 · 2·M²`` per branch where
      ``M`` is the largest absolute mass-flow magnitude allowed by
      the Var bounds (default 100 kg/s when none is set).
    * **McCormick-DHS t_pu pull** (ε = 1e-6 per node): bounded by
      ``1e-6 · max(|1-t_min|, |1-t_max|)`` per junction, capped at
      ``1e-6 · 2`` since the formulation pins ``t_pu ∈ [0, 2]``.

    All numbers are absolute upper bounds; signs are absorbed by the
    triangle inequality.  Components lacking the relevant attributes
    contribute 0 — the routine is formulation-agnostic and adds new
    contributors automatically as long as they expose the same
    naming conventions used today.

    Args:
        network: A monee ``Network`` (already built; formulations may
            or may not be attached at the point of the call).
        vm_pu_fallback: Voltage upper bound to use when a power grid
            does not define ``vm_pu_max``.  ``1.1`` matches the
            ``bounds_el`` default of :func:`create_min_load_shedding_problem`.
        max_line_loading: When not ``None``, tighten the MISOCP current
            bound using the line-loading constraint with this
            upper limit.  ``None`` means line-loading limits are not
            enforced and only the SOC physics bound is used (loose but
            correct).

    Returns:
        A non-negative float ``A_max`` such that any feasible solution
        satisfies ``|Σ aux| ≤ A_max``.
    """
    total = 0.0
    for component in network.all_components():
        m = component.model
        # MISOCP-style current·R contribution.  ``br_r``/``br_x`` on
        # PowerLine/Trafo are computed inside ``equations()`` (via
        # ``calc_r_x``) and are still 0 at ``_apply`` time when this
        # hook runs.  Read them via ``calc_r_x`` when available so the
        # MISOCP aux contribution — typically the dominant term —
        # isn't silently dropped from the bound.
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
                # Tighten with the line-loading constraint when enforced.
                # ``line_loading_limit`` writes
                # ``current_pu · (I_base/max_i_ka)² ≤ max_loading²``,
                # i.e. ``current_pu ≤ max_loading² · (max_i_ka/I_base)²``.
                # The from-side base is divided by the tap; use the
                # tighter of the two side-specific bounds.
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
        # LinearHX delivered-heat contribution.  ``q_mw_set`` is set
        # to ``-q_mw`` by the constructor; magnitude is what matters.
        q_mw_set = getattr(m, "q_mw_set", None)
        if isinstance(q_mw_set, (int, float)):
            total += abs(q_mw_set)
        # NL gas / water epigraph tightener: ε·(m_pos² + m_neg²).
        # ε is currently 1e-5 (per nonlinear/{gas,water}.py); we use
        # 1e-4 to be safe against future ε bumps within the same OoM.
        m_pos_sq = getattr(m, "mass_flow_pos_squared", None)
        m_neg_sq = getattr(m, "mass_flow_neg_squared", None)
        if m_pos_sq is not None and m_neg_sq is not None:
            # Read the Var's declared upper bound when available;
            # otherwise assume 100 kg/s — a generous limit for any
            # piped network monee currently models.
            ub_pos = getattr(m_pos_sq, "max", None) or 100.0**2
            ub_neg = getattr(m_neg_sq, "max", None) or 100.0**2
            total += 1e-4 * (ub_pos + ub_neg)
        # McCormick-DHS t_pu-pull tightener: ε·(1 − t_pu), t_pu ∈ [0, 2].
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
):
    """Build a callback that retunes *weights* from the live network.

    Designed to be appended to
    :attr:`OptimizationProblem._controllable_appliables` so the
    framework runs it during ``_apply(network)`` before objective
    expressions are evaluated.  At call time it:

    1. Computes :func:`_aux_objective_upper_bound` over the network.
    2. Sets ``weights['demand'] = max(weights['demand'], α · A_max)``
       so a user-supplied floor is always honoured as a lower bound.
    3. Scales ``weights['generator']`` by the same factor that
       ``weights['demand']`` was scaled by, preserving the
       demand:generator ratio the user encoded with
       ``generator_weight``.

    *max_line_loading* is forwarded to the bound so the MISOCP term is
    tightened via the line-loading constraint when active.  Pass
    ``None`` when ``check_line_loading=False`` so the bound falls back
    to the (loose but correct) SOC physics limit.
    """
    import logging

    _log = logging.getLogger(__name__)

    def _hook(network):
        a_max = _aux_objective_upper_bound(network, max_line_loading=max_line_loading)
        floor = alpha * a_max
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
                "covered by user-supplied demand_weight=%.3g — no change.",
                a_max,
                floor,
                old_demand,
            )

    return _hook


def _shedding_mw(model, gas_mw_factor=None):
    """Return the unserved-energy expression for *model* in MW-equivalent.

    *gas_mw_factor* is the kg/s→MW conversion factor for the enclosing
    gas grid (``3.6 · HHV``); only consulted for ``Sink`` / ``Source``
    models.  When omitted, the :data:`_HHV_DEFAULT` (≈ natural gas) is
    used.

    For "hard" load / generator components (``PowerLoad``,
    ``PowerGenerator``, ``Sink``, ``Source``) the entire setpoint is fed
    into the node balance through ``setpoint · regulation``, so the only
    way to under-serve is to drop ``regulation`` below 1.  The shed is
    therefore ``|setpoint| · (1 − regulation)``.

    For ``HeatExchanger`` branches the LinearHeatExchanger formulation
    pins ``mass_flow_design_kgs`` and only requires
    ``|q_mw_delivered| ≤ |q_mw_set · regulation|``, *not* equality.  The
    actual heat reaching the consumer can therefore fall short of the
    design rating even at ``regulation = 1`` whenever the hydraulic
    temperature spread on the network is too small.  Penalising
    ``(1 − regulation)`` alone misses this physical under-delivery and
    the optimiser has no incentive to push the supply-return ΔT wider.
    Use the *gap* between the design and the actually delivered heat:

        load (q_mw_set > 0):  shed = q_mw_set − q_mw_delivered
        gen  (q_mw_set < 0):  shed = q_mw_delivered − q_mw_set

    When ``regulation = 1`` the gap captures pure physical shortfall;
    when ``regulation < 1`` the gap automatically grows because
    ``|q_mw_delivered| ≤ |q_mw_set · regulation|`` shrinks the upper bound.

    External grids and unknown types return ``0`` (their deviation is
    captured through constraints, not the objective).
    """
    reg = getattr(model, "regulation", 1)

    if isinstance(model, (PowerLoad, PowerGenerator)):
        p = nan_to_zero(model.p_mw)
        # abs(p) * (1 - reg):  p is a Var so use p*p trick? No — we know
        # sign at construction time: PowerLoad.p_mw >= 0, PowerGenerator.p_mw <= 0.
        if isinstance(model, PowerLoad):
            return p * (1 - reg)
        # PowerGenerator: p_mw is negative, so -p_mw is the magnitude
        return (-p) * (1 - reg)

    if isinstance(model, (HeatLoad, HeatGenerator)):
        # HeatLoad.q_mw_heat ≥ 0 (consumption); HeatGenerator.q_mw_heat ≤ 0
        # (load convention — constructor negates the user magnitude).
        q = nan_to_zero(model.q_mw_heat)
        if isinstance(model, HeatLoad):
            return q * (1 - reg)
        return (-q) * (1 - reg)

    if isinstance(model, _HE_OBJECTIVE_TYPES) or type(model) is HeatExchanger:
        q_mw_set = getattr(model, "q_mw_set", 0)
        q_mw_delivered = getattr(model, "q_mw_delivered", None)
        if q_mw_delivered is not None and isinstance(q_mw_set, (int, float)):
            # Use the q_mw_delivered gap to capture both regulation cuts and
            # physical under-delivery from limited supply-return ΔT.
            if q_mw_set > 0:
                # Load: q_mw_delivered ∈ [0, q_mw_set · reg]; gap ≥ 0.
                return q_mw_set - q_mw_delivered
            if q_mw_set < 0:
                # Generator: q_mw_delivered ∈ [q_mw_set · reg, 0]; gap ≥ 0.
                return q_mw_delivered - q_mw_set
        # Fallback (q_mw_set is a Var or q_mw_delivered missing):
        # use the regulation-only proxy.
        if isinstance(q_mw_set, (int, float)):
            return abs(q_mw_set) * (1 - reg)
        return q_mw_set * (1 - reg)

    if isinstance(model, (Sink, Source)):
        mf = nan_to_zero(model.mass_flow)
        factor = gas_mw_factor if gas_mw_factor is not None else 3.6 * _HHV_DEFAULT
        # Sink.mass_flow > 0 (load), Source.mass_flow < 0 (generator)
        if isinstance(model, Sink):
            return mf * factor * (1 - reg)
        return (-mf) * factor * (1 - reg)

    return 0


def _calc_objective(model_to_data):
    """Sum weighted unserved energy across all components.

    *model_to_data* maps each model to a ``(weight, gas_mw_factor)`` tuple.
    ``gas_mw_factor`` is the (kg/s → MW) conversion derived from the
    enclosing :class:`GasGrid`'s ``higher_heating_value`` for ``Sink``/
    ``Source`` models, and ``None`` for everything else.
    """
    return sum(
        _shedding_mw(model, gas_mw_factor=factor) * weight
        for model, (weight, factor) in model_to_data.items()
    )


def create_min_load_shedding_problem(
    *,
    demand_weight=WEIGHT_DEMAND,
    generator_weight=WEIGHT_GENERATOR,
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
    check_vm=True,
    check_pressure=True,
    check_temperature=True,
    check_line_loading=True,
    lex_objectives=False,
    auto_priority_floor=True,
    priority_safety_factor=10.0,
    debug=False,
):
    """Create a minimal load shedding optimisation problem for multi-energy grids.

    Minimises total unserved energy across electrical, gas, and thermal
    carriers.  Each demand, generator, and coupling point receives a
    ``regulation`` variable in [0, 1].  The objective penalises deviation
    from full supply (``regulation = 1``), weighted per component category.

    Gas Sink/Source shed is converted to MW-equivalent using the
    enclosing :class:`~monee.model.grid.GasGrid`'s
    ``higher_heating_value`` (kWh/kg), so multi-fuel networks
    (natural gas, hydrogen, biogas, …) are weighted on a common
    energy basis.  External grids carry no objective contribution —
    their interaction with the rest of the system is governed solely by
    the ``ext_grid_*_bounds`` constraints.

    All component types in a multi-energy grid are covered:

    * **Electrical**: PowerLoad, PowerGenerator, PowerLine (backup)
    * **Gas**: Sink, Source
    * **Thermal**: HeatLoad, HeatGenerator, HeatExchanger variants,
      PassiveHeatExchanger variants
    * **Coupling**: CHP, CHPHG, PowerToHeat, PowerToHeatHG, GasToHeat,
      GasToHeatHG, PowerToGas, GasToPower
    * **Storage**: ElectricStorage, GasStorage, ThermalStorage

    Args:
        demand_weight: Objective penalty for shedding demand components.
        generator_weight: Objective penalty for shedding generators.
        bounds_el: Bus voltage magnitude bounds ``(min, max)`` in pu.
        bounds_gas: Gas junction pressure bounds ``(min, max)`` in pu.
        bounds_heat: Water junction temperature bounds ``(min, max)`` in pu.
        max_line_loading: Maximum line loading in pu (upper bound only;
            ``loading_*_percent`` is non-negative by construction).
        ext_grid_el_bounds: Power bounds for electrical external grids (MW).
        ext_grid_gas_bounds: Mass-flow bounds for gas external grids (kg/s).
        ext_grid_heat_bounds: Mass-flow bounds for water/heat ext grids (kg/s).
        regulation_ramp_limit: Maximum change of ``regulation`` per period.
            ``None`` = no ramp limit.
        include_storages: Make storage components controllable.
        include_ext_grids: Enable external-grid bound constraints
            (``ext_grid_*_bounds``) and add a quadratic slack penalty
            ``(p_mw)²`` / ``(mass_flow)²`` to the objective at weight
            ``demand_weight · EXT_SLACK_WEIGHT_RATIO`` (currently 0.1
            of demand) so ties are resolved toward zero exchange with
            the upstream grid.  ``False`` disables both.
        check_vm: Enforce electrical voltage magnitude bounds.
        check_pressure: Enforce gas pressure bounds.
        check_temperature: Enforce water/heat temperature bounds.
        check_line_loading: Enforce line loading limits.
        lex_objectives: Solve in two-phase lexicographic mode (Pyomo
            backend only).  Phase 1 minimises the load-shedding sum;
            phase 2 minimises the formulation-level tightening terms
            (MISOCP Joule losses, HX delivered-heat slack, etc.) with
            the phase-1 optimum pinned via a cap constraint.  Removes
            the dependence on hand-tuned ``demand_weight`` /
            ``generator_weight`` magnitudes relative to the tightening
            objectives.  Default ``False`` preserves the single
            weighted-sum solve.
        auto_priority_floor: Auto-set ``demand_weight`` to
            ``priority_safety_factor · A_max`` where ``A_max`` is a
            provable over-estimate of the formulation-level
            tightening objective (computed by
            :func:`_aux_objective_upper_bound` from the network at
            ``_apply`` time).  Ensures the shed term dominates the
            aux term in the weighted sum regardless of network size.
            ``generator_weight`` is scaled by the same factor so the
            demand:generator ratio is preserved.  A user-supplied
            ``demand_weight`` is honoured as a *lower bound* — the
            auto floor can raise but never lower it.  No-op when
            ``lex_objectives=True`` (lex enforces priority
            structurally).  Default ``False``.
        priority_safety_factor: Multiplier ``α`` used to set the
            auto floor as ``α · A_max``.  ``10`` is the standard safe
            choice; ``100`` is paranoid.  Only consulted when
            ``auto_priority_floor=True``.  Default ``10.0``.
        debug: Enable debug logging for variable promotion and the
            auto-priority-floor decision.

    Returns:
        An :class:`OptimizationProblem` for single-period or multi-period
        solves.

    Example -- multi-period with ramp limits::

        prob = create_min_load_shedding_problem(
            regulation_ramp_limit=0.3,
            ext_grid_el_bounds=(-3.0, 3.0),
        )
        result = run_multi_period(net, td, steps=4, optimization_problem=prob)
    """
    problem = OptimizationProblem(debug=debug, lex_objectives=lex_objectives)

    # Mutable weight container so ``_set_auto_priority_floor`` can
    # retune the floor at ``_apply`` time once it has seen the network.
    # The objective closure (``weight_fn``) reads from this dict, so
    # any update before the objective is built becomes effective.
    _weights = {"demand": float(demand_weight), "generator": float(generator_weight)}

    # --- Controllable components ---
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
            )
        )

    # # --- Variable bounds ---
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

    # --- Objective: minimise total unserved energy ---
    # All lookups go through ``_weights`` so the auto-priority-floor
    # hook above can retune the weights at ``_apply`` time without
    # rebuilding the closure.
    def weight_fn(model):
        if isinstance(model, _DEMAND_TYPES):
            return _weights["demand"]
        # Subclasses of ``HeatExchanger`` / ``PassiveHeatExchanger`` were
        # already handled by ``_DEMAND_TYPES`` / ``_GENERATOR_TYPES`` above.
        # The branches below cover the bare base classes (rare in user
        # code but used internally) by routing on the sign of ``q_mw_set``
        # — positive ⇒ load, negative ⇒ generator (post-express
        # convention, see :class:`HeatExchanger`).
        if isinstance(model, (HeatExchanger, PassiveHeatExchanger)):
            q = getattr(model, "q_mw_set", 0)
            if isinstance(q, (int, float)):
                return _weights["demand"] if q > 0 else _weights["generator"]
            return _weights["demand"]
        if isinstance(model, _GENERATOR_TYPES):
            return _weights["generator"]
        return 1

    objective_types = _DEMAND_TYPES + _GENERATOR_TYPES + _HE_OBJECTIVE_TYPES

    def _is_objective_model(m):
        # isinstance covers all leaf types; type() is needed for plain
        # HeatExchanger to exclude SubHE (internal compound-model branch).
        return isinstance(m, objective_types) or type(m) is HeatExchanger

    def _is_gas_grid(g):
        # ``Sink``/``Source`` shed in ``_shedding_mw`` is weighted via the
        # gas higher-heating-value factor.  Applying it to water-grid Sinks
        # (heating-loop mass flows) gives them a bogus ~MW-scale penalty
        # in the objective, which dominates real demand shed.  Water-side
        # heat demand is already captured by ``HeatLoad`` children; the
        # water mass flows are derived, not independently sched.
        if g is None:
            return False
        grids = g if isinstance(g, list) else [g]
        return any(
            gg is not None and hasattr(gg, "higher_heating_value") for gg in grids
        )

    # Per-evaluation map populated by ``_objective_models`` and consumed
    # by ``_data_attacher`` to thread the enclosing :class:`GasGrid`'s
    # higher-heating-value into the Sink/Source kg/s → MW conversion.
    model_to_grid: dict = {}

    def _objective_models(network):
        """Like ``Objectives.select`` but with grid-aware Sink/Source filtering.

        Water-grid Sinks/Sources represent heating-loop mass flow, not
        gas demand.  Applying the gas higher-heating-value factor in
        ``_shedding_mw`` to them would yield a ~MW-scale phantom demand
        penalty that has no physical meaning (real heat demand is
        captured by the ``HeatLoad`` children attached to the same
        water junctions).
        """
        model_to_grid.clear()
        out = []
        for model, grid in network.all_models_with_grid():
            if not _is_objective_model(model):
                continue
            if isinstance(model, (Sink, Source)) and not _is_gas_grid(grid):
                continue
            out.append(model)
            model_to_grid[model] = grid
        return out

    def _data_attacher(model):
        """Bundle weight + per-grid gas conversion factor for the objective.

        The ``Objective`` machinery only exposes the *model* to its
        data-attacher; the enclosing grid is recovered from
        :data:`model_to_grid` (populated by :func:`_objective_models` on
        the same evaluation).  For non-gas components ``factor`` is
        ``None`` and :func:`_shedding_mw` ignores it.
        """
        weight = weight_fn(model)
        factor = None
        if isinstance(model, (Sink, Source)):
            factor = _gas_mw_factor(model_to_grid.get(model))
        return (weight, factor)

    objectives = Objectives()
    objectives.with_models(_objective_models).data(_data_attacher).calculate(
        _calc_objective
    )

    # --- Ext-grid slack penalty -----------------------------------------
    # Soft nudge for ExtPowerGrid / ExtHydrGrid exchange toward zero (the
    # natural lower bound of |p_mw| / |mass_flow|) so ties are resolved
    # toward self-sufficiency.  Weight is ``demand_weight ·
    # EXT_SLACK_WEIGHT_RATIO`` and is read live from ``_weights`` so it
    # tracks the auto-priority-floor scaling.  Quadratic (``x²``) so we
    # don't need aux vars — the MISOCP problem already has quadratic
    # constraints.  Off when ``include_ext_grids=False`` since there
    # would be no ext-grid bounds either.
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

    # --- Constraints ---
    constraints = Constraints()

    if check_line_loading:
        from monee.problem.utils import line_loading_limit

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
