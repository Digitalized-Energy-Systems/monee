import warnings

from monee.model.branch import (
    GenericPowerBranch,
    HeatExchanger,
    PassiveHeatExchanger,
    hx_is_generating,
)
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    HeatGenerator,
    PowerGenerator,
    Source,
)
from monee.model.core import is_plain_number
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
    SubHE,
)
from monee.model.storage import ElectricStorage, GasStorage
from monee.problem.core import (
    AttributeParameter,
    Constraints,
    Objectives,
    OptimizationProblem,
)
from monee.problem.utils import (
    _live,
    gas_mw_per_kgs,
    line_loading_limit,
    make_vm_bounds_hook,
    make_water_temperature_bounds_hook,
    water_slack_heat_terms,
)

_UNSET = object()
_DEFAULT_BOUNDS_T_K = (278.15, 423.15)
_SUPPLY_ONLY = (None, 0.0)
_CARRIERS = frozenset({"power", "gas", "heat"})
_ENABLE_GAS = "pass gas_cost_default to dispatch it"
_ENABLE_HEAT = "pass heat_cost_default to dispatch it"

_CP_CARRIERS = (
    (
        (CHPControlNode, CHPHGControlNode, GasToHeatControlNode, GasToHeatHG),
        {"gas", "heat"},
    ),
    ((PowerToHeatControlNode, PowerToHeatHG), {"heat"}),
    ((GasToPower, PowerToGas), {"gas"}),
)
_CP_ELECTRIC = (
    CHPControlNode,
    CHPHGControlNode,
    PowerToHeatControlNode,
    PowerToHeatHG,
    GasToPower,
    PowerToGas,
)

_CP_REGULATION = [
    (
        "regulation",
        AttributeParameter(
            min=lambda attr, val: 0,
            max=lambda attr, val: val if is_plain_number(val) else 1,
            val=lambda attr, val: val if is_plain_number(val) else 1,
        ),
    )
]


def _cost_or(model, default):
    if not hasattr(model, "cost") or model.cost is None:
        return default
    return model.cost


def _served(model, attr):
    value = getattr(model, attr)
    regulation = model.regulation
    if is_plain_number(regulation) and regulation == 1:
        return value
    return value * regulation


def _is_gas_supply(model, component):
    return type(model) in (Source, ExtHydrGrid) and isinstance(component.grid, GasGrid)


def _is_gas_source(model, component):
    return (
        type(model) is Source
        and isinstance(component.grid, GasGrid)
        and _live(component)
    )


def _is_water_slack(model, component):
    return type(model) is ExtHydrGrid and isinstance(component.grid, WaterGrid)


def _hx_generates(model):
    """Classify by the plain setpoint, so call it before promotion; afterwards
    read the ``_ed_heat_generator`` stamp via :func:`_is_hx_generator`."""
    if isinstance(model, SubHE) or not hx_is_generating(model):
        return False
    on_off = getattr(model, "on_off", 1)
    return not (is_plain_number(on_off) and on_off == 0)


def _is_hx_generator(model):
    return getattr(model, "_ed_heat_generator", False)


def _cp_carriers(model):
    for classes, carriers in _CP_CARRIERS:
        if isinstance(model, classes):
            return carriers
    return None


def _validate(bounds_lp, heat_on, bounds_ext_heat_mw, bounds_t_k):
    if bounds_lp[0] != 0:
        raise ValueError(
            "bounds_lp[0] must be 0: a non-zero minimum line loading would "
            f"force flow onto every line, got bounds_lp={bounds_lp!r}."
        )
    ext_heat_given = bounds_ext_heat_mw is not _UNSET and bounds_ext_heat_mw is not None
    t_k_given = bounds_t_k is not _UNSET and bounds_t_k is not None
    if not heat_on:
        for name, given, value in (
            ("bounds_ext_heat_mw", ext_heat_given, bounds_ext_heat_mw),
            ("bounds_t_k", t_k_given, bounds_t_k),
        ):
            if given:
                raise ValueError(
                    f"{name} only applies when heat is dispatched; pass "
                    f"heat_cost_default as well, got {name}={value!r}."
                )
        return
    if ext_heat_given:
        lo, hi = bounds_ext_heat_mw
        if lo is not None and hi is not None and lo > hi:
            raise ValueError(
                "bounds_ext_heat_mw must be (lo, hi) with lo <= hi, got "
                f"{bounds_ext_heat_mw!r}."
            )
    if t_k_given and not 0 < bounds_t_k[0] < bounds_t_k[1]:
        raise ValueError(
            f"bounds_t_k must be (lo, hi) in K with 0 < lo < hi, got {bounds_t_k!r}."
        )


def _ramp_limits(ramp_limit, enabled):
    if ramp_limit is None:
        return {}
    if not isinstance(ramp_limit, dict):
        return dict.fromkeys(enabled, ramp_limit)
    unknown = set(ramp_limit) - _CARRIERS
    if unknown:
        raise ValueError(
            f"ramp_limit keys must be a subset of {sorted(_CARRIERS)}, got {sorted(unknown)}."
        )
    disabled = set(ramp_limit) - enabled
    if disabled:
        raise ValueError(
            f"ramp_limit names {sorted(disabled)}, which this problem does not "
            "dispatch; pass gas_cost_default / heat_cost_default for them."
        )
    return dict(ramp_limit)


def _scaled(value, regulation):
    if is_plain_number(regulation) and regulation != 1:
        return regulation * value
    return value


def _ramp_rows(attr, limit, factor=None):
    """Rows bounding the change of the served quantity (setpoint times a plain
    ``regulation``) between consecutive periods; a period whose regulation is
    unknown reuses the current one."""

    def _rows(model, cid, ts):
        model_type = type(model)
        prev = ts.get(cid, attr, model_type=model_type)
        if prev is None:
            return []
        regulation = getattr(model, "regulation", 1)
        prev_regulation = ts.get(cid, "regulation", model_type=model_type)
        if prev_regulation is None:
            prev_regulation = regulation
        step = _scaled(getattr(model, attr), regulation) - _scaled(
            prev, prev_regulation
        )
        if factor is not None:
            step = factor(model) * step
        return [step <= limit, -step <= limit]

    return _rows


def create_economic_dispatch_problem(
    gen_cost_default=1.0,  # NOSONAR
    ext_grid_cost_default=None,
    bounds_vm=(0.9, 1.1),
    bounds_lp=(0, 1.0),
    ext_grid_bounds=None,
    ramp_limit=None,
    check_vm=True,
    check_lp=True,
    debug=False,
    *,
    gas_cost_default=None,
    heat_cost_default=None,
    bounds_ext_heat_mw=_UNSET,
    bounds_t_k=_UNSET,
    dispatch_coupling_points=True,
    include_storages=False,
):
    """Economic dispatch minimising the cost of supplying power, gas and heat.
    Every price is currency per MW and period.

    Power: ``Sigma cost * p_gen`` over the PowerGenerators, priced on the power
    the bus balance sees (``p_mw * regulation``). Set each generator's
    ``cost`` at creation (``mx.create_power_generator(..., cost=...)``),
    directly on the model, or per period via
    TimeseriesData.add_objective_data; unset costs fall back to
    ``gen_cost_default``.

    ``ext_grid_cost_default=None`` (the default) leaves the external power grid
    out of the objective entirely: slack import is free, so the optimum
    degenerates to importing everything and dispatching no local generation.
    Pass a cost to price the exchange: ``ExtPowerGrid.p_mw`` follows the load
    convention (import negative, export positive), so import is charged at
    ``cost`` and export is credited at the same price. ``ext_grid_bounds``
    bounds ``ExtPowerGrid.p_mw`` in that convention: capping the import at X MW
    is ``ext_grid_bounds=(-X, 0)``.

    ``bounds_lp`` only supports a zero lower bound: line loading is capped at
    ``bounds_lp[1]``, and a non-zero minimum loading is rejected.

    Gas (``gas_cost_default``, None keeps gas out): every gas Source becomes a
    decision in [0, its setpoint], and every gas Source and gas ExtHydrGrid is
    priced at its own ``cost`` or this default, per MW of higher heating value
    of its grid's gas (``3.6 * higher_heating_value_kwh_per_kg`` per kg/s).
    Gas grid import is charged and export credited; cap export with the ext
    grid's ``max_export_kgs``.

    Heat (``heat_cost_default``, None keeps heat out): every HeatGenerator and
    every generating HeatExchanger / PassiveHeatExchanger becomes a decision
    in [0, its setpoint] and is priced at its own ``cost`` or this default.
    Each water ExtHydrGrid is priced on the heat it supplies to its island.
    Heat exchanger duties become exact, so heat demand is hard.
    ``bounds_ext_heat_mw`` = (lo, hi) bounds that slack exchange in MW in the
    load convention (supply negative); the default (None, 0.0) lets the slack
    supply heat but not absorb a surplus, None removes the bound (a surplus
    is then credited). ``bounds_t_k`` bounds every water temperature in K
    (default 278.15 to 423.15; None disables it). Unlike ``bounds_t`` of the
    load-shedding problem it is in K, not per unit. Neither may be passed
    without heat.

    Coupling points: with ``dispatch_coupling_points=True`` each coupler's
    ``regulation`` becomes a decision in [0, its current regulation] once all
    of its non-electric carriers are dispatched (CHP and gas-to-heat need gas
    and heat, power-to-heat needs heat, gas-to-power and power-to-gas need
    gas). Couplers carry no price: fuel is paid where it enters the network.

    ``include_storages=True`` makes ElectricStorage (and GasStorage when gas
    is dispatched) controllable in run_multi_period / run_mpc; other solves
    keep their setpoints and warn.

    ``ramp_limit``: a float limits every dispatched unit of every dispatched
    carrier by that many MW per period (power only: today's generator ramp);
    a dict with keys among "power", "gas" and "heat" sets one limit per
    carrier. Seed the first period through ``initial_state``.

    Units without their own ``cost`` tie with a slack priced at the same
    default, so their split is arbitrary; give either one its own cost."""
    gas_on = gas_cost_default is not None
    heat_on = heat_cost_default is not None
    _validate(bounds_lp, heat_on, bounds_ext_heat_mw, bounds_t_k)
    enabled = (
        {"power"} | ({"gas"} if gas_on else set()) | ({"heat"} if heat_on else set())
    )
    ramps = _ramp_limits(ramp_limit, enabled)

    problem = OptimizationProblem(debug=debug)
    objectives = Objectives()
    constraints = Constraints()

    _add_power(
        problem,
        objectives,
        constraints,
        optional=gas_on or heat_on,
        gen_cost_default=gen_cost_default,
        ext_grid_cost_default=ext_grid_cost_default,
        ext_grid_bounds=ext_grid_bounds,
        bounds_vm=bounds_vm if check_vm else None,
        max_loading=bounds_lp[1] if check_lp else None,
    )
    if gas_on:
        _add_gas(problem, objectives, gas_cost_default)
    if heat_on:
        _add_heat(
            problem,
            objectives,
            constraints,
            heat_cost_default,
            _SUPPLY_ONLY if bounds_ext_heat_mw is _UNSET else bounds_ext_heat_mw,
            _DEFAULT_BOUNDS_T_K if bounds_t_k is _UNSET else bounds_t_k,
        )
    if dispatch_coupling_points and (gas_on or heat_on):
        _add_coupling_points(problem, frozenset(enabled), ext_grid_cost_default)
    if include_storages:
        problem._controllable_appliables.append(_storage_hook(gas_on))
    _add_ramps(constraints, ramps, optional=gas_on or heat_on)
    problem._controllable_appliables.append(_ignored_costs_hook(gas_on, heat_on))

    problem.objectives = objectives
    problem.constraints = constraints
    return problem


def _add_power(
    problem,
    objectives,
    constraints,
    optional,
    gen_cost_default,
    ext_grid_cost_default,
    ext_grid_bounds,
    bounds_vm,
    max_loading,
):
    problem.controllable_generators(["p_mw"], optional=optional)
    include_ext_grid = ext_grid_cost_default is not None
    if include_ext_grid:
        problem.controllable_ext()
        if ext_grid_bounds is not None:
            problem.bounds(
                ext_grid_bounds,
                lambda m, _: isinstance(m, ExtPowerGrid),
                ["p_mw"],
            )
    if bounds_vm is not None:
        # Bound the actual decision var (vm_pu under the AC NLP, vm_pu_squared
        # under MISOCP); a static bound on vm_pu alone is a no-op when it is
        # only a reporting Intermediate.
        problem._controllable_appliables.append(make_vm_bounds_hook(bounds_vm))

    objectives.select(
        lambda m: isinstance(m, PowerGenerator),
        optional=optional,
        looked_for="PowerGenerator components (dispatch cost objective)",
    ).calculate(
        lambda models: sum(
            _cost_or(m, gen_cost_default) * (-_served(m, "p_mw")) for m in models
        )
    )
    if include_ext_grid:
        objectives.select(
            lambda m: isinstance(m, ExtPowerGrid),
            looked_for="ExtPowerGrid components (exchange cost objective)",
        ).calculate(
            lambda models: sum(
                _cost_or(m, ext_grid_cost_default) * (-m.p_mw) for m in models
            )
        )

    # loading_*_pu are passive intermediates in every electricity formulation,
    # so this constraint, not a var-bounds override, enforces the line cap.
    if max_loading is not None:
        constraints.select_types(GenericPowerBranch, optional=True).equation(
            lambda model: line_loading_limit(model, "from", max_loading)
        ).equation(lambda model: line_loading_limit(model, "to", max_loading))
    if include_ext_grid and ext_grid_bounds is not None:
        constraints.select_types(ExtPowerGrid).equation(
            lambda model, _b=ext_grid_bounds: model.p_mw >= _b[0]
        ).equation(lambda model, _b=ext_grid_bounds: model.p_mw <= _b[1])


def _add_gas(problem, objectives, gas_cost_default):
    problem.controllable(
        ["mass_flow_kgs"],
        _is_gas_source,
        optional=True,
        looked_for="Source components on a GasGrid (gas dispatch)",
    )

    def _stamp_gas_factors(network):
        for child in network.childs:
            if _is_gas_supply(child.model, child):
                child.model._ed_mw_per_kgs = gas_mw_per_kgs(child.grid)

    problem._controllable_appliables.append(_stamp_gas_factors)
    objectives.select(
        _is_gas_supply,
        optional=True,
        looked_for="gas Source/ExtHydrGrid components (gas supply cost)",
    ).calculate(
        lambda models: sum(
            _cost_or(m, gas_cost_default)
            * m._ed_mw_per_kgs
            * (-_served(m, "mass_flow_kgs"))
            for m in models
        )
    )


def _add_heat(
    problem, objectives, constraints, heat_cost_default, bounds_ext_heat_mw, bounds_t_k
):
    _add_heat_controls(problem, bounds_t_k)
    _add_heat_objectives(objectives, heat_cost_default)
    if bounds_ext_heat_mw is not None:
        _bound_heat_exchange(constraints, bounds_ext_heat_mw)


def _add_heat_controls(problem, bounds_t_k):
    problem.controllable(
        ["q_mw_heat"],
        lambda m, c: isinstance(m, HeatGenerator) and _live(c),
        optional=True,
        looked_for="HeatGenerator components (heat dispatch)",
    )
    # q_mw_set, not regulation: regulation also scales a fixed-flow
    # exchanger's through-flow, which ties it to the loop's mass balance.
    problem.controllable(
        ["q_mw_set"],
        lambda m, c: _hx_generates(m) and _live(c),
        optional=True,
        looked_for="generating HeatExchanger/PassiveHeatExchanger branches (heat dispatch)",
    )

    def _mark_heat_exchangers(network):
        for component in network.all_components():
            if not _live(component):
                continue
            model = component.model
            if isinstance(model, HeatExchanger):
                model._he_duty_exact = True
            if _hx_generates(model):
                model._ed_heat_generator = True

    problem._controllable_appliables.append(_mark_heat_exchangers)
    if bounds_t_k is not None:
        problem._controllable_appliables.append(
            make_water_temperature_bounds_hook(bounds_t_k)
        )


def _add_heat_objectives(objectives, heat_cost_default):
    objectives.select(
        lambda m, _c: isinstance(m, HeatGenerator),
        optional=True,
        looked_for="HeatGenerator components (heat supply cost)",
    ).calculate(
        lambda models: sum(
            _cost_or(m, heat_cost_default) * (-_served(m, "q_mw_heat")) for m in models
        )
    )
    objectives.select(
        lambda m, _c: _is_hx_generator(m),
        optional=True,
        looked_for="generating heat exchangers (heat supply cost)",
    ).calculate(
        lambda models: sum(_cost_or(m, heat_cost_default) * (-m.q_mw) for m in models)
    )
    objectives.with_models(water_slack_heat_terms).calculate(
        lambda terms: sum(
            _cost_or(slack.model, heat_cost_default) * q_mw for slack, q_mw in terms
        )
    )


def _bound_heat_exchange(constraints, bounds_ext_heat_mw):
    lo, hi = bounds_ext_heat_mw
    exchange = constraints.with_models(
        lambda network: [q_mw for _, q_mw in water_slack_heat_terms(network)]
    )
    if lo is not None:
        exchange.equation(lambda q_mw: -q_mw >= lo)
    if hi is not None:
        exchange.equation(lambda q_mw: -q_mw <= hi)


def _add_coupling_points(problem, enabled, ext_grid_cost_default):
    def _dispatched(model, component):
        carriers = _cp_carriers(model)
        return carriers is not None and carriers <= enabled and _live(component)

    problem.controllable(
        _CP_REGULATION,
        _dispatched,
        optional=True,
        looked_for="coupling points whose carriers are dispatched",
    )
    if ext_grid_cost_default is None:
        problem._controllable_appliables.append(_free_power_hook(_dispatched))


def _free_power_hook(dispatched):
    reported = []

    def _warn_free_power_for_couplers(network):
        if reported:
            return
        components = network.all_components()
        has_ext_power = any(
            isinstance(c.model, ExtPowerGrid) and _live(c) for c in components
        )
        electric_cp = any(
            isinstance(c.model, _CP_ELECTRIC) and dispatched(c.model, c)
            for c in components
        )
        if has_ext_power and electric_cp:
            reported.append(True)
            warnings.warn(
                "coupling points are dispatched but ext_grid_cost_default is "
                "None: electricity from the external grid is free, so their "
                "merit order is degenerate. See the problems/economic_dispatch "
                "docs page.",
                stacklevel=2,
            )

    return _warn_free_power_for_couplers


def _storage_hook(gas_on):
    storage_types = (ElectricStorage, GasStorage) if gas_on else (ElectricStorage,)
    reported = []

    def _dispatch_storages(network):
        if getattr(network, "_solve_multi_period", False):
            for component in network.all_components():
                if isinstance(component.model, storage_types) and _live(component):
                    component.model.make_controllable()
            return
        if not reported:
            reported.append(True)
            warnings.warn(
                "include_storages only dispatches storages in run_multi_period "
                "/ run_mpc; this solve keeps every storage at its setpoint.",
                stacklevel=2,
            )

    return _dispatch_storages


def _add_ramps(constraints, ramps, optional):
    if "power" in ramps:
        constraints.select(
            lambda comp: (
                isinstance(comp.model, PowerGenerator)
                and comp.active
                and (not comp.ignored)
            ),
            optional=optional,
            looked_for="PowerGenerator components (ramp_limit)",
        ).temporal_equation(_ramp_rows("p_mw", ramps["power"]))
    if "gas" in ramps:
        constraints.select(
            _is_gas_source,
            optional=True,
            looked_for="gas Source components (ramp_limit)",
        ).temporal_equation(
            _ramp_rows("mass_flow_kgs", ramps["gas"], lambda m: m._ed_mw_per_kgs)
        )
    if "heat" in ramps:
        constraints.select(
            lambda m, c: isinstance(m, HeatGenerator) and _live(c),
            optional=True,
            looked_for="HeatGenerator components (ramp_limit)",
        ).temporal_equation(_ramp_rows("q_mw_heat", ramps["heat"]))
        constraints.select(
            lambda m, c: _is_hx_generator(m) and _live(c),
            optional=True,
            looked_for="generating heat exchangers (ramp_limit)",
        ).temporal_equation(_ramp_rows("q_mw_set", ramps["heat"]))


def _unread_cost(model, component, gas_on, heat_on):
    """Why the objective ignores the ``cost`` on *model*, or None when it is
    read or absent."""
    if getattr(model, "cost", None) is None:
        return None
    if _is_gas_supply(model, component):
        return None if gas_on else _ENABLE_GAS
    if isinstance(model, HeatGenerator) or _is_water_slack(model, component):
        return None if heat_on else _ENABLE_HEAT
    if isinstance(model, (HeatExchanger, PassiveHeatExchanger)):
        if not _hx_generates(model):
            return (
                "the economic dispatch never prices a consuming, zero-duty or "
                "switched-off exchanger"
            )
        return None if heat_on else _ENABLE_HEAT
    if type(model) is Source:
        return "the economic dispatch never prices a water Source"
    return None


def _ignored_costs_hook(gas_on, heat_on):
    reported = []

    def _warn_ignored_costs(network):
        if reported:
            return
        for component in network.all_components():
            if not _live(component):
                continue
            reason = _unread_cost(component.model, component, gas_on, heat_on)
            if reason is None:
                continue
            reported.append(True)
            warnings.warn(
                f"{type(component.model).__name__} {component.id} carries a "
                f"cost the objective ignores: {reason}.",
                stacklevel=2,
            )
            return

    return _warn_ignored_costs


#: Alias for :func:`create_economic_dispatch_problem` (mirrors load_shedding naming).
create_multi_period_economic_dispatch_problem = create_economic_dispatch_problem
