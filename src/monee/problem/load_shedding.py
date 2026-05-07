from monee.model.branch import (
    GenericPowerBranch,
    HeatExchanger,
    HeatExchangerGenerator,
    HeatExchangerLoad,
    PassiveHeatExchanger,
    PassiveHeatExchangerGenerator,
    PassiveHeatExchangerLoad,
    PowerLine,
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
    PowerToHeat,
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

HHV = 15.3


_CP_ZERO_TYPES = (
    CHPControlNode,
    CHPHGControlNode,
    PowerToHeatControlNode,
    PowerToHeat,
    PowerToHeatHG,
    PowerToGas,
    GasToPower,
    GasToHeatControlNode,
    GasToHeatHG,
)


def retrieve_power_uniform(model):
    if isinstance(
        model,
        HeatExchangerLoad
        | HeatExchangerGenerator
        | HeatExchanger
        | PassiveHeatExchangerLoad
        | PassiveHeatExchangerGenerator
        | PassiveHeatExchanger,
    ):
        return (model.q_mw_set * model.regulation, model.q_mw_set)
    elif isinstance(model, HeatLoad | HeatGenerator):
        return (nan_to_zero(model.q_mw_heat) * model.regulation, model.q_mw_heat)
    elif isinstance(model, PowerLoad | PowerGenerator):
        return (nan_to_zero(model.p_mw) * model.regulation, model.p_mw)
    elif isinstance(model, ExtPowerGrid):
        return model.p_mw, 0
    elif isinstance(model, ExtHydrGrid):
        return model.mass_flow * 3.6 * HHV, 0
    elif isinstance(model, Sink | Source):
        return (
            nan_to_zero(model.mass_flow) * 3.6 * HHV * model.regulation,
            model.mass_flow * 3.6 * HHV,
        )
    elif isinstance(model, _CP_ZERO_TYPES):
        return (0, 0)
    elif isinstance(model, PowerLine):
        if model.backup:
            return (0, model.on_off * 0.1)
        return (0, 0)
    raise ValueError(f"The model {type(model)} is not a known load.")


def calculate_objective(model_to_data):
    power_coeff = [
        (
            model,
            (retrieve_power_uniform(model)[1] - retrieve_power_uniform(model)[0])
            * data,
        )
        for model, data in model_to_data.items()
    ]
    return sum([t[1] for t in power_coeff])


def create_multi_period_load_shedding_optimization_problem(
    load_weight=10,
    bounds_el=(0.9, 1.1),
    bounds_heat=(0.9, 1.1),
    bounds_gas=(0.9, 1.1),
    bounds_lp=(0, 1.5),
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
    regulation_ramp_limit=None,
    use_ext_grid_bounds=True,
    use_ext_grid_objective=True,
    check_lp=True,
    check_vm=True,
    check_pressure=True,
    check_t=True,
    debug=False,
):
    """Create an optimization problem for multi-period minimal load shedding.

    Minimises total unserved energy across all periods.  Each demand and
    generator receives a ``regulation`` variable (0–1) that scales its
    output.  The objective penalises deviation from full supply
    (``regulation = 1``), weighted by component type.

    Compared to :func:`create_load_shedding_optimization_problem`, this
    variant adds an optional **regulation ramp-rate constraint** that
    limits how fast load shedding can change between consecutive periods
    (e.g. to model operator reaction time or equipment limits).

    Args:
        load_weight: Penalty weight for shedding demand-side components.
        bounds_el: Voltage magnitude bounds ``(min, max)`` in pu.
        bounds_heat: Temperature bounds ``(min, max)`` in pu.
        bounds_gas: Pressure bounds ``(min, max)`` in pu.
        bounds_lp: Line loading bounds ``(min, max)`` in pu.
        ext_grid_el_bounds: Active power bounds for electric external grids.
        ext_grid_gas_bounds: Mass-flow bounds for gas external grids.
        regulation_ramp_limit: Maximum change of ``regulation`` between
            consecutive periods.  ``None`` (default) means no ramp limit.
            A value of e.g. ``0.3`` means regulation can change by at
            most 0.3 per period (prevents abrupt load shedding swings).
        use_ext_grid_bounds: Apply ext-grid power/flow constraints.
        use_ext_grid_objective: Include ext-grid in controllable set.
        check_lp: Enforce line-loading limits.
        check_vm: Enforce voltage magnitude bounds.
        check_pressure: Enforce gas pressure bounds.
        check_t: Enforce temperature bounds.
        debug: Enable debug logging for variable promotion.

    Returns:
        An :class:`OptimizationProblem` suitable for
        :func:`~monee.simulation.multi_period.run_multi_period` or
        :func:`~monee.simulation.multi_period.run_mpc`.

    Example::

        import monee
        from monee.problem.load_shedding import (
            create_multi_period_load_shedding_optimization_problem,
        )

        td = monee.TimeseriesData()
        td.add_child_series(load_id, "p_mw", [2.0, 5.0, 3.0, 1.0])

        prob = create_multi_period_load_shedding_optimization_problem(
            regulation_ramp_limit=0.3,
            ext_grid_el_bounds=(-3.0, 3.0),
        )
        result = monee.run_multi_period(net, td, steps=4, optimization_problem=prob)
    """
    problem = OptimizationProblem(debug=debug)
    problem.controllable_demands(REGULATION_ATTR)
    problem.controllable_generators(REGULATION_ATTR)
    problem.controllable_cps(REGULATION_ATTR)
    if use_ext_grid_objective:
        problem.controllable_ext()
    problem.controllable_backup_lines()
    if check_vm:
        problem.bounds(bounds_el, lambda m, _: type(m) is Bus, ["vm_pu"])
    if check_t:
        problem.bounds(
            bounds_heat,
            lambda m, g: type(m) is Junction and type(g) is WaterGrid,
            ["t_pu"],
        )
    if check_pressure:
        problem.bounds(bounds_gas, lambda m, _: type(m) is Junction, ["pressure_pu"])

    # --- Objective: minimise total unserved energy across all periods ---
    # Use select() instead of with_models(controllables_link) because
    # controllables_link references models from the last _apply() call,
    # which in multi-period mode points to the last period's network copy
    # rather than the current period being evaluated.
    objectives = Objectives()

    _controllable_types = (
        PowerLoad,
        HeatLoad,
        HeatExchangerLoad,
        PassiveHeatExchangerLoad,
        Sink,
        HeatExchanger,
        PassiveHeatExchanger,
        PowerGenerator,
        HeatGenerator,
        HeatExchangerGenerator,
        PassiveHeatExchangerGenerator,
        Source,
        CHPControlNode,
        CHPHGControlNode,
        PowerToHeatControlNode,
        PowerToHeatHG,
        PowerToGas,
        GasToPower,
        GasToHeatControlNode,
        GasToHeatHG,
    )
    if use_ext_grid_objective:
        _controllable_types = _controllable_types + (ExtPowerGrid, ExtHydrGrid)

    _DEMAND_WEIGHT_TYPES = (
        HeatExchangerLoad,
        PassiveHeatExchangerLoad,
        Sink,
        PowerLoad,
        HeatLoad,
    )
    _CP_WEIGHT_TYPES = (
        CHPControlNode,
        CHPHGControlNode,
        PowerToHeatControlNode,
        PowerToHeatHG,
        GasToHeatControlNode,
        GasToHeatHG,
        PowerToGas,
        GasToPower,
        PowerToHeat,
    )

    def calc_weight(model):
        weight = 1
        if isinstance(model, _DEMAND_WEIGHT_TYPES):
            weight = load_weight
        elif isinstance(model, _CP_WEIGHT_TYPES):
            weight = load_weight - 1
        elif isinstance(model, ExtPowerGrid | ExtHydrGrid):
            weight = 5
        return weight

    objectives.select(lambda m, _types=_controllable_types: isinstance(m, _types)).data(
        calc_weight
    ).calculate(calculate_objective)

    # --- Constraints ---
    constraints = Constraints()
    if use_ext_grid_bounds:
        constraints.select_types(ExtPowerGrid).equation(
            lambda model: model.p_mw >= ext_grid_el_bounds[0]
        ).equation(lambda model: model.p_mw <= ext_grid_el_bounds[1])
        constraints.select(
            lambda comp: type(comp.grid) is GasGrid and type(comp.model) is ExtHydrGrid
        ).equation(lambda model: model.mass_flow >= ext_grid_gas_bounds[0]).equation(
            lambda model: model.mass_flow <= ext_grid_gas_bounds[1]
        )

    if check_lp:
        constraints.select_types(GenericPowerBranch).equation(
            lambda model: model.loading_from_percent <= bounds_lp[1]
        ).equation(lambda model: model.loading_to_percent <= bounds_lp[1])

    if regulation_ramp_limit is not None:
        constraints.regulation_ramp(regulation_ramp_limit)

    problem.constraints = constraints
    problem.objectives = objectives
    return problem


def create_load_shedding_optimization_problem(
    load_weight=10,
    bounds_el=(0.9, 1.1),
    bounds_heat=(0.9, 1.1),
    bounds_gas=(0.9, 1.1),
    bounds_lp=(0, 1.5),
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
    use_ext_grid_bounds=True,
    use_ext_grid_objective=True,
    check_lp=True,
    check_vm=True,
    check_pressure=True,
    check_t=True,
    debug=False,
):
    problem = OptimizationProblem(debug=debug)
    problem.controllable_demands(REGULATION_ATTR)
    problem.controllable_generators(REGULATION_ATTR)
    problem.controllable_cps(REGULATION_ATTR)
    if use_ext_grid_objective:
        problem.controllable_ext()
    problem.controllable_backup_lines()
    if check_vm:
        problem.bounds(bounds_el, lambda m, _: type(m) is Bus, ["vm_pu"])
    if check_t:
        problem.bounds(
            bounds_heat,
            lambda m, g: type(m) is Junction and type(g) is WaterGrid,
            ["t_pu"],
        )
    if check_pressure:
        problem.bounds(bounds_gas, lambda m, _: type(m) is Junction, ["pressure_pu"])

    objectives = Objectives()

    _DEMAND_WEIGHT_TYPES = (
        HeatExchangerLoad,
        PassiveHeatExchangerLoad,
        Sink,
        PowerLoad,
        HeatLoad,
    )
    _CP_WEIGHT_TYPES = (
        CHPControlNode,
        CHPHGControlNode,
        PowerToHeatControlNode,
        PowerToHeatHG,
        GasToHeatControlNode,
        GasToHeatHG,
        PowerToGas,
        GasToPower,
        PowerToHeat,
    )

    def calc_weight(model):
        weight = 1
        if isinstance(model, _DEMAND_WEIGHT_TYPES):
            weight = load_weight
        elif isinstance(model, _CP_WEIGHT_TYPES):
            weight = load_weight - 1
        elif isinstance(model, ExtPowerGrid | ExtHydrGrid):
            weight = 5
        return weight

    objectives.with_models(problem.controllables_link).data(calc_weight).calculate(
        calculate_objective
    )

    constraints = Constraints()
    if use_ext_grid_bounds:
        constraints.select_types(ExtPowerGrid).equation(
            lambda model: model.p_mw >= ext_grid_el_bounds[0]
        ).equation(lambda model: model.p_mw <= ext_grid_el_bounds[1])
        constraints.select(
            lambda comp: type(comp.grid) is GasGrid and type(comp.model) is ExtHydrGrid
        ).equation(lambda model: model.mass_flow >= ext_grid_gas_bounds[0]).equation(
            lambda model: model.mass_flow <= ext_grid_gas_bounds[1]
        )

    if check_lp:
        constraints.select_types(GenericPowerBranch).equation(
            lambda model: model.loading_from_percent <= bounds_lp[1]
        ).equation(lambda model: model.loading_to_percent <= bounds_lp[1])
    problem.constraints = constraints
    problem.objectives = objectives
    return problem
