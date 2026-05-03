"""Minimal load shedding optimisation for multi-energy grids.

Provides a formulation that minimises total unserved energy across
electrical, gas, and thermal carriers.  Works for both single-period
and multi-period solves.

Typical usage::

    from monee.problem.min_load_shedding import (
        create_min_load_shedding_problem,
    )
    import monee.simulation as sim

    prob = create_min_load_shedding_problem()
    result = sim.run(net, optimization_problem=prob)
"""

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

WEIGHT_DEMAND = 10
WEIGHT_GENERATOR = 1
WEIGHT_EXT_GRID = 5

# Gas higher heating value for MW-equivalent conversion.
_HHV_DEFAULT = 15.3  # kWh/kg
_GAS_MW_FACTOR = 3.6 * _HHV_DEFAULT  # mass_flow [kg/s] * factor ≈ MW

# Types that participate in the load-shedding objective.
_DEMAND_TYPES = (
    PowerLoad,
    HeatExchangerLoad,
    PassiveHeatExchangerLoad,
    Sink,
)
_GENERATOR_TYPES = (
    PowerGenerator,
    HeatExchangerGenerator,
    PassiveHeatExchangerGenerator,
    Source,
)
_EXT_GRID_TYPES = (ExtPowerGrid, ExtHydrGrid)
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


def _shedding_mw(model):
    """Return the unserved-energy expression for *model* in MW-equivalent.

    The returned value is ``|setpoint| * (1 - regulation)`` — always
    non-negative when ``regulation`` drops below 1, regardless of the
    load-convention sign of the underlying attribute.

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

    if isinstance(model, _HE_OBJECTIVE_TYPES) or type(model) is HeatExchanger:
        q_mw = getattr(model, "q_w_set", 0) / 1e6
        # q_w_set > 0 for loads, < 0 for generators
        if isinstance(q_mw, (int, float)):
            return abs(q_mw) * (1 - reg)
        return q_mw * (1 - reg)  # fallback for Var

    if isinstance(model, (Sink, Source)):
        mf = nan_to_zero(model.mass_flow)
        # Sink.mass_flow > 0 (load), Source.mass_flow < 0 (generator)
        if isinstance(model, Sink):
            return mf * _GAS_MW_FACTOR * (1 - reg)
        return (-mf) * _GAS_MW_FACTOR * (1 - reg)

    return 0


def _calc_objective(model_to_data):
    """Sum weighted unserved energy across all components."""
    return sum(_shedding_mw(model) * weight for model, weight in model_to_data.items())


def create_min_load_shedding_problem(
    *,
    demand_weight=WEIGHT_DEMAND,
    generator_weight=WEIGHT_GENERATOR,
    ext_grid_weight=WEIGHT_EXT_GRID,
    bounds_el=(0.9, 1.1),
    bounds_gas=(0.9, 1.1),
    bounds_heat=(0.9, 1.1),
    bounds_line_loading=(0, 1.5),
    ext_grid_el_bounds=(-3, 3),
    ext_grid_gas_bounds=(-10, 10),
    ext_grid_heat_bounds=(-10, 10),
    regulation_ramp_limit=None,
    include_storages=True,
    include_ext_grids=True,
    check_vm=False,
    check_pressure=False,
    check_temperature=False,
    check_line_loading=False,
    debug=False,
):
    """Create a minimal load shedding optimisation problem for multi-energy grids.

    Minimises total unserved energy across electrical, gas, and thermal
    carriers.  Each demand, generator, and coupling point receives a
    ``regulation`` variable in [0, 1].  The objective penalises deviation
    from full supply (``regulation = 1``), weighted per component category.

    All component types in a multi-energy grid are covered:

    * **Electrical**: PowerLoad, PowerGenerator, ExtPowerGrid, PowerLine (backup)
    * **Gas**: Sink, Source, ExtHydrGrid
    * **Thermal**: HeatExchanger variants, PassiveHeatExchanger variants,
      ExtHydrGrid (water)
    * **Coupling**: CHP, PowerToHeat, GasToHeat, PowerToGas, GasToPower
    * **Storage**: ElectricStorage, GasStorage, ThermalStorage

    Args:
        demand_weight: Objective penalty for shedding demand components.
        generator_weight: Objective penalty for shedding generators.
        ext_grid_weight: Objective penalty for ext grid deviations.
        bounds_el: Bus voltage magnitude bounds ``(min, max)`` in pu.
        bounds_gas: Gas junction pressure bounds ``(min, max)`` in pu.
        bounds_heat: Water junction temperature bounds ``(min, max)`` in pu.
        bounds_line_loading: Line loading bounds ``(min, max)`` in pu.
        ext_grid_el_bounds: Power bounds for electrical external grids (MW).
        ext_grid_gas_bounds: Mass-flow bounds for gas external grids (kg/s).
        ext_grid_heat_bounds: Mass-flow bounds for water/heat ext grids (kg/s).
        regulation_ramp_limit: Maximum change of ``regulation`` per period.
            ``None`` = no ramp limit.
        include_storages: Make storage components controllable.
        include_ext_grids: Make external grids controllable.
        check_vm: Enforce electrical voltage magnitude bounds.
        check_pressure: Enforce gas pressure bounds.
        check_temperature: Enforce water/heat temperature bounds.
        check_line_loading: Enforce line loading limits.
        debug: Enable debug logging for variable promotion.

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
    problem = OptimizationProblem(debug=debug)

    # --- Controllable components ---
    problem.controllable_demands(REGULATION_ATTR)
    problem.controllable_generators(REGULATION_ATTR)
    problem.controllable_cps(REGULATION_ATTR)
    problem.controllable_backup_lines()
    if include_ext_grids:
        problem.controllable_ext()
    if include_storages:
        problem.controllable_storages()

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
    def weight_fn(model):
        if isinstance(model, _DEMAND_TYPES):
            return demand_weight
        if isinstance(model, HeatExchanger):
            q = getattr(model, "q_w_set", 0)
            if isinstance(q, (int, float)):
                return demand_weight if q > 0 else generator_weight
            return demand_weight
        if isinstance(model, _GENERATOR_TYPES):
            return generator_weight
        if isinstance(model, _EXT_GRID_TYPES):
            return ext_grid_weight
        return 1

    objective_types = _DEMAND_TYPES + _GENERATOR_TYPES + _HE_OBJECTIVE_TYPES
    if include_ext_grids:
        objective_types = objective_types + _EXT_GRID_TYPES

    def _is_objective_model(m):
        # isinstance covers all leaf types; type() is needed for plain
        # HeatExchanger to exclude SubHE (internal compound-model branch).
        return isinstance(m, objective_types) or type(m) is HeatExchanger

    objectives = Objectives()
    objectives.select(_is_objective_model).data(weight_fn).calculate(_calc_objective)
    problem.objectives = objectives

    # --- Constraints ---
    constraints = Constraints()

    if check_line_loading:
        constraints.select_types(GenericPowerBranch).equation(
            lambda m: m.loading_from_percent <= bounds_line_loading[1]
        ).equation(lambda m: m.loading_to_percent <= bounds_line_loading[1])

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
