import math

from monee.problem.core import OptimizationProblem, Objectives, Constraints

from monee.model.child import (
    Sink,
    PowerLoad,
    ExtPowerGrid,
    ExtHydrGrid,
    PowerGenerator,
    Source,
)
from monee.model.branch import (
    HeatExchangerLoad,
    HeatExchangerGenerator,
    HeatExchanger,
    GenericPowerBranch,
)
from monee.model.node import Bus, Junction
from monee.model.grid import WaterGrid, GasGrid
from monee.model.multi import CHP, PowerToHeat, PowerToGas
from monee.model.core import Var

CONTROLLABLE_ATTRIBUTES = ["p_mw", "mass_flow", "q_w"]
CONTROLLABLE_ATTRIBUTES_CP = ["mass_flow", "heat_energy_mw", "to_mass_flow"]


def _or_zero(var):
    if type(var) == Var and math.isnan(var.value):
        return 0
    if hasattr(var.value, "value") and math.isnan(var.value.value):
        return 0
    return var


def retrieve_power_uniform(model):
    if isinstance(model, (HeatExchangerLoad, HeatExchangerGenerator, HeatExchanger)):
        return _or_zero(model.q_w), model.q_w.max
    elif isinstance(model, (PowerLoad, PowerGenerator)):
        return _or_zero(model.p_mw), model.p_mw.max
    elif isinstance(model, (Sink, Source)):
        return -_or_zero(model.mass_flow), -model.mass_flow.min
    elif isinstance(model, CHP):
        return 0, 0
    elif isinstance(model, PowerToHeat):
        return 0, 0
    elif isinstance(model, PowerToGas):
        return 0, 0

    raise ValueError(f"The model {type(model)} is not a known load.")


def calculate_objective(model_to_data):
    return sum(
        [
            (retrieve_power_uniform(model)[1] - retrieve_power_uniform(model)[0]) * data
            for model, data in model_to_data.items()
        ]
    )


def create_load_shedding_optimization_problem(
    load_weight=100,
    bounds_el=(0.9, 1.1),
    bounds_heat=(340, 390),
    bounds_gas=(900000, 1100000),
    bounds_lp=(0, 1.5),
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
):
    problem = OptimizationProblem()

    problem.controllable_demands(CONTROLLABLE_ATTRIBUTES)
    problem.controllable_generators(CONTROLLABLE_ATTRIBUTES)
    problem.controllable_cps(CONTROLLABLE_ATTRIBUTES_CP)

    problem.bounds(bounds_el, lambda m, _: type(m) == Bus, ["vm_pu"])
    problem.bounds(
        bounds_heat, lambda m, g: type(m) == Junction and type(g) == WaterGrid, ["t_k"]
    )
    problem.bounds(bounds_gas, lambda m, _: type(m) == Junction, ["pressure_pa"])

    objectives = Objectives()
    objectives.with_models(problem.controllables_link).data(
        lambda model: load_weight
        if isinstance(model, (HeatExchangerLoad, Sink, PowerLoad))
        else (
            load_weight - 1 if isinstance(model, (CHP, PowerToGas, PowerToHeat)) else 1
        )
    ).calculate(calculate_objective)

    constraints = Constraints()
    constraints.select_types(ExtPowerGrid).equation(
        lambda model: model.p_mw >= ext_grid_el_bounds[0]
    ).equation(lambda model: model.p_mw <= ext_grid_el_bounds[1])

    constraints.select(
        lambda comp: type(comp.grid) == GasGrid and type(comp.model) == ExtHydrGrid
    ).equation(lambda model: model.mass_flow >= ext_grid_gas_bounds[0]).equation(
        lambda model: model.mass_flow <= ext_grid_gas_bounds[1]
    )

    constraints.select_types(GenericPowerBranch).equation(
        lambda model: model.loading_from_percent <= bounds_lp[1]
    ).equation(lambda model: model.loading_to_percent <= bounds_lp[1])

    problem.objectives = objectives
    problem.constraints = constraints

    return problem


def create_ls_init_optimization_problem(
    bounds_el=(0.9, 1.1),
    bounds_heat=(340, 390),
    bounds_gas=(900000, 1100000),
    bounds_lp=(0, 1.5),
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
):
    problem = OptimizationProblem()

    problem.controllable_generators(CONTROLLABLE_ATTRIBUTES)

    problem.bounds(bounds_el, lambda m, _: type(m) == Bus, ["vm_pu"])
    problem.bounds(
        bounds_heat, lambda m, g: type(m) == Junction and type(g) == WaterGrid, ["t_k"]
    )
    problem.bounds(bounds_gas, lambda m, _: type(m) == Junction, ["pressure_pa"])

    constraints = Constraints()
    constraints.select_types(ExtPowerGrid).equation(
        lambda model: model.p_mw >= ext_grid_el_bounds[0]
    ).equation(lambda model: model.p_mw <= ext_grid_el_bounds[1])

    constraints.select(
        lambda comp: type(comp.grid) == GasGrid and type(comp.model) == ExtHydrGrid
    ).equation(lambda model: model.mass_flow >= ext_grid_gas_bounds[0]).equation(
        lambda model: model.mass_flow <= ext_grid_gas_bounds[1]
    )

    constraints.select_types(GenericPowerBranch).equation(
        lambda model: model.loading_from_percent <= bounds_lp[1]
    ).equation(lambda model: model.loading_to_percent <= bounds_lp[1])

    problem.constraints = constraints

    return problem
