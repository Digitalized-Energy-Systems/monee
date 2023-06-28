from monee.problem.core import OptimizationProblem, Objectives, Constraints

from monee.model.child import (
    Sink,
    PowerLoad,
    ExtPowerGrid,
    ExtHydrGrid,
    PowerGenerator,
    Source,
)
from monee.model.branch import HeatExchangerLoad, HeatExchangerGenerator
from monee.model.node import Bus, Junction
from monee.model.grid import WaterGrid


def retrieve_power_uniform(model):
    if isinstance(model, (HeatExchangerLoad, HeatExchangerGenerator)):
        return model.q_w, model.q_w.max
    elif isinstance(model, (PowerLoad, PowerGenerator)):
        return model.p_mw, model.p_mw.max
    elif isinstance(model, (Sink, Source)):
        return -model.mass_flow, -model.mass_flow.min
    raise ValueError(f"The model {type(model)} is not a known load.")


def calculate_objective(model_to_data):
    return sum(
        [
            (retrieve_power_uniform(model)[1] - retrieve_power_uniform(model)[0]) * data
            for model, data in model_to_data.items()
        ]
    )


def create_load_shedding_optimization_problem(
    load_weight=10,
    bounds_el=(0.9, 1.1),
    bound_heat=(340, 390),
    bounds_gas=(900000, 1100000),
):
    problem = OptimizationProblem()

    controllable_attributes = ["p_mw", "mass_flow", "q_w"]
    problem.controllable_demands(controllable_attributes)
    problem.controllable_generators(controllable_attributes)

    problem.bounds(bounds_el, lambda m, _: type(m) == Bus, ["vm_pu"])
    problem.bounds(
        bound_heat, lambda m, g: type(m) == Junction and type(g) == WaterGrid, ["t_k"]
    )
    problem.bounds(bounds_gas, lambda m, _: type(m) == Junction, ["pressure_pa"])

    objectives = Objectives()
    objectives.with_models(problem.controllables_link).data(
        lambda model: load_weight
        if isinstance(model, (HeatExchangerLoad, Sink, PowerLoad))
        else 1
    ).calculate(calculate_objective)

    constraints = Constraints()
    constraints.select_types(ExtPowerGrid).equation(lambda model: model.p_mw == 0)
    constraints.select_types(ExtHydrGrid).equation(lambda model: model.mass_flow == 0)

    problem.objectives = objectives
    problem.constraints = constraints

    return problem
