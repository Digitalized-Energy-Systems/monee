import math

from monee.model.branch import (
    GenericPowerBranch,
    HeatExchanger,
    HeatExchangerGenerator,
    HeatExchangerLoad,
    PowerLine,
)
from monee.model.child import (
    ExtHydrGrid,
    ExtPowerGrid,
    PowerGenerator,
    PowerLoad,
    Sink,
    Source,
)
from monee.model.core import Var
from monee.model.grid import GasGrid, WaterGrid
from monee.model.multi import CHPControlNode, PowerToGas, PowerToHeat
from monee.model.node import Bus, Junction
from monee.problem.core import (
    AttributeParameter,
    Constraints,
    Objectives,
    OptimizationProblem,
)

CONTROLLABLE_ATTRIBUTES = [
    (
        "regulation",
        AttributeParameter(
            min=lambda attr, val: 0, max=lambda attr, val: 1, val=lambda attr, val: 1
        ),
    )
]
CONTROLLABLE_ATTRIBUTES_CP = [
    (
        "regulation",
        AttributeParameter(
            min=lambda attr, val: 0, max=lambda attr, val: 1, val=lambda attr, val: 1
        ),
    )
]


def _or_zero(var):
    """
    No docstring provided.
    """
    if type(var) is Var and math.isnan(var.value):
        return 0
    if (
        hasattr(var, "value")
        and hasattr(var.value, "value")
        and math.isnan(var.value.value)
    ):
        return 0
    return var


HHV = 15.3


def retrieve_power_uniform(model):
    """
    No docstring provided.
    """

    if isinstance(model, HeatExchangerLoad | HeatExchangerGenerator | HeatExchanger):
        return (model.q_w_set / 1000000.0 * model.regulation, model.q_w_set / 1000000.0)
    elif isinstance(model, PowerLoad | PowerGenerator):
        return (_or_zero(model.p_mw) * model.regulation, model.p_mw)
    elif isinstance(model, ExtPowerGrid):
        return model.p_mw, 0
    elif isinstance(model, ExtHydrGrid):
        return model.mass_flow * 3.6 * HHV, 0
    elif isinstance(model, Sink | Source):
        return (
            _or_zero(model.mass_flow) * 3.6 * HHV * model.regulation,
            model.mass_flow * 3.6 * HHV,
        )
    elif isinstance(model, CHPControlNode):
        return (0, 0)
    elif isinstance(model, PowerToHeat):
        return (0, 0)
    elif isinstance(model, PowerToGas):
        return (0, 0)
    elif isinstance(model, PowerLine):
        if model.backup:
            return (0, model.on_off * 0.1)
        return (0, 0)
    raise ValueError(f"The model {type(model)} is not a known load.")


def calculate_objective(model_to_data):
    """
    No docstring provided.
    """
    power_coeff = [
        (
            model,
            (retrieve_power_uniform(model)[1] - retrieve_power_uniform(model)[0])
            * data,
        )
        for model, data in model_to_data.items()
    ]
    print(power_coeff)
    return sum([t[1] for t in power_coeff])


def create_load_shedding_optimization_problem(
    load_weight=10,
    bounds_el=(0.9, 1.1),
    bounds_heat=(0.9, 1.1),
    bounds_gas=(0.9, 1.1),
    bounds_lp=(0, 1.5),
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
    use_ext_grid_bounds=False,
    use_ext_grid_objective=True,
    debug=False,
):
    """
    No docstring provided.
    """
    problem = OptimizationProblem(debug=debug)
    problem.controllable_demands(CONTROLLABLE_ATTRIBUTES)
    problem.controllable_generators(CONTROLLABLE_ATTRIBUTES)
    problem.controllable_cps(CONTROLLABLE_ATTRIBUTES_CP)
    if use_ext_grid_objective:
        problem.controllable_ext()
    problem.controllable(
        component_condition=lambda component: "backup" in component.model.vars
        and component.model.backup,
        attributes=[
            (
                "on_off",
                AttributeParameter(
                    min=lambda attr, val: 0,
                    max=lambda attr, val: 1,
                    val=lambda attr, val: 1,
                    integer=True,
                ),
            )
        ],
    )
    problem.bounds(bounds_el, lambda m, _: type(m) is Bus, ["vm_pu"])
    problem.bounds(
        bounds_heat, lambda m, g: type(m) is Junction and type(g) is WaterGrid, ["t_pu"]
    )
    problem.bounds(bounds_gas, lambda m, _: type(m) is Junction, ["pressure_pu"])

    objectives = Objectives()

    def calc_weight(model):
        weight = 1
        if isinstance(model, HeatExchangerLoad | Sink | PowerLoad):
            weight = load_weight
        elif isinstance(model, CHPControlNode | PowerToGas | PowerToHeat):
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

    constraints.select_types(GenericPowerBranch).equation(
        lambda model: model.loading_from_percent <= bounds_lp[1]
    ).equation(lambda model: model.loading_to_percent <= bounds_lp[1])
    problem.constraints = constraints
    problem.objectives = objectives
    return problem


def create_ls_init_optimization_problem(
    bounds_el=(0.9, 1.1),
    bounds_heat=(340, 390),
    bounds_gas=(900000, 1100000),
    bounds_lp=(0, 1.5),
    ext_grid_el_bounds=(-0.25, 0.25),
    ext_grid_gas_bounds=(-1.5, 1.5),
):
    """
    No docstring provided.
    """
    problem = OptimizationProblem()
    problem.controllable_generators(CONTROLLABLE_ATTRIBUTES)
    problem.bounds(bounds_el, lambda m, _: type(m) is Bus, ["vm_pu"])
    problem.bounds(
        bounds_heat, lambda m, g: type(m) is Junction and type(g) is WaterGrid, ["t_k"]
    )
    problem.bounds(bounds_gas, lambda m, _: type(m) is Junction, ["pressure_pa"])
    constraints = Constraints()
    constraints.select_types(ExtPowerGrid).equation(
        lambda model: model.p_mw >= ext_grid_el_bounds[0]
    ).equation(lambda model: model.p_mw <= ext_grid_el_bounds[1])
    constraints.select(
        lambda comp: type(comp.grid) is GasGrid and type(comp.model) is ExtHydrGrid
    ).equation(lambda model: model.mass_flow >= ext_grid_gas_bounds[0]).equation(
        lambda model: model.mass_flow <= ext_grid_gas_bounds[1]
    )
    constraints.select_types(GenericPowerBranch).equation(
        lambda model: model.loading_from_percent <= bounds_lp[1]
    ).equation(lambda model: model.loading_to_percent <= bounds_lp[1])
    problem.constraints = constraints
    return problem
