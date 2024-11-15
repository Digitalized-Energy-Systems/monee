import functools

from monee.model.branch import HeatExchanger, HeatExchangerGenerator, HeatExchangerLoad
from monee.model.child import PowerGenerator, PowerLoad, Sink, Source
from monee.model.core import GenericModel, Network, Var
from monee.model.grid import GasGrid
from monee.model.multi import CHP, PowerToGas, PowerToHeat


class Objective:
    def __init__(self, selected_models_link) -> None:
        self._selected_models_link = selected_models_link
        self._data_attacher = None
        self._calculator = lambda _: 0

    def data(self, data_attacher):
        self._data_attacher = data_attacher
        return self

    def calculate(self, calculator):
        self._calculator = calculator

    def _eval(self, network):
        model_objectives = []
        if self._data_attacher is not None:
            model_to_data = {}
            for model in self._selected_models_link(network):
                model_to_data[model] = self._data_attacher(model)
            model_objectives.append(self._calculator(model_to_data))
        else:
            model_objectives.append(
                self._calculator(self._selected_models_link(network))
            )
        return model_objectives


class Objectives:
    def __init__(self) -> None:
        self._objectives = []

    def select(self, model_selection_function) -> Objective:
        objective = Objective(
            lambda network: [
                model
                for model in network.all_models()
                if model_selection_function(model)
            ],
        )
        self._objectives.append(objective)
        return objective

    def with_models(self, models_link) -> Objective:
        objective = Objective(models_link)
        self._objectives.append(objective)
        return objective

    def all(self, network):
        if self._objectives:
            return functools.reduce(
                lambda a, b: a + b,
                [objective._eval(network) for objective in self._objectives],
            )
        return []


class Constraint:
    def __init__(self, selected_models_link) -> None:
        self._selected_models_link = selected_models_link
        self._data_attacher = None
        self._model_to_data = {}
        self._equations = []
        self._comp_equations = []

    def data(self, data_attacher):
        self._data_attacher = data_attacher
        return self

    def equation(self, equation_lambda):
        self._equations.append(equation_lambda)
        return self

    def comp_equation(self, equation_lambda):
        self._comp_equations.append(equation_lambda)
        return self

    def _eval(self, network):
        model_equations = []
        selected_models = self._selected_models_link(network)
        for equation in self._equations:
            if len(self._model_to_data) > 0:
                model_to_data = {}
                for model in selected_models:
                    model_to_data[model] = self._data_attacher(model)
                for item in model_to_data.items():
                    model_equations.append(equation(item))
            else:
                for model in selected_models:
                    model_equations.append(equation(model))

        for comp_equation in self._comp_equations:
            if len(self._model_to_data) > 0:
                model_to_data = {}
                for model in selected_models:
                    model_to_data[model] = self._data_attacher(model)
                model_equations.append(comp_equation(model_to_data))
            else:
                model_equations.append(comp_equation(selected_models))

        return model_equations


class Constraints:
    def __init__(self) -> None:
        self._constraints = []

    def select(self, component_selection_function) -> Constraint:
        constraint = Constraint(
            lambda network: [
                component.model
                for component in network.all_components()
                if component_selection_function(component)
                and component.active
                and not component.ignored
            ],
        )
        self._constraints.append(constraint)
        return constraint

    def select_types(self, model_cls_tuple) -> Constraint:
        return self.select(
            lambda component: isinstance(component.model, model_cls_tuple)
        )

    def select_grids(self, grid_cls_tuple) -> Constraint:
        return self.select(lambda component: isinstance(component.grid, grid_cls_tuple))

    def with_models(self, models) -> Constraint:
        constraint = Constraint(models)
        self._constraints.append(constraint)
        return constraint

    def all(self, network):
        if self._constraints:
            return functools.reduce(
                lambda a, b: a + b,
                [constraint._eval(network) for constraint in self._constraints],
            )
        return []

    @property
    def empty(self):
        return len(self._constraints) == 0


class OptimizationProblem:
    def __init__(self) -> None:
        self._controllable_appliables: list = []
        self._controllable_to_attr: dict[GenericModel, str] = {}
        self._bounds_for_controllables: list = []
        self._objectives: Objectives = None
        self._constraints: Constraints = None

    def _apply(self, network: Network):
        for appliable in self._controllable_appliables:
            appliable(network)
        for model, attributes in self._controllable_to_attr.items():
            for attribute in attributes:
                if hasattr(model, attribute):
                    val = getattr(model, attribute)
                    if type(val) is not Var:
                        setattr(
                            model,
                            attribute,
                            Var(
                                val,
                                max=0 if val <= 0 else val,
                                min=0 if val > 0 else val,
                            ),
                        )

        for min, max, component_condition, attributes in self._bounds_for_controllables:
            component_list = network.all_components()
            for component in component_list:
                if (
                    component_condition(component.model, component.grid)
                    and component.independent
                ):
                    for attribute in attributes:
                        var = getattr(component.model, attribute)
                        var.max = max
                        var.min = min

    def add_to_controllable(self, model, attributes=None):
        if model not in self._controllable_to_attr:
            self._controllable_to_attr[model] = []
        self._controllable_to_attr[model] += attributes or list(model.vars.keys())

    def bounds(self, minmax, component_condition=lambda _: True, attributes=None):
        self._bounds_for_controllables.append(
            (minmax[0], minmax[1], component_condition, attributes)
        )

    def controllable(self, component_condition=lambda _: True, attributes=None):
        def apply_controllable(network: Network):
            component_list = network.all_components()
            for component in component_list:
                if component.independent and component_condition(component):
                    self.add_to_controllable(component.model, attributes)

        self._controllable_appliables.append(apply_controllable)
        return self

    def controllable_all(self, attributes):
        self.controllable(
            component_condition=lambda _: True,
            attributes=attributes,
        )
        return self

    def controllable_demands(self, attributes):
        self.controllable(
            component_condition=lambda component: (
                isinstance(component.model, HeatExchangerLoad | PowerLoad)
                or (type(component.model) is Sink and type(component.grid) is GasGrid)
                or (
                    type(component.model) is HeatExchanger
                    and type(component.model.q_w) is not Var
                    and component.model.q_w > 0
                )
            )
            and component.active
            and not component.ignored,
            attributes=attributes,
        )
        return self

    def controllable_generators(self, attributes):
        self.controllable(
            component_condition=lambda component: isinstance(
                component.model, HeatExchangerGenerator | PowerGenerator | Source
            )
            and component.active
            and not component.ignored,
            attributes=attributes,
        )
        return self

    def controllable_cps(self, attributes):
        self.controllable(
            component_condition=lambda component: isinstance(
                component.model, CHP | PowerToHeat | PowerToGas
            )
            and component.active
            and not component.ignored,
            attributes=attributes,
        )
        return self

    @property
    def objectives(self):
        return self._objectives

    @property
    def constraints(self):
        return self._constraints

    @constraints.setter
    def constraints(self, constraints):
        self._constraints = constraints

    @objectives.setter
    def objectives(self, objectives):
        self._objectives = objectives

    @property
    def controllables_link(self):
        return lambda _: self._controllable_to_attr.keys()
