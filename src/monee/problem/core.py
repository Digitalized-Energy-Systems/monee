import functools
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

from monee.model import (
    CHPControlNode,
    CHPHGControlNode,
    ExtHydrGrid,
    ExtPowerGrid,
    GasGrid,
    GasToHeatControlNode,
    GasToHeatHG,
    GasToPower,
    GenericModel,
    HeatExchanger,
    HeatExchangerGenerator,
    HeatExchangerLoad,
    Network,
    PassiveHeatExchanger,
    PassiveHeatExchangerGenerator,
    PassiveHeatExchangerLoad,
    PowerGenerator,
    PowerLoad,
    PowerToGas,
    PowerToHeatControlNode,
    PowerToHeatHG,
    Sink,
    Source,
    Var,
)

logger = logging.getLogger(__name__)


def nan_to_zero(v):
    """Return *v* unchanged unless it is (or wraps) ``NaN``, in which case return ``0``.

    Works with plain floats, :class:`~monee.model.core.Var` objects, and
    anything with a ``.value`` attribute.  Useful inside objective
    functions where ``NaN`` initial values should not propagate into the
    solver expression.
    """
    if isinstance(v, (int, float)):
        return 0 if math.isnan(v) else v
    if isinstance(v, Var):
        if isinstance(v.value, (int, float)) and math.isnan(v.value):
            return 0
        return v
    if hasattr(v, "value"):
        val = v.value
        if isinstance(val, (int, float)) and math.isnan(val):
            return 0
    return v


class Objective:
    def __init__(self, selected_models_link) -> None:
        self._selected_models_link = selected_models_link
        self._data_attacher = None
        self._calculator = lambda _: 0
        self._period_filter = None

    def data(self, data_attacher):
        self._data_attacher = data_attacher
        return self

    def calculate(self, calculator):
        self._calculator = calculator

    def when_period(self, period_filter):
        """Only activate this objective for periods where *period_filter* is truthy.

        Args:
            period_filter: A callable ``(t: int) -> bool``, or a collection
                of period indices (set, list, range).  Has no effect in
                single-period solves.

        Returns:
            ``self`` for method chaining.
        """
        if callable(period_filter):
            self._period_filter = period_filter
        else:
            allowed = set(period_filter)
            self._period_filter = lambda t: t in allowed
        return self

    def _eval(self, network, period_index=None):
        if self._period_filter is not None and period_index is not None:
            if not self._period_filter(period_index):
                return [0]
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
            ]
        )
        self._objectives.append(objective)
        return objective

    def with_models(self, models_link) -> Objective:
        objective = Objective(models_link)
        self._objectives.append(objective)
        return objective

    def all(self, network, period_index=None):
        if self._objectives:
            return functools.reduce(
                lambda a, b: a + b,
                [
                    objective._eval(network, period_index)
                    for objective in self._objectives
                ],
            )
        return []


class Constraint:
    def __init__(
        self, selected_models_link, selected_models_with_ids_link=None
    ) -> None:
        self._selected_models_link = selected_models_link
        self._selected_models_with_ids_link = selected_models_with_ids_link
        self._data_attacher = None
        self._model_to_data = {}
        self._equations = []
        self._comp_equations = []
        self._temporal_equations = []
        self._period_filter = None

    def data(self, data_attacher):
        self._data_attacher = data_attacher
        return self

    def equation(self, equation_lambda):
        self._equations.append(equation_lambda)
        return self

    def comp_equation(self, equation_lambda):
        self._comp_equations.append(equation_lambda)
        return self

    def temporal_equation(self, equation_lambda):
        """Add a cross-period constraint that has access to temporal state.

        The lambda receives ``(model, component_id, temporal_state)`` and
        must return a single equation or a list of equations.

        *temporal_state* is a :class:`~monee.simulation.step_state.PeriodState`
        (multi-period) or :class:`~monee.simulation.step_state.StepState`
        (timeseries). Use ``temporal_state.get(component_id, attr)`` to read
        variables from other periods.

        These constraints are only evaluated when a temporal state is available
        (multi-period or timeseries mode). They are silently skipped in
        single-period solves.

        Example — ramp-rate limit on a generator::

            constraints.select_types(PowerGenerator).temporal_equation(
                lambda m, cid, ts: [
                    m.p_mw - ts.get(cid, "p_mw") <= 10,   # ramp up
                    ts.get(cid, "p_mw") - m.p_mw <= 10,   # ramp down
                ]
            )

        Example — storage-like coupling (SoC tracks previous value)::

            constraints.select_types(PowerLoad).temporal_equation(
                lambda m, cid, ts: m.e_mwh == ts.get(cid, "e_mwh") + ts.dt_h * m.p_mw
            )
        """
        self._temporal_equations.append(equation_lambda)
        return self

    def when_period(self, period_filter):
        """Only activate this constraint for periods where *period_filter* is truthy.

        Args:
            period_filter: A callable ``(t: int) -> bool``, or a collection
                of period indices (set, list, range).  Has no effect in
                single-period solves.

        Returns:
            ``self`` for method chaining.

        Example — force generator offline at period 5::

            constraints.select_types(PowerGenerator)
                .equation(lambda m: m.p_mw == 0)
                .when_period(lambda t: t == 5)
        """
        if callable(period_filter):
            self._period_filter = period_filter
        else:
            allowed = set(period_filter)
            self._period_filter = lambda t: t in allowed
        return self

    def _eval(self, network, period_index=None):
        if self._period_filter is not None and period_index is not None:
            if not self._period_filter(period_index):
                return []
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

    @property
    def has_temporal(self):
        """True if this constraint has any temporal equations."""
        return len(self._temporal_equations) > 0

    def _eval_temporal(self, network, temporal_state, period_index=None):
        """Evaluate temporal equations that need access to temporal state."""
        if not self._temporal_equations:
            return []
        if self._period_filter is not None and period_index is not None:
            if not self._period_filter(period_index):
                return []
        if self._selected_models_with_ids_link is None:
            return []
        model_equations = []
        for model, comp_id in self._selected_models_with_ids_link(network):
            for eq_fn in self._temporal_equations:
                result = eq_fn(model, comp_id, temporal_state)
                if hasattr(result, "__iter__") and not isinstance(result, str):
                    model_equations.extend(result)
                else:
                    model_equations.append(result)
        return model_equations


class Constraints:
    def __init__(self) -> None:
        self._constraints = []

    def select(self, component_selection_function) -> Constraint:
        def _filter(component):
            return (
                component_selection_function(component)
                and component.active
                and (not component.ignored)
            )

        constraint = Constraint(
            lambda network: [
                component.model
                for component in network.all_components()
                if _filter(component)
            ],
            selected_models_with_ids_link=lambda network: [
                (component.model, component.id)
                for component in network.all_components()
                if _filter(component)
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

    def all(self, network, period_index=None):
        if self._constraints:
            return functools.reduce(
                lambda a, b: a + b,
                [
                    constraint._eval(network, period_index)
                    for constraint in self._constraints
                ],
            )
        return []

    def all_temporal(self, network, temporal_state, period_index=None):
        """Evaluate all temporal constraints that couple across periods.

        Called during inter-step/inter-period equation assembly where
        *temporal_state* is available.
        """
        results = []
        for constraint in self._constraints:
            results.extend(
                constraint._eval_temporal(network, temporal_state, period_index)
            )
        return results

    @property
    def has_temporal(self):
        """True if any constraint has temporal equations."""
        return any(c.has_temporal for c in self._constraints)

    def regulation_ramp(self, limit):
        """Add a ramp-rate constraint on ``regulation`` across periods.

        Limits how fast the ``regulation`` variable can change between
        consecutive periods.  Components without a ``regulation`` attribute
        (or where it is a plain float) are silently skipped.

        Args:
            limit: Maximum change of ``regulation`` per period (e.g. 0.3).

        Returns:
            ``self`` for method chaining.

        Example::

            constraints.regulation_ramp(0.3)
        """

        def _regulation_ramp(model, cid, ts):
            if not hasattr(model, "regulation") or isinstance(
                model.regulation, (int, float)
            ):
                return []
            prev_reg = ts.get(cid, "regulation")
            if prev_reg is None:
                return []
            return [
                model.regulation - prev_reg <= limit,
                prev_reg - model.regulation <= limit,
            ]

        self.select(
            lambda comp: (
                hasattr(comp.model, "regulation") and comp.active and not comp.ignored
            )
        ).temporal_equation(_regulation_ramp)
        return self

    @property
    def empty(self):
        return len(self._constraints) == 0


@dataclass
class AttributeParameter:
    min: Callable[[str, float], float]
    max: Callable[[str, float], float]
    val: Callable[[str, float], float]
    integer: bool = False


# Regulation attribute parameter (0–1) used across load-shedding formulations.
REGULATION_ATTR = [
    (
        "regulation",
        AttributeParameter(
            min=lambda attr, val: 0,
            max=lambda attr, val: 1,
            val=lambda attr, val: 1,
        ),
    )
]


class OptimizationProblem:
    """
    Declares which network components are free variables and, optionally,
    what objective function the solver should minimise.

    The typical workflow is:

    1. Create a problem object.
    2. Call one or more ``controllable_*`` methods to declare which components
       the solver may dispatch freely.
    3. Optionally set ``problem.objectives`` and ``problem.constraints``.
    4. Pass the object to :func:`~monee.simulation.run_multi_period` or
       :func:`~monee.simulation.run_mpc`.

    Example::

        from monee.problem.core import OptimizationProblem
        from monee.simulation import run_multi_period

        prob = OptimizationProblem()
        prob.controllable_storages()
        result = run_multi_period(net, td, optimization_problem=prob, dt_h=1.0)

    **Time-varying objectives** — register per-period data (e.g. electricity
    prices) via :meth:`~monee.simulation.timeseries.TimeseriesData.add_objective_data`
    and read it inside the objective lambda.  ``TimeseriesData`` applies the
    correct period's value to the model *before* the objective is evaluated::

        td = TimeseriesData()
        td.add_objective_data(gen_id, "price", [40, 80, 60, 30])

        prob = OptimizationProblem()
        prob.controllable_generators(["p_mw"])
        obj = Objectives()
        obj.select(
            lambda m: isinstance(m, PowerGenerator)
        ).calculate(
            lambda models: sum(m.price * m.p_mw for m in models)
        )
        prob.objectives = obj

    **Per-period constraints** — use :meth:`Constraint.when_period` to
    restrict a constraint to specific periods::

        cons = Constraints()
        cons.select_types(PowerGenerator).equation(
            lambda m: m.p_mw <= 0.5
        ).when_period(lambda t: t >= 3)  # cap output from period 3 onward
        prob.constraints = cons

    **Cross-period constraints** — use :meth:`Constraint.temporal_equation`
    to couple variables across periods (ramp rates, custom storage, etc.)::

        cons = Constraints()
        cons.select_types(ExtPowerGrid).temporal_equation(
            lambda m, cid, ts: (
                [] if ts.get(cid, "p_mw") is None else
                [m.p_mw - ts.get(cid, "p_mw") <= 1.0,
                 ts.get(cid, "p_mw") - m.p_mw <= 1.0]
            )
        )
        prob.constraints = cons
    """

    def __init__(self, debug=False) -> None:
        self._controllable_appliables: list = []
        self._controllable_to_attr: dict[GenericModel, str] = {}
        self._bounds_for_controllables: list = []
        self._objectives: Objectives = None
        self._constraints: Constraints = None
        self._debug = debug

    def _apply(self, network: Network):
        self._controllable_to_attr.clear()
        for appliable in self._controllable_appliables:
            appliable(network)
        for model, attributes in self._controllable_to_attr.items():
            for attribute_param in attributes:
                attribute = attribute_param
                param = None
                if type(attribute_param) is tuple:
                    attribute = attribute_param[0]
                    param: AttributeParameter = attribute_param[1]
                if hasattr(model, attribute):
                    val = getattr(model, attribute)
                    if type(val) is not Var:
                        if param is None:
                            if val == 0.0:
                                logger.warning(
                                    "Attribute '%s' on %s has value 0.0 and no "
                                    "explicit bounds — inferred bounds [0, 0] "
                                    "will lock this variable. Use an "
                                    "AttributeParameter or prob.bounds() to "
                                    "set meaningful bounds.",
                                    attribute,
                                    type(model).__name__,
                                )
                            variable = Var(
                                val,
                                max=0 if val <= 0 else val,
                                min=0 if val > 0 else val,
                                name=attribute,
                            )
                        else:
                            variable = Var(
                                param.val(attribute, val),
                                param.max(attribute, val),
                                param.min(attribute, val),
                                param.integer,
                                name=attribute,
                            )
                        setattr(model, attribute, variable)
                        if self._debug:
                            logger.warning("From the model %s", model)
                            logger.warning(
                                "The attribute %s has been replaced", attribute
                            )
        for (
            min_value,
            max_value,
            component_condition,
            attributes,
        ) in self._bounds_for_controllables:
            component_list = network.all_components()
            for component in component_list:
                if (
                    component_condition(component.model, component.grid)
                    and component.independent
                ):
                    if self._debug:
                        logger.info("From the model %s", component.model)
                        logger.info("The attributes %s are bounded", attributes)
                    for attribute in attributes:
                        var = getattr(component.model, attribute)
                        var.max = max_value
                        var.min = min_value

    def add_to_controllable(
        self, model, attributes: list[str | tuple[str, AttributeParameter]]
    ):
        if model not in self._controllable_to_attr:
            self._controllable_to_attr[model] = []
        self._controllable_to_attr[model] += attributes

    def bounds(self, minmax, component_condition=lambda _: True, attributes=None):
        """Override the min/max bounds of specific ``Var`` attributes on matching components.

        Args:
            minmax: A ``(min_value, max_value)`` tuple applied to all matched variables.
            component_condition: Callable ``(model, grid) -> bool`` that selects
                which components to bound.  Defaults to all components.
            attributes: List of attribute names (strings) whose ``Var`` bounds to
                override.

        Example — cap all generator output at 0.8 MW::

            from monee.model import PowerGenerator

            prob.bounds(
                (0.0, 0.8),
                component_condition=lambda m, g: isinstance(m, PowerGenerator),
                attributes=["p_mw"],
            )
        """
        self._bounds_for_controllables.append(
            (minmax[0], minmax[1], component_condition, attributes)
        )

    def controllable(
        self,
        attributes: list[str | tuple[str, AttributeParameter]],
        component_condition=lambda _: True,
    ):
        """Make specific attributes on matching components free optimisation variables.

        This is the low-level primitive underlying all ``controllable_*`` helpers.
        Prefer the typed helpers (``controllable_demands``, ``controllable_generators``,
        etc.) unless you need custom component selection.

        Args:
            attributes: List of attribute names to promote to ``Var``.  Each
                entry is either a plain string (bounds inferred from the current
                value) or a ``(name, AttributeParameter)`` tuple for explicit
                bounds/initial value.
            component_condition: Callable ``(component) -> bool`` that selects
                which components to affect.  Defaults to all components.

        Returns:
            ``self`` for method chaining.
        """

        def apply_controllable(network: Network):
            component_list = network.all_components()
            for component in component_list:
                if component_condition(component):
                    self.add_to_controllable(component.model, attributes)

        self._controllable_appliables.append(apply_controllable)
        return self

    def controllable_all(self, attributes):
        self.controllable(component_condition=lambda _: True, attributes=attributes)
        return self

    def controllable_demands(
        self, attributes: list[str | tuple[str, AttributeParameter]]
    ):
        """Make demand-side components (loads, gas sinks, heat exchangers as loads) controllable.

        Targets :class:`~monee.model.PowerLoad`, :class:`~monee.model.HeatExchangerLoad`,
        gas :class:`~monee.model.Sink`, and :class:`~monee.model.HeatExchanger` instances
        that are currently consuming (positive ``q_w``).

        Args:
            attributes: Attribute names to promote to ``Var`` (e.g. ``["p_mw"]``).

        Returns:
            ``self`` for method chaining.

        Example — controllable load shedding::

            prob.controllable_demands(["p_mw"])

        .. note::

            When an attribute's current value is ``0.0`` and no explicit
            ``AttributeParameter`` is given, the inferred bounds are
            ``[0, 0]``, effectively locking the variable.  This commonly
            happens when the base network uses ``p_mw=0.0`` and real
            values come from ``TimeseriesData``.  Use ``prob.bounds()``
            or pass an ``(attr, AttributeParameter)`` tuple to set
            meaningful bounds.
        """
        self.controllable(
            component_condition=lambda component: (
                (
                    isinstance(
                        component.model,
                        HeatExchangerLoad | PassiveHeatExchangerLoad | PowerLoad,
                    )
                    or (
                        type(component.model) is Sink
                        and type(component.grid) is GasGrid
                    )
                    or (
                        isinstance(
                            component.model, HeatExchanger | PassiveHeatExchanger
                        )
                        and type(component.model.q_mw) is not Var
                        and (component.model.q_mw > 0)
                    )
                )
                and component.active
                and (not component.ignored)
            ),
            attributes=attributes,
        )
        return self

    def controllable_generators(self, attributes):
        """Make generation-side components controllable.

        Targets :class:`~monee.model.PowerGenerator`,
        :class:`~monee.model.HeatExchangerGenerator`, and gas
        :class:`~monee.model.Source` instances.

        Args:
            attributes: Attribute names to promote to ``Var`` (e.g. ``["p_mw"]``).

        Returns:
            ``self`` for method chaining.

        Example — dispatchable generator output::

            prob.controllable_generators(["p_mw"])

        .. note::

            When an attribute's current value is ``0.0`` and no explicit
            ``AttributeParameter`` is given, the inferred bounds are
            ``[0, 0]``, effectively locking the variable.  Use
            ``prob.bounds()`` or pass an ``(attr, AttributeParameter)``
            tuple to set meaningful bounds.
        """
        self.controllable(
            component_condition=lambda component: (
                isinstance(
                    component.model,
                    HeatExchangerGenerator
                    | PassiveHeatExchangerGenerator
                    | PowerGenerator
                    | Source,
                )
                and component.active
                and (not component.ignored)
            ),
            attributes=attributes,
        )
        return self

    def controllable_ext(self):
        """Make external grid connections controllable.

        Targets :class:`~monee.model.ExtPowerGrid` and
        :class:`~monee.model.ExtHydrGrid` (both gas and water/heat grids).
        """
        self.controllable(
            component_condition=lambda component: (
                isinstance(component.model, ExtPowerGrid | ExtHydrGrid)
                and component.active
                and (not component.ignored)
            ),
            attributes=[],
        )
        return self

    def controllable_cps(self, attributes):
        """Make coupling-point components controllable.

        Targets :class:`~monee.model.CHPControlNode`,
        :class:`~monee.model.PowerToHeatControlNode`,
        :class:`~monee.model.GasToHeatControlNode`,
        :class:`~monee.model.PowerToGas`,
        :class:`~monee.model.GasToPower`,
        :class:`~monee.model.PowerToHeatHG`, and
        :class:`~monee.model.GasToHeatHG` instances.

        Args:
            attributes: Attribute names to promote to ``Var``
                (e.g. ``["regulation"]`` for P2H, ``["p_mw"]`` for CHP).

        Returns:
            ``self`` for method chaining.

        Example — optimise P2H modulation::

            prob.controllable_cps(["regulation"])
        """
        self.controllable(
            component_condition=lambda component: (
                isinstance(
                    component.model,
                    CHPControlNode
                    | CHPHGControlNode
                    | PowerToHeatControlNode
                    | GasToHeatControlNode
                    | PowerToGas
                    | GasToPower
                    | PowerToHeatHG
                    | GasToHeatHG,
                )
                and component.active
                and (not component.ignored)
            ),
            attributes=attributes,
        )
        return self

    def controllable_storages(self):
        """Make all storage components (electric, gas, thermal) controllable.

        Calls :meth:`~monee.model.storage.ElectricStorage.make_controllable` /
        :meth:`~monee.model.storage.GasStorage.make_controllable` /
        :meth:`~monee.model.storage.ThermalStorage.make_controllable` on every
        matching component in the network during :meth:`_apply`.  This converts
        ``p_mw`` / ``mass_flow`` (and efficiency-split variables for lossy models)
        from plain floats to :class:`~monee.model.core.Var` objects so the
        solver can optimise storage dispatch.

        Returns:
            ``self`` for method chaining.

        Example::

            problem = OptimizationProblem()
            problem.controllable_storages()
            result = run_multi_period(net, td, optimization_problem=problem)
        """
        from monee.model.storage import ElectricStorage, GasStorage, ThermalStorage

        def _apply_storages(network: Network):
            for component in network.all_components():
                if (
                    isinstance(
                        component.model, (ElectricStorage, GasStorage, ThermalStorage)
                    )
                    and component.active
                    and not component.ignored
                ):
                    component.model.make_controllable()

        self._controllable_appliables.append(_apply_storages)
        return self

    def controllable_backup_lines(self):
        """Make backup lines switchable (on/off) during optimisation.

        Targets branch components that have ``backup=True``.  Adds an
        integer ``on_off`` variable in {0, 1} so the solver can decide
        whether to activate each backup line.

        Returns:
            ``self`` for method chaining.
        """
        self.controllable(
            component_condition=lambda component: (
                "backup" in component.model.vars and component.model.backup
            ),
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
