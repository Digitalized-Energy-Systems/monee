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
    HeatExchangerGenerator,
    HeatExchangerLoad,
    HeatGenerator,
    HeatLoad,
    Network,
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
    hx_is_consuming,
    hx_is_generating,
)
from monee.model.storage import ElectricStorage, GasStorage, ThermalStorage

logger = logging.getLogger(__name__)


def nan_to_zero(v):
    """Return *v*, replacing NaN values (in Var/Const wrappers too) with 0."""
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


def _normalize_period_filter(period_filter):
    if callable(period_filter):
        return period_filter
    allowed = set(period_filter)
    return lambda t: t in allowed


def _period_inactive(period_filter, period_index) -> bool:
    return (
        period_filter is not None
        and period_index is not None
        and not period_filter(period_index)
    )


def _attach_data(data_attacher, models) -> dict:
    return {model: data_attacher(model) for model in models}


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
        return self

    def when_period(self, period_filter):
        """Only activate for periods where *period_filter* is truthy. Accepts a
        callable ``(t: int) -> bool`` or a collection of indices."""
        self._period_filter = _normalize_period_filter(period_filter)
        return self

    def _eval(self, network, period_index=None):
        if _period_inactive(self._period_filter, period_index):
            return [0]
        selected_models = self._selected_models_link(network)
        if self._data_attacher is not None:
            return [self._calculator(_attach_data(self._data_attacher, selected_models))]
        return [self._calculator(selected_models)]


class Objectives:
    def __init__(self) -> None:
        self._objectives = []

    def select(self, model_selection_function) -> Objective:
        def _filter(component):
            return (
                model_selection_function(component.model)
                and component.active
                and (not component.ignored)
            )

        objective = Objective(
            lambda network: [
                component.model
                for component in network.all_components()
                if _filter(component)
            ]
        )
        self._objectives.append(objective)
        return objective

    def with_models(self, models_link) -> Objective:
        objective = Objective(models_link)
        self._objectives.append(objective)
        return objective

    def all(self, network, period_index=None):
        return [
            expr
            for objective in self._objectives
            for expr in objective._eval(network, period_index)
        ]


class Constraint:
    def __init__(
        self, selected_models_link, selected_models_with_ids_link=None
    ) -> None:
        self._selected_models_link = selected_models_link
        self._selected_models_with_ids_link = selected_models_with_ids_link
        self._data_attacher = None
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
        """Add a cross-period constraint. Lambda signature
        ``(model, component_id, temporal_state) -> eq | list[eq]``. Silently
        skipped in single-period solves."""
        self._temporal_equations.append(equation_lambda)
        return self

    def when_period(self, period_filter):
        """Only activate for periods where *period_filter* is truthy. Accepts a
        callable ``(t: int) -> bool`` or a collection of indices."""
        self._period_filter = _normalize_period_filter(period_filter)
        return self

    def _eval(self, network, period_index=None):
        if _period_inactive(self._period_filter, period_index):
            return []
        model_equations = []
        selected_models = self._selected_models_link(network)
        # The attacher is re-run per equation so its call counts stay identical
        # to the historical behavior (attachers may be stateful).
        for equation in self._equations:
            if self._data_attacher is not None:
                for item in _attach_data(self._data_attacher, selected_models).items():
                    model_equations.append(equation(item))
            else:
                for model in selected_models:
                    model_equations.append(equation(model))
        for comp_equation in self._comp_equations:
            if self._data_attacher is not None:
                model_equations.append(
                    comp_equation(_attach_data(self._data_attacher, selected_models))
                )
            else:
                model_equations.append(comp_equation(selected_models))
        return model_equations

    @property
    def has_temporal(self):
        return len(self._temporal_equations) > 0

    def _eval_temporal(self, network, temporal_state, period_index=None):
        if not self._temporal_equations:
            return []
        if _period_inactive(self._period_filter, period_index):
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

    def with_models(self, models) -> Constraint:
        constraint = Constraint(models)
        self._constraints.append(constraint)
        return constraint

    def all(self, network, period_index=None):
        return [
            eq
            for constraint in self._constraints
            for eq in constraint._eval(network, period_index)
        ]

    def all_temporal(self, network, temporal_state, period_index=None):
        results = []
        for constraint in self._constraints:
            results.extend(
                constraint._eval_temporal(network, temporal_state, period_index)
            )
        return results

    @property
    def has_temporal(self):
        return any(c.has_temporal for c in self._constraints)

    def regulation_ramp(self, limit):
        """Limit per-period change of ``regulation`` to ``limit`` (skips models
        without a regulation Var)."""

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
    Declares which components are free variables plus optional objectives/constraints.

    Workflow: create → call ``controllable_*`` helpers → set ``objectives`` and
    ``constraints`` → pass to :func:`run_multi_period` / :func:`run_mpc`.
    """

    def __init__(self, debug=False, lex_objectives: bool = False) -> None:
        """
        Args:
            debug: verbose logging during variable promotion.
            lex_objectives: Pyomo-only two-phase solve: first user objectives,
                then formulation-tightening terms (``branch/node/child.minimize``)
                with the phase-1 optimum pinned. Removes weight tuning. GEKKO
                falls back to single-objective sum.
        """
        self._controllable_appliables: list = []
        self._controllable_to_attr: dict[GenericModel, str] = {}
        self._bounds_for_controllables: list = []
        self._objectives: Objectives | None = None
        self._constraints: Constraints | None = None
        self._debug = debug
        self._lex_objectives = lex_objectives

    @property
    def lex_objectives(self) -> bool:
        return self._lex_objectives

    @staticmethod
    def _promote_to_var(model, attribute: str, val, param: AttributeParameter | None):
        if param is not None:
            return Var(
                param.val(attribute, val),
                param.max(attribute, val),
                param.min(attribute, val),
                param.integer,
                name=attribute,
            )
        if val == 0.0:  # NOSONAR
            logger.warning(
                "Attribute '%s' on %s has value 0.0 and no "
                "explicit bounds - inferred bounds [0, 0] "
                "will lock this variable. Use an "
                "AttributeParameter or prob.bounds() to "
                "set meaningful bounds.",
                attribute,
                type(model).__name__,
            )
        return Var(
            val,
            max=0 if val <= 0 else val,
            min=0 if val > 0 else val,
            name=attribute,
        )

    def _apply(self, network: Network):
        self._controllable_to_attr.clear()
        for appliable in self._controllable_appliables:
            appliable(network)
        for model, attributes in self._controllable_to_attr.items():
            for attribute_param in attributes:
                attribute, param = (
                    attribute_param
                    if isinstance(attribute_param, tuple)
                    else (attribute_param, None)
                )
                if not hasattr(model, attribute):
                    continue
                val = getattr(model, attribute)
                if type(val) is Var:
                    continue
                setattr(
                    model, attribute, self._promote_to_var(model, attribute, val, param)
                )
                if self._debug:
                    logger.warning("From the model %s", model)
                    logger.warning("The attribute %s has been replaced", attribute)
        for (
            min_value,
            max_value,
            component_condition,
            attributes,
        ) in self._bounds_for_controllables:
            for component in network.all_components():
                if not (
                    component_condition(component.model, component.grid)
                    and component.independent
                ):
                    continue
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

    def bounds(self, minmax, component_condition=lambda _m, _g: True, attributes=None):
        """Override min/max for ``Var`` attributes on matching components.
        ``component_condition`` is ``(model, grid) -> bool``."""
        if not attributes:
            raise ValueError(
                "bounds() requires a non-empty list of attribute names to bound, "
                f"got {attributes!r}."
            )
        self._bounds_for_controllables.append(
            (minmax[0], minmax[1], component_condition, attributes)
        )

    def controllable(
        self,
        attributes: list[str | tuple[str, AttributeParameter]],
        component_condition=lambda _: True,
    ):
        """Promote attributes on matching components to free Vars. Low-level
        primitive; prefer the typed ``controllable_*`` helpers. Each attribute
        entry is a name or ``(name, AttributeParameter)`` tuple."""

        def apply_controllable(network: Network):
            component_list = network.all_components()
            for component in component_list:
                if component_condition(component):
                    self.add_to_controllable(component.model, attributes)

        self._controllable_appliables.append(apply_controllable)
        return self

    def controllable_demands(
        self, attributes: list[str | tuple[str, AttributeParameter]]
    ):
        """Make PowerLoad/HeatLoad/HeatExchangerLoad/PassiveHeatExchangerLoad,
        gas Sinks, and consuming HeatExchanger/PassiveHeatExchanger controllable.

        Note: an attribute currently at 0.0 with no AttributeParameter is locked
        to [0,0]. Use prob.bounds() or pass an AttributeParameter for real bounds.
        """
        self.controllable(
            component_condition=lambda component: (
                (
                    isinstance(
                        component.model,
                        HeatExchangerLoad
                        | PassiveHeatExchangerLoad
                        | PowerLoad
                        | HeatLoad,
                    )
                    or (
                        type(component.model) is Sink
                        and type(component.grid) is GasGrid
                    )
                    or hx_is_consuming(component.model)
                )
                and component.active
                and (not component.ignored)
            ),
            attributes=attributes,
        )
        return self

    def controllable_generators(self, attributes):
        """Make PowerGenerator/HeatGenerator/HeatExchangerGenerator/
        PassiveHeatExchangerGenerator/Source and generating (bare)
        HeatExchanger/PassiveHeatExchanger controllable. See
        :meth:`controllable_demands` for the 0.0-locks-to-[0,0] caveat."""
        self.controllable(
            component_condition=lambda component: (
                (
                    isinstance(
                        component.model,
                        HeatExchangerGenerator
                        | PassiveHeatExchangerGenerator
                        | PowerGenerator
                        | HeatGenerator
                        | Source,
                    )
                    or hx_is_generating(component.model)
                )
                and component.active
                and (not component.ignored)
            ),
            attributes=attributes,
        )
        return self

    def optimize_bus_voltages(self, vm_min=None, vm_max=None):
        """Make the bus voltage magnitudes optimisation variables.

        Bounds every bus voltage magnitude to ``[vm_min, vm_max]`` (when given)
        and frees the slack: each :class:`ExtPowerGrid` stops pinning its bus
        ``vm_pu``, so the reference voltage becomes a decision variable in the
        same band and only the reference *angle* stays fixed - the optimal power
        flow convention (MATPOWER/pandapower ``runopp`` optimise the slack
        voltage within [VMIN, VMAX]).

        Freeing the slack only makes sense when the buses carry a voltage band,
        so the two go together. Pass no bounds to free the slack against limits
        already on the buses (e.g. the per-bus VMIN/VMAX of a MATPOWER import).
        """

        def _apply_voltages(network: Network):
            if vm_min is not None or vm_max is not None:
                for node in network.nodes:
                    self._bound_var(getattr(node.model, "vm_pu", None), vm_min, vm_max)
                    self._bound_var(
                        getattr(node.model, "vm_pu_squared", None),
                        None if vm_min is None else vm_min * vm_min,
                        None if vm_max is None else vm_max * vm_max,
                    )
            for child in network.childs:
                if isinstance(child.model, ExtPowerGrid):
                    child.model.regulate_vm = False

        self._controllable_appliables.append(_apply_voltages)
        return self

    @staticmethod
    def _bound_var(var, lo, hi):
        if isinstance(var, Var):
            if lo is not None:
                var.min = lo
            if hi is not None:
                var.max = hi

    def controllable_ext(self):
        """Declare ExtPowerGrid / ExtHydrGrid connections controllable.

        Purely declarative: these models already expose their exchange
        (``p_mw`` / ``mass_flow_kgs``) as free Vars from their own ``__init__``, so
        no attribute is (re)bound here (``attributes=[]``). The call documents
        intent and registers the components with the controllable set; it does
        not itself make the ext-grid exchange free.
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
        """Make all coupling-point components (CHP / P2H / G2H / P2G / G2P /
        their HG variants) controllable on *attributes*."""
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
        """Promote dispatch on all storage components via their ``make_controllable``."""

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
        """Add a binary ``on_off`` ∈ {0,1} on every branch with ``backup=True``."""
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
