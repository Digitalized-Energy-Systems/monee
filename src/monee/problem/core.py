import inspect
import logging
import math
import warnings
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


def _bound_attribute(model, attribute: str, min_value, max_value):
    """Apply a user bound to *attribute*, redirecting it onto the paired
    ``<attribute>_squared`` Var when the formulation reports *attribute* itself
    as an Intermediate (branch-flow electricity: ``vm_pu`` is derived,
    ``vm_pu_squared`` is the decision variable). Bounding a non-Var is a no-op
    the user cannot see, so it is reported instead of silently dropped."""
    var = getattr(model, attribute, None)
    if type(var) is Var:
        var.min, var.max = min_value, max_value
        return
    squared = getattr(model, f"{attribute}_squared", None)
    if type(squared) is Var and min_value >= 0 and max_value >= 0:
        squared.min, squared.max = min_value * min_value, max_value * max_value
        logger.info(
            "Bound on %s.%s redirected onto %s_squared: the active formulation "
            "uses the squared quantity as its decision variable.",
            type(model).__name__,
            attribute,
            attribute,
        )
        return
    logger.warning(
        "bounds() on %s.%s has no effect: the attribute is %s, not a decision "
        "variable of the active formulation.",
        type(model).__name__,
        attribute,
        "missing" if var is None else type(var).__name__,
    )


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


def _required_positional_args(predicate):
    try:
        signature = inspect.signature(predicate)
    except (TypeError, ValueError):
        return None
    required = 0
    for parameter in signature.parameters.values():
        if parameter.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            if parameter.default is inspect.Parameter.empty:
                required += 1
        elif parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            break
    return required


def _accepts(predicate, argument, errors):
    try:
        return bool(predicate(argument))
    except Exception as exc:  # noqa: BLE001 - a predicate written for the other
        errors.append(exc)  # convention may raise; the other pass decides
        return False


_MODEL_FALLBACK_HINT = (
    "matched no Component but the same predicate accepts the component "
    "model(s); applying the model interpretation."
)
_COMPONENT_FALLBACK_HINT = (
    "matched no model but the same predicate accepts the Component "
    "container(s); applying the Component interpretation."
)


def _select_dual_convention(
    predicate, components, native_of, other_of, where, applied_hint, reported
):
    """Return the components accepted by *predicate*. A two-argument predicate
    is called with ``(model, component)``; a one-argument predicate first with
    the selector's native argument and, when that matches nothing, with the
    other convention's argument (reported once as a warning)."""
    required = _required_positional_args(predicate)
    if required is not None and required >= 2:
        return [
            component
            for component in components
            if predicate(component.model, component)
        ]
    errors = []
    selected = [
        component
        for component in components
        if _accepts(predicate, native_of(component), errors)
    ]
    if selected:
        return selected
    fallback_errors = []
    fallback = [
        component
        for component in components
        if _accepts(predicate, other_of(component), fallback_errors)
    ]
    if fallback:
        if not reported:
            reported.append(True)
            warnings.warn(
                f"{where} {applied_hint} Prefer the two-argument form "
                "'lambda model, component: ...'. See the api/monee.problem "
                "docs page.",
                stacklevel=3,
            )
        return fallback
    if errors:
        raise errors[0]
    return []


def _report_zero_match(where, components, reported, looked_for=None, optional=False):
    if not components or reported:
        return
    reported.append(True)
    detail = f" It looked for {looked_for}." if looked_for else ""
    if optional:
        logger.debug(
            "%s matched no component; skipped (optional selector).%s", where, detail
        )
        return
    warnings.warn(
        f"{where} matched no component and has no effect.{detail} "
        "See the tutorials/01_optimization_basics docs page.",
        stacklevel=2,
    )


def _class_names(cls_tuple):
    classes = cls_tuple if isinstance(cls_tuple, tuple) else (cls_tuple,)
    return "/".join(cls.__name__ for cls in classes)


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
            return [
                self._calculator(_attach_data(self._data_attacher, selected_models))
            ]
        return [self._calculator(selected_models)]


class Objectives:
    def __init__(self) -> None:
        self._objectives = []

    def select(
        self, model_selection_function, optional=False, looked_for=None
    ) -> Objective:
        """Select the models the objective is computed over.

        Recommended: the two-argument form ``lambda model, component: ...``,
        which works identically on :meth:`Constraints.select` and
        ``OptimizationProblem.controllable``. A one-argument predicate
        receives the model (``lambda m: isinstance(m, PowerGenerator)``); one
        written for the Component container is still applied, with a warning,
        when the model interpretation matches nothing. A predicate that
        matches nothing at all is reported as a warning naming *looked_for*
        when given; ``optional=True`` marks an empty match as expected
        configuration (a component family the network may simply lack, as in
        the built-in problem factories) and downgrades the warning to a DEBUG
        log entry. Selectors without ``optional=True`` keep the warning. See
        the api/monee.problem docs page."""

        reported = []

        def _selected(network):
            components = [
                component
                for component in network.all_components()
                if component.active and not component.ignored
            ]
            selected = _select_dual_convention(
                model_selection_function,
                components,
                native_of=lambda component: component.model,
                other_of=lambda component: component,
                where="Objectives.select",
                applied_hint=_COMPONENT_FALLBACK_HINT,
                reported=reported,
            )
            if not selected:
                _report_zero_match(
                    "Objectives.select",
                    components,
                    reported,
                    looked_for=looked_for,
                    optional=optional,
                )
            return [component.model for component in selected]

        objective = Objective(_selected)
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

    def select(
        self, component_selection_function, optional=False, looked_for=None
    ) -> Constraint:
        """Constrain the components accepted by *component_selection_function*.

        Recommended: the two-argument form ``lambda model, component: ...``,
        which works identically on :meth:`Objectives.select` and
        ``OptimizationProblem.controllable``. A one-argument predicate
        receives the :class:`~monee.model.core.Component` container
        (``lambda c: isinstance(c.model, PowerLoad)``); one written for the
        model is still applied, with a warning, when the Component
        interpretation matches nothing. The equations registered on the
        returned :class:`Constraint` are then called with the model. A
        predicate that matches nothing at all is reported as a warning naming
        *looked_for* when given; ``optional=True`` marks an empty match as
        expected configuration (a component family the network may simply
        lack, as in the built-in problem factories) and downgrades the warning
        to a DEBUG log entry. Selectors without ``optional=True`` keep the
        warning. See the api/monee.problem docs page."""

        reported = []

        def _selected(network):
            components = [
                component
                for component in network.all_components()
                if component.active and not component.ignored
            ]
            selected = _select_dual_convention(
                component_selection_function,
                components,
                native_of=lambda component: component,
                other_of=lambda component: component.model,
                where="Constraints.select",
                applied_hint=_MODEL_FALLBACK_HINT,
                reported=reported,
            )
            if not selected:
                _report_zero_match(
                    "Constraints.select",
                    components,
                    reported,
                    looked_for=looked_for,
                    optional=optional,
                )
            return selected

        constraint = Constraint(
            lambda network: [component.model for component in _selected(network)],
            selected_models_with_ids_link=lambda network: [
                (component.model, component.id) for component in _selected(network)
            ],
        )
        self._constraints.append(constraint)
        return constraint

    def select_types(self, model_cls_tuple, optional=False) -> Constraint:
        """Constrain every component whose *model* is an instance of
        *model_cls_tuple*. Type-based shorthand for :meth:`select`, which takes a
        predicate over the Component."""
        return self.select(
            lambda component: isinstance(component.model, model_cls_tuple),
            optional=optional,
            looked_for=f"{_class_names(model_cls_tuple)} components",
        )

    def select_grids(self, grid_cls_tuple, optional=False) -> Constraint:
        return self.select(
            lambda component: isinstance(component.grid, grid_cls_tuple),
            optional=optional,
            looked_for=f"components on a {_class_names(grid_cls_tuple)} grid",
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
            ),
            looked_for="components with a regulation attribute (regulation_ramp)",
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


# Regulation attribute parameter (0 to 1) used across load-shedding formulations.
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

    Workflow: create, call ``controllable_*`` helpers, set ``objectives`` and
    ``constraints``, then pass to :func:`run_multi_period` / :func:`run_mpc`.
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
        self._bounds_reported: set[int] = set()
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
        for index, (
            min_value,
            max_value,
            component_condition,
            attributes,
            optional,
            looked_for,
        ) in enumerate(self._bounds_for_controllables):
            matched = False
            for component in network.all_components():
                if not (
                    component_condition(component.model, component.grid)
                    and component.independent
                ):
                    continue
                matched = True
                if self._debug:
                    logger.info("From the model %s", component.model)
                    logger.info("The attributes %s are bounded", attributes)
                for attribute in attributes:
                    _bound_attribute(component.model, attribute, min_value, max_value)
            if not matched and index not in self._bounds_reported:
                self._bounds_reported.add(index)
                detail = f" It looked for {looked_for}." if looked_for else ""
                if optional:
                    logger.debug(
                        "bounds() on attributes %r matched no component; skipped "
                        "(optional selector).%s",
                        attributes,
                        detail,
                    )
                    continue
                warnings.warn(
                    f"bounds() on attributes {attributes!r} matched no component "
                    f"and has no effect.{detail} The component_condition receives "
                    "(model, grid). See the tutorials/01_optimization_basics "
                    "docs page.",
                    stacklevel=2,
                )

    def add_to_controllable(
        self, model, attributes: list[str | tuple[str, AttributeParameter]]
    ):
        if model not in self._controllable_to_attr:
            self._controllable_to_attr[model] = []
        self._controllable_to_attr[model] += attributes

    def bounds(
        self,
        minmax,
        component_condition=lambda _m, _g: True,
        attributes=None,
        optional=False,
        looked_for=None,
    ):
        """Override min/max for ``Var`` attributes on matching components.
        ``component_condition`` is ``(model, grid) -> bool``; a one-argument
        condition receives the model alone (``lambda m: isinstance(m, Bus)``).
        A condition that matches nothing at apply time is reported as a
        warning naming *looked_for* when given; ``optional=True`` marks an
        empty match as expected configuration (used by the problem factories
        for carriers a network may lack) and downgrades the warning to a
        DEBUG log entry. See the api/monee.problem docs page."""
        if not attributes:
            raise ValueError(
                "bounds() requires a non-empty list of attribute names to bound, "
                f"got {attributes!r}."
            )
        condition = component_condition
        if _required_positional_args(component_condition) == 1:

            def condition(model, _grid):
                return component_condition(model)

        self._bounds_for_controllables.append(
            (minmax[0], minmax[1], condition, attributes, optional, looked_for)
        )

    def controllable(
        self,
        attributes: list[str | tuple[str, AttributeParameter]],
        component_condition=lambda _: True,
        optional=False,
        looked_for=None,
    ):
        """Promote attributes on matching components to free Vars. Low-level
        primitive; prefer the typed ``controllable_*`` helpers. Each attribute
        entry is a name or ``(name, AttributeParameter)`` tuple.

        Recommended: the two-argument form
        ``component_condition=lambda model, component: ...``, which works
        identically on :meth:`Objectives.select` and
        :meth:`Constraints.select`. A one-argument condition receives the
        :class:`~monee.model.core.Component` container
        (``lambda c: isinstance(c.model, PowerLoad)``); one written for the
        model is still applied, with a warning, when the Component
        interpretation matches nothing. A condition that matches nothing at
        all is reported as a warning naming *looked_for* when given;
        ``optional=True`` marks an empty match as expected configuration (the
        built-in problem factories pass it, since a network may simply lack a
        component family) and downgrades the warning to a DEBUG log entry.
        Conditions without ``optional=True`` keep the warning. See the
        api/monee.problem docs page."""

        reported = []
        attr_names = [a[0] if isinstance(a, tuple) else a for a in attributes]

        def apply_controllable(network: Network):
            component_list = network.all_components()
            selected = _select_dual_convention(
                component_condition,
                component_list,
                native_of=lambda component: component,
                other_of=lambda component: component.model,
                where=f"controllable({attr_names!r})",
                applied_hint=_MODEL_FALLBACK_HINT,
                reported=reported,
            )
            for component in selected:
                self.add_to_controllable(component.model, attributes)
            if not selected:
                _report_zero_match(
                    f"controllable({attr_names!r})",
                    component_list,
                    reported,
                    looked_for=looked_for,
                    optional=optional,
                )

        self._controllable_appliables.append(apply_controllable)
        return self

    def controllable_all(self, attributes):
        self.controllable(component_condition=lambda _: True, attributes=attributes)
        return self

    def controllables_link(self):
        return lambda _: self._controllable_to_attr.keys()

    def controllable_demands(
        self, attributes: list[str | tuple[str, AttributeParameter]], optional=False
    ):
        """Make PowerLoad/HeatLoad/HeatExchangerLoad/PassiveHeatExchangerLoad,
        gas Sinks, and consuming HeatExchanger/PassiveHeatExchanger controllable.

        Note: an attribute currently at 0.0 with no AttributeParameter is locked
        to [0,0]. Use prob.bounds() or pass an AttributeParameter for real bounds.
        """
        self.controllable(
            optional=optional,
            looked_for=(
                "demand components (PowerLoad/HeatLoad/HeatExchangerLoad/"
                "PassiveHeatExchangerLoad, gas Sink, consuming HeatExchanger)"
            ),
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

    def controllable_generators(self, attributes, optional=False):
        """Make PowerGenerator/HeatGenerator/HeatExchangerGenerator/
        PassiveHeatExchangerGenerator/Source and generating (bare)
        HeatExchanger/PassiveHeatExchanger controllable. See
        :meth:`controllable_demands` for the 0.0-locks-to-[0,0] caveat."""
        self.controllable(
            optional=optional,
            looked_for=(
                "generator components (PowerGenerator/HeatGenerator/"
                "HeatExchangerGenerator/PassiveHeatExchangerGenerator/Source, "
                "generating HeatExchanger)"
            ),
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

    def controllable_ext(self, optional=False):
        """Declare ExtPowerGrid / ExtHydrGrid connections controllable.

        Purely declarative: these models already expose their exchange
        (``p_mw`` / ``mass_flow_kgs``) as free Vars from their own ``__init__``, so
        no attribute is (re)bound here (``attributes=[]``). The call documents
        intent and registers the components with the controllable set; it does
        not itself make the ext-grid exchange free.
        """
        self.controllable(
            optional=optional,
            looked_for="external grid components (ExtPowerGrid/ExtHydrGrid)",
            component_condition=lambda component: (
                isinstance(component.model, ExtPowerGrid | ExtHydrGrid)
                and component.active
                and (not component.ignored)
            ),
            attributes=[],
        )
        return self

    def controllable_cps(self, attributes, optional=False):
        """Make all coupling-point components (CHP / P2H / G2H / P2G / G2P /
        their HG variants) controllable on *attributes*."""
        self.controllable(
            optional=optional,
            looked_for=(
                "coupling-point components (CHP/P2H/G2H/P2G/G2P and HG variants)"
            ),
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

    def controllable_backup_lines(self, optional=False):
        """Add a binary ``on_off`` in {0,1} on every branch with ``backup=True``.

        Under an electricity islanding config the branch-energisation gate
        (``ElectricityIslandingMode.branch_gating``) only bounds a backup branch
        from above (``on_off <= e_from``, ``on_off <= e_to``), so closing it
        remains this method's decision; see docs/source/concepts/islanding.md.
        """
        self.controllable(
            optional=optional,
            looked_for="branches with backup=True",
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
