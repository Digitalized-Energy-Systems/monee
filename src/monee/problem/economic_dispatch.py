from monee.model.branch import GenericPowerBranch
from monee.model.child import ExtPowerGrid, PowerGenerator
from monee.model.node import Bus
from monee.problem.core import (
    Constraints,
    Objectives,
    OptimizationProblem,
)


def _gen_cost(model):
    """Return the generation cost term for a single generator.

    ``PowerGenerator`` follows the load convention (``p_mw`` is stored
    negative), so ``-model.p_mw`` gives positive generation output.
    The ``cost`` attribute is expected to be in currency/MWh (or
    currency/MW for a single snapshot).

    If the model does not carry a ``cost`` attribute the contribution
    is zero, allowing mixed fleets where only some units have costs.
    """
    cost = getattr(model, "cost", None)
    if cost is None:
        return 0
    return cost * (-model.p_mw)


def _ext_grid_cost(model):
    """Return the import cost for an external grid connection.

    Positive ``p_mw`` = import (buying from the upstream grid).  The
    ``cost`` attribute represents the price per MW of imported power.
    """
    cost = getattr(model, "cost", None)
    if cost is None:
        return 0
    return cost * model.p_mw


def create_economic_dispatch_problem(
    gen_cost_default=1.0,
    ext_grid_cost_default=None,
    bounds_vm=(0.9, 1.1),
    bounds_lp=(0, 1.0),
    ext_grid_bounds=None,
    ramp_limit=None,
    check_vm=True,
    check_lp=True,
    debug=False,
):
    """Create an economic dispatch OPF that minimises total generation cost.

    Each :class:`~monee.model.PowerGenerator` is made controllable on
    ``p_mw``.  The solver dispatches generators to minimise the sum of
    ``cost * p_gen`` across all units while respecting network constraints.

    **Assigning costs** -- attach a ``cost`` attribute to each generator
    model *before* solving.  For static costs set it directly on the model;
    for time-varying prices use
    :meth:`~monee.simulation.timeseries.TimeseriesData.add_objective_data`::

        net.child_models[gen_id].cost = 40          # EUR/MWh
        # -- or for multi-period --
        td.add_objective_data(gen_id, "cost", [40, 80, 60, 30])

    Generators without a ``cost`` attribute use *gen_cost_default*.

    Args:
        gen_cost_default: Default marginal cost applied to generators that
            have no explicit ``cost`` attribute.  Set to ``None`` to skip
            generators without costs in the objective.
        ext_grid_cost_default: Marginal cost for importing power from
            :class:`~monee.model.ExtPowerGrid`.  ``None`` (default) means
            the external grid is *not* part of the objective (it still
            balances the system as the slack bus).  Set a value to include
            grid import cost and make it controllable.
        bounds_vm: Voltage magnitude bounds ``(min, max)`` in pu.
        bounds_lp: Line loading bounds ``(min, max)`` in pu.
        ext_grid_bounds: Active power bounds ``(min, max)`` in MW for
            ``ExtPowerGrid``.  Required when *ext_grid_cost_default* is not
            ``None``.  Ignored otherwise.
        ramp_limit: Maximum change of ``p_mw`` (in MW) between consecutive
            periods.  ``None`` = no ramp limit.  Only effective in
            multi-period / MPC mode.
        check_vm: Enforce bus voltage magnitude bounds.
        check_lp: Enforce line loading limits.
        debug: Enable debug logging for variable promotion.

    Returns:
        An :class:`OptimizationProblem` suitable for
        :func:`~monee.simulation.run`,
        :func:`~monee.simulation.multi_period.run_multi_period`, or
        :func:`~monee.simulation.multi_period.run_mpc`.

    Example -- single-period dispatch::

        from monee.problem.economic_dispatch import (
            create_economic_dispatch_problem,
        )
        import monee.simulation as sim

        # assign costs to generators
        net.child_models[gen1_id].cost = 40   # cheap baseload
        net.child_models[gen2_id].cost = 80   # expensive peaker

        prob = create_economic_dispatch_problem()
        result = sim.run(net, optimization_problem=prob)

    Example -- multi-period with time-varying prices::

        from monee.problem.economic_dispatch import (
            create_economic_dispatch_problem,
        )
        from monee.simulation.multi_period import run_multi_period
        from monee.simulation.timeseries import TimeseriesData

        td = TimeseriesData()
        td.add_child_series(load_id, "p_mw", [2.0, 5.0, 3.0, 1.0])
        td.add_objective_data(gen1_id, "cost", [40, 40, 80, 80])
        td.add_objective_data(gen2_id, "cost", [60, 60, 60, 60])

        prob = create_economic_dispatch_problem(ramp_limit=2.0)
        result = run_multi_period(net, td, steps=4, optimization_problem=prob)
    """
    problem = OptimizationProblem(debug=debug)

    # --- Controllable generators (p_mw) ---
    problem.controllable_generators(["p_mw"])

    # --- Optionally make ext grid controllable with bounds ---
    include_ext_grid = ext_grid_cost_default is not None
    if include_ext_grid:
        problem.controllable_ext()
        if ext_grid_bounds is not None:
            problem.bounds(
                ext_grid_bounds,
                lambda m, _: isinstance(m, ExtPowerGrid),
                ["p_mw"],
            )

    # --- Voltage and line loading bounds ---
    if check_vm:
        problem.bounds(bounds_vm, lambda m, _: type(m) is Bus, ["vm_pu"])
    if check_lp:
        problem.bounds(
            bounds_lp,
            lambda m, _: isinstance(m, GenericPowerBranch),
            ["loading_from_percent", "loading_to_percent"],
        )

    # --- Objective: minimise total generation cost ---
    objectives = Objectives()

    def _apply_default_cost(model):
        if not hasattr(model, "cost") or model.cost is None:
            return gen_cost_default
        return model.cost

    def _apply_ext_default_cost(model):
        if not hasattr(model, "cost") or model.cost is None:
            return ext_grid_cost_default
        return model.cost

    # Generator cost objective
    objectives.select(lambda m: isinstance(m, PowerGenerator)).calculate(
        lambda models: sum(_apply_default_cost(m) * (-m.p_mw) for m in models)
    )

    # External grid import cost (if enabled)
    if include_ext_grid:
        objectives.select(lambda m: isinstance(m, ExtPowerGrid)).calculate(
            lambda models: sum(_apply_ext_default_cost(m) * m.p_mw for m in models)
        )

    problem.objectives = objectives

    # --- Constraints ---
    constraints = Constraints()

    if check_lp:
        from monee.problem.utils import line_loading_limit

        constraints.select_types(GenericPowerBranch).equation(
            lambda model: line_loading_limit(model, "from", bounds_lp[1])
        ).equation(lambda model: line_loading_limit(model, "to", bounds_lp[1]))

    if include_ext_grid and ext_grid_bounds is not None:
        constraints.select_types(ExtPowerGrid).equation(
            lambda model, _b=ext_grid_bounds: model.p_mw >= _b[0]
        ).equation(lambda model, _b=ext_grid_bounds: model.p_mw <= _b[1])

    # --- Ramp constraints (cross-period coupling) ---
    if ramp_limit is not None:
        _ramp = ramp_limit

        def _gen_ramp(model, cid, ts):
            prev_p = ts.get(cid, "p_mw")
            if prev_p is None:
                return []
            return [
                model.p_mw - prev_p <= _ramp,
                prev_p - model.p_mw <= _ramp,
            ]

        constraints.select(
            lambda comp: (
                isinstance(comp.model, PowerGenerator)
                and comp.active
                and (not comp.ignored)
            )
        ).temporal_equation(_gen_ramp)

    problem.constraints = constraints
    return problem


def create_multi_period_economic_dispatch_problem(
    gen_cost_default=1.0,
    ext_grid_cost_default=None,
    bounds_vm=(0.9, 1.1),
    bounds_lp=(0, 1.0),
    ext_grid_bounds=None,
    ramp_limit=None,
    check_vm=True,
    check_lp=True,
    debug=False,
):
    """Convenience alias for :func:`create_economic_dispatch_problem`.

    All parameters are forwarded directly. This function exists to mirror
    the naming convention of
    :func:`~monee.problem.load_shedding.create_multi_period_load_shedding_optimization_problem`.

    See :func:`create_economic_dispatch_problem` for full documentation.
    """
    return create_economic_dispatch_problem(
        gen_cost_default=gen_cost_default,
        ext_grid_cost_default=ext_grid_cost_default,
        bounds_vm=bounds_vm,
        bounds_lp=bounds_lp,
        ext_grid_bounds=ext_grid_bounds,
        ramp_limit=ramp_limit,
        check_vm=check_vm,
        check_lp=check_lp,
        debug=debug,
    )
