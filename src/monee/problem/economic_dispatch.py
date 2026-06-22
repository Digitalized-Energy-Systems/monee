from monee.model.branch import GenericPowerBranch
from monee.model.child import ExtPowerGrid, PowerGenerator
from monee.model.node import Bus
from monee.problem.core import (
    Constraints,
    Objectives,
    OptimizationProblem,
)
from monee.problem.utils import line_loading_limit


def create_economic_dispatch_problem(  # NOSONAR
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
    """Economic dispatch OPF minimising ``Σ cost · p_gen``. Set each generator's
    ``cost`` (currency/MW) directly or per-period via TimeseriesData.add_objective_data."""
    problem = OptimizationProblem(debug=debug)

    problem.controllable_generators(["p_mw"])

    include_ext_grid = ext_grid_cost_default is not None
    if include_ext_grid:
        problem.controllable_ext()
        if ext_grid_bounds is not None:
            problem.bounds(
                ext_grid_bounds,
                lambda m, _: isinstance(m, ExtPowerGrid),
                ["p_mw"],
            )

    if check_vm:
        problem.bounds(bounds_vm, lambda m, _: type(m) is Bus, ["vm_pu"])
    # The line-loading cap is enforced by the line_loading_limit *constraint*
    # below (added when check_lp). loading_*_pu are passive intermediates in
    # every electricity formulation (NLP and MISOCP), so a var-bounds override on
    # them is a no-op - the constraint is the single enforcement path.

    objectives = Objectives()

    def _apply_default_cost(model):
        if not hasattr(model, "cost") or model.cost is None:
            return gen_cost_default
        return model.cost

    def _apply_ext_default_cost(model):
        if not hasattr(model, "cost") or model.cost is None:
            return ext_grid_cost_default
        return model.cost

    objectives.select(lambda m: isinstance(m, PowerGenerator)).calculate(
        lambda models: sum(_apply_default_cost(m) * (-m.p_mw) for m in models)
    )

    if include_ext_grid:
        objectives.select(lambda m: isinstance(m, ExtPowerGrid)).calculate(
            lambda models: sum(_apply_ext_default_cost(m) * m.p_mw for m in models)
        )

    problem.objectives = objectives

    constraints = Constraints()

    if check_lp:
        constraints.select_types(GenericPowerBranch).equation(
            lambda model: line_loading_limit(model, "from", bounds_lp[1])
        ).equation(lambda model: line_loading_limit(model, "to", bounds_lp[1]))

    if include_ext_grid and ext_grid_bounds is not None:
        constraints.select_types(ExtPowerGrid).equation(
            lambda model, _b=ext_grid_bounds: model.p_mw >= _b[0]
        ).equation(lambda model, _b=ext_grid_bounds: model.p_mw <= _b[1])

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
    """Alias for :func:`create_economic_dispatch_problem` (mirrors load_shedding naming)."""
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
