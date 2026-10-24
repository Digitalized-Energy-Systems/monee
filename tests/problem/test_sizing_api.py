"""Regressions of the optimization API found while building a pipe/cable
sizing study: heat-duty semantics, switchable water branches on quadratic
solvers, list-returning constraints, decision ratings in loading limits,
add_to_controllable, multi-period post-processing, the velocity report of a
decision diameter, and shared design decisions across periods."""

from types import SimpleNamespace

import pytest

import monee.express as mx
import monee.model as mm
import monee.problem as mp
from monee import run_energy_flow_optimization
from monee.model.formulation.common import ensure_velocity_report
from monee.model.phys.core.hydraulics import calc_pipe_area
from monee.problem.core import REGULATION_ATTR
from monee.problem.utils import line_loading_limit
from monee.simulation import run_multi_period
from monee.simulation.timeseries import TimeseriesData
from monee.solver.core import apply_post_process, prepare_solve_network


def _gurobipy_available() -> bool:
    try:
        import gurobipy as gp

        m = gp.Model()
        m.setParam("OutputFlag", 0)
        m.optimize()
        return True
    except Exception:
        return False


requires_gurobi = pytest.mark.skipif(
    not _gurobipy_available(), reason="gurobipy not installed or no licence"
)

DIAMETERS = (0.05, 0.08, 0.15)


def _dhs(q=(0.3, 0.4), candidates=True):
    """DHS line with two consumers; with *candidates*, the first supply pipe is
    replaced by three parallel candidate pipes of which only 0.15 m carries the
    full design flow (5.6 kg/s) at 1 m/s."""
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=300, unidirectional=False)
    seg = dhs.line(3, heat_exchanger_q_mw=[0, *q])
    dhs.attach_heat_plant(seg.supply.first, seg.return_.first, t_k=358.0)
    if not candidates:
        return net, {}, seg
    s0, s1 = seg.supply.nodes[:2]
    old = next(b for b in net.branches if {b.from_node_id, b.to_node_id} == {s0, s1})
    net.deactivate_by_id(mm.Branch, old.id)
    old.grid.v_max_mps = 1.0
    cands = {
        d: mx.create_water_pipe(net, s0, s1, diameter_m=d, length_m=300)
        for d in DIAMETERS
    }
    return net, cands, seg


def _priced_heat(**kwargs):
    """A heat-supply price far above the formulation's 1/MW duty pull."""
    problem = mp.OptimizationProblem(**kwargs)
    objectives = mp.Objectives()
    objectives.select(lambda m, _c: isinstance(m, mm.HeatExchangerLoad)).calculate(
        lambda ms: sum(50 * m.q_mw_delivered for m in ms)
    )
    problem.objectives = objectives
    return problem


def _size_cost(cands, periods=None):
    ids = set(cands.values())
    objectives = mp.Objectives()
    objective = objectives.select(lambda m, c: isinstance(c, mm.Branch) and c.id in ids)
    if periods is not None:
        objective.when_period(periods)
    objective.calculate(lambda ms: sum(100 * m.diameter_m * m.on_off for m in ms))
    return objectives


def _chosen(frame, cands):
    on = dict(zip(frame["id"], frame["on_off"]))
    return [d for d, bid in cands.items() if on[bid] > 0.5]


# --- heat duty --------------------------------------------------------------


def test_fixed_duty_is_exact_by_default():
    net, _, _ = _dhs(candidates=False)

    result = run_energy_flow_optimization(
        net, _priced_heat(), solver="scip", formulation="nonconvex_miqcqp"
    )

    assert result.success
    hx = result.dataframes["HeatExchangerLoad"]
    assert hx["q_mw_delivered"].tolist() == pytest.approx([0.3, 0.4], abs=1e-6)
    assert not [k for k in result.violations if k.endswith("shortfall")]


def test_heat_duty_at_most_keeps_the_upper_bound_semantics():
    net, _, _ = _dhs(candidates=False)

    result = run_energy_flow_optimization(
        net,
        _priced_heat(heat_duty="at_most"),
        solver="scip",
        formulation="nonconvex_miqcqp",
    )

    assert result.success
    assert result.dataframes["HeatExchangerLoad"]["q_mw_delivered"].max() < 1e-6
    assert len([k for k in result.violations if k.endswith("shortfall")]) == 2


def test_heat_duty_rejects_unknown_mode():
    with pytest.raises(ValueError, match="heat_duty"):
        mp.OptimizationProblem(heat_duty="soft")


def test_load_shedding_keeps_at_most():
    assert mp.create_min_load_shedding_problem().heat_duty == "at_most"
    assert mp.create_economic_dispatch_problem().heat_duty == "exact"


def _dhs_with_boiler():
    net, _, seg = _dhs(candidates=False)
    gas = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, gas)
    boiler = mx.create_g2h(
        net,
        gas_node_id=gas,
        cold_node_id=seg.return_.nodes[-1],
        hot_node_id=seg.supply.nodes[-1],
        heat_energy_mw=0.5,
        diameter_m=0.15,
        efficiency=0.9,
    )
    return net, boiler


def _duty_exact(net, problem):
    network = prepare_solve_network(net, optimization_problem=problem)[0]
    return {
        type(b.model).__name__: b.model._he_duty_exact
        for b in network.branches
        if isinstance(b.model, mm.HeatExchanger)
    }


def test_idle_unit_keeps_at_most_so_multi_period_ipopt_converges():
    net, boiler = _dhs_with_boiler()
    net.compound_by_id(boiler).set(regulation=0.0)
    assert _duty_exact(net, mp.OptimizationProblem()) == {
        "HeatExchangerLoad": True,
        "SubHE": False,
    }

    net, boiler = _dhs_with_boiler()
    series = TimeseriesData()
    series.add_compound_series(boiler, "regulation", [0.6, 0.0])
    problem = mp.OptimizationProblem()
    problem.objectives = mp.Objectives()

    result = run_multi_period(
        net, series, steps=2, optimization_problem=problem, solver="ipopt"
    )

    assert result.success
    assert result.termination_condition == "Solve_Succeeded"


def test_hook_promoting_an_idle_unit_gets_exact_duty():
    net, boiler = _dhs_with_boiler()
    net.compound_by_id(boiler).set(regulation=0.0)
    problem = mp.OptimizationProblem()

    def promote(network):
        control = next(
            sub.model
            for sub in network.compound_by_id(boiler).subcomponents
            if isinstance(sub, mm.Node)
        )
        problem.add_to_controllable(control, REGULATION_ATTR)

    problem.add_hook(promote)

    assert _duty_exact(net, problem)["SubHE"] is True


# --- switchable water branches on quadratic solvers --------------------------


def _choose_one(cands):
    ids = set(cands.values())
    problem = mp.OptimizationProblem()
    problem.controllable(
        [
            (
                "on_off",
                mp.AttributeParameter(
                    min=lambda a, v: 0,
                    max=lambda a, v: 1,
                    val=lambda a, v: 1,
                    integer=True,
                ),
            )
        ],
        component_condition=lambda m, c: isinstance(c, mm.Branch) and c.id in ids,
    )
    constraints = mp.Constraints()
    constraints.select(
        lambda m, c: isinstance(c, mm.Branch) and c.id in ids
    ).comp_equation(lambda ms: sum(m.on_off for m in ms) == 1)
    problem.constraints = constraints
    problem.objectives = _size_cost(cands)
    return problem


@requires_gurobi
@pytest.mark.parametrize("backend", [None, "pyomo"])
def test_candidate_pipes_solve_on_gurobi(backend):
    net, cands, _ = _dhs()

    result = run_energy_flow_optimization(
        net,
        _choose_one(cands),
        solver="gurobi",
        backend=backend,
        formulation="nonconvex_miqcqp",
    )

    assert result.success
    assert _chosen(result.dataframes["WaterPipe"], cands) == [0.15]


def test_decision_on_off_stays_out_of_the_heat_balance():
    """A decision on_off is gated by the branch formulation; the nodal heat
    balance must stay bilinear in it (degree 2), a fixed on_off multiplies."""

    class Sym:
        def __init__(self, degree=1):
            self.degree = degree

        def __mul__(self, other):
            return Sym(self.degree + getattr(other, "degree", 0))

        __rmul__ = __mul__

        def __add__(self, other):
            return Sym(max(self.degree, getattr(other, "degree", 0)))

        __radd__ = __add__
        __sub__ = __add__
        __rsub__ = __add__

        def __neg__(self):
            return self

    def branch(on_off):
        return SimpleNamespace(
            vars={
                "mass_flow_pos_kgs": Sym(),
                "mass_flow_neg_kgs": Sym(),
                "t_from_pu": Sym(),
                "t_to_pu": Sym(),
                "on_off": on_off,
            }
        )

    junction = mm.Junction()
    junction.t_pu = Sym()
    switched = junction.calc_signed_heat_flow(
        [branch(Sym())], [branch(Sym())], [], None
    )
    assert [term.degree for term in switched] == [2, 2]
    fixed = junction.calc_signed_heat_flow([branch(1)], [branch(1)], [], None)
    assert [term.degree for term in fixed] == [2, 2]


# --- constraints -------------------------------------------------------------


def _power_net():
    net = mm.Network()
    b0, b1 = mx.create_bus(net), mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, b1, p_mw=1.0, q_mvar=0.0)
    gen = mx.create_power_generator(net, b1, p_mw=1.0, q_mvar=0.0)
    mx.create_line(net, b0, b1, 1000, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    return net, gen


def _min_import(problem):
    objectives = mp.Objectives()
    objectives.select(lambda m, _c: isinstance(m, mm.ExtPowerGrid)).calculate(
        lambda ms: -sum(m.p_mw for m in ms)
    )
    problem.objectives = objectives
    return problem


@pytest.mark.parametrize("solver", ["ipopt", "scip"])
def test_list_returning_comp_equation_is_applied(solver):
    net, _ = _power_net()
    problem = _min_import(mp.OptimizationProblem())
    problem.controllable_generators(REGULATION_ATTR)
    constraints = mp.Constraints()
    constraints.select(lambda m, _c: isinstance(m, mm.PowerGenerator)).comp_equation(
        lambda ms: [m.regulation <= 0.4 for m in ms]
    )
    problem.constraints = constraints

    result = run_energy_flow_optimization(net, problem, solver=solver)

    assert result.success
    assert result.dataframes["PowerGenerator"]["regulation"].iloc[0] == pytest.approx(
        0.4, abs=1e-5
    )


def test_casadi_rejects_a_non_constraint():
    from monee.solver.casadi import CasModel

    with pytest.raises(TypeError, match="Not a constraint"):
        CasModel().Equation([1, 2])


@pytest.mark.parametrize(
    "solver, formulation",
    [("ipopt", None), ("scip", "el_misocp")],
)
def test_loading_limit_keeps_a_decision_rating(solver, formulation):
    net, gen = _power_net()
    net.deactivate_by_id(mm.Child, gen)
    problem = mp.OptimizationProblem()
    problem.controllable(
        [
            (
                "max_i_ka",
                mp.AttributeParameter(
                    min=lambda a, v: 0.01, max=lambda a, v: 1.0, val=lambda a, v: 0.5
                ),
            )
        ],
        component_condition=lambda m, _c: isinstance(m, mm.PowerLine),
    )
    objectives = mp.Objectives()
    objectives.select(lambda m, _c: isinstance(m, mm.PowerLine)).calculate(
        lambda ms: sum(m.max_i_ka for m in ms)
    )
    problem.objectives = objectives
    constraints = mp.Constraints()
    for side in ("from", "to"):
        constraints.select(lambda m, _c: isinstance(m, mm.PowerLine)).equation(
            lambda m, s=side: line_loading_limit(m, s, 1.0)
        )
    problem.constraints = constraints

    result = run_energy_flow_optimization(
        net, problem, solver=solver, formulation=formulation
    )

    assert result.success
    line = result.dataframes["PowerLine"].iloc[0]
    assert line["max_i_ka"] == pytest.approx(line["i_from_ka"], rel=1e-3)
    assert line["loading_from_pu"] == pytest.approx(1.0, rel=1e-3)


# --- add_to_controllable -----------------------------------------------------


def test_add_to_controllable_with_component_before_solve():
    net, gen = _power_net()
    net.child_by_id(gen).model.regulation = 0.3
    problem = _min_import(mp.OptimizationProblem())
    problem.add_to_controllable(net.child_by_id(gen), REGULATION_ATTR)

    result = run_energy_flow_optimization(net, problem, solver="ipopt")

    assert result.success
    assert result.dataframes["PowerGenerator"]["regulation"].iloc[0] == pytest.approx(
        1.0, abs=1e-5
    )


def test_add_to_controllable_maps_a_caller_component_inside_a_hook():
    # GIVEN a component of the caller's network, passed from a hook
    net, gen = _power_net()
    net.child_by_id(gen).model.regulation = 0.3
    problem = _min_import(mp.OptimizationProblem())
    component = net.child_by_id(gen)
    problem.add_hook(lambda n: problem.add_to_controllable(component, REGULATION_ATTR))

    # WHEN
    result = run_energy_flow_optimization(net, problem, solver="ipopt")

    # THEN the decision reaches the solve and the caller's model is untouched
    assert result.dataframes["PowerGenerator"]["regulation"].iloc[0] == pytest.approx(
        1.0, abs=1e-5
    )
    assert net.child_by_id(gen).model.regulation == 0.3


def test_add_to_controllable_rejects_a_bare_model_outside_hooks():
    net, gen = _power_net()
    with pytest.raises(TypeError, match="needs the component"):
        mp.OptimizationProblem().add_to_controllable(
            net.child_by_id(gen).model, REGULATION_ATTR
        )


def test_add_to_controllable_accepts_the_model_inside_a_hook():
    net, gen = _power_net()
    problem = _min_import(mp.OptimizationProblem())
    problem.add_hook(
        lambda n: problem.add_to_controllable(n.child_by_id(gen).model, REGULATION_ATTR)
    )

    result = run_energy_flow_optimization(net, problem, solver="ipopt")

    assert result.dataframes["PowerGenerator"]["regulation"].iloc[0] == pytest.approx(
        1.0, abs=1e-5
    )


# --- multi-period post-processing ---------------------------------------------


def test_pyomo_multi_period_runs_the_post_processing():
    net, _, _ = _dhs(candidates=False)

    result = run_multi_period(
        net,
        steps=2,
        optimization_problem=mp.OptimizationProblem(),
        solver="scip",
        formulation="nonconvex_miqcqp",
    )

    assert result.success
    assert result.termination_condition == "optimal"
    assert result.solver_status == "ok"
    t_k = result.get_result_for(mm.Junction, "t_k").iloc[1]
    assert t_k.max() == pytest.approx(358.0, abs=1e-3)
    assert (result.get_result_for(mm.Junction, "pressure_pa").iloc[1] > 9e5).all()


# --- multi-period solver options and limit stops ---------------------------------


def _sizing_horizon(solver_options, solver="gurobi"):
    net, cands, seg = _dhs()
    problem = mp.OptimizationProblem()
    problem.candidate_branches([cands.values()])
    problem.objectives = _size_cost(cands, periods=[0])
    return run_multi_period(
        net,
        _light_then_peak(seg),
        steps=2,
        optimization_problem=problem,
        solver=solver,
        formulation="nonconvex_miqcqp",
        solver_options=solver_options,
    )


@requires_gurobi
def test_multi_period_limit_stop_returns_the_incumbent_and_gap():
    """Regression: a multi-period solve stopped at a limit raised RuntimeError
    and discarded its incumbent and gap. NodeLimit stops deterministically
    after the root, where Gurobi holds an incumbent but no proof."""
    result = _sizing_horizon({"NodeLimit": 0, "Threads": 1})

    stats = result.solver_stats
    assert result.success
    assert result.solver_status == "aborted"
    assert result.termination_condition == "maxEvaluations"
    assert stats["obj"] == pytest.approx(result.objective)
    assert stats["bound"] < stats["obj"]
    assert stats["gap"] > 0
    on = result.get_result_for(mm.WaterPipe, "on_off").iloc[0]
    assert on.max() == pytest.approx(1.0)


@requires_gurobi
@pytest.mark.parametrize("solver", ["gurobi", "gurobi_direct"])
def test_multi_period_time_limit_without_incumbent_names_the_limit(solver):
    """Pyomo's own solution loading raised 'Cannot load a SolverResults object
    with bad status: aborted' here, which hid the time limit. gurobi_direct
    always inserts a solution entry, which read as an incumbent and returned
    the start values with success=True."""
    with pytest.raises(RuntimeError, match="termination=maxTimeLimit"):
        _sizing_horizon({"TimeLimit": 0}, solver=solver)


@requires_gurobi
def test_candidate_pipes_solve_with_lumped_thermal_capacitance():
    """Regression: the capacitance's nodal heat balance multiplied a decision
    on_off into m * t (degree three), which gurobi cannot take."""
    net, cands, seg = _dhs()
    net.add_extension(mm.LumpedThermalCapacitance())
    problem = mp.OptimizationProblem()
    problem.candidate_branches([cands.values()])
    problem.objectives = _size_cost(cands, periods=[0])

    result = run_multi_period(
        net,
        _light_then_peak(seg),
        steps=2,
        optimization_problem=problem,
        solver="gurobi",
        formulation="nonconvex_miqcqp",
    )

    assert result.success
    assert result.termination_condition == "optimal"


@requires_gurobi
def test_multi_period_optimum_reports_a_closed_gap():
    result = _sizing_horizon({"MIPGap": 0})

    assert result.termination_condition == "optimal"
    assert result.solver_stats["gap"] < 1e-9


def test_multi_period_solver_options_need_the_pyomo_back_end():
    net, _, _ = _dhs(candidates=False)

    with pytest.raises(ValueError, match="Pyomo multi-period back-end only"):
        run_multi_period(net, steps=1, solver="ipopt", solver_options={"max_iter": 10})


# --- velocity report -----------------------------------------------------------


def test_velocity_uses_the_solved_diameter():
    grid = mm.WaterGrid(name="water")
    pipe = mm.WaterPipe(diameter_m=0.1, length_m=100)
    ensure_velocity_report(pipe, grid)
    pipe.diameter_m = mm.Var(0.2)  # promoted by a problem after the attach
    pipe.mass_flow_pos_kgs, pipe.mass_flow_neg_kgs = mm.Var(0.0), mm.Var(10.0)

    apply_post_process(pipe)

    expected = -10.0 / (grid.fluid_density_kg_per_m3 * calc_pipe_area(0.2))
    assert pipe.velocity_mps.value == pytest.approx(expected)


# --- design decisions shared across periods -----------------------------------


def _light_then_peak(seg):
    data = TimeseriesData()
    for hx, light in zip(seg.heat_exchangers, (0.0, 0.5)):
        data.add_branch_series(hx, "regulation", [light, 1.0])
    return data


@requires_gurobi
def test_candidate_branches_pick_one_build_for_all_periods():
    net, cands, seg = _dhs()
    problem = mp.OptimizationProblem()
    problem.candidate_branches([cands.values()])
    problem.objectives = _size_cost(cands, periods=[0])

    result = run_multi_period(
        net,
        _light_then_peak(seg),
        steps=2,
        optimization_problem=problem,
        solver="gurobi",
        formulation="nonconvex_miqcqp",
    )

    assert result.success
    on = result.get_result_for(mm.WaterPipe, "on_off")
    for t in range(2):
        assert [d for d, bid in cands.items() if on.iloc[t][bid] > 0.5] == [0.15]
    assert result.user_objective == pytest.approx(15.0)


@requires_gurobi
def test_without_design_each_period_sizes_on_its_own():
    """Control for the test above: per-period binaries undersize the light hour."""
    net, cands, seg = _dhs()

    result = run_multi_period(
        net,
        _light_then_peak(seg),
        steps=2,
        optimization_problem=_choose_one(cands),
        solver="gurobi",
        formulation="nonconvex_miqcqp",
    )

    on = result.get_result_for(mm.WaterPipe, "on_off")
    assert [d for d, bid in cands.items() if on.iloc[0][bid] > 0.5] == [0.05]


@requires_gurobi
@pytest.mark.parametrize(
    "allow_none, expected", [(True, [0.15]), (False, [0.05, 0.15])]
)
def test_candidate_branches_allow_none(allow_none, expected):
    net, cands, _ = _dhs()
    problem = mp.OptimizationProblem()
    problem.candidate_branches(
        [[cands[0.15]], [cands[0.05], cands[0.08]]], allow_none=allow_none
    )
    problem.objectives = _size_cost(cands)

    result = run_energy_flow_optimization(
        net, problem, solver="gurobi", formulation="nonconvex_miqcqp"
    )

    assert result.success
    assert _chosen(result.dataframes["WaterPipe"], cands) == expected


def test_controllable_design_ties_later_periods_to_the_first():
    from monee.problem.core import _same_as_first_period

    first, later = object(), SimpleNamespace(on_off="x1")

    class Period:
        def __init__(self, t):
            self.current_t = t

        def get(self, cid, attr, step=-1, model_type=None):
            return first if step == 0 else None

    class Pinned:
        def __eq__(self, other):
            return ("tie", other)

    model = SimpleNamespace(on_off=Pinned())
    assert _same_as_first_period(model, 1, Period(1), ["on_off"]) == [("tie", first)]
    model_t0 = SimpleNamespace(on_off=first)
    assert _same_as_first_period(model_t0, 1, Period(0), ["on_off"]) == []
    # a timeseries StepState has no current_t: no tie
    assert _same_as_first_period(later, 1, SimpleNamespace(get=None), ["on_off"]) == []


# --- per-category ids ---------------------------------------------------------


def test_tid_is_unique_where_id_is_not():
    net, _ = _power_net()
    components = list(net.all_components())
    ids = [c.id for c in components]
    tids = [c.tid for c in components]
    assert len(set(ids)) < len(ids)
    assert len(set(tids)) == len(tids)
