"""Local optima of the multi-energy town's dispatch, and a global solver.

This follows examples/mes_network_limits.py: the same town, the same three
coupling units, the same eight hourly prices and the same two weak links to
the plant site, the 1.5 MVA cable and the 30 mm gas pipe. There IPOPT, the
third-party solver monee calls by default through CasADi, reaches the
optimum in every hour. A local solver need not:

* APOPT, a local MINLP solver that the GEKKO package ships, reports every
  hour of that day as optimal, yet leaves the CHP off in the four hours it
  should run in. Its point is feasible and is the best dispatch without the
  CHP: the CHP's feed-in exchanger sits at zero flow with its outlet off the
  30 K design spread, and from there no small step turns the unit back on.
* IPOPT can stop short of the optimum near these settings. The script
  solves hours 4 and 5 with a 34 mm pipe, a floor of 0.72 and cable ratings
  of 1.5 to 2.1 MVA. Whether and where IPOPT stops early there depends on
  the CasADi build, so IPOPT's numbers stay out of the report; they are
  printed after it. One check still rests on IPOPT: that no result
  undercuts the global optimum, which holds on every build.

SCIP, a third-party global solver that monee calls through Pyomo and
pyscipopt, finds and proves the optimum
of every hour on the exact quadratic formulation (nonconvex_miqcqp). Since
neither that formulation nor APOPT's is the NLP that IPOPT solves, IPOPT then
solves every hour on its own formulation with every unit pinned at SCIP's and
at APOPT's operating points, and must reach each solver's cost. A relaxation (SOC
power flow, epigraph gas flow, exact heat) solved by SCIP bounds the optimum
from below.

The script checks that every SCIP solve ends optimal, each exact SCIP
result as mes_network_limits.py checks IPOPT's (balances, conversions,
limits, the recomputed cost, the hand dispatch and the Weymouth drop of the
thin pipe), that the relaxation lies at or just below the optimum with a
tight power flow cone, that APOPT's point is a valid operating point that
costs exactly the best dispatch without the CHP and exceeds the optimum by
the CHP's margins plus line losses, that hour 0's cost falls all the way
along the CHP's operating point, that pinning the e-boiler off costs its
lost margins plus line losses, that IPOPT confirms SCIP's dispatch on the
NLP formulation within the limits, and that no IPOPT result undercuts the
global optimum. It exits with status 1 if a check fails.

Run::

    python examples/mes_local_optima.py
"""

from __future__ import annotations

import logging
import math
import statistics
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import monee.model as mm
from monee import TimeseriesData, run_timeseries
from monee.model.multi import SubHE
from monee.problem import create_economic_dispatch_problem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mes_economic_dispatch as ed  # noqa: E402
import mes_network_limits as nl  # noqa: E402

APOPT = {"solver": "apopt"}
# The squared gas flows are about 1e-4 (kg/s)^2 here, so SCIP's default
# feasibility tolerance of 1e-6 moves the thin pipe's pressure drop by up to
# 0.4 %; the gap of 1e-4 that monee passes to SCIP by default lets the
# relaxation stop with a loose cone.
SCIP_OPTIONS = {"numerics/feastol": 1e-9, "limits/gap": 1e-6, "limits/time": 60}
EXACT = {
    "solver": "scip",
    "formulation": "nonconvex_miqcqp",
    "solver_options": SCIP_OPTIONS,
}
RELAXATION = {
    "solver": "scip",
    "formulation": ("el_misocp", "gas_convex_miqcqp", "heat_nonconvex_miqcqp"),
    "solver_options": SCIP_OPTIONS,
}
DAY_LIMITS = {
    "cable_mva": nl.CABLE_MVA,
    "pipe_diameter_m": nl.PIPE_DIAMETER_M,
    "floor": nl.PRESSURE_FLOOR,
}
CHP_HOURS = [t for t, price in enumerate(ed.EL_PRICE) if price > ed.CHP_ABOVE]
CHP_PINS = (0.0, 0.25, 0.5, 0.75)
# At this pipe and floor the pipe carries just the CHP's full fuel, and IPOPT
# stops early in many of these hours on some CasADi builds.
SWEEP_PIPE_DIAMETER_M = 0.034
SWEEP_FLOOR = 0.72
SWEEP_CABLE_MVA = (1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1)
SWEEP_HOURS = (4, 5)

UNIT_MODELS = {
    "chp": mm.CHPControlNode,
    "e_boiler": mm.PowerToHeatControlNode,
    "electrolyser": mm.PowerToGas,
}
FEED_INS = ("chp", "e_boiler")


@dataclass
class Exchanger:
    """A feed-in station's exchanger: the heat it feeds in, its flow from the
    return to the supply side, the temperature rise across it, and the
    temperature of the unit's control node."""

    heat_mw: float
    flow_kgs: float
    spread_k: float
    node_k: float


@dataclass
class Hour:
    """One hour solved one way. cost is recomputed from the result tables and
    objective is the solver's user objective (None for APOPT, which reports
    none). A failed solve has the status "failed" and NaN in place of its
    results."""

    t: int
    price: float
    expected: dict
    status: str = "failed"
    retried: bool = False
    seconds: float = 0.0
    dispatch: dict = field(default_factory=lambda: dict.fromkeys(nl.UNITS, math.nan))
    cost: float = math.nan
    objective: float | None = math.nan
    line_loss_mw: float = math.nan
    pipe_drop: float = math.nan
    weymouth_drop: float = math.nan
    soc_residual: float | None = None
    exchangers: dict | None = None
    quantities: dict | None = None

    @property
    def failed(self):
        return self.quantities is None

    @property
    def deviation(self):
        """Largest distance of a unit's operating point from the hand dispatch."""
        if self.failed:
            return math.nan
        return max(abs(self.dispatch[u] - self.expected[u]) for u in nl.UNITS)


@dataclass
class Results:
    """Every solve, keyed by approach. day[label][t]: the network limits day by
    "APOPT", "SCIP" (the exact formulation), "relaxation", "IPOPT", "SCIP on
    NLP" and "APOPT on NLP" (IPOPT at each solver's dispatch) in every hour,
    and by "CHP off" (SCIP with the CHP pinned off) in CHP_HOURS.
    chp_pinned[x]: hour 0 by SCIP with the CHP pinned at x.
    sweep[label][(cable_mva, t)]: hours 4 and 5 near the day's settings by
    "SCIP", "relaxation", "e-boiler off", "IPOPT" and "SCIP on NLP"."""

    day: dict
    chp_pinned: dict
    sweep: dict


class StepLog(logging.Handler):
    """monee's warnings and the wall time of each step of a run_timeseries."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.steps = []
        self._messages = []
        self._since = time.perf_counter()

    def emit(self, record):
        self._messages.append(record.getMessage())

    def step_done(self, step, steps):
        now = time.perf_counter()
        self.steps.append((now - self._since, self._messages))
        self._since, self._messages = now, []


def unit_models(net):
    return {
        unit: component.model
        for component in (*net.nodes, *net.branches)
        for unit, model_type in UNIT_MODELS.items()
        if isinstance(component.model, model_type)
    }


def pin_unit(net, unit, x):
    unit_models(net)[unit].regulation = mm.Var(x, min=x, max=x, name="regulation")


def hand_limits(net, site, settings):
    return nl.Day(
        hours=[],
        cost=0.0,
        cable_mva=settings["cable_mva"],
        cable_mw=nl.cable_capacity_mw(net, site, settings["cable_mva"]),
        floor=settings["floor"],
        pipe_capacity_kgs=nl.pipe_capacity_kgs(net, site, settings["floor"]),
        k_pipe=nl.weymouth_k(net, site.pipe),
    )


def feed_in_exchangers(result, net, ids):
    exchangers = result.get(SubHE, index="id")
    states = {}
    for unit in FEED_INS:
        model_type = UNIT_MODELS[unit]
        node = next(n.id for n in net.nodes if isinstance(n.model, model_type))
        branch = next(
            b.id
            for b in net.branches
            if isinstance(b.model, SubHE) and node in (b.from_node_id, b.to_node_id)
        )
        row = exchangers.loc[branch]
        states[unit] = Exchanger(
            heat_mw=-float(row["q_mw"]),
            flow_kgs=-float(row["mass_flow_kgs"]),
            spread_k=float(row["t_out_pu"] - row["t_in_pu"]) * ids.water_t_ref_k,
            node_k=float(result.get(model_type, index="id").loc[node, "t_k"]),
        )
    return states


def soc_residual(result, net):
    """Largest gap of the power flow cone w l = p^2 + q^2 over the lines, in
    per unit, or None for a formulation without the lifted current l."""
    lines = result.get(mm.PowerLine, index="id")
    if "current_pu_squared" not in lines or lines["current_pu_squared"].isna().any():
        return None
    buses = result.get(mm.Bus, index="id")
    worst = 0.0
    for line_id, line in lines.iterrows():
        sn_mva = net.branch_by_id(line_id).grid.sn_mva
        w = buses.loc[line_id[0], "vm_pu_squared"] / line["tap"] ** 2
        p = line["p_series_from_mw"] / sn_mva
        q = line["q_series_from_mvar"] / sn_mva
        worst = max(worst, abs(w * line["current_pu_squared"] - p * p - q * q))
    return float(worst)


def describe_hour(t, result, network, limits, step):
    """An Hour from one step of solve_hours: result is None if it failed,
    network is what nl.build_network returned and step the step's seconds and
    monee's warnings."""
    net, ids, site = network
    seconds, messages = step
    common = {
        "t": t,
        "price": ed.EL_PRICE[t],
        "expected": nl.expected_dispatch(limits, ed.EL_PRICE[t], ids.gas_mw_per_kgs),
        # monee reports its automatic IPOPT retry only in its log.
        "retried": any("retrying automatically" in m for m in messages),
        "seconds": seconds,
    }
    if result is None:
        return Hour(**common)
    h = nl.quantities([result], ids, site, hours=[t])[0]
    return Hour(
        **common,
        status=str(result.termination_condition),
        dispatch={unit: h[unit] for unit in nl.UNITS},
        cost=float(ed.hourly_cost(h)),
        objective=result.user_objective,
        line_loss_mw=h["line_loss_mw"],
        pipe_drop=h["psq"][1] - h["psq"][2],
        weymouth_drop=limits.k_pipe * h["pipe_kgs"] * abs(h["pipe_kgs"]),
        soc_residual=soc_residual(result, net),
        exchangers=feed_in_exchangers(result, net, ids),
        quantities=h,
    )


def solve_hours(settings, hours, pin=None, **solver):
    """Solve the given hours of the day on nl.build_network(**settings), each
    hour on its own as in mes_network_limits.py. pin fixes the named units'
    operating points. A failed hour is kept as data, not raised."""
    network = nl.build_network(**settings)
    net, ids, site = network
    for unit, x in (pin or {}).items():
        pin_unit(net, unit, x)
    td = TimeseriesData()
    td.add_objective_data(ids.grid, "cost", [ed.EL_PRICE[t] for t in hours])
    td.add_child_series(ids.pv, "regulation", [ed.PV_AVAILABILITY[t] for t in hours])
    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=ed.EL_PRICE[0],
        gas_cost_default=ed.GAS_PRICE,
        heat_cost_default=ed.BOILER_PRICE,
    )
    log = StepLog()
    logging.getLogger("monee").addHandler(log)
    try:
        result = run_timeseries(
            net,
            td,
            steps=len(hours),
            optimization_problem=problem,
            on_step_error="skip",
            progress_callback=log.step_done,
            **solver,
        )
    finally:
        logging.getLogger("monee").removeHandler(log)
    limits = hand_limits(net, site, settings)
    return [
        describe_hour(t, step.result, network, limits, timing)
        for t, step, timing in zip(hours, result.step_results, log.steps)
    ]


def pinned_dispatch(hour):
    """Every unit at the operating point of hour's dispatch, one that is off
    at 1e-6: at zero flow its feed-in exchanger would sit in the corner,
    where IPOPT can fail, and 1e-6 of a unit's output moves an hour's cost
    by far less than 0.001."""
    return {unit: max(x, 1e-6) for unit, x in hour.dispatch.items()}


def solve_at_dispatch(settings, hour):
    """IPOPT on the default NLP formulation at another solver's dispatch."""
    return solve_hours(settings, [hour.t], pin=pinned_dispatch(hour))[0]


def solve_day():
    """The network limits day by each approach, and its CHP hours with the
    CHP pinned off, keyed by hour."""
    hours = range(len(ed.EL_PRICE))
    runs = {
        "APOPT": solve_hours(DAY_LIMITS, hours, **APOPT),
        "SCIP": solve_hours(DAY_LIMITS, hours, **EXACT),
        "relaxation": solve_hours(DAY_LIMITS, hours, **RELAXATION),
        "IPOPT": solve_hours(DAY_LIMITS, hours),
        "CHP off": solve_hours(DAY_LIMITS, CHP_HOURS, pin={"chp": 0.0}, **EXACT),
    }
    for solver in ("SCIP", "APOPT"):
        runs[f"{solver} on NLP"] = [
            solve_at_dispatch(DAY_LIMITS, h) for h in runs[solver]
        ]
    return {label: {hour.t: hour for hour in run} for label, run in runs.items()}


def sweep_settings(cable_mva):
    return {
        "cable_mva": cable_mva,
        "pipe_diameter_m": SWEEP_PIPE_DIAMETER_M,
        "floor": SWEEP_FLOOR,
    }


def solve_sweep():
    """Hours 4 and 5 at each cable rating of the sweep by each approach,
    keyed by (cable rating, hour)."""
    sweep = {}
    for cable_mva in SWEEP_CABLE_MVA:
        settings = sweep_settings(cable_mva)
        runs = {
            "SCIP": solve_hours(settings, SWEEP_HOURS, **EXACT),
            "relaxation": solve_hours(settings, SWEEP_HOURS, **RELAXATION),
            "e-boiler off": solve_hours(
                settings, SWEEP_HOURS, pin={"e_boiler": 0.0}, **EXACT
            ),
            "IPOPT": solve_hours(settings, SWEEP_HOURS),
        }
        runs["SCIP on NLP"] = [solve_at_dispatch(settings, h) for h in runs["SCIP"]]
        for label, run in runs.items():
            sweep.setdefault(label, {}).update({(cable_mva, h.t): h for h in run})
    return sweep


def added_line_losses(hour, optimum):
    """The line losses hour's dispatch adds to the optimum's, at the hour's
    power price."""
    return hour.price * (hour.line_loss_mw - optimum.line_loss_mw)


def apopt_excess(results):
    """What APOPT's day costs more than SCIP's, as solved and by hand: the
    CHP's margins at its hand operating point in the hours it should run in,
    plus the line losses APOPT's dispatch adds."""
    apopt, scip = results.day["APOPT"], results.day["SCIP"]
    solved = sum(apopt[t].cost - scip[t].objective for t in scip)
    margins = sum(
        ed.unit_margins(scip[t].price)["chp"] * scip[t].expected["chp"]
        for t in CHP_HOURS
    )
    losses = sum(added_line_losses(apopt[t], scip[t]) for t in scip)
    return solved, margins, losses


def chp_cost_curve(results):
    """Hour 0's cost with the CHP pinned at each of CHP_PINS and at SCIP's
    optimum, as (operating point, cost) pairs."""
    optimum = results.day["SCIP"][0]
    points = [(x, hour.objective) for x, hour in results.chp_pinned.items()]
    return points + [(optimum.dispatch["chp"], optimum.objective)]


def slopes(points):
    return [(c1 - c0) / (x1 - x0) for (x0, c0), (x1, c1) in zip(points, points[1:])]


def seconds_per_solve(runs):
    """Mean wall time of one hourly solve of each approach in Results.day or
    Results.sweep."""
    return {
        label: statistics.fmean(hour.seconds for hour in hours.values())
        for label, hours in runs.items()
    }


def check_status(checks, label, hour):
    checks.expect(hour.status == "optimal", f"{label}: SCIP ended {hour.status}")


def check_operating_point(checks, label, hour):
    """The balance, conversion and limit checks of mes_economic_dispatch.py,
    one check each so that the count does not depend on the dispatch."""
    for check in (ed.check_balances, ed.check_conversions, ed.check_limits):
        found = ed.Checks()
        if hour.failed:
            found.failed.append("no result")
        else:
            check(found, hour.quantities, hour.t)
        checks.expect(not found.failed, f"{label}: {'; '.join(found.failed)}")


def check_optimum(checks, label, hour):
    """SCIP's exact solve: optimal, at the hand dispatch, a valid operating
    point that costs its objective, and on the Weymouth curve."""
    check_status(checks, label, hour)
    for unit in nl.UNITS:
        checks.expect(
            abs(hour.dispatch[unit] - hour.expected[unit]) <= 1e-3,
            f"{label}: {unit} {hour.dispatch[unit]:.4f} but by hand "
            f"{hour.expected[unit]:.4f}",
        )
    check_operating_point(checks, label, hour)
    checks.expect(
        abs(hour.objective - hour.cost) <= 1e-6 * abs(hour.cost),
        f"{label}: objective {hour.objective:.4f}, recomputed cost {hour.cost:.4f}",
    )
    checks.expect(
        abs(hour.pipe_drop - hour.weymouth_drop)
        <= 1e-3 * abs(hour.weymouth_drop) + 1e-7,
        f"{label}: pipe drop {hour.pipe_drop:.6f}, Weymouth {hour.weymouth_drop:.6f}",
    )


def check_bound(checks, label, relaxation, optimum):
    """The relaxation: optimal, at or below the optimum and within 0.01 of
    it, with a tight power flow cone."""
    check_status(checks, f"{label}, relaxation", relaxation)
    checks.expect(
        -1e-3 <= optimum.objective - relaxation.objective <= 1e-2,
        f"{label}: relaxation {relaxation.objective:.4f}, optimum "
        f"{optimum.objective:.4f}",
    )
    checks.expect(
        relaxation.soc_residual is not None and relaxation.soc_residual <= 1e-6,
        f"{label}: the relaxation's power flow cone is off by "
        f"{relaxation.soc_residual}",
    )


def check_apopt(checks, label, apopt, optimum):
    """APOPT's point: a valid operating point, not cheaper than the optimum."""
    check_operating_point(checks, f"{label}, APOPT", apopt)
    checks.expect(
        apopt.cost >= optimum.objective - 1e-3,
        f"{label}: APOPT costs {apopt.cost:.4f}, less than the optimum "
        f"{optimum.objective:.4f}",
    )


def check_apopt_stop(checks, results):
    """APOPT's point is the best dispatch without the CHP, and costs the
    CHP's margins plus the line losses more than the optimum."""
    day = results.day
    for t, off in day["CHP off"].items():
        check_status(checks, f"day, hour {t}, CHP off", off)
        checks.expect(
            abs(day["APOPT"][t].cost - off.objective) <= 1e-3,
            f"day, hour {t}: APOPT costs {day['APOPT'][t].cost:.4f}, the best "
            f"dispatch without the CHP {off.objective:.4f}",
        )
    solved, margins, losses = apopt_excess(results)
    checks.expect(
        abs(solved - margins - losses) <= 1e-2,
        f"day: APOPT costs {solved:.4f} more, the CHP's margins and the line "
        f"losses {margins + losses:.4f}",
    )


def check_chp_curve(checks, results):
    """Hour 0's cost falls, convex, all the way to the CHP's optimum."""
    for x, hour in results.chp_pinned.items():
        check_status(checks, f"hour 0, CHP pinned at {x:g}", hour)
    rates = slopes(chp_cost_curve(results))
    checks.expect(
        all(rate < 0 for rate in rates)
        and all(b >= a - 1e-3 for a, b in zip(rates, rates[1:])),
        f"hour 0: the cost along the CHP's operating point changes by {rates}",
    )


def check_e_boiler_off(checks, label, off, optimum):
    """Pinning the e-boiler off costs the margins its units give up plus the
    line losses it adds, the congestion identity of mes_network_limits.py."""
    check_status(checks, f"{label}, e-boiler off", off)
    by_hand = (
        math.nan
        if off.failed
        else nl.lost_margin(off.quantities, off.expected)
        + added_line_losses(off, optimum)
    )
    extra = off.objective - optimum.objective
    checks.expect(
        abs(extra - by_hand) <= 1e-3,
        f"{label}: the e-boiler off costs {extra:.4f} more, its lost margins "
        f"and line losses {by_hand:.4f}",
    )


def check_ipopt(checks, label, ipopt, optimum):
    """What holds for IPOPT on every build: no result undercuts the global
    optimum, and one off the hand dispatch costs more than it."""
    checks.expect(
        ipopt.failed
        or (
            ipopt.cost >= optimum.objective - 1e-3
            and (ipopt.deviation <= 1e-3 or ipopt.cost > optimum.objective + 1e-3)
        ),
        f"{label}: IPOPT {ipopt.status} at {ipopt.cost:.4f}, off the hand dispatch "
        f"by {ipopt.deviation:.4f}, optimum {optimum.objective:.4f}",
    )


def confirms(nlp, hour):
    """Whether IPOPT at hour's dispatch solved at that dispatch and its cost."""
    return (
        not nlp.failed
        and abs(nlp.cost - hour.cost) <= 1e-3
        and all(abs(nlp.dispatch[u] - hour.dispatch[u]) <= 1e-3 for u in nl.UNITS)
    )


def check_nlp(checks, label, nlp, hour, settings):
    """IPOPT at hour's dispatch: at that dispatch and its cost, a valid
    operating point, and within the pipe's floor and the cable's rating."""
    checks.expect(
        confirms(nlp, hour),
        f"{label}: IPOPT at the dispatch ended {nlp.status} at {nlp.cost:.4f}, "
        f"the solver's own cost {hour.cost:.4f}",
    )
    check_operating_point(checks, f"{label}, NLP", nlp)
    q = nlp.quantities
    checks.expect(
        not nlp.failed
        and min(q["psq"]) >= settings["floor"] - 1e-6
        and q["cable_mva"] <= settings["cable_mva"] + 1e-6,
        f"{label}: IPOPT at the dispatch breaks the pipe's floor or the cable's rating",
    )


def sweep_label(key):
    return f"{key[0]:.2f} MVA, hour {key[1]}"


def check_scip_and_apopt(checks, results):
    """Every check that does not depend on IPOPT, and so not on the CasADi
    build."""
    day = results.day
    for t, optimum in day["SCIP"].items():
        label = f"day, hour {t}"
        check_optimum(checks, label, optimum)
        check_bound(checks, label, day["relaxation"][t], optimum)
        check_apopt(checks, label, day["APOPT"][t], optimum)
    check_apopt_stop(checks, results)
    check_chp_curve(checks, results)
    sweep = results.sweep
    for key, optimum in sweep["SCIP"].items():
        label = sweep_label(key)
        check_optimum(checks, label, optimum)
        check_bound(checks, label, sweep["relaxation"][key], optimum)
        check_e_boiler_off(checks, label, sweep["e-boiler off"][key], optimum)


def check_ipopt_results(checks, results):
    day = results.day
    for solver in ("SCIP", "APOPT"):
        for t, hour in day[solver].items():
            nlp = day[f"{solver} on NLP"][t]
            check_nlp(checks, f"day, hour {t}, {solver}", nlp, hour, DAY_LIMITS)
    for key, optimum in results.sweep["SCIP"].items():
        label = sweep_label(key)
        nlp = results.sweep["SCIP on NLP"][key]
        check_nlp(checks, label, nlp, optimum, sweep_settings(key[0]))
        check_ipopt(checks, label, results.sweep["IPOPT"][key], optimum)


def run_checks(results):
    checks = ed.Checks()
    check_scip_and_apopt(checks, results)
    check_ipopt_results(checks, results)
    return checks


def listing(items):
    words = [str(item) for item in items]
    return (
        ", ".join(words[:-1]) + " and " + words[-1]
        if len(words) > 1
        else "".join(words)
    )


def print_day(results):
    apopt, scip, bound = (results.day[k] for k in ("APOPT", "SCIP", "relaxation"))
    print("The network limits day by APOPT and by SCIP (1 = unit at full output)")
    print(
        f"{'':11} {'SCIP dispatch':^20} {'APOPT':>7} {'cost of the hour':^27}".rstrip()
    )
    print(
        f"{'hour':>4} {'price':>6} {'chp':>6} {'e-boil':>6} {'elyser':>6} "
        f"{'chp':>7} {'APOPT':>9} {'SCIP':>8} {'bound':>8}"
    )
    for t, e in scip.items():
        print(
            f"{t:>4} {e.price:>6.1f} {e.dispatch['chp']:>6.3f} "
            f"{e.dispatch['e_boiler']:>6.3f} {e.dispatch['electrolyser']:>6.3f} "
            f"{apopt[t].dispatch['chp']:>7.3f} {apopt[t].cost:>9.2f} "
            f"{e.objective:>8.2f} {bound[t].objective:>8.2f}"
        )
    print(
        f"{'day':<40} {sum(h.cost for h in apopt.values()):>9.2f} "
        f"{sum(h.objective for h in scip.values()):>8.2f} "
        f"{sum(h.objective for h in bound.values()):>8.2f}"
    )

    solved, margins, losses = apopt_excess(results)
    print(
        f"\nAPOPT costs {solved:.2f} more: {margins:.2f} of CHP margins in hours "
        f"{listing(CHP_HOURS)}\nand {losses:.2f} of added line losses."
    )
    off = listing(f"{h.objective:.2f}" for h in results.day["CHP off"].values())
    print(f"SCIP with the CHP pinned off in these hours: {off},")
    print("exactly APOPT's costs.")

    points = chp_cost_curve(results)
    print(
        "\nHour 0 by SCIP with the CHP pinned: no second valley on the way to the optimum"
    )
    print(f"{'chp':<8}" + "".join(f"{x:>9.3f}" for x, _ in points))
    print(f"{'cost':<8}" + "".join(f"{c:>9.2f}" for _, c in points))
    day = results.day
    scip, apopt = (confirmed(day[s], day[f"{s} on NLP"]) for s in ("SCIP", "APOPT"))
    print(
        "\nIPOPT on the NLP formulation at each solver's dispatch, cost within 0.001"
        f"\nof the solver's own: SCIP in {scip} of {len(day['SCIP'])} hours, "
        f"APOPT in {apopt} of {len(day['APOPT'])}."
    )


def confirmed(hours, nlp):
    return sum(confirms(nlp[key], hour) for key, hour in hours.items())


def print_sweep(results):
    print(
        f"Hours {listing(SWEEP_HOURS)} with the {SWEEP_PIPE_DIAMETER_M * 1000:g} mm "
        f"pipe at a floor of {SWEEP_FLOOR:g} (1 = unit at full output)"
    )
    print(f"{'':19} {'SCIP dispatch':^15} {'cost of the hour':^31}".rstrip())
    print(
        f"{'cable':>6} {'hour':>5} {'price':>6} {'e-boil':>7} {'elyser':>7} "
        f"{'SCIP':>8} {'bound':>8} {'e-boiler off':>13}"
    )
    sweep = results.sweep
    for key, e in sweep["SCIP"].items():
        print(
            f"{key[0]:>6.2f} {key[1]:>5} {e.price:>6.1f} "
            f"{e.dispatch['e_boiler']:>7.3f} {e.dispatch['electrolyser']:>7.3f} "
            f"{e.objective:>8.2f} {sweep['relaxation'][key].objective:>8.2f} "
            f"{sweep['e-boiler off'][key].objective:>13.2f}"
        )
    print(
        "\nIPOPT on the NLP formulation at SCIP's dispatch, cost within 0.001 of"
        f"\nSCIP's own: {confirmed(sweep['SCIP'], sweep['SCIP on NLP'])} of "
        f"{len(sweep['SCIP'])} hours of the sweep."
    )


def print_report(results, checks):
    print_day(results)
    print()
    print_sweep(results)
    print(
        f"\nPlausibility checks: {checks.count - len(checks.failed)} of {checks.count} passed"
    )
    for label in checks.failed:
        print(f"  FAILED {label}")


def ipopt_row(key, h, optimum):
    line = f"{key[0]:>6.2f} {key[1]:>5}  {h.status:<27}"
    if h.failed:
        return line
    dispatch = "/".join(f"{h.dispatch[u]:.3f}" for u in nl.UNITS)
    e_boiler = h.exchangers["e_boiler"]
    return line + (
        f" {dispatch:<17} {h.cost:>9.4f} {h.cost - optimum.objective:>7.3f} "
        f"{'yes' if h.retried else 'no':>5}  {e_boiler.spread_k:>17.2f} "
        f"{e_boiler.node_k:>7.2f}"
    )


def print_ipopt(results):
    day, sweep = results.day["IPOPT"], results.sweep
    off = [t for t, h in day.items() if not h.deviation <= 1e-3]
    print("IPOPT as monee calls it by default (depends on the CasADi build)")
    print(
        f"network limits day: {sum(h.cost for h in day.values()):.4f}, hours off "
        f"the hand dispatch: {listing(off) or 'none'}, retries: "
        f"{sum(h.retried for h in day.values())}"
    )
    print(
        f"{'cable':>6} {'hour':>5}  {'status':<27} {'chp/e-boil/elyser':<17} "
        f"{'cost':>9} {'excess':>7} {'retry':>5}  {'e-boiler spread K':>17} {'node K':>7}"
    )
    for key, h in sweep["IPOPT"].items():
        print(ipopt_row(key, h, sweep["SCIP"][key]))
    print_ipopt_summary(results)


def ipopt_stops(sweep, t):
    hours = [h for (_, hour_t), h in sweep["IPOPT"].items() if hour_t == t]
    missed = [h for h in hours if not h.deviation <= 1e-3]
    silent = [h for h in missed if h.status == "Solve_Succeeded" and not h.retried]
    return (
        f"hour {t}: off the optimum at {len(missed)} of {len(hours)} ratings, "
        f"{len(silent)} of them Solve_Succeeded without a retry; retries: "
        f"{sum(h.retried for h in hours)}"
    )


def print_ipopt_summary(results):
    sweep = results.sweep
    for t in SWEEP_HOURS:
        print(ipopt_stops(sweep, t))
    for part, runs in (("day", results.day), ("sweep", sweep)):
        print(
            f"seconds per hourly solve, {part}: "
            + ", ".join(
                f"{label} {s:.2f}" for label, s in seconds_per_solve(runs).items()
            )
        )


def run():
    day = solve_day()
    chp_pinned = {
        x: solve_hours(DAY_LIMITS, [0], pin={"chp": x}, **EXACT)[0] for x in CHP_PINS
    }
    results = Results(day, chp_pinned, solve_sweep())
    return results, run_checks(results)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    results, checks = run()
    print_report(results, checks)
    print()
    print_ipopt(results)
    sys.exit(1 if checks.failed else 0)
