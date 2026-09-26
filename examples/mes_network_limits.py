"""Economic dispatch of the multi-energy town with binding network limits.

This extends examples/mes_economic_dispatch.py: the same town, the same three
coupling units and the same eight hourly prices. There no network limit
binds, so the dispatch is the merit order up to small losses. Here the plant
site, the bus and the gas junction where the CHP, the electric boiler and
the electrolyser sit, hangs on two weaker links:

* the cable from the town bus is rated at 1.5 MVA, while the electric
  boiler and the electrolyser draw 2.05 MW in the two cheapest hours;
* the gas pipe to the plant junction is 30 mm instead of 150 mm, too thin
  to bring the CHP's full fuel without the pressure at its end falling
  below the gas grid's floor of 0.7 in squared per unit.

Both limits bind, in different hours, and the dispatch leaves the merit
order: the scarce capacity goes to the unit that earns most per MW of it,
and one unit runs at part load. Nothing links one hour to the next, so the
day is still eight independent dispatches solved with ``run_timeseries``.

The script solves the day with both limits, with each limit alone, without
either (the economic dispatch tutorial's day), and with each limit relaxed
by one small step. It then checks each result: the balances, conversions,
limits and supply temperature as in mes_economic_dispatch.py, every
operating point against a hand calculation of the scarce capacity, the
pressure drop of the gas pipe against the Weymouth equation, that each limit
binds exactly where the day without it would overload it, that each limit's
congestion cost equals the margins it takes away, and that one more MW
through each bottleneck is worth the margin of the unit it lets run. It
exits with status 1 if a check fails.

Run::

    python examples/mes_network_limits.py
"""

from __future__ import annotations

import math
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import monee.model as mm
from monee import TimeseriesData, run_timeseries
from monee.problem import create_economic_dispatch_problem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mes_economic_dispatch as ed  # noqa: E402

CABLE_MVA = 1.5
PIPE_DIAMETER_M = 0.03
PRESSURE_FLOOR = 0.7
CABLE_STEP_MVA = 0.1
FLOOR_STEP = 0.02

UNITS = ("chp", "e_boiler", "electrolyser")
E_BOILER_EL_MW = ed.E_BOILER_HEAT_MW / ed.E_BOILER_EFF
ELECTROLYSER_EL_MW = ed.ELECTROLYSER_GAS_MW / ed.ELECTROLYSER_EFF
CABLE_RELAXED = f"cable at {CABLE_MVA + CABLE_STEP_MVA:g} MVA"
FLOOR_RELAXED = f"floor at {PRESSURE_FLOOR - FLOOR_STEP:g}"
# None keeps the economic dispatch tutorial's value: an unrated cable, whose
# default current rating of 3.19 kA is about 110 MVA, and a 150 mm pipe.
DAYS = {
    "both limits": {},
    "cable only": {"pipe_diameter_m": None},
    "gas pipe only": {"cable_mva": None},
    "neither": {"cable_mva": None, "pipe_diameter_m": None},
    CABLE_RELAXED: {"cable_mva": CABLE_MVA + CABLE_STEP_MVA},
    FLOOR_RELAXED: {"floor": PRESSURE_FLOOR - FLOOR_STEP},
}
LIMITS = {
    "cable": ("cable only", CABLE_RELAXED),
    "gas pipe": ("gas pipe only", FLOOR_RELAXED),
}


@dataclass
class Site:
    cable: tuple
    main_pipe: tuple
    pipe: tuple
    junctions: list


@dataclass
class Day:
    hours: list
    cost: float
    cable_mva: float | None
    cable_mw: float | None
    floor: float
    pipe_capacity_kgs: float
    k_pipe: float


def build_network(
    cable_mva=CABLE_MVA, pipe_diameter_m=PIPE_DIAMETER_M, floor=PRESSURE_FLOOR
):
    net, ids = ed.build_network()
    town_bus = net.child_by_id(ids.pv).node_id
    plant_junction = net.child_by_id(ids.biogas).node_id
    cable = next(
        b
        for b in net.branches
        if isinstance(b.model, mm.PowerLine) and b.from_node_id == town_bus
    )
    pipes = [b for b in net.branches if isinstance(b.model, mm.GasPipe)]
    pipe = next(b for b in pipes if b.to_node_id == plant_junction)
    main = next(b for b in pipes if b is not pipe)
    cable.model.max_s_mva = cable_mva
    if pipe_diameter_m is not None:
        pipe.model.diameter_m = pipe_diameter_m
    pipe.grid.pressure_squared_pu_min = floor
    junctions = [main.from_node_id, main.to_node_id, plant_junction]
    return net, ids, Site(cable.id, main.id, pipe.id, junctions)


def solve_day(net, ids):
    td = TimeseriesData()
    td.add_objective_data(ids.grid, "cost", ed.EL_PRICE)
    td.add_child_series(ids.pv, "regulation", ed.PV_AVAILABILITY)
    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=ed.EL_PRICE[0],
        gas_cost_default=ed.GAS_PRICE,
        heat_cost_default=ed.BOILER_PRICE,
    )
    # No solver options: the adaptive mu_strategy of the economic dispatch
    # tutorial fails once the gas pipe is thin, and setting any mu_strategy
    # switches off monee's automatic retry.
    result = run_timeseries(
        net, td, steps=len(ed.EL_PRICE), optimization_problem=problem
    )
    return [step.result for step in result.step_results]


def weymouth_k(net, pipe_id):
    """Drop of the squared pressure per (kg/s)^2 of a gas pipe,
    p_i^2 - p_j^2 = K m |m| in per unit, with the constant friction of a
    fully rough pipe that the default gas formulation uses."""
    branch = net.branch_by_id(pipe_id)
    grid, pipe = branch.grid, branch.model
    friction = 0.25 / math.log10(pipe.roughness_m / (3.7 * pipe.diameter_m)) ** 2
    c_squared = (math.pi**2 * pipe.diameter_m**5) / (
        16
        * pipe.length_m
        * (grid.universal_gas_constant / grid.molar_mass)
        * grid.t_k
        * grid.compressibility
    )
    return friction / (c_squared * grid.pressure_ref_pa**2)


def pipe_capacity_kgs(net, site, floor):
    """Largest flow to the plant junction with its pressure at the floor. The
    main pipe carries that flow plus the town's gas demand from the grid
    connection, held at a squared pressure of 1, so the two drops add up:
    K_main (m + d)^2 + K m^2 = 1 - floor."""
    k_main, k = weymouth_k(net, site.main_pipe), weymouth_k(net, site.pipe)
    d = ed.GAS_DEMAND_KGS
    a, b, c = k_main + k, 2 * k_main * d, k_main * d * d - (1 - floor)
    return (-b + math.sqrt(b * b - 4 * a * c)) / (2 * a)


def cable_capacity_mw(net, site, cable_mva):
    """Power the cable delivers to the plant bus at its rating: the rating
    less the series loss r S^2 / U^2, at the 20 kV of the town bus and with
    next to no reactive power on the cable."""
    if cable_mva is None:
        return None
    branch = net.branch_by_id(site.cable)
    r_ohm = branch.model.r_ohm_per_m * branch.model.length_m
    u_kv = net.node_by_id(branch.from_node_id).model.base_kv
    return cable_mva - r_ohm * cable_mva**2 / u_kv**2


def quantities(hour_results, ids, site, hours=None):
    rows = ed.hourly_quantities(hour_results, ids, hours)
    for h, r in zip(rows, hour_results):
        cable = r.get(mm.PowerLine, index="id").loc[site.cable]
        junctions = r.get(mm.Junction, index="id")
        h["cable_mva"] = max(
            math.hypot(cable["p_from_mw"], cable["q_from_mvar"]),
            math.hypot(cable["p_to_mw"], cable["q_to_mvar"]),
        )
        h["cable_in_mw"] = -float(cable["p_to_mw"])
        # mass_flow_kgs is negative for a flow from the from to the to node.
        h["pipe_kgs"] = -float(
            r.get(mm.GasPipe, index="id").loc[site.pipe, "mass_flow_kgs"]
        )
        h["psq"] = [
            float(junctions.loc[j, "pressure_squared_pu"]) for j in site.junctions
        ]
    return rows


def solve(changes, solver=solve_day):
    settings = {
        "cable_mva": CABLE_MVA,
        "pipe_diameter_m": PIPE_DIAMETER_M,
        "floor": PRESSURE_FLOOR,
        **changes,
    }
    net, ids, site = build_network(**settings)
    results = solver(net, ids)
    return Day(
        quantities(results, ids, site),
        ed.day_cost(results),
        settings["cable_mva"],
        cable_capacity_mw(net, site, settings["cable_mva"]),
        settings["floor"],
        pipe_capacity_kgs(net, site, settings["floor"]),
        weymouth_k(net, site.pipe),
    )


def expected_dispatch(day, price, k):
    """The merit order of mes_economic_dispatch.py, cut back where a limit
    binds. The gas pipe caps the CHP's fuel at the pipe's capacity plus the
    biogas that enters at the plant junction. The cable serves the electric
    boiler first: it earns 38 - price per MW of power, 15.25 more than the
    electrolyser's 22.75 - price. The CHP's 1.05 MW never fills the cable."""
    chp = 1.0 if price > ed.CHP_ABOVE else 0.0
    e_boiler = 1.0 if price < ed.E_BOILER_BELOW else 0.0
    electrolyser = 1.0 if price < ed.ELECTROLYSER_BELOW else 0.0
    chp_fuel_kgs = ed.CHP_FUEL_MW / k
    chp = min(chp, (day.pipe_capacity_kgs + ed.BIOGAS_KGS) / chp_fuel_kgs)
    if day.cable_mw is not None:
        e_boiler = min(e_boiler, day.cable_mw / E_BOILER_EL_MW)
        spare_mw = day.cable_mw - e_boiler * E_BOILER_EL_MW
        electrolyser = min(electrolyser, max(spare_mw, 0.0) / ELECTROLYSER_EL_MW)
    return {"chp": chp, "e_boiler": e_boiler, "electrolyser": electrolyser}


def binds(limit, day, h):
    if limit == "cable":
        return day.cable_mva is not None and h["cable_mva"] >= day.cable_mva - 1e-4
    return h["psq"][-1] <= day.floor + 1e-4


def overloads(limit, h, limited):
    """Whether hour h of the day without limits carries more than the
    limited day's cable or pipe could."""
    if limit == "cable":
        return (
            limited.cable_mva is not None and h["cable_mva"] > limited.cable_mva + 1e-4
        )
    return h["pipe_kgs"] > limited.pipe_capacity_kgs + 1e-6


def binding_limits(day, h):
    return [limit for limit in LIMITS if binds(limit, day, h)]


def bottleneck_mw(limit, h):
    """What flows through the limit's bottleneck in MW: power arriving at the
    plant bus, or gas arriving at the plant junction."""
    return h["cable_in_mw"] if limit == "cable" else h["pipe_kgs"] * h["k"]


def margin_per_mw(limit, price):
    """Margin per MW of the scarce flow of the unit that runs at part load:
    the electrolyser per MW of power, the CHP per MW of fuel."""
    if limit == "cable":
        return ed.ELECTROLYSER_BELOW - price
    return ed.CHP_EFF_EL * price + ed.CHP_EFF_HEAT * ed.BOILER_PRICE - ed.GAS_PRICE


def lost_margin(h, unlimited):
    margins = ed.unit_margins(h["price"])
    return sum(margins[u] * (unlimited[u] - h[u]) for u in UNITS)


def congestion(days, label):
    """Per hour: what the day costs more than the day without limits, and
    the margins its units lose against that day."""
    return [
        (h["objective"] - n["objective"], lost_margin(h, n))
        for h, n in zip(days[label].hours, days["neither"].hours)
    ]


def marginal_values(days):
    """Per hour a limit binds in: the saving of the relaxed day per MW more
    through the bottleneck, and the margin per MW it should equal."""
    both = days["both limits"]
    values = {}
    for limit, (_, relaxed) in LIMITS.items():
        for t, (h, r) in enumerate(zip(both.hours, days[relaxed].hours)):
            if binds(limit, both, h):
                saving = h["objective"] - r["objective"]
                more_mw = bottleneck_mw(limit, r) - bottleneck_mw(limit, h)
                values[t] = (limit, saving / more_mw, margin_per_mw(limit, h["price"]))
    return dict(sorted(values.items()))


def check_hours(checks, label, day):
    for t, h in enumerate(day.hours):
        ed.check_balances(checks, h, t)
        ed.check_conversions(checks, h, t)
        ed.check_limits(checks, h, t)
        by_hand = ed.hourly_cost(h)
        checks.expect(
            abs(by_hand - h["objective"]) <= 1e-6 * abs(h["objective"]),
            f"{label}, hour {t}: objective {h['objective']:.4f} differs from "
            f"recomputed cost {by_hand:.4f}",
        )
        expected = expected_dispatch(day, h["price"], h["k"])
        for unit in UNITS:
            checks.expect(
                abs(h[unit] - expected[unit]) <= 1e-3,
                f"{label}, hour {t}: {unit} {h[unit]:.4f} but by hand "
                f"{expected[unit]:.4f}",
            )


def check_network(checks, label, day):
    for t, h in enumerate(day.hours):
        if day.cable_mva is not None:
            checks.expect(
                h["cable_mva"] <= day.cable_mva + 1e-6,
                f"{label}, hour {t}: cable carries {h['cable_mva']:.4f} MVA",
            )
        checks.expect(
            min(h["psq"]) >= day.floor - 1e-6,
            f"{label}, hour {t}: gas pressure below the floor",
        )
        drop = h["psq"][1] - h["psq"][2]
        weymouth = day.k_pipe * h["pipe_kgs"] * abs(h["pipe_kgs"])
        checks.expect(
            abs(drop - weymouth) <= 1e-3 * abs(weymouth) + 1e-7,
            f"{label}, hour {t}: pipe drop {drop:.6f}, Weymouth {weymouth:.6f}",
        )
        if binds("gas pipe", day, h):
            checks.expect(
                abs(h["pipe_kgs"] - day.pipe_capacity_kgs) <= 1e-5,
                f"{label}, hour {t}: pipe carries {h['pipe_kgs']:.6f} kg/s at the "
                f"floor, capacity {day.pipe_capacity_kgs:.6f}",
            )


def check_active_sets(checks, days):
    neither = days["neither"]
    for label, day in days.items():
        if label == "neither":
            continue
        for limit in LIMITS:
            binding = [t for t, h in enumerate(day.hours) if binds(limit, day, h)]
            overloaded = [
                t for t, h in enumerate(neither.hours) if overloads(limit, h, day)
            ]
            checks.expect(
                binding == overloaded,
                f"{label}: {limit} binds in hours {binding}, the day without "
                f"limits overloads it in {overloaded}",
            )


def check_congestion(checks, days):
    neither = days["neither"]
    for label in ("both limits", *(alone for alone, _ in LIMITS.values())):
        day = days[label]
        for t, ((extra, lost), h, n) in enumerate(
            zip(congestion(days, label), day.hours, neither.hours)
        ):
            losses = h["price"] * (h["line_loss_mw"] - n["line_loss_mw"])
            checks.expect(
                abs(extra - lost - losses) <= 1e-3,
                f"{label}, hour {t}: costs {extra:.4f} more, the margins lost "
                f"and the line losses {lost + losses:.4f}",
            )
            if binding_limits(day, h):
                checks.expect(extra > 1e-2, f"{label}, hour {t}: binds for free")
            else:
                checks.expect(
                    abs(extra) <= 1e-3,
                    f"{label}, hour {t}: costs {extra:.4f} without a binding limit",
                )
    both = congestion(days, "both limits")
    split = zip(*(congestion(days, alone) for alone, _ in LIMITS.values()))
    for t, ((extra, _), parts) in enumerate(zip(both, split)):
        # Four solves enter here, each with up to about 5e-4 of solver noise.
        checks.expect(
            abs(extra - sum(part for part, _ in parts)) <= 2e-3,
            f"hour {t}: the limits' congestion costs do not add up",
        )


def check_marginal_values(checks, days):
    for t, (limit, value, by_hand) in marginal_values(days).items():
        # A MW more through the bottleneck also changes the line losses,
        # which shift its value by a fraction of the power price.
        checks.expect(
            abs(value - by_hand) <= 0.01 * ed.EL_PRICE[t],
            f"hour {t}: one more MW through the {limit} saves {value:.3f}, "
            f"its margin is {by_hand:.3f}",
        )


def check_order(checks, days):
    for tighter, looser in (
        ("both limits", CABLE_RELAXED),
        ("both limits", FLOOR_RELAXED),
        ("both limits", "cable only"),
        ("both limits", "gas pipe only"),
        ("cable only", "neither"),
        ("gas pipe only", "neither"),
        (CABLE_RELAXED, "neither"),
        (FLOOR_RELAXED, "neither"),
    ):
        for t, (a, b) in enumerate(zip(days[tighter].hours, days[looser].hours)):
            checks.expect(
                b["objective"] <= a["objective"] + 1e-3,
                f"hour {t}: {looser} costs more than {tighter}",
            )


def run_checks(days):
    checks = ed.Checks()
    for label, day in days.items():
        check_hours(checks, label, day)
        check_network(checks, label, day)
    check_active_sets(checks, days)
    check_congestion(checks, days)
    check_marginal_values(checks, days)
    check_order(checks, days)
    return checks


def print_report(days, checks):
    both = days["both limits"]
    print("Hourly dispatch with both limits (1 = unit at full output)")
    print(
        f"{'hour':>4} {'price':>6} {'chp':>6} {'e-boil':>6} {'elyser':>6} "
        f"{'cable':>6} {'psq':>6}  binding"
    )
    for t, h in enumerate(both.hours):
        print(
            f"{t:>4} {h['price']:>6.1f} {h['chp']:>6.3f} {h['e_boiler']:>6.3f} "
            f"{h['electrolyser']:>6.3f} {h['cable_mva']:>6.3f} {h['psq'][-1]:>6.3f}  "
            f"{', '.join(binding_limits(both, h)) or 'none'}"
        )
    print("\nWhere a limit binds: congestion cost and the value of one more MW")
    print(
        f"{'hour':>4} {'price':>6}  {'limit':<9} {'congestion':>11} "
        f"{'lost margin':>12} {'saving/MW':>10} {'margin/MW':>10}"
    )
    extra = congestion(days, "both limits")
    for t, (limit, value, by_hand) in marginal_values(days).items():
        cost, lost = extra[t]
        print(
            f"{t:>4} {ed.EL_PRICE[t]:>6.1f}  {limit:<9} {cost:>11.2f} {lost:>12.2f} "
            f"{value:>10.2f} {by_hand:>10.2f}"
        )
    supply_t_k = [
        v for day in days.values() for h in day.hours for v in h["supply_t_k"]
    ]
    lo_k, hi_k = ed.SUPPLY_T_BAND_K
    print(
        f"\nSupply temperature in every run: {min(supply_t_k):.1f} to "
        f"{max(supply_t_k):.1f} K, setpoint band {lo_k:.0f} to {hi_k:.0f} K"
    )
    print(f"\n{'day':<18} {'cost':>8} {'congestion':>11} {'lost margins':>13}")
    for label, day in days.items():
        line = f"{label:<18} {day.cost:>8.2f}"
        if label in ("both limits", *(alone for alone, _ in LIMITS.values())):
            cost, lost = (sum(v) for v in zip(*congestion(days, label)))
            line += f" {cost:>11.2f} {lost:>13.2f}"
        print(line)
    print(
        f"\nPlausibility checks: {checks.count - len(checks.failed)} of {checks.count} passed"
    )
    for label in checks.failed:
        print(f"  FAILED {label}")


def run():
    days = {
        # The day without limits is the economic dispatch tutorial's network,
        # solved as there.
        name: solve(changes, ed.solve_day if name == "neither" else solve_day)
        for name, changes in DAYS.items()
    }
    return days, run_checks(days)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    days, checks = run()
    print_report(days, checks)
    sys.exit(1 if checks.failed else 0)
