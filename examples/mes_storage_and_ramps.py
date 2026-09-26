"""Economic dispatch of the multi-energy town with a battery and ramp limits.

This extends examples/mes_economic_dispatch.py: the same town, the same three
coupling units and the same eight hourly prices, with two additions:

* a battery at the town bus: 1 MW, 2 MWh, 90 % efficient when charging and
  when discharging, holding 1 MWh at the start of the day and required to
  hold 1 MWh again at its end;
* ramp limits: the operating point of the CHP, the electric boiler and the
  electrolyser may change by at most 0.5 from one hour to the next.

Both couple the hours. What the battery sells in one hour it must have
bought in an earlier one or held since the start of the day, and how far a
unit can run in one hour depends on where it ran in the hour before. The
day is therefore solved as one problem with ``run_multi_period`` instead of
eight independent hours.

The script solves the day four times: without limits, hour by hour as in
mes_economic_dispatch.py; with ramp limits hour by hour; with ramp limits
jointly; and with ramp limits and the battery jointly. It then checks each
result: the balances, conversions, limits and supply temperature as in
mes_economic_dispatch.py, the ramp limits, the battery's state of charge,
the day cost recomputed from the result tables, and every dispatch against a
copper plate twin, a small linear program with the same prices, efficiencies
and limits but no network. The cost of the ramp limits, the value of
foresight and the value of the battery must match the twin's within the
network losses. It exits with status 1 if a check fails.

Run::

    python examples/mes_storage_and_ramps.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

import monee.express as mx
import monee.model as mm
from monee import TimeseriesData, run_multi_period, run_timeseries
from monee.problem import create_economic_dispatch_problem

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mes_economic_dispatch as ed  # noqa: E402

RAMP_LIMIT = 0.5
BATTERY_MW = 1.0
BATTERY_MWH = 2.0
BATTERY_EFF = 0.9
BATTERY_START_MWH = 1.0

UNITS = ("chp", "e_boiler", "electrolyser")
UNIT_HEAT_MW = (ed.CHP_EFF_HEAT * ed.CHP_FUEL_MW, ed.E_BOILER_HEAT_MW, 0.0)
DAYS = (
    "no limits, hour by hour",
    "ramp limits, hour by hour",
    "ramp limits, joint",
    "ramps and battery, joint",
)


@dataclass
class Day:
    hours: list
    cost: float


def build_network(battery=True):
    net, ids = ed.build_network()
    if not battery:
        return net, ids, None
    town_bus = net.child_by_id(ids.pv).node_id
    battery_id = mx.create_el_child(
        net,
        mm.ElectricStorage(
            e_mwh_initial=BATTERY_START_MWH,
            e_mwh_max=BATTERY_MWH,
            p_max_mw=BATTERY_MW,
            efficiency_charge=BATTERY_EFF,
            efficiency_discharge=BATTERY_EFF,
        ),
        town_bus,
        name="battery",
    )
    return net, ids, battery_id


def price_day(ids):
    td = TimeseriesData()
    td.add_objective_data(ids.grid, "cost", ed.EL_PRICE)
    td.add_child_series(ids.pv, "regulation", ed.PV_AVAILABILITY)
    return td


def dispatch_problem(battery):
    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=ed.EL_PRICE[0],
        gas_cost_default=ed.GAS_PRICE,
        heat_cost_default=ed.BOILER_PRICE,
        include_storages=battery,
    )
    problem.constraints.regulation_ramp(RAMP_LIMIT)
    return problem


def solve_jointly(battery=True):
    net, ids, battery_id = build_network(battery)
    result = run_multi_period(
        net,
        price_day(ids),
        steps=len(ed.EL_PRICE),
        optimization_problem=dispatch_problem(battery),
        terminal_state={(battery_id, "e_mwh"): BATTERY_START_MWH} if battery else None,
    )
    periods = [result.get_period_result(t) for t in range(result.T)]
    return Day(quantities(periods, ids), result.user_objective)


def solve_hour_by_hour():
    net, ids, _ = build_network(battery=False)
    result = run_timeseries(
        net,
        price_day(ids),
        steps=len(ed.EL_PRICE),
        optimization_problem=dispatch_problem(battery=False),
        solver_options={"ipopt.mu_strategy": "adaptive"},
    )
    periods = [step.result for step in result.step_results]
    return Day(quantities(periods, ids), ed.day_cost(periods))


def solve_without_limits():
    net, ids = ed.build_network()
    periods = ed.solve_day(net, ids)
    return Day(quantities(periods, ids), ed.day_cost(periods))


def quantities(period_results, ids):
    hours = ed.hourly_quantities(period_results, ids)
    for h, r in zip(hours, period_results):
        frames = r.dataframes
        h["battery_mw"] = ed._one(frames, "ElectricStorage", "p_mw")
        h["charge_mw"] = ed._one(frames, "ElectricStorage", "p_charge_mw")
        h["discharge_mw"] = ed._one(frames, "ElectricStorage", "p_discharge_mw")
        h["stored_mwh"] = (
            ed._one(frames, "ElectricStorage", "e_mwh")
            if "ElectricStorage" in frames
            else BATTERY_START_MWH
        )
    return hours


def copper_plate(prices, battery=True, start=None, ramp_limit=RAMP_LIMIT):
    """The dispatch without the network: each unit earns its margin at full
    output, the battery pays the price for what it charges and earns it for
    what it discharges, under the ramp limits, the battery's limits and the
    heat plant's (it cannot absorb heat). ``start`` holds the operating
    points of the hour before the first one; without it the first hour is
    free, as in the solves."""
    n, width = len(prices), len(UNITS) + 3
    charge, discharge, stored = len(UNITS), len(UNITS) + 1, len(UNITS) + 2

    def row(*entries):
        coefficients = np.zeros(n * width)
        for t, j, value in entries:
            coefficients[t * width + j] += value
        return coefficients

    cost = np.zeros(n * width)
    a_ub, b_ub, a_eq, b_eq = [], [], [], []
    for t, price in enumerate(prices):
        margins = ed.unit_margins(price)
        for j, unit in enumerate(UNITS):
            cost[t * width + j] = -margins[unit]
        cost[t * width + charge] = price
        cost[t * width + discharge] = -price
        a_ub.append(row(*((t, j, UNIT_HEAT_MW[j]) for j in range(len(UNITS)))))
        b_ub.append(sum(ed.HEAT_DEMAND_MW))
        for j, unit in enumerate(UNITS):
            for sign in (1, -1):
                if t > 0:
                    a_ub.append(row((t, j, sign), (t - 1, j, -sign)))
                    b_ub.append(ramp_limit)
                elif start is not None:
                    a_ub.append(row((t, j, sign)))
                    b_ub.append(ramp_limit + sign * start[unit])
        previous = ((t - 1, stored, -1.0),) if t > 0 else ()
        a_eq.append(
            row(
                (t, stored, 1.0),
                (t, charge, -BATTERY_EFF),
                (t, discharge, 1 / BATTERY_EFF),
                *previous,
            )
        )
        b_eq.append(0.0 if t > 0 else BATTERY_START_MWH)
    a_eq.append(row((n - 1, stored, 1.0)))
    b_eq.append(BATTERY_START_MWH)
    power = BATTERY_MW if battery else 0.0
    bounds = [(0, 1)] * len(UNITS) + [(0, power), (0, power), (0, BATTERY_MWH)]
    solution = linprog(cost, a_ub, b_ub, a_eq, b_eq, bounds=bounds * n, method="highs")
    x = solution.x.reshape(n, width)
    plate = {unit: x[:, j].tolist() for j, unit in enumerate(UNITS)}
    plate.update(
        charge_mw=x[:, charge].tolist(),
        discharge_mw=x[:, discharge].tolist(),
        stored_mwh=x[:, stored].tolist(),
        value=-solution.fun,
    )
    return plate


def copper_plate_hour_by_hour():
    points = {unit: [] for unit in UNITS}
    value, start = 0.0, None
    for price in ed.EL_PRICE:
        hour = copper_plate([price], battery=False, start=start)
        start = {unit: hour[unit][0] for unit in UNITS}
        for unit in UNITS:
            points[unit].append(start[unit])
        value += hour["value"]
    return {**points, "value": value}


def check_hours(checks, hours):
    for t, h in enumerate(hours):
        # The battery is one more use of power: without its draw the import
        # must balance the town exactly as in mes_economic_dispatch.py.
        ed.check_balances(
            checks, {**h, "import_mw": h["import_mw"] - h["battery_mw"]}, t
        )
        ed.check_conversions(checks, h, t)
        ed.check_limits(checks, h, t)


def check_ramps(checks, hours, label):
    for unit in UNITS:
        for t in range(1, len(hours)):
            step = hours[t][unit] - hours[t - 1][unit]
            checks.expect(
                abs(step) <= RAMP_LIMIT + 1e-6,
                f"{label}, hour {t}: {unit} moves by {step:.3f}",
            )


def check_against_plate(checks, hours, plate, label, keys=UNITS):
    for key in keys:
        for t, h in enumerate(hours):
            checks.expect(
                abs(h[key] - plate[key][t]) <= 1e-3,
                f"{label}, hour {t}: {key} {h[key]:.4f} but copper plate "
                f"{plate[key][t]:.4f}",
            )


def check_battery(checks, hours):
    stored = BATTERY_START_MWH
    for t, h in enumerate(hours):
        charge, discharge = h["charge_mw"], h["discharge_mw"]
        checks.expect(
            abs(h["battery_mw"] - (charge - discharge)) <= 1e-6,
            f"hour {t}: battery power is not charge minus discharge",
        )
        stored += BATTERY_EFF * charge - discharge / BATTERY_EFF
        checks.expect(
            abs(h["stored_mwh"] - stored) <= 1e-6,
            f"hour {t}: stored {h['stored_mwh']:.4f} MWh, bookkeeping says {stored:.4f}",
        )
        stored = h["stored_mwh"]
        checks.expect(
            -1e-6 <= h["stored_mwh"] <= BATTERY_MWH + 1e-6,
            f"hour {t}: state of charge outside 0 to {BATTERY_MWH} MWh",
        )
        checks.expect(
            max(charge, discharge) <= BATTERY_MW + 1e-6,
            f"hour {t}: battery above {BATTERY_MW} MW",
        )
        checks.expect(
            min(charge, discharge) <= 1e-4,
            f"hour {t}: battery charges and discharges at once",
        )
    checks.expect(
        abs(hours[-1]["stored_mwh"] - BATTERY_START_MWH) <= 1e-6,
        "battery does not end the day at its starting charge",
    )
    for t, h in enumerate(hours):
        if h["charge_mw"] <= 1e-4:
            continue
        sale = next(
            (
                later["price"]
                for later in hours[t + 1 :]
                if later["discharge_mw"] > 1e-4
            ),
            None,
        )
        checks.expect(
            sale is None or h["price"] <= BATTERY_EFF**2 * sale,
            f"hour {t}: battery buys at {h['price']} what it next sells at {sale} "
            "after losses",
        )


def check_day_cost(checks, hours, cost, label):
    by_hand = sum(ed.hourly_cost(h) for h in hours)
    checks.expect(
        abs(by_hand - cost) <= 1e-6 * abs(cost),
        f"{label}: day cost {cost:.4f} differs from recomputed {by_hand:.4f}",
    )


def check_battery_cash_flow(checks, joint, battery):
    for t, (c, d) in enumerate(zip(joint, battery)):
        cash = c["price"] * (
            d["discharge_mw"] - d["charge_mw"] - (d["line_loss_mw"] - c["line_loss_mw"])
        )
        saving = ed.hourly_cost(c) - ed.hourly_cost(d)
        checks.expect(
            abs(saving - cash) <= 1e-3,
            f"hour {t}: battery saves {saving:.4f}, its cash flow is {cash:.4f}",
        )


def effects(days, plates):
    """Cost of the ramp limits, value of foresight and value of the battery,
    from the solver's day costs and from the copper plate."""
    free, hourly, joint, battery = (days[name].cost for name in DAYS)
    unlimited = plates["no limits"]["value"]
    return {
        "cost of the ramp limits": (joint - free, unlimited - plates["joint"]["value"]),
        "value of foresight": (
            hourly - joint,
            plates["joint"]["value"] - plates["hour by hour"]["value"],
        ),
        "value of the battery": (
            joint - battery,
            plates["battery"]["value"] - plates["joint"]["value"],
        ),
    }


def run_checks(days, plates):
    checks = ed.Checks()
    free, hourly, joint, battery = (days[name] for name in DAYS)
    for label, day in days.items():
        check_hours(checks, day.hours)
        check_day_cost(checks, day.hours, day.cost, label)
        if label != DAYS[0]:
            check_ramps(checks, day.hours, label)
    check_against_plate(checks, hourly.hours, plates["hour by hour"], DAYS[1])
    check_against_plate(checks, joint.hours, plates["joint"], DAYS[2])
    check_against_plate(
        checks,
        battery.hours,
        plates["battery"],
        DAYS[3],
        keys=(*UNITS, "charge_mw", "discharge_mw", "stored_mwh"),
    )
    check_battery(checks, battery.hours)
    check_battery_cash_flow(checks, joint.hours, battery.hours)
    checks.expect(free.cost <= joint.cost + 1e-6, "ramp limits made the day cheaper")
    checks.expect(joint.cost <= hourly.cost + 1e-6, "foresight made the day dearer")
    checks.expect(battery.cost <= joint.cost + 1e-6, "the battery made the day dearer")
    for name, (solver, plate) in effects(days, plates).items():
        # Network losses shift the margins slightly; 1 % covers them.
        checks.expect(
            abs(solver - plate) <= 0.01 * abs(plate),
            f"{name}: solver {solver:.2f}, copper plate {plate:.2f}",
        )
    return checks


def print_report(days, plates, checks):
    hourly, battery = days[DAYS[1]].hours, days[DAYS[3]].hours
    print("Operating points with ramp limits, joint and hour by hour")
    print(
        " " * 13
        + "".join(f"{name:<14}" for name in ("chp", "e-boiler"))
        + "electrolyser"
    )
    print(f"{'hour':>4} {'price':>6}" + f"{'joint':>7}{'hourly':>7}" * 3)
    for t, (j, h) in enumerate(zip(battery, hourly)):
        points = "".join(f"{max(j[u], 0.0):>7.2f}{max(h[u], 0.0):>7.2f}" for u in UNITS)
        print(f"{t:>4} {j['price']:>6.1f}{points}")
    print("\nBattery, joint (MW; MWh stored at the end of the hour)")
    print(
        f"{'hour':>4} {'price':>6} {'charge':>7} {'disch':>7} {'stored':>7} {'import':>7}"
    )
    for t, h in enumerate(battery):
        print(
            f"{t:>4} {h['price']:>6.1f} {max(h['charge_mw'], 0.0):>7.3f} "
            f"{max(h['discharge_mw'], 0.0):>7.3f} {max(h['stored_mwh'], 0.0):>7.3f} "
            f"{h['import_mw']:>7.3f}"
        )
    supply_t_k = [
        v for day in days.values() for h in day.hours for v in h["supply_t_k"]
    ]
    lo_k, hi_k = ed.SUPPLY_T_BAND_K
    print(
        f"\nSupply temperature in every run: {min(supply_t_k):.1f} to "
        f"{max(supply_t_k):.1f} K, setpoint band {lo_k:.0f} to {hi_k:.0f} K"
    )
    print(f"\n{'day cost':<28} {'total':>8}")
    for name in DAYS:
        print(f"{name:<28} {days[name].cost:>8.2f}")
    print(f"\n{'effect':<28} {'solver':>8} {'copper plate':>13}")
    for name, (solver, plate) in effects(days, plates).items():
        print(f"{name:<28} {solver:>8.2f} {plate:>13.2f}")
    print(
        f"\nPlausibility checks: {checks.count - len(checks.failed)} of {checks.count} passed"
    )
    for label in checks.failed:
        print(f"  FAILED {label}")


def run():
    days = dict(
        zip(
            DAYS,
            (
                solve_without_limits(),
                solve_hour_by_hour(),
                solve_jointly(battery=False),
                solve_jointly(battery=True),
            ),
        )
    )
    plates = {
        "no limits": copper_plate(ed.EL_PRICE, battery=False, ramp_limit=1.0),
        "hour by hour": copper_plate_hour_by_hour(),
        "joint": copper_plate(ed.EL_PRICE, battery=False),
        "battery": copper_plate(ed.EL_PRICE, battery=True),
    }
    checks = run_checks(days, plates)
    return days, plates, checks


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    days, plates, checks = run()
    print_report(days, plates, checks)
    sys.exit(1 if checks.failed else 0)
