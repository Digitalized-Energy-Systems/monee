"""Economic dispatch of a small multi-energy system over one price day.

A town is supplied by three coupled carriers:

* power: a grid connection with an hourly price, a PV plant and the town load;
* gas: a gas grid connection, a small biogas source and the town gas demand;
* heat: a district heating line run at a fixed supply temperature. Its plant
  (a gas boiler fired from outside the modelled gas network, modelled as the
  water external grid with a fixed heat price) holds the supply at 358 K and
  feeds three consumers designed for a 30 K spread.

Three coupling units link them: a CHP (gas to power and heat), an electric
boiler (power to heat) and an electrolyser (power to gas). The CHP and the
electric boiler are feed-in stations at the plant: they take return water,
heat it by the same 30 K and feed it into the supply header, so the supply
temperature stays at its setpoint whichever unit runs. The economic dispatch
decides every hour which unit runs. With the prices below the merit order has
closed form breakeven prices, so the dispatch can be checked by hand:

* electric boiler: heat at ``el_price / 0.95`` beats the boiler plant at 40
  below 38 per MWh of power;
* electrolyser: gas at ``el_price / 0.65`` beats the gas grid at 35 below
  22.75 per MWh of power;
* CHP: 0.35 MWh of power plus 0.5 MWh of heat (worth the boiler plant's 40)
  out of 1 MWh of gas at 35 pays off above 42.86 per MWh of power.

Nothing links one hour to the next (no storage, no ramp limit), so the day
is a timeseries of independent hourly dispatches, solved with
``run_timeseries``. Storages or ramp limits would couple the hours and call
for a joint solve with ``run_multi_period``.

The script solves the day, prints the hourly dispatch and then checks that the
result is plausible: the merit order matches those breakevens, every carrier
balances, the coupling units convert at their efficiencies, the supply
temperature stays at its setpoint, all operating limits hold, each hour's
objective equals the cost recomputed from its result tables, and the optimum
is cheaper than two rule based baselines (boiler only, and a must run CHP).
It exits with status 1 if a check fails.

Run::

    python examples/mes_economic_dispatch.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import dataclass, field

import monee.express as mx
import monee.model as mm
from monee import TimeseriesData, run_timeseries
from monee.model.phys.nonlinear.hf import SPECIFIC_HEAT_CAP_WATER
from monee.problem import create_economic_dispatch_problem
from monee.problem.utils import gas_mw_per_kgs

# Prices in currency per MWh; every period is one hour, so they enter as
# currency per MW and period. Gas prices are per MWh of higher heating value.
EL_PRICE = [120.0, 80.0, 55.0, 40.0, 30.0, 18.0, 10.0, 60.0]
PV_AVAILABILITY = [0.0, 0.2, 0.5, 0.8, 1.0, 1.0, 0.9, 0.3]
GAS_PRICE = 35.0
BIOGAS_PRICE = 20.0
BOILER_PRICE = 40.0

LOAD_MW = 2.0
PV_MW = 1.5
GAS_DEMAND_KGS = 0.03
BIOGAS_KGS = 0.008
HEAT_DEMAND_MW = [0.6, 0.7, 0.7]
SUPPLY_T_K = 358.0
SUPPLY_T_BAND_K = (356.0, 360.0)

CHP_FUEL_MW = 3.0
CHP_EFF_EL, CHP_EFF_HEAT = 0.35, 0.5
E_BOILER_HEAT_MW, E_BOILER_EFF = 1.0, 0.95
ELECTROLYSER_GAS_MW, ELECTROLYSER_EFF = 0.65, 0.65

# The analytic breakevens the dispatch must reproduce.
E_BOILER_BELOW = BOILER_PRICE * E_BOILER_EFF
ELECTROLYSER_BELOW = GAS_PRICE * ELECTROLYSER_EFF
CHP_ABOVE = (GAS_PRICE - CHP_EFF_HEAT * BOILER_PRICE) / CHP_EFF_EL

ALL_COUPLERS = ("chp", "e_boiler", "electrolyser")
T_BAND_K = (278.15, 423.15)


@dataclass
class Ids:
    grid: int = None
    pv: int = None
    gas_grid: int = None
    biogas: int = None
    plant: int = None
    return_junction: int = None
    gas_mw_per_kgs: float = None
    water_t_ref_k: float = None
    supply_junctions: list = field(default_factory=list)
    water_junctions: list = field(default_factory=list)


def build_network(couplers=ALL_COUPLERS):
    net = mx.create_multi_energy_network()
    ids = Ids()

    b0, b1, b2 = (mx.create_bus(net, base_kv=20) for _ in range(3))
    for a, b in ((b0, b1), (b1, b2)):
        mx.create_line(net, a, b, length_m=1000, r_ohm_per_m=1.6e-4, x_ohm_per_m=1e-4)
    ids.grid = mx.create_ext_power_grid(net, b0, name="grid")
    mx.create_power_load(net, b1, p_mw=LOAD_MW, q_mvar=0.0, name="town")
    ids.pv = mx.create_power_generator(
        net, b1, p_mw=PV_MW, q_mvar=0.0, cost=0.0, name="pv"
    )

    g0, g1, g2 = (mx.create_gas_junction(net) for _ in range(3))
    ids.gas_mw_per_kgs = gas_mw_per_kgs(net.node_by_id(g0).grid)
    for a, b in ((g0, g1), (g1, g2)):
        mx.create_gas_pipe(net, a, b, diameter_m=0.15, length_m=1000)
    # max_export_kgs=0: the town cannot sell gas back to the grid.
    ids.gas_grid = mx.create_gas_ext_grid(
        net, g0, max_export_kgs=0, cost=GAS_PRICE, name="gas_grid"
    )
    mx.create_gas_sink(net, g1, mass_flow_kgs=GAS_DEMAND_KGS, name="gas_demand")
    ids.biogas = mx.create_gas_source(
        net, g2, mass_flow_kgs=BIOGAS_KGS, cost=BIOGAS_PRICE, name="biogas"
    )

    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=300)
    seg = dhs.line(3, heat_exchanger_q_mw=HEAT_DEMAND_MW)
    ids.plant, _ = dhs.attach_heat_plant(
        seg.supply.first, seg.return_.last, t_k=SUPPLY_T_K, name="boiler"
    )
    net.child_by_id(ids.plant).model.cost = BOILER_PRICE
    ids.return_junction = seg.return_.last
    ids.water_t_ref_k = net.node_by_id(seg.supply.first).grid.t_ref_k
    ids.supply_junctions = list(seg.supply.nodes)
    ids.water_junctions = ids.supply_junctions + list(seg.return_.nodes)

    # Feed-in stations: their internal exchanger heats return water by its
    # 30 K design spread at any output, so what they feed in is at the supply
    # setpoint and only the flow they take over from the plant changes.
    cold, hot = seg.return_.last, seg.supply.first
    if "chp" in couplers:
        mx.create_chp(
            net,
            power_node_id=b2,
            cold_node_id=cold,
            hot_node_id=hot,
            gas_node_id=g2,
            diameter_m=0.15,
            efficiency_power=CHP_EFF_EL,
            efficiency_heat=CHP_EFF_HEAT,
            mass_flow_setpoint_kgs=CHP_FUEL_MW / ids.gas_mw_per_kgs,
            name="chp",
        )
    if "e_boiler" in couplers:
        mx.create_p2h(
            net,
            power_node_id=b2,
            cold_node_id=cold,
            hot_node_id=hot,
            heat_energy_mw=E_BOILER_HEAT_MW,
            diameter_m=0.15,
            efficiency=E_BOILER_EFF,
            name="e_boiler",
        )
    if "electrolyser" in couplers:
        mx.create_p2g(
            net,
            b2,
            g2,
            efficiency=ELECTROLYSER_EFF,
            mass_flow_setpoint_kgs=ELECTROLYSER_GAS_MW / ids.gas_mw_per_kgs,
            name="electrolyser",
        )
    return net, ids


def solve_day(net, ids, dispatch_coupling_points=True):
    td = TimeseriesData()
    td.add_objective_data(ids.grid, "cost", EL_PRICE)
    # regulation scales the PV setpoint: the hourly availability factor.
    td.add_child_series(ids.pv, "regulation", PV_AVAILABILITY)
    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=EL_PRICE[0],
        gas_cost_default=GAS_PRICE,
        heat_cost_default=BOILER_PRICE,
        dispatch_coupling_points=dispatch_coupling_points,
    )
    # The CHP's feed-in exchanger next to the AC lines can stall IPOPT's
    # default barrier update from a cold start in the high price hours.
    result = run_timeseries(
        net,
        td,
        steps=len(EL_PRICE),
        optimization_problem=problem,
        solver_options={"ipopt.mu_strategy": "adaptive"},
    )
    return [step.result for step in result.step_results]


def _one(frames, table, column, row=0):
    df = frames.get(table)
    return 0.0 if df is None else float(df[column].iloc[row])


def day_cost(hour_results):
    return sum(r.user_objective for r in hour_results)


def plant_heat_mw(plant_kgs, t_return_k, feed_ins, t_ref_k):
    """Heat of the boiler plant: its own flow heated from the return to the
    setpoint, plus the top up of the feed-in streams. Pipe losses cool the
    return slightly, so the feed-in stations, adding exactly their design
    spread, deliver a little below the setpoint the plant holds its header at."""
    kgs_k = plant_kgs * (SUPPLY_T_K - t_return_k)
    if feed_ins is not None:
        outlet_k = feed_ins["t_to_pu"] * t_ref_k
        kgs_k += float(
            (feed_ins["mass_flow_kgs"].abs() * (SUPPLY_T_K - outlet_k)).sum()
        )
    return SPECIFIC_HEAT_CAP_WATER * kgs_k / 1e6


def hourly_quantities(hour_results, ids, hours=None):
    """Per hour: the dispatch and every flow the checks need, in MW (gas also
    in kg/s), with the load convention undone so supply is positive. hours
    gives the hour of the day of each result when they are not the whole day."""
    k = ids.gas_mw_per_kgs
    rows = []
    if hours is None:
        hours = range(len(hour_results))
    for t, r in zip(hours, hour_results):
        frames = r.dataframes
        ext_gas = r.get(mm.ExtHydrGrid, index="id")
        junctions = r.get(mm.Junction, index="id")
        plant_kgs = -ext_gas.loc[ids.plant, "mass_flow_kgs"]
        t_return = junctions.loc[ids.return_junction, "t_k"]
        chp_reg = _one(frames, "CHPControlNode", "regulation")
        rows.append(
            {
                "price": EL_PRICE[t],
                "objective": r.user_objective,
                "import_mw": -_one(frames, "ExtPowerGrid", "p_mw"),
                "pv_mw": -_one(frames, "PowerGenerator", "p_mw") * PV_AVAILABILITY[t],
                "line_loss_mw": float(
                    (
                        frames["PowerLine"]["p_from_mw"]
                        + frames["PowerLine"]["p_to_mw"]
                    ).sum()
                ),
                "chp": chp_reg,
                "chp_fuel_kgs": chp_reg
                * _one(frames, "CHPControlNode", "gas_mass_flow_kgs"),
                "chp_el_mw": -_one(frames, "CHPControlNode", "el_mw"),
                "chp_heat_mw": -_one(frames, "CHPControlNode", "heat_mw"),
                "e_boiler": _one(frames, "PowerToHeatControlNode", "regulation"),
                "e_boiler_el_mw": _one(frames, "PowerToHeatControlNode", "el_mw"),
                "e_boiler_heat_mw": -_one(frames, "PowerToHeatControlNode", "heat_mw"),
                "electrolyser": _one(frames, "PowerToGas", "regulation"),
                "electrolyser_el_mw": _one(frames, "PowerToGas", "p_from_mw"),
                "electrolyser_gas_kgs": -_one(frames, "PowerToGas", "to_mass_flow_kgs"),
                "gas_import_kgs": -ext_gas.loc[ids.gas_grid, "mass_flow_kgs"],
                "biogas_kgs": -r.get(mm.Source, index="id").loc[
                    ids.biogas, "mass_flow_kgs"
                ],
                "plant_heat_mw": plant_heat_mw(
                    plant_kgs, t_return, frames.get("SubHE"), ids.water_t_ref_k
                ),
                "hx_duty_mw": float(frames["HeatExchangerLoad"]["q_mw"].sum()),
                "hx_delivered_mw": float(
                    frames["HeatExchangerLoad"]["q_mw_delivered"].sum()
                ),
                "vm_pu": r.get(mm.Bus)["vm_pu"].tolist(),
                "line_loading_pu": frames["PowerLine"]["loading_from_pu"].tolist(),
                "water_t_k": junctions.loc[ids.water_junctions, "t_k"].tolist(),
                "supply_t_k": junctions.loc[ids.supply_junctions, "t_k"].tolist(),
                "return_t_k": t_return,
                "k": k,
            }
        )
    return rows


def unit_margins(price):
    """Hourly saving of each coupling unit at full output against the boiler
    plant and the gas grid, ignoring network losses."""
    return {
        "chp": (CHP_EFF_EL * price + CHP_EFF_HEAT * BOILER_PRICE - GAS_PRICE)
        * CHP_FUEL_MW,
        "e_boiler": (BOILER_PRICE - price / E_BOILER_EFF) * E_BOILER_HEAT_MW,
        "electrolyser": (GAS_PRICE - price / ELECTROLYSER_EFF) * ELECTROLYSER_GAS_MW,
    }


def expected_savings():
    """Saving of the optimum against each baseline from the margins alone: the
    optimum runs every unit whose margin is positive, the must run CHP
    baseline runs the CHP in every hour."""
    margins = [unit_margins(price) for price in EL_PRICE]
    positive = sum(max(m, 0.0) for hour in margins for m in hour.values())
    chp_losses = sum(max(-hour["chp"], 0.0) for hour in margins)
    chp_gains = sum(max(hour["chp"], 0.0) for hour in margins)
    return {
        "boiler only": positive,
        "must run CHP": positive - chp_gains + chp_losses,
    }


def hourly_cost(h):
    k = h["k"]
    return (
        h["price"] * h["import_mw"]
        + GAS_PRICE * k * h["gas_import_kgs"]
        + BIOGAS_PRICE * k * h["biogas_kgs"]
        + BOILER_PRICE * h["plant_heat_mw"]
    )


class Checks:
    def __init__(self):
        self.failed = []
        self.count = 0

    def expect(self, ok, label):
        self.count += 1
        if not ok:
            self.failed.append(label)


def _on(regulation):
    return regulation > 0.99


def _off(regulation):
    return regulation < 0.01


def check_merit_order(checks, h, t):
    price = h["price"]
    for unit, run in (
        ("e_boiler", price < E_BOILER_BELOW),
        ("electrolyser", price < ELECTROLYSER_BELOW),
        ("chp", price > CHP_ABOVE),
    ):
        state = _on(h[unit]) if run else _off(h[unit])
        checks.expect(
            state, f"hour {t}: {unit} should be {'on' if run else 'off'} at {price}"
        )


def check_balances(checks, h, t):
    power = (
        h["import_mw"]
        + h["pv_mw"]
        + h["chp_el_mw"]
        - h["e_boiler_el_mw"]
        - h["electrolyser_el_mw"]
        - LOAD_MW
        - h["line_loss_mw"]
    )
    checks.expect(abs(power) < 1e-4, f"hour {t}: power balance off by {power:.2e} MW")
    checks.expect(0 <= h["line_loss_mw"] < 0.02 * LOAD_MW, f"hour {t}: line losses")
    gas = (
        h["gas_import_kgs"]
        + h["biogas_kgs"]
        + h["electrolyser_gas_kgs"]
        - GAS_DEMAND_KGS
        - h["chp_fuel_kgs"]
    )
    checks.expect(abs(gas) < 1e-6, f"hour {t}: gas balance off by {gas:.2e} kg/s")
    heat_loss = (
        h["plant_heat_mw"] + h["chp_heat_mw"] + h["e_boiler_heat_mw"] - h["hx_duty_mw"]
    )
    checks.expect(
        0 <= heat_loss < 0.03 * sum(HEAT_DEMAND_MW),
        f"hour {t}: heat supply minus demand {heat_loss:.4f} MW is not a small pipe loss",
    )
    checks.expect(
        abs(h["hx_delivered_mw"] - sum(HEAT_DEMAND_MW)) < 1e-5,
        f"hour {t}: heat demand not served in full",
    )


def check_conversions(checks, h, t):
    fuel_mw = h["chp_fuel_kgs"] * h["k"]
    if fuel_mw > 1e-3:
        checks.expect(
            abs(h["chp_el_mw"] - CHP_EFF_EL * fuel_mw) < 1e-6, f"hour {t}: CHP power"
        )
        checks.expect(
            abs(h["chp_heat_mw"] - CHP_EFF_HEAT * fuel_mw) < 1e-6, f"hour {t}: CHP heat"
        )
    checks.expect(
        abs(h["e_boiler_heat_mw"] - E_BOILER_EFF * h["e_boiler_el_mw"]) < 1e-6,
        f"hour {t}: electric boiler efficiency",
    )
    checks.expect(
        abs(
            h["electrolyser_gas_kgs"] * h["k"]
            - ELECTROLYSER_EFF * h["electrolyser_el_mw"]
        )
        < 1e-6,
        f"hour {t}: electrolyser efficiency",
    )


def check_limits(checks, h, t):
    checks.expect(all(0.9 <= v <= 1.1 for v in h["vm_pu"]), f"hour {t}: voltage band")
    checks.expect(
        all(v <= 1.0 for v in h["line_loading_pu"]), f"hour {t}: line loading"
    )
    lo, hi = T_BAND_K
    checks.expect(
        all(lo - 1e-6 <= v <= hi + 1e-6 for v in h["water_t_k"]),
        f"hour {t}: water temperature band",
    )
    lo, hi = SUPPLY_T_BAND_K
    checks.expect(
        all(lo - 1e-6 <= v <= hi + 1e-6 for v in h["supply_t_k"]),
        f"hour {t}: supply temperature off its setpoint",
    )
    checks.expect(h["gas_import_kgs"] >= -1e-9, f"hour {t}: gas export although capped")
    checks.expect(h["plant_heat_mw"] >= -1e-6, f"hour {t}: heat plant absorbs heat")


def run_checks(hours, baselines):
    checks = Checks()
    for t, h in enumerate(hours):
        check_merit_order(checks, h, t)
        check_balances(checks, h, t)
        check_conversions(checks, h, t)
        check_limits(checks, h, t)
        by_hand = hourly_cost(h)
        checks.expect(
            abs(by_hand - h["objective"]) <= 1e-3 * abs(h["objective"]),
            f"hour {t}: objective {h['objective']:.3f} differs from recomputed "
            f"cost {by_hand:.3f}",
        )
    optimum = sum(h["objective"] for h in hours)
    expected = expected_savings()
    for name, cost in baselines.items():
        saving = cost - optimum
        checks.expect(saving >= 0, f"optimum {optimum:.2f} is above baseline {name}")
        # Network losses shift the margins slightly; 1 % of the saving covers them.
        checks.expect(
            abs(saving - expected[name]) <= 0.01 * expected[name],
            f"saving against {name} {saving:.2f} differs from the margins {expected[name]:.2f}",
        )
    return checks


def print_report(hours, baselines, checks):
    print("Hourly dispatch (1 = unit at full output)")
    print(
        f"{'hour':>4} {'price':>6} {'import':>7} {'pv':>5} {'chp':>5} "
        f"{'e-boil':>6} {'elyser':>6} {'boiler':>6}  expected"
    )
    for t, h in enumerate(hours):
        expected = [
            name
            for name, run in (
                ("e-boiler", h["price"] < E_BOILER_BELOW),
                ("electrolyser", h["price"] < ELECTROLYSER_BELOW),
                ("CHP", h["price"] > CHP_ABOVE),
            )
            if run
        ]
        print(
            f"{t:>4} {h['price']:>6.1f} {h['import_mw']:>7.3f} {h['pv_mw']:>5.2f} "
            f"{h['chp']:>5.2f} {h['e_boiler']:>6.2f} {h['electrolyser']:>6.2f} "
            f"{h['plant_heat_mw']:>6.3f}  {', '.join(expected) or 'boiler plant only'}"
        )
    print(
        f"\nBreakevens: e-boiler below {E_BOILER_BELOW:.2f}, "
        f"electrolyser below {ELECTROLYSER_BELOW:.2f}, CHP above {CHP_ABOVE:.2f}"
    )
    supply_t_k = [v for h in hours for v in h["supply_t_k"]]
    print(
        f"Supply temperature: {min(supply_t_k):.1f} to {max(supply_t_k):.1f} K, "
        f"setpoint band {SUPPLY_T_BAND_K[0]:.0f} to {SUPPLY_T_BAND_K[1]:.0f} K"
    )
    expected = expected_savings()
    optimum = sum(h["objective"] for h in hours)
    print(f"\n{'day cost':<20} {'total':>8} {'saving':>8} {'%':>6} {'margins':>8}")
    print(f"{'optimised dispatch':<20} {optimum:>8.2f}")
    for name, cost in baselines.items():
        saving = cost - optimum
        print(
            f"{name:<20} {cost:>8.2f} {saving:>8.2f} {100 * saving / cost:>6.1f} "
            f"{expected[name]:>8.2f}"
        )
    print(
        f"\nPlausibility checks: {checks.count - len(checks.failed)} of {checks.count} passed"
    )
    for label in checks.failed:
        print(f"  FAILED {label}")


def run():
    net, ids = build_network()
    hours = hourly_quantities(solve_day(net, ids), ids)
    baselines = {
        "boiler only": day_cost(solve_day(*build_network(couplers=()))),
        "must run CHP": day_cost(
            solve_day(*build_network(couplers=("chp",)), dispatch_coupling_points=False)
        ),
    }
    checks = run_checks(hours, baselines)
    return hours, baselines, checks


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    hours, baselines, checks = run()
    print_report(hours, baselines, checks)
    sys.exit(1 if checks.failed else 0)
