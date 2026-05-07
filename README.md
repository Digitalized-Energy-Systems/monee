<p align="center">
  <img src="docs/source/_static/monee-logo.drawio.svg" width="220" alt="monee"/>
</p>

<p align="center">
  <strong>Modular Network-based Energy Grid Optimization</strong><br>
  Steady-state simulation and optimal energy flow for coupled electricity, gas, and heat networks
</p>

<p align="center">
  <a href="https://pypi.org/project/monee/"><img src="https://img.shields.io/pypi/v/monee?color=blue" alt="PyPI"/></a>
  <a href="https://monee.readthedocs.io"><img src="https://img.shields.io/badge/docs-readthedocs-blue" alt="Docs"/></a>
  <img src="https://img.shields.io/badge/lifecycle-maturing-blue.svg" alt="lifecycle"/>
  <a href="https://github.com/Digitalized-Energy-Systems/monee/blob/development/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"/></a>
  <a href="https://github.com/Digitalized-Energy-Systems/monee/actions/workflows/test-monee.yml"><img src="https://github.com/Digitalized-Energy-Systems/monee/actions/workflows/test-monee.yml/badge.svg" alt="Tests"/></a>
  <a href="https://codecov.io/gh/Digitalized-Energy-Systems/monee"><img src="https://codecov.io/gh/Digitalized-Energy-Systems/monee/graph/badge.svg?token=KSBSBQGNBZ" alt="Coverage"/></a>
  <a href="https://sonarcloud.io/summary/new_code?id=Digitalized-Energy-Systems_monee"><img src="https://sonarcloud.io/api/project_badges/measure?project=Digitalized-Energy-Systems_monee&metric=alert_status" alt="Quality Gate"/></a>
</p>

---

**monee** is a Python framework for steady-state simulation and optimal energy flow of **multi-energy systems (MES)**. It models electricity, gas, and district-heat networks in a single unified directed graph, couples them via standard conversion units (CHP, P2H, P2G, G2P), and exposes the coupled system to two solver back-ends. The framework is designed for research workflows that require flexible problem formulation, reproducible benchmarks, and straightforward integration with external optimisers.

## Capabilities

**Physical modelling**
- AC power flow (full nonlinear or MISOCP relaxation)
- Weymouth gas flow (*p*² formulation, Swamee–Jain friction)
- Darcy–Weisbach hydraulic flow with piecewise-linear friction factor and temperature propagation
- Coupling components: CHP, P2H (heat pump / electric boiler), P2G, G2P, G2H

**Optimisation**
- Optimal energy flow via GEKKO (IPOPT) or Pyomo (HiGHS, Gurobi, GLPK, CBC)
- MISOCP relaxation for convex AC OPF
- Built-in minimum-curtailment (load shedding) formulation with per-carrier bounds
- Capacity limits on all external grid connections (`max_import` / `max_export`)
- Custom objectives and constraints through a composable problem API

**Simulation**
- Sequential multi-step timeseries simulation with time-varying profiles
- Ramp constraints and inter-step state tracking
- Per-step hooks for custom logic

**Interoperability**
- Import from MATPOWER, pandapower, and SimBench
- NetworkX as the internal graph structure
- Typed result DataFrames, one row per component

---

## Installation

```bash
pip install monee
```

The default GEKKO solver (IPOPT) is bundled. For MILP/MIQCP problems, install a Pyomo-compatible solver:

```bash
pip install highspy   # HiGHS — open source, recommended
pip install gurobipy  # Gurobi — commercial, free academic licence available
```

---

## Quick start

Build a coupled electricity and district-heat network and run an energy-flow calculation:

```python
from monee import mx, run_energy_flow

net = mx.create_multi_energy_network()

# Electricity grid
bus_0 = mx.create_bus(net)
bus_1 = mx.create_bus(net)
mx.create_line(net, bus_0, bus_1, length_m=100,
               r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
mx.create_ext_power_grid(net, bus_0)
mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

# District heating loop
j_supply = mx.create_water_junction(net)
j_mid    = mx.create_water_junction(net)
j_return = mx.create_water_junction(net)
mx.create_ext_hydr_grid(net, j_supply)
mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.12, length_m=100)
mx.create_sink(net, j_return, mass_flow=1.0)

# Coupling: bus_1 drives a heat pump that feeds the heating loop
mx.create_p2h(net, bus_1, j_mid, j_return,
              heat_energy_w=100_000, diameter_m=0.1, efficiency=0.9)

result = run_energy_flow(net)
print(result.get("Bus")[["vm_pu", "va_degree"]])
print(result.get("WaterPipe")[["mass_flow"]])
```

---

## Optimal energy flow and load shedding

Solve minimum-curtailment problems with per-carrier operational bounds:

```python
import monee

net.apply_formulation(monee.MISOCP_NETWORK_FORMULATION)

problem = monee.create_load_shedding_optimization_problem(
    bounds_el=(0.9, 1.1),    # voltage bounds (pu)
    bounds_heat=(0.9, 1.1),  # temperature bounds (pu)
    bounds_gas=(0.9, 1.1),   # pressure bounds (pu)
    use_ext_grid_bounds=False,
)

result = monee.run_energy_flow_optimization(
    net, problem,
    solver=monee.PyomoSolver(), solver_name="highs",
    exclude_unconnected_nodes=True,
)

# regulation == 1.0: fully served; regulation == 0.0: completely shed
print(result.dataframes["PowerLoad"][["regulation"]])
```

---

## Formulations

monee separates the **physical equations** from the **network topology** through a `NetworkFormulation` layer. Each formulation covers a **single energy domain** and maps the component types in that domain to a set of equations. Calling `apply_formulation()` overwrites the equations for only the component types included in that formulation, leaving all other domains untouched.

Every new `Network` starts with three single-domain defaults:

| Formulation constant | Domain | Equations |
|---|---|---|
| `AC_NETWORK_FORMULATION` | Electricity | Nonlinear AC power flow (voltage magnitude + angle) |
| `MISOCP_NETWORK_FORMULATION` | Electricity | MISOCP relaxation (lifted voltages, SOC constraints) |
| `NL_WEYMOUTH_NETWORK_FORMULATION` | Gas | Weymouth equation (*p*² formulation) |
| `NL_DARCY_WEISBACH_NETWORK_FORMULATION` | Water / Heat | Darcy–Weisbach + temperature propagation |

To use the convex MISOCP relaxation for electricity optimal power flow, apply it over the defaults — gas and heat equations remain unchanged:

```python
import monee

net.apply_formulation(monee.MISOCP_NETWORK_FORMULATION)   # replaces electricity equations only
```

Custom formulations follow the same pattern — subclass the appropriate base class and register it for the target component types:

```python
from monee.model.formulation.core import BranchFormulation, NetworkFormulation
from monee.model.branch import GasPipe

class MyGasPipeFormulation(BranchFormulation):
    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [from_node_model.pressure_pu - to_node_model.pressure_pu
                == branch.resistance * branch.mass_flow]

net.apply_formulation(NetworkFormulation(
    branch_type_to_formulations={GasPipe: MyGasPipeFormulation()},
))
```

---

## Extensibility

Every layer of the framework is designed to be subclassed or replaced without modifying the library internals.

**Custom components** — subclass `NodeModel`, `BranchModel`, `ChildModel`, or `CompoundModel` and register the model with `NetworkFormulation`. The solver infrastructure (variable injection, result extraction, timeseries state tracking) handles the new type automatically.

**Custom objectives and constraints** — the optimisation problem is assembled with a composable builder API. `Objectives` and `Constraints` objects are attached to an `OptimizationProblem`; the solver evaluates them without any solver-specific glue:

```python
import monee.model as mm
from monee import run_energy_flow_optimization
from monee.problem import AttributeParameter, Constraints, Objectives, OptimizationProblem

problem = OptimizationProblem()

# Make regulation ∈ [0, 1] a solver decision variable for each demand
problem.controllable_demands([
    ("regulation", AttributeParameter(
        min=lambda attr, val: 0,
        max=lambda attr, val: 1,
        val=lambda attr, val: 1,
    ))
])

# Objective: minimise curtailment cost weighted by load priority
objectives = Objectives()
objectives.select(
    lambda model: isinstance(model, mm.PowerLoad) and hasattr(model, "_priority")
).data(
    lambda model: (1 - model.regulation) * model.p_mw * model._priority
).calculate(
    lambda model_to_data: sum(model_to_data.values())
)
problem.objectives = objectives

# Constraint: cap the substation import
constraints = Constraints()
constraints.select_types(mm.ExtPowerGrid).equation(lambda model: model.p_mw <= 0.6)
problem.constraints = constraints

result = run_energy_flow_optimization(net, problem)
```

**Network-level extensions** — implement `NetworkAspect` (with `prepare` and `equations` methods) and attach it via `network.add_extension(aspect)`. The extension participates in both variable injection and equation registration without any solver-specific glue code.

---

## Timeseries simulation

```python
from monee import run_timeseries, TimeseriesData
import numpy as np

td = TimeseriesData()
td.add_child_series(load_id, "p_mw", np.linspace(0.5, 1.5, 96))

ts_result = run_timeseries(net, td)
# DataFrame indexed by timestep, columns by component ID
print(ts_result.get_result_for("Bus", "vm_pu"))
```

---

## Documentation

Full documentation with tutorials, concept explanations, and API reference:
**[monee.readthedocs.io](https://monee.readthedocs.io)**

---

## License

MIT © [Digitalized Energy Systems](https://github.com/Digitalized-Energy-Systems)
