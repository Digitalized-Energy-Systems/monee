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

**monee** is a Python framework for steady-state simulation and optimal energy flow of **multi-energy systems (MES)**. It models electricity, gas, and district-heat networks in a single unified directed graph, couples them via standard conversion units (CHP, P2H, P2G, G2P, G2H), and exposes the coupled system to three solver back-ends (CasADi, GEKKO, and Pyomo). The framework is designed for research workflows that require flexible problem formulation, reproducible benchmarks, and straightforward integration with external optimisers.

## Capabilities

**Physical modelling**
- AC power flow (full nonlinear or MISOCP relaxation)
- Weymouth gas flow (*p*² formulation, Swamee–Jain friction)
- Darcy–Weisbach hydraulic flow with piecewise-linear friction factor and temperature propagation
- Coupling components: CHP, P2H (heat pump / electric boiler), P2G, G2P, G2H

**Optimisation**
- Optimal energy flow via CasADi or GEKKO (IPOPT) or Pyomo (SCIP, GLPK, CBC, Gurobi)
- MISOCP relaxation for convex AC OPF
- Built-in minimum-curtailment (load shedding) formulation with per-carrier bounds
- Capacity limits on all external grid connections (`max_import` / `max_export`)
- Islanding-aware solves with grid-forming units per carrier (`monee.enable_islanding`)
- Structured infeasibility diagnostics on failed solves, for both back-ends
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

Requires **Python 3.11+**.

```bash
pip install monee
```

The core install bundles everything needed for both simulation and the default
optimisation workflows:

- **CasADi**: in-process IPOPT, the default solver when present.
- **GEKKO**: bundles its own APOPT/BPOPT/IPOPT binaries; used as the IPOPT fallback if CasADi is unavailable.
- **Pyomo**: for MILP/MIQCP problems such as MISOCP optimal power flow.

Optional solvers and import formats are pulled in only when you need them:

```bash
pip install monee[simbench]   # SimBench / pandapower grid import
pip install gurobipy          # Gurobi - commercial, free academic licence available
```

---

## Quick start

Build a coupled electricity and district-heat network and run an energy-flow calculation:

```python
import monee.model as mm
from monee import mx, run_energy_flow

net = mx.create_multi_energy_network()

# Electricity grid
bus_0 = mx.create_bus(net)
bus_1 = mx.create_bus(net)
mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
mx.create_ext_power_grid(net, bus_0)
mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

# District heating loop
j_supply = mx.create_water_junction(net)
j_mid = mx.create_water_junction(net)
j_return = mx.create_water_junction(net)
mx.create_ext_hydr_grid(net, j_supply)
mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.12, length_m=100)
mx.create_sink(net, j_return, mass_flow_kgs=1.0)

# Coupling: bus_1 drives a heat pump that feeds the heating loop
mx.create_p2h(
    net, bus_1, j_mid, j_return, heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9
)

result = run_energy_flow(net)
print(result.get(mm.Bus)[["vm_pu", "va_degree"]])
print(result.get(mm.WaterPipe)[["mass_flow_kgs"]])
```

Solvers are selected by name - `ipopt` routes to CasADi when installed (otherwise GEKKO), `apopt` and `bpopt` route to GEKKO, and any other name (e.g. `scip`, `gurobi`) is forwarded to the matching Pyomo solver. Override the back-end explicitly with `backend=...` if needed. By default `run_energy_flow` runs in **simulation mode** (`simulation=True`), squaring the model for a fast steady-state solve; pass `simulation=False` (or an optimisation problem) to solve in optimisation mode. Check `result.mode_used` to see which path actually ran.

---

## Optimal energy flow

Swap `run_energy_flow` for `run_energy_flow_optimization` and pass a problem.
monee ships ready-made formulations - economic dispatch and minimum load
shedding - and you can also build your own (see *Extensibility* below).

Economic dispatch minimises the total generation cost `Σ cost · p_gen` while
respecting the network's physical limits. Give each generator a marginal
`cost`, and the optimiser dispatches the cheaper units first:

```python
import monee
import monee.model as mm
from monee import mx

net = mx.create_multi_energy_network()
bus_ext = mx.create_bus(net)
bus_a = mx.create_bus(net)
bus_b = mx.create_bus(net)
mx.create_line(net, bus_ext, bus_a, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
mx.create_line(net, bus_ext, bus_b, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
mx.create_ext_power_grid(net, bus_ext)
mx.create_power_load(net, bus_a, p_mw=0.1, q_mvar=0.0)

# Two dispatchable generators with different marginal costs (currency/MW)
mx.create_power_generator(net, bus_a, p_mw=0.15, q_mvar=0.0, cost=10.0)  # cheap
mx.create_power_generator(net, bus_b, p_mw=0.15, q_mvar=0.0, cost=40.0)  # expensive

problem = monee.create_economic_dispatch_problem(
    ext_grid_cost_default=0.0,  # include the slack so it can be bounded
    ext_grid_bounds=(-0.01, 0.01),  # substation import/export limit (MW)
)

result = monee.run_energy_flow_optimization(net, problem, solver="ipopt")

# The cheap generator carries the load; the expensive one stays near zero.
# (generation uses the load convention, so dispatched power is negative)
print(result.get(mm.PowerGenerator)[["p_mw"]])
```

For resilience studies, `create_min_load_shedding_problem` instead minimises
curtailment with per-carrier operational bounds, returning a `regulation` per
demand (`1.0` fully served, `0.0` completely shed). Both problems accept the
same solver and formulation options.

---

## Formulations

monee separates the **physical equations** from the **network topology** through a `NetworkFormulation` layer. A formulation maps component types to a set of equations - most cover a **single energy domain**. Calling `apply_formulation()` overwrites the equations for only the component types included in that formulation, leaving all other domains untouched.

Formulations are named by **optimization class** (`NLP`, `MILP`, convex/non-convex `MIQCQP`) and are a property of the **solve**, not the network: pass `formulation=` to any solve call (a registry key string, a `NetworkFormulation`, or a mix), or record a network-level choice with `apply_formulation()`. Components without a choice fall back to `DEFAULT_SIMULATION_FORMULATION` - a deliberate per-domain hybrid:

| Formulation constant | Domain | Equations | Applied |
|---|---|---|---|
| `EL_NLP_FORMULATION` | Electricity | Nonlinear polar AC power flow (voltage magnitude + angle) | default |
| `GAS_CONVEX_MIQCQP_FORMULATION` | Gas | Weymouth equation (*p*² formulation, convex epigraph relaxation) | default |
| `HEAT_NONCONVEX_MIQCQP_FORMULATION` | Water / Heat | Darcy–Weisbach + bilinear temperature propagation | default |
| `EL_MISOCP_FORMULATION` | Electricity | MISOCP relaxation (lifted voltages, SOC constraints) | opt-in |
| `HEAT_CONVEX_MILP_FORMULATION` | Water / Heat | McCormick district-heating relaxation (LP/MILP) | opt-in |
| `SMOOTH_NLP_FORMULATION` | All three | Binary-free smooth NLP (AC + smooth Weymouth + smooth Darcy–Weisbach) | opt-in bundle |
| `CONVEX_MIQCQP_FORMULATION` | All three | Certifiable convex relaxations (MISOCP + relaxed Weymouth + McCormick heat) | opt-in bundle |
| `NONCONVEX_MIQCQP_FORMULATION` | All three | Exact quadratic models for global solvers (exact branch flow + exact Weymouth + bilinear heat) | opt-in bundle |

The smooth bundle replaces all direction binaries with smooth approximations, making the full multi-energy system tractable for pure NLP solvers such as IPOPT; import it (or its factory `make_smooth_nlp_formulation()`) from `monee.model.formulation`.

To use the convex MISOCP relaxation for electricity optimal power flow, select it at solve time - gas and heat equations remain on their defaults:

```python
import monee

result = monee.run_energy_flow(
    net, solver="gurobi", formulation="el_misocp", simulation=False
)

# or record it as the network-level choice for all subsequent solves
net.apply_formulation(monee.EL_MISOCP_FORMULATION)
```

Custom formulations follow the same pattern - subclass the appropriate base class and register it for the target component types:

```python
from monee.model.core import Const, Var
from monee.model.formulation.core import BranchFormulation, NetworkFormulation
from monee.model.branch import GasPipe


class MyGasPipeFormulation(BranchFormulation):
    """Toy linear pipe: pressure drop proportional to the mass flow."""

    def ensure_var(self, branch, simulation=False, grid=None):
        # One signed mass-flow decision variable plus a geometric resistance the
        # equation references; neutralise the Vars this linear model does not use.
        branch.mass_flow_kgs = Var(0.1, name="mass_flow_kgs")
        branch.resistance = Const(1e-3 * branch.length_m / branch.diameter_m**2)
        branch.direction = Const(1)
        for unused in (
            "velocity_mps",
            "reynolds_scaled",
            "gas_density_kg_per_m3",
            "friction",
            "mass_flow_pos_kgs_squared",
            "mass_flow_neg_kgs_squared",
        ):
            setattr(branch, unused, Const(0.0))

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        m = branch.mass_flow_kgs
        mag = kwargs["sqrt_impl"](m * m)  # |m|, so it works in either flow direction
        return [
            from_node_model.pressure_pu - to_node_model.pressure_pu
            == branch.resistance * m,
            # the nodal balance consumes the positive / negative flow split
            branch.mass_flow_pos_kgs == 0.5 * (mag + m),
            branch.mass_flow_neg_kgs == 0.5 * (mag - m),
        ]


net.apply_formulation(
    NetworkFormulation(
        branch_type_to_formulations={GasPipe: MyGasPipeFormulation()},
    )
)
```

---

## Extensibility

Every layer of the framework is designed to be subclassed or replaced without modifying the library internals.

**Custom components** - subclass `NodeModel`, `BranchModel`, `ChildModel`, or `CompoundModel` and register the model with `NetworkFormulation`. The solver infrastructure (variable injection, result extraction, timeseries state tracking) handles the new type automatically.

**Custom objectives and constraints** - the optimisation problem is assembled with a composable builder API. `Objectives` and `Constraints` objects are attached to an `OptimizationProblem`; the solver evaluates them without any solver-specific glue:

```python
import monee.model as mm
from monee import run_energy_flow_optimization
from monee.problem import (
    AttributeParameter,
    Constraints,
    Objectives,
    OptimizationProblem,
)

problem = OptimizationProblem()

# Make regulation ∈ [0, 1] a solver decision variable for each demand
problem.controllable_demands(
    [
        (
            "regulation",
            AttributeParameter(
                min=lambda attr, val: 0,
                max=lambda attr, val: 1,
                val=lambda attr, val: 1,
            ),
        )
    ]
)

# Objective: minimise curtailment cost weighted by load priority
objectives = Objectives()
objectives.select(
    lambda model: isinstance(model, mm.PowerLoad) and hasattr(model, "_priority")
).data(lambda model: (1 - model.regulation) * model.p_mw * model._priority).calculate(
    lambda model_to_data: sum(model_to_data.values())
)
problem.objectives = objectives

# Constraint: cap the substation import
constraints = Constraints()
constraints.select_types(mm.ExtPowerGrid).equation(lambda model: model.p_mw <= 0.6)
problem.constraints = constraints

result = run_energy_flow_optimization(net, problem)
```

**Network-level extensions** - implement `NetworkAspect` (with `prepare` and `equations` methods) and attach it via `network.add_extension(aspect)`. The extension participates in both variable injection and equation registration without any solver-specific glue code.

---

## Timeseries simulation

```python
import monee.model as mm
from monee import run_timeseries, TimeseriesData
import numpy as np

td = TimeseriesData()
td.add_child_series(load_id, "p_mw", np.linspace(0.5, 1.5, 96))

ts_result = run_timeseries(net, td)
# DataFrame indexed by timestep, columns by component ID
print(ts_result.get_result_for(mm.Bus, "vm_pu"))
```

---

## Documentation

Full documentation with tutorials, concept explanations, and API reference:
**[monee.readthedocs.io](https://monee.readthedocs.io)**

---

## License

MIT © [Digitalized Energy Systems](https://github.com/Digitalized-Energy-Systems)
