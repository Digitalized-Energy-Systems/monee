
# Solve islanded networks

This guide shows how to enable islanding for one or more carriers so that a
network with multiple disconnected islands converges correctly.

For background on the formulation see {doc}`../concepts/islanding`.

---

## Prerequisites

* monee installed with a MIP-capable solver (GEKKO or Pyomo + HiGHS / CBC /
  Gurobi).
* A `Network` object with at least one junction that has no reachable path to
  the external grid — i.e. a second island.

---

## Quick start — electricity

```python
import monee as mn
import monee.model as mm
import monee.express as mx

# 1. Build the network
net = mm.Network()

bus_0 = mx.create_bus(net)   # island A — reference
bus_1 = mx.create_bus(net)   # island A — load bus
bus_2 = mx.create_bus(net)   # island B — isolated

mx.create_ext_power_grid(net, bus_0)
mx.create_power_load(net, bus_1, p_mw=0.05, q_mvar=0)
mx.create_line(net, bus_0, bus_1, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

# Island B: a grid-forming generator anchors the island and absorbs its load
mx.create_grid_forming_generator(net, bus_2, p_mw_max=1.0, q_mvar_max=0.5)
mx.create_power_load(net, bus_2, p_mw=0.08, q_mvar=0)

# 2. Enable islanding
mn.enable_islanding(net, electricity=True)

# 3. Solve
result = mn.run_energy_flow(net)
print(result)
```

`enable_islanding` returns a `NetworkIslandingConfig` and registers it on the
network.  The solver picks it up automatically when `run_energy_flow` (or
`solve`) is called.

---

## Multi-carrier islanding

Pass `True` (or a custom mode instance) for each carrier you want to island:

```python
mn.enable_islanding(net, electricity=True, gas=True, water=True)
```

You can mix custom modes with defaults:

```python
from monee.model.islanding import ElectricityIslandingMode, GasIslandingMode

mn.enable_islanding(
    net,
    electricity=ElectricityIslandingMode(angle_bound=3.15, big_m_conn=50),
    gas=GasIslandingMode(big_m_conn=50),
)
```

Set `big_m_conn` to at least the number of nodes in the carrier sub-network.
The default of 200 is safe for most networks.

---

## Adding a grid-forming gas source

Use `mx.create_gas_grid_forming_source` to anchor a gas island at a fixed
pressure setpoint:

```python
junction_island = mx.create_gas_junction(net)
mx.create_gas_grid_forming_source(net, junction_island, pressure_pu=1.0, t_k=356.0)

mn.enable_islanding(net, gas=True)
result = mn.run_energy_flow(net)
```

The source pins the junction pressure to `pressure_pu` (like `ExtHydrGrid`)
and exposes a variable `mass_flow` that absorbs the island's supply–demand
imbalance.

For water/heat networks use `mx.create_water_grid_forming_source` identically:

```python
mx.create_water_grid_forming_source(net, junction_island, pressure_pu=1.0)
mn.enable_islanding(net, water=True)
```

---

## Express API reference — islanding components

| Function | Carrier | Description |
|---|---|---|
| `mx.create_grid_forming_generator(net, node_id, p_mw_max, q_mvar_max)` | Electricity | Variable-output slack generator; angle pinned to 0 by islanding formulation |
| `mx.create_gas_grid_forming_source(net, node_id)` | Gas | Pressure-reference source for a gas island |
| `mx.create_water_grid_forming_source(net, node_id)` | Water / Heat | Pressure-reference source for a water island |
| `mx.create_grid_forming_source(net, node_id, grid_key=...)` | Gas or Water | Generic version; use `mm.GAS_KEY` or `mm.WATER_KEY` |

---

## Accessing results

Results for the islanding-specific components appear in the standard result
dataframes under their class name:

```python
# Electricity grid-forming generator output
gf_df = result.dataframes.get("GridFormingGenerator")
print(gf_df[["p_mw", "q_mvar"]])

# Gas grid-forming source output
src_df = result.dataframes.get("GridFormingSource")
print(src_df[["mass_flow"]])
```

---

## Choosing the right solver

Islanding adds binary variables, making the problem a **MILP / MIQP**.  Both
back-ends support this:

| Solver | Notes |
|---|---|
| `GEKKOSolver` | Built-in APOPT handles MILP; sufficient for most islanding problems |
| `PyomoSolver` | Use HiGHS (`highs`), CBC (`cbc`), or Gurobi (`gurobi`) for larger problems |

```python
result = mn.run_energy_flow(net, solver=mn.PyomoSolver(solver_name="highs"))
```

---

## Common pitfalls

**Island not detected — bus still ignored**

Check that the island's grid-forming child inherits `GridFormingMixin` and
is marked `active=True`.  `ExtPowerGrid`, `ExtHydrGrid`, `GridFormingGenerator`,
and `GridFormingSource` all qualify automatically.  A custom component must
explicitly inherit `GridFormingMixin`.

**`big_m_conn` too small**

If the solver returns a trivially infeasible or degenerate solution, increase
`big_m_conn` to be at least the number of carrier nodes.

**Solver does not support integers**

Some Pyomo solver interfaces (e.g. IPOPT) do not support integer variables and
will raise an error.  Switch to HiGHS, CBC, or Gurobi.
