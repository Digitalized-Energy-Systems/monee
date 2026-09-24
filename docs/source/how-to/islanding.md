
# Solve islanded networks

Enable islanding for one or more carriers so a network with disconnected
islands converges. For the formulation, see {doc}`../concepts/islanding`.

---

## Prerequisites

* monee installed with a MIP-capable solver (GEKKO or Pyomo + SCIP / CBC /
  Gurobi).
* A `Network` with at least one node that has no reachable path to the external
  grid, that is, a second island.

---

## Quick start: electricity

```python
import monee as mn
import monee.model as mm
import monee.express as mx

# 1. Build the network
net = mm.Network()

bus_0 = mx.create_bus(net)  # island A - reference
bus_1 = mx.create_bus(net)  # island A - load bus
bus_2 = mx.create_bus(net)  # island B - isolated

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
network. The solver picks it up automatically when `run_energy_flow` (or
`solve`) runs.

---

## Multi-carrier islanding

Pass `True` (or a custom mode instance) for each carrier you want to island:

```python
mn.enable_islanding(net, electricity=True, gas=True, water=True)
```

You can mix custom modes with defaults:

```python
import monee as mn

mn.enable_islanding(
    net,
    electricity=mn.ElectricityIslandingMode(angle_bound=3.15),
    gas=mn.GasIslandingMode(),
)
```

`angle_bound` (in radians) caps the voltage angle on energised buses. The
big-M constant of the connectivity-flow constraints is sized automatically
from the node count, so you do not need to tune it.

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
and exposes a `mass_flow_kgs` variable that absorbs the island's supply and
demand imbalance.

For water/heat networks use `mx.create_water_grid_forming_source` identically:

```python
mx.create_water_grid_forming_source(net, junction_island, pressure_pu=1.0)
mn.enable_islanding(net, water=True)
```

---

## Express API reference: islanding components

| Function | Carrier | Description |
|---|---|---|
| `mx.create_grid_forming_generator(net, node_id, p_mw_max, q_mvar_max)` | Electricity | Variable-output slack generator; angle pinned to 0 by islanding formulation. While it does not lead an island it holds `nominal_p_mw` / `nominal_q_mvar` (0 by default, `None` leaves the injection free) |
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
print(src_df[["mass_flow_kgs"]])
```

Results use the load convention that every child component follows, so a
grid-forming generator created with `p_mw_max=2.0` and injecting its full
output reports `p_mw = -2.0`. The capacity argument is a positive magnitude
and bounds the injection symmetrically, that is `-2.0 <= p_mw <= 2.0`.

---

## Changing a unit capacity between scenarios

`p_mw_max` and `q_mvar_max` stay writable on the model, so a scenario study can
rescale a grid-forming generator in place and solve again:

```python
gid = mx.create_grid_forming_generator(net, bus_2, p_mw_max=1.0, q_mvar_max=0.5)
result_full = mn.run_energy_flow(net)

net.child_by_id(gid).model.p_mw_max = 0.4
result_derated = mn.run_energy_flow(net)
```

Set the capacity before the solve; assigning it while the network is injected
into a solver back end raises an `AttributeError`.

`overwrite_id` is not a replace switch: it assigns an explicit child id that
must still be free, and raises if the id is taken. To swap a unit out rather
than rescale it, call `net.remove_child(gid)` first.

---

## Choosing the right solver

Islanding adds binary variables, making the problem a MILP or MIQP. Both
back-ends support this:

| Solver | Notes |
|---|---|
| `GEKKOSolver` | Built-in APOPT handles MILP; sufficient for most islanding problems |
| `PyomoSolver` | Use SCIP (`scip`), CBC (`cbc`), or Gurobi (`gurobi`) for larger problems |

```python
result = mn.run_energy_flow(net, solver=mn.PyomoSolver(solver_name="scip"))
```

---

## Common pitfalls

**Island not detected, bus still ignored.** Check that the island's
grid-forming child inherits `GridFormingMixin` and is marked `active=True`.
`ExtPowerGrid`, `ExtHydrGrid`, `GridFormingGenerator`, and `GridFormingSource`
all qualify automatically. A custom component must inherit `GridFormingMixin`
explicitly.

**All energization variables are zero.** The big-M is sized automatically from
the node count, so an all-zero solution is never a tuning issue. Check instead
that an `active` grid-forming child exists in each island, and that the back-end
is MIP-capable. The energization variables `e_*` are binary, so use GEKKO
(APOPT) or Pyomo with `scip`, `cbc`, or `gurobi`.

**The grid-forming unit stays at zero.** A unit only leads an island whose
component has lost its ext grid. Anywhere else it holds its nominal setpoint,
which is 0 MW for `GridFormingGenerator`, so an intact ext-led grid runs as if
the unit were not there. That is deliberate: nothing prices a free grid-forming
injection, so the dispatch would be arbitrary. Pass `nominal_p_mw` /
`nominal_q_mvar` for a real setpoint, or `None` to leave the injection free.
A unit solved without any islanding configuration logs a warning and behaves
the same way.

**How far along a feeder an island reaches.** De-energising a bus takes the
branches at that bus out of service with it, so the unit serves as many
neighbours as its capacity covers and sheds the rest. A five-bus feeder with
0.15 MW per bus and a 0.4 MW unit at the far end settles at
`e_el = [0, 0, 0, 1, 1]`, two buses and 0.30 MW served; raise the unit to
0.6 MW and it becomes `e_el = [0, 0, 1, 1, 1]`, 0.45 MW served. This gating
turns the AC branch equations into a MIQCQP, so solve with SCIP or Gurobi.
`ElectricityIslandingMode(branch_gating=...)` controls where it applies:
`"auto"` (the default) restricts it to components anchored solely by
capacity-limited grid-forming units, `"all"` applies it everywhere, and
`"off"` returns to the older formulation in which a de-energised bus stays
coupled to its neighbours and only pre-separated node sets can be shed. See
{doc}`../concepts/islanding` for the formulation.

**A backup line still needs `controllable_backup_lines`.** The gate only bounds
a `backup=True` branch from above, so islanding never closes it for you.
Mark the candidates with `model.backup = True` and call
`OptimizationProblem.controllable_backup_lines()` to make closing it a
decision.

**Islanding lost after saving and loading.** The islanding configuration is
registered as a network extension and is not persisted by
{mod}`monee.io.native`. Call `enable_islanding` again after loading a network
from JSON.

**Solver does not support integers.** Some Pyomo solver interfaces (for example
IPOPT) reject integer variables and raise an error. Switch to SCIP, CBC, or
Gurobi.
