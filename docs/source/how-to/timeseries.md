
# Timeseries simulation

This guide shows how to drive a multi-energy network through a sequence of
timesteps — varying load profiles, generator setpoints, mass flows, and any
other model attribute — and how to query the results.

For background on the underlying architecture see
{doc}`../concepts/timeseries`.

---

## Prerequisites

* A solved base network (run `run_energy_flow` once to check it converges).
* One or more time series for component attributes (plain Python lists, pandas
  `Series`, or a pandas `DataFrame`).

---

## Quick start

```python
import monee as mn
import monee.model as mm
import monee.express as mx
from monee.simulation import TimeseriesData, run_timeseries

# 1. Build a simple two-bus power network
net = mm.Network()
bus_0 = mx.create_bus(net)
bus_1 = mx.create_bus(net)
mx.create_ext_power_grid(net, bus_0)
load = mx.create_power_load(net, bus_1, p_mw=1.0, q_mvar=0.0, name="demand")
mx.create_line(net, bus_0, bus_1, length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

# 2. Define time-varying load profile (24 hourly values)
td = TimeseriesData()
td.add_child_series_by_name("demand", "p_mw", [0.4, 0.5, 0.6, 0.8, 1.0, 1.2,
                                                1.3, 1.1, 1.0, 0.9, 0.8, 0.7,
                                                0.6, 0.7, 0.8, 1.0, 1.2, 1.3,
                                                1.1, 0.9, 0.7, 0.6, 0.5, 0.4])

# 3. Run — steps inferred from series length
result = run_timeseries(net, td)
print(f"Simulated {len(result.raw)} steps, {len(result.failed_steps)} failures")
```

---

## Registering time series

### By component id

Use the integer id returned when you add a component to the network:

```python
td = TimeseriesData()
td.add_child_series(load.id, "p_mw", profile)        # child (load, generator …)
td.add_branch_series(pipe.id, "on_off", [1, 1, 0])   # branch
td.add_compound_series(chp.id, "regulation", ramps)  # compound (CHP, P2H …)
td.add_node_series(bus.id, "some_attr", values)       # node model attribute
```

### By name

Components added with a `name` keyword can be referenced by that name:

```python
mx.create_power_load(net, bus, p_mw=1.0, q_mvar=0, name="factory_load")
mx.create_gas_pipe(net, j0, j1, diameter_m=0.5, length_m=500, name="main_pipe")

td.add_child_series_by_name("factory_load", "p_mw", values)
td.add_branch_series_by_name("main_pipe", "on_off", [1, 0, 1])
td.add_compound_series_by_name("boiler_1", "regulation", values)
```

### From a pandas DataFrame

When your data lives in a DataFrame (one column per attribute, one row per
timestep):

```python
import pandas

df = pandas.read_csv("load_profile.csv")   # columns: p_mw, q_mvar
td = TimeseriesData.from_dataframe(df, component_type="child", component_id=load.id)

# or by name
td = TimeseriesData.from_dataframe(df, component_type="child", component_name="demand")
```

`component_type` is one of `'node'`, `'child'`, `'branch'`, or `'compound'`.
Name-based lookup is not available for nodes.

### Validation

All registered series must have the same length.  A mismatch raises
`ValueError` at registration time, not during the run:

```python
td.add_child_series(1, "p_mw", [1.0, 2.0, 3.0])
td.add_child_series(2, "p_mw", [1.0, 2.0])       # ← ValueError immediately
```

---

## Step count

By default `steps` is inferred from the registered series length:

```python
result = run_timeseries(net, td)         # steps = len(series)
result = run_timeseries(net, td, steps=8)  # explicit override — must not exceed series length
```

---

## Combining TimeseriesData objects

```python
td_loads  = TimeseriesData()
td_loads.add_child_series(load_id, "p_mw", load_profile)

td_pipes  = TimeseriesData()
td_pipes.add_branch_series(pipe_id, "on_off", switch_profile)

td_combined = td_loads + td_pipes
```

`extend()` merges a second `TimeseriesData` into an existing one.  For
duplicate (component, attribute) pairs the **existing value wins**.

---

## Querying results

### By model class

```python
df = result.get_result_for(mm.PowerLoad, "p_mw")
# DataFrame: rows = timesteps, columns = positional component index
```

### By component id

```python
s = result.get_result_for_id(load.id, "p_mw")
# pandas Series: index = step number (or datetime), values = p_mw per step
```

### Datetime index

Pass a `pd.DatetimeIndex` to `run_timeseries` to label results with real
timestamps:

```python
import pandas as pd

idx = pd.date_range("2024-01-01", periods=24, freq="h")
result = run_timeseries(net, td, datetime_index=idx)

df = result.get_result_for(mm.PowerLoad, "p_mw")
print(df.index)   # DatetimeIndex(['2024-01-01 00:00', '2024-01-01 01:00', ...])
```

---

## Error handling

By default the run raises immediately on any step failure.  Set
`on_step_error='skip'` to record the failure and continue:

```python
result = run_timeseries(net, td, on_step_error="skip")

print("Failed steps:", result.failed_steps)   # e.g. [3, 17]
for sr in result.step_results:
    if sr.failed:
        print(f"  step {sr.step}: {sr.error}")
```

The `StepResult` dataclass exposes `step`, `result`, `failed`, and `error`.

---

## Progress reporting

For long runs pass a callback that receives `(current_step, total_steps)`:

```python
from tqdm import tqdm

bar = tqdm(total=8760)
result = run_timeseries(
    net, td,
    progress_callback=lambda step, total: bar.update(1),
)
bar.close()
```

---

## Step hooks

Hooks let you inspect or modify the network copy before and after each solve.

### Class-based hook

```python
from monee.simulation import StepHook

class MyHook(StepHook):
    def pre_run(self, net, base_net, step, step_state):
        # Called after timeseries data is applied, before the solve.
        print(f"Step {step}: starting solve")

    def post_run(self, net, base_net, step, step_state, step_result):
        # Called after the solve (even on failure).
        if step_result.failed:
            print(f"Step {step}: FAILED — {step_result.error}")

result = run_timeseries(net, td, step_hooks=[MyHook()])
```

### Callable hook (post-step only)

```python
def log_step(net_copy, base_net, step):
    print(f"Step {step} done")

result = run_timeseries(net, td, step_hooks=[log_step])
```

Hooks also receive the `StepState` object, which holds solved values from the
previous timestep — useful for building custom inter-step logic inside a hook.

---

## Inter-step coupling: ramp constraints

Use `tracked` in place of `Var` to automatically carry a variable's solved
value into the next timestep.  Pair it with `inter_step_equations` to impose
constraints that link consecutive steps.

```python
from monee.model import tracked
from monee.model.child import PowerGenerator

class RampGenerator(PowerGenerator):
    """Generator with up/down ramp limits between consecutive timesteps."""

    def __init__(self, p_mw, ramp_up, ramp_down, **kwargs):
        super().__init__(p_mw, **kwargs)
        self.p_mw = tracked(p_mw, min=0.0, max=500.0)  # track across steps
        self.ramp_up   = ramp_up
        self.ramp_down = ramp_down

    def inter_step_equations(self, prev_state, component_id, **kwargs):
        prev_p = prev_state.get(component_id, "p_mw")
        if prev_p is None:
            return []   # first timestep — no previous value yet
        return [
            self.p_mw - prev_p <= self.ramp_up,
            prev_p - self.p_mw <= self.ramp_down,
        ]
```

Attaching it to the network:

```python
gen = RampGenerator(p_mw=100.0, ramp_up=20.0, ramp_down=30.0, q_mvar=0)
mx.create_el_child(net, gen, node_id=bus.id, name="ramp_gen")

result = run_timeseries(net, td)
```

The framework automatically:
1. Detects `tracked` Vars at injection time and records them.
2. Extracts their solved values after each step into `StepState`.
3. Passes `StepState` to `inter_step_equations` before the next solve.

No `inter_step_vars()` method is needed when using `tracked`.

---

## Multi-energy example

```python
# Gas network with varying demand
td_gas = TimeseriesData()
td_gas.add_child_series_by_name("industrial_sink", "mass_flow",
                                 [0.05, 0.08, 0.12, 0.10, 0.06])

# Coupled electricity network with varying load
td_el = TimeseriesData()
td_el.add_child_series_by_name("factory_load", "p_mw",
                                [0.4, 0.6, 0.9, 0.7, 0.5])

result = run_timeseries(mes_net, td_gas + td_el)

# Retrieve gas flow across all steps
gas_s = result.get_result_for_id(sink_id, "mass_flow")
# Retrieve electricity consumption
el_df = result.get_result_for(mm.PowerLoad, "p_mw")
```

---

## API reference

| Symbol | Description |
|---|---|
| `TimeseriesData` | Container for per-component time series |
| `TimeseriesData.from_dataframe(df, type, id/name)` | Build from a pandas DataFrame |
| `TimeseriesData.length` | Inferred step count from registered series |
| `run_timeseries(net, td, ...)` | Execute the timeseries simulation |
| `TimeseriesResult.get_result_for(ModelClass, attr)` | DataFrame: steps × components |
| `TimeseriesResult.get_result_for_id(id, attr)` | Series: step values for one component |
| `TimeseriesResult.failed_steps` | List of step indices that failed |
| `TimeseriesResult.step_results` | List of `StepResult` objects (all steps) |
| `StepResult` | Dataclass: `step`, `result`, `failed`, `error` |
| `StepHook` | Base class for pre/post step callbacks |
| `tracked` | `Var` subclass that participates in inter-step state |
