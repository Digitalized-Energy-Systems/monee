
# Timeseries simulation

Drive a multi-energy network through a sequence of timesteps with time-varying
profiles, ramp constraints, and step hooks.

For the architectural background see {doc}`../concepts/timeseries`.

---

## Quick start

```{testcode}
import monee.model as mm
import monee.express as mx
from monee.simulation import TimeseriesData, run_timeseries

# 1. Build a simple two-bus power network
net = mx.create_multi_energy_network()
bus0 = mx.create_bus(net)
bus1 = mx.create_bus(net)
mx.create_ext_power_grid(net, bus0)
mx.create_line(net, bus0, bus1,
               length_m=1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
mx.create_power_load(net, bus1, p_mw=1.0, q_mvar=0.0, name="demand")

# 2. Define a 6-step load profile
td = TimeseriesData()
td.add_child_series_by_name("demand", "p_mw",
                             [0.4, 0.6, 1.0, 1.2, 0.9, 0.5])

# 3. Run
result = run_timeseries(net, td)
print(f"{len(result.raw)} steps,  {len(result.failed_steps)} failures")
```

```{testoutput}
6 steps,  0 failures
```

---

## Registering time series

::::{tab-set}

:::{tab-item} By name
Components added with a `name=` keyword can be referenced directly:

```{testcode}
from monee.simulation import TimeseriesData

td2 = TimeseriesData()
td2.add_child_series_by_name("demand",    "p_mw",  [0.4, 0.8, 1.2])
td2.add_branch_series_by_name("main_pipe","on_off", [1, 0, 1])
td2.add_compound_series_by_name("boiler", "regulation", [1.0, 0.5, 1.0])
```
:::

:::{tab-item} By id
Use the integer id returned when the component was added:

```{testcode}
import monee.model as mm
import monee.express as mx
from monee.simulation import TimeseriesData

net3 = mx.create_multi_energy_network()
b0 = mx.create_bus(net3)
b1 = mx.create_bus(net3)
mx.create_ext_power_grid(net3, b0)
mx.create_line(net3, b0, b1, length_m=500,
               r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
ld = mx.create_power_load(net3, b1, p_mw=1.0, q_mvar=0.0)

td3 = TimeseriesData()
td3.add_child_series(ld, "p_mw", [0.5, 0.9, 1.3])
```
:::

:::{tab-item} From DataFrame
```python
import pandas as pd
from monee.simulation import TimeseriesData

df = pd.read_csv("load_profile.csv")   # columns: p_mw, q_mvar
td = TimeseriesData.from_dataframe(
    df, component_type="child", component_name="demand"
)
```
`component_type` is one of `'node'`, `'child'`, `'branch'`, or `'compound'`.
:::

::::

### Merging

```{testcode}
from monee.simulation import TimeseriesData

td_a = TimeseriesData()
td_a.add_child_series_by_name("demand", "p_mw", [0.4, 0.8])

td_b = TimeseriesData()
td_b.add_branch_series_by_name("main_pipe", "on_off", [1, 0])

combined = td_a + td_b
print(combined.length)
```

```{testoutput}
2
```

For duplicate `(component, attribute)` pairs the conflict rules are:

- `td_a.extend(td_b)` merges in place - the **receiver** (`td_a`) **wins**.
- `td_a + td_b` builds a new object - the **left operand** (`td_a`) **wins**.

Both raise a `ValueError` if the two objects have different series lengths.

---

## Querying results

::::{tab-set}

:::{tab-item} By model class
```{testcode}
# DataFrame: rows = successful steps, columns = component ids
vm = result.get_result_for(mm.Bus, "vm_pu")
p  = result.get_result_for(mm.PowerLoad, "p_mw")
```
:::

:::{tab-item} By component id
```python
# pandas Series: one value per successful step
s = result.get_result_for_id(load.id, "p_mw")
```
:::

:::{tab-item} Datetime index
```python
import pandas as pd

idx = pd.date_range("2024-01-01", periods=6, freq="h")
result = run_timeseries(net, td, datetime_index=idx)
vm = result.get_result_for(mm.Bus, "vm_pu")
print(vm.index)   # DatetimeIndex
```
:::

::::

---

## Error handling

By default the runner raises immediately on any step failure.  Set
`on_step_error='skip'` to record failures and continue:

```{testcode}
result = run_timeseries(net, td, on_step_error="skip")

print("Failed steps:", result.failed_steps)
for sr in result.step_results:
    if sr.failed:
        print(f"  step {sr.step}: {sr.error}")
```

```{testoutput}
Failed steps: []
```

```{note}
Failed and skipped steps are excluded from all DataFrame queries
(`get_result_for`, `get_result_for_id`, `result[id]`);
{attr}`~monee.simulation.TimeseriesResult.failed_steps` lists the indices of
the steps that failed.
```

---

## Applying data without solving

Set `solve_flag=False` to run the loop without invoking the solver.  Each step
still copies the network, applies the timeseries data, and calls the step
hooks - the resulting `StepResult` objects are marked `skipped=True`.  This is
useful for dry-running profiles and hooks, or when an external process
performs the actual solve:

```{testcode}
result_dry = run_timeseries(net, td, solve_flag=False)
print(all(sr.skipped for sr in result_dry.step_results))
```

```{testoutput}
True
```

---

## Progress reporting

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

```{testcode}
from monee.simulation import StepHook

class LogHook(StepHook):
    def pre_run(self, base_net, step, step_state):
        pass   # called before the per-step copy and timeseries application

    def post_run(self, net, step, step_state, step_result, base_net):
        if step_result.failed:
            print(f"Step {step}: FAILED - {step_result.error}")
```

Plain callables in `step_hooks` are treated as post-run hooks and called with
the same arguments as `post_run`:

```{testcode}
def log_step(net_copy, step, step_state, step_result, base_net):
    pass   # net_copy is the solved network for this step
```

```{note}
Both `pre_run` and `post_run` receive the live `StepState` - hooks can read
or write inter-step values directly.
```

---

## Inter-step coupling: ramp constraints

Implement `inter_temporal_equations` to add constraints that link consecutive
steps.  The previous step's solved values are provided via
`temporal_state.get(component_id, attribute)`:

```{testcode}
from monee.model.core import Var, model
from monee.model.child import ChildModel

@model
class RampGenerator(ChildModel):
    """Generator with up/down ramp limits between consecutive timesteps."""

    def __init__(self, p_mw, ramp_up, ramp_down, **kwargs):
        super().__init__(**kwargs)
        self.p_mw      = Var(p_mw, min=0.0, max=500.0, name="p_mw")
        self.q_mvar    = 0.0
        self.ramp_up   = ramp_up
        self.ramp_down = ramp_down

    def inter_temporal_equations(self, temporal_state, component_id, **kwargs):
        prev_p = temporal_state.get(component_id, "p_mw")
        if prev_p is None:
            return []   # first step - no history yet
        return [
            self.p_mw - prev_p <= self.ramp_up,
            prev_p - self.p_mw <= self.ramp_down,
        ]
```

```{tip}
`inter_temporal_equations` is called in **both** timeseries and multi-period
solves.  The same model works in both contexts without any changes - see
{doc}`multi_period` for the multi-period workflow.
```

---

## Thermal and gas storage extensions

For network-wide time coupling - thermal inertia in junctions, gas stored in
pipelines - attach a {doc}`../concepts/network_aspects` extension before the
first run:

```{testcode}
import monee.model as mm
import monee.express as mx
from monee.model import LumpedThermalCapacitance, GasLinepack

net_ext = mx.create_multi_energy_network()

# Water side
jw0 = mx.create_water_junction(net_ext)
jw1 = mx.create_water_junction(net_ext)
mx.create_ext_hydr_grid(net_ext, jw0)
mx.create_water_pipe(net_ext, jw0, jw1, diameter_m=0.3, length_m=500)

# Gas side
jg0 = mx.create_gas_junction(net_ext)
jg1 = mx.create_gas_junction(net_ext)
mx.create_gas_ext_grid(net_ext, jg0)
pipe_ext_id = mx.create_gas_pipe(net_ext, jg0, jg1,
                                 diameter_m=0.4, length_m=20_000)

net_ext.add_extension(LumpedThermalCapacitance())
net_ext.add_extension(GasLinepack(overrides={
    pipe_ext_id: dict(linepack_kg_initial=1_000, linepack_kg_max=5_000)
}))
```

See {doc}`../concepts/temporal_extensions` for step-by-step walkthroughs.

---

## Externally paced simulation

`run_timeseries` owns the time loop: it iterates over a fixed number of
equally indexed steps.  When an external process drives the clock - a
co-simulation framework, a real-time loop, an event-based scheduler - use
{class}`~monee.simulation.Conductor` instead.  It keeps a persistent
`StepState` and lets you call `step(dt_h)` whenever *you* decide, with a
variable step size and per-step data overrides.  See {doc}`conductor` for the
full workflow.

---

## API reference

### TimeseriesData

| Symbol | Description |
|---|---|
| `TimeseriesData()` | Container for per-component time series |
| `td.add_child_series(id, attr, values)` | Register series by component id |
| `td.add_child_series_by_name(name, attr, values)` | Register series by component name |
| `td.add_branch_series(id, attr, values)` | Register series for a branch |
| `td.add_branch_series_by_name(name, attr, values)` | Register series for a branch by name |
| `td.add_node_series(id, attr, values)` | Register series for a node |
| `td.add_compound_series(id, attr, values)` | Register series for a compound |
| `td.add_compound_series_by_name(name, attr, values)` | Register series for a compound by name |
| `TimeseriesData.from_dataframe(df, type, id/name)` | Build from a pandas DataFrame |
| `td.length` | Inferred step count |
| `td_a + td_b` | Merge two `TimeseriesData` objects (left wins on conflicts) |
| `td.extend(other)` | In-place merge |

### run_timeseries

| Symbol | Description |
|---|---|
| `run_timeseries(net, td, ...)` | Execute the timeseries simulation |
| `steps=None` | Number of steps; defaults to `td.length`.  Raises if neither is given or `steps` exceeds the registered series length |
| `step_hooks=None` | List of `StepHook` instances; plain callables are treated as post-run hooks |
| `solver=None` / `backend=None` | Solver name or instance, and `'gekko'`/`'pyomo'` backend (default: GEKKO + IPOPT) |
| `optimization_problem=None` | `OptimizationProblem` applied at every step |
| `solve_flag=True` | If `False`, apply data and run hooks without solving; steps are marked `skipped=True` |
| `on_step_error='raise'` | `'raise'` (default) or `'skip'` to record failures and continue |
| `progress_callback=None` | Callable `(step, steps)` invoked after each step |
| `datetime_index=None` | `pandas.DatetimeIndex` labelling result rows and setting `step_state.dt_h` per step |

### Results

| Symbol | Description |
|---|---|
| `result.get_result_for(ModelClass, attr)` | DataFrame (steps × component ids) |
| `result.get_result_for_id(id, attr)` | Series: one value per successful step |
| `result[component_id]` | DataFrame of all attributes for one component |
| `result.failed_steps` | List of step indices that failed to converge |
| `result.step_results` | List of `StepResult` objects (incl. failed/skipped steps) |
| `result.raw` | List of successful `SolverResult` objects (backward compat) |

Failed and skipped steps are excluded from all DataFrame queries.

### StepResult

| Attribute | Description |
|---|---|
| `step` | Zero-based step index |
| `result` | `SolverResult` for this step, or `None` if failed/skipped |
| `failed` | `True` if the solve raised an exception |
| `skipped` | `True` if the solve was not attempted (e.g. `solve_flag=False`) |
| `error` | The exception that caused the failure, or `None` |

### StepHook

| Symbol | Description |
|---|---|
| `StepHook` | Base class for pre/post step callbacks |
| `hook.pre_run(base_net, step, step_state)` | Called before the per-step copy and timeseries application |
| `hook.post_run(net, step, step_state, step_result, base_net)` | Called after each step's solve (success or failure) |

Plain callables `(net, step, step_state, step_result, base_net) -> None` are
accepted in `step_hooks` and treated as post-run hooks without subclassing.

### StepState

`StepState` is passed to `inter_temporal_equations` and to hooks.

| Symbol | Description |
|---|---|
| `state.get(component_id, attr, step=-1)` | Float from a prior solved step (`-1` = most recent) |
| `state.has(component_id, attr)` | `True` if a non-`None` value is available |
| `state.dt_h` | Duration of the current timestep in hours |
