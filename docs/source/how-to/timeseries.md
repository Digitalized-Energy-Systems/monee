
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

A `name=` value lands on the Component container, not on the model object, so
read it back from the container (`net.child_by_id(load_id).name`) or from the
`name` column a result frame carries once a component of that type was named.
`component.model.name` raises an
`AttributeError`, because the model carries only the physics attributes; see
{doc}`../concepts/data_model` for the full container/model split.
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

df = pd.read_csv("load_profile.csv")  # columns: p_mw, q_mvar
td = TimeseriesData.from_dataframe(df, component_type="child", component_name="demand")
```
`component_type` is one of `'node'`, `'child'`, `'branch'`, or `'compound'`.
:::

::::

### Series must target settable inputs

A series drives a model attribute as an input, so it should target a plain
float attribute (such as `PowerLoad.p_mw`) or a `Const`. A series registered
on a solver `Var` attribute does not act as an input in a simulation run: the
solver owns the value, the series only pins the Var through its bounds, and
the model stays non-square, so every step reports a dof finding.
`run_timeseries` warns once at run start in that case, naming the attribute
and listing the component's settable inputs. Declare the attribute as a plain
float for simulation runs, or pass an optimization problem that controls the
Var.

The common case is a heat demand profile: a heat exchanger's `q_mw` is a
solved output, so drive `regulation` (variable flow) or `q_mw_set` (constant
flow) instead. {ref}`Part-load control <dh-part-load>` in the district
heating guide shows both patterns end to end.

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

- `td_a.extend(td_b)` merges in place: the receiver (`td_a`) wins.
- `td_a + td_b` builds a new object: the left operand (`td_a`) wins.

Objects of unequal length merge by truncating every series (on both sides) to
the shorter length, with a single warning naming both lengths. So a
short study profile merges directly into a year-long imported object without
tiling it first; call `head(n)` or `slice(start, stop)` beforehand to choose
the window yourself and silence the warning.

### Windowing

`slice(start, stop)` and `head(n)` return a new `TimeseriesData` with every
registered series cut to the same positional window. Use them to fit a long
profile (a year of quarter hours, say) to a shorter study, or to make two
objects of different lengths mergeable:

```{testcode}
from monee.simulation import TimeseriesData

year = TimeseriesData()
year.add_child_series_by_name("demand", "p_mw",
                              [0.4 + 0.01 * i for i in range(96)])

day = year.head(24)          # steps 0 to 23
window = year.slice(24, 48)  # steps 24 to 47
print(day.length, window.length)
```

```{testoutput}
24 24
```

### Length checks against the run horizon

The step count of a run must fit the registered series:

- `steps` larger than the series length raises a `ValueError` up front.
- `steps` smaller than the series length runs the requested window: an
  explicit `steps=` is a deliberate truncation, so the run only logs a note
  (at INFO level) that the rows past `steps` are ignored. Use `head(steps)` or
  `slice(start, stop)` when you want the truncation visible in the data
  itself. A length mismatch created by merging unequal series still warns at
  merge time.
- A `Stepper` raises a `ValueError` when `step(ts_index=...)` points past the
  end of the registered series.

### Series that bind to nothing

A registered id or name that matches no component of the network would
otherwise be silently ignored for the whole run. `run_timeseries`,
`run_multi_period`, `run_mpc` and the `Stepper` constructor therefore report
every unmatched key once, up front, as a `UserWarning`. To turn the check into
an error, call `td.validate_bound(net)` yourself before the run; it raises a
`KeyError` listing the unmatched keys.

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
s = result.get_result_for_id(load.id, "p_mw", mm.PowerLoad)

# external grids collide with the node they sit on: child 0 vs node 0
p_ext = result.get_result_for_id(ext_id, "p_mw", mm.ExtPowerGrid)
m_ext = result.get_result_for_id(gas_ext_id, "mass_flow_kgs", mm.ExtHydrGrid)
```

Component ids are handed out per category, so the same number addresses one
node, one child and one branch. Pass the model class (or `result[id, mm.PowerLoad]`
for the full row) whenever an id is shared; a lookup that matches several result
tables raises a `ValueError` naming them and the `model_type=` form that picks
one. Passing the class is always safe, so these examples do it everywhere the
queried id belongs to a child. The class is the model class, not the container:
`mm.ExtPowerGrid` for an electrical slack and `mm.ExtHydrGrid` for a gas one.
:::

:::{tab-item} Datetime index
```python
import pandas as pd

idx = pd.date_range("2024-01-01", periods=6, freq="h")
result = run_timeseries(net, td, datetime_index=idx)
vm = result.get_result_for(mm.Bus, "vm_pu")
print(vm.index)  # DatetimeIndex
```
:::

::::

:::{admonition} Timeseries frames are keyed by id along the columns
:class: warning

A timeseries frame spans the run, so its rows are the steps and its *columns*
are the component ids: `result.get_result_for(mm.Bus, "vm_pu")[node_id]` is a
column lookup and gives that bus over time.

A snapshot frame is laid out the other way round. The per step
`SolverResult` frames (`result.step_results[i].result.get(mm.Bus)`, and the
frames from `run_energy_flow`) have one row per component under a positional
`RangeIndex`, with the id in an `id` column, so `df.loc[node_id]` there selects
a row number and quietly returns a different bus. Ask for the id-indexed view
when you want to join a snapshot frame against an id map:

```python
step_df = result.step_results[0].result.get(mm.Bus, index="id")
vm_at_node = step_df.loc[node_id, "vm_pu"]
```

The same holds for `result.raw`, which is a list of `SolverResult` objects.
See {doc}`../concepts/data_model` for the two index modes and for branch ids,
which are `(from_node_id, to_node_id, key)` tuples.
:::

---

## Step timing: dt_h and datetime_index

`dt_h` sets a constant inter-step interval in hours (default 1.0), seen by
inter-step dynamics as `step_state.dt_h`. A `datetime_index` overrides it
with per-step intervals derived from consecutive timestamps, and labels the
result rows.

The convention: a label marks the end of the interval it integrates over.
Step `k` integrates over `index[k] - index[k - 1]` hours; the first step has
no predecessor and borrows the first difference, so `dt_0 = index[1] -
index[0]`. The same rule is used by `run_multi_period`, and the `Stepper`
follows it in reverse when you build an index from its per-step `dt_h`
values (see {doc}`stepper`).

An irregular grid works directly. Here the first four steps are quarter
hours and the last two are full hours:

```{testcode}
import pandas as pd

idx = pd.DatetimeIndex(
    ["2024-01-01 00:15", "2024-01-01 00:30", "2024-01-01 00:45",
     "2024-01-01 01:00", "2024-01-01 02:00", "2024-01-01 03:00"]
)
result_irregular = run_timeseries(net, td, datetime_index=idx)
print(len(result_irregular.raw))
```

```{testoutput}
6
```

Step 0 and step 1 both integrate over 0.25 h (step 0 borrows the first
difference), steps 4 and 5 over 1.0 h each. The index must be strictly
increasing and cover at least `steps` entries.

---

## Error handling

By default the runner raises immediately on any step failure. Set
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

Set `solve_flag=False` to run the loop without invoking the solver. Each step
still copies the network, applies the timeseries data, and calls the step
hooks, but the resulting `StepResult` objects are marked `skipped=True`. Use this
to dry-run profiles and hooks, or when an external process performs the solve:

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
    net,
    td,
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
Both `pre_run` and `post_run` receive the live `StepState`; hooks can read
or write inter-step values directly.
```

---

## Inter-step coupling: ramp constraints

Implement `inter_temporal_equations` to add constraints that link consecutive
steps. On a model class (any `ChildModel`, `BranchModel`, or node model) the
hook has the signature

```python
def inter_temporal_equations(self, temporal_state, component_id, **kwargs): ...
```

and returns a list of equations. Network-wide extensions implement a
different, aspect-level variant instead; see
{doc}`../concepts/network_aspects`. The previous step's solved values are
provided via `temporal_state.get(component_id, attribute)`:

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
`inter_temporal_equations` is called in both timeseries and multi-period
solves.  The same model works in both contexts without any changes; see
{doc}`multi_period` for the multi-period workflow.
```

### Prescribed replay semantics

A registered series overrides the attribute it targets at the start of every
step. That attribute is then no longer decided by the solve, so any constraint
that depends only on it is decided before the solve too. Two rules follow.

Constraints over solver variables are enforced. A `Var` attribute (`p_mw` on
`RampGenerator` above) is still solved for, so its `equations` and its
`inter_temporal_equations` bind the step exactly as they would without a
series, and a profile the model cannot follow makes the step fail.

Constraints over prescribed values alone are overridden, not enforced. When
every term of an `inter_temporal_equations` entry is a prescribed value or a
value replayed from the previous step, the constraint is a constant. The solve
cannot repair it, so the backends drop it and the step still succeeds with the
prescribed trajectory. monee reports the dropped constraint instead of leaving
it silent: the step's `SolverResult.warnings` gains a `temporal` entry naming
the component, the equation index and the step, and the run emits one Python
warning per affected step. Passing `strict=True` to `run_timeseries` turns the
finding into a `ValidationError`, the same way a strict solve reports its own
validation findings (see {doc}`diagnose_infeasibility`).

If you want a profile to be checked against a ramp limit rather than to
override it, declare the attribute as a `Var` and drive the profile through an
objective or a bound instead of a prescribed series.

---

## Thermal and gas storage extensions

For network-wide time coupling (thermal inertia in junctions, gas stored in
pipelines) attach a {doc}`../concepts/network_aspects` extension before the
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
equally indexed steps. When an external process drives the clock (a
co-simulation framework, a real-time loop, an event-based scheduler) use
{class}`~monee.simulation.Stepper` instead. It keeps a persistent
`StepState` and lets you call `step(dt_h)` on demand, with a variable step
size and per-step data overrides. See {doc}`stepper` for the full workflow.

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
| `td.slice(start, stop)` | New object with every series cut to `[start, stop)` |
| `td.head(n)` | New object with the first `n` steps of every series |
| `td.validate_bound(net)` | Raise `KeyError` for series that match no component |

### run_timeseries

| Symbol | Description |
|---|---|
| `run_timeseries(net, td, ...)` | Execute the timeseries simulation |
| `steps=None` | Number of steps; defaults to `td.length`.  Raises if neither is given or `steps` exceeds the registered series length |
| `step_hooks=None` | List of `StepHook` instances; plain callables are treated as post-run hooks |
| `solver=None` / `backend=None` | Solver name or instance, and `'casadi'`, `'gekko'`, `'pyomo'`, or `'gurobipy'` backend. Default is IPOPT on CasADi when installed, otherwise GEKKO's bundled IPOPT |
| `optimization_problem=None` | `OptimizationProblem` applied at every step |
| `solve_flag=True` | If `False`, apply data and run hooks without solving; steps are marked `skipped=True` |
| `on_step_error='raise'` | `'raise'` (default) or `'skip'` to record failures and continue |
| `progress_callback=None` | Callable `(step, steps)` invoked after each step |
| `datetime_index=None` | `pandas.DatetimeIndex` labelling result rows and setting `step_state.dt_h` per step |
| `dt_h=None` | Constant inter-step interval in hours (default 1.0); ignored when `datetime_index` is given |
| `simulation=True` | Square steady-state simulation, the default without an `optimization_problem`, so a single step matches `run_energy_flow`. Pass `False` for the optimize-the-feasibility-problem path |
| `**solver_kwargs` | Forwarded to `solver.solve(...)`, e.g. `formulation="smooth_nlp"` |

### Results

| Symbol | Description |
|---|---|
| `result.get_result_for(ModelClass, attr)` | DataFrame: rows are steps, columns are component ids |
| `result.get_result_for_id(id, attr, model_type=None)` | Series: one value per successful step; `model_type` picks the category when the id is shared |
| `result[component_id]` | DataFrame of all attributes for one component; use `result[id, ModelClass]` when the id is shared |
| `result.failed_steps` | List of step indices that failed to converge |
| `result.step_results` | List of `StepResult` objects (incl. failed/skipped steps) |
| `result.raw` | List of successful `SolverResult` objects (backward compat) |

Failed and skipped steps are excluded from all DataFrame queries.

A per step `SolverResult` frame is indexed the other way round from the tables
above: rows are components under a positional index, with the id in an `id`
column. Use `solver_result.get(ModelClass, index="id")` to index those rows by
component id.

### StepResult

| Attribute | Description |
|---|---|
| `step` | Zero-based step index |
| `result` | `SolverResult` for this step, or `None` if failed/skipped |
| `failed` | `True` if the solve raised an exception |
| `skipped` | `True` if the solve was not attempted (e.g. `solve_flag=False`) |
| `error` | The exception that caused the failure, or `None` |
| `dt_h` / `effective_dt_h` / `t_h` | Step interval, integrated interval and clock time; set by the {class}`~monee.simulation.Stepper`, `None` here |

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
