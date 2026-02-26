
# Timeseries simulation

The timeseries module drives a network through a sequence of timesteps, each
solved independently.  This page explains the architecture, data flow, and the
mechanisms for coupling consecutive timesteps.

---

## Sequential solve architecture

Each timestep follows a fixed four-phase pipeline:

```
for step in range(steps):
    1. Copy base network
    2. Inject timeseries values into model attributes
    3. Solve (energy flow / optimisation)
    4. Extract inter-step state
```

**Copy** — the base network is never modified.  A fresh `Network.copy()` is
taken each step so that model attributes set in step *k* do not bleed into
step *k+1*.

**Inject** — `TimeseriesData` writes scalar values (`series[step]`) directly
onto the model's Python attributes before the solver runs.  This is identical
to setting the attribute by hand before calling `solve()`; the solver picks up
whatever value is on the model at injection time.

**Solve** — the copied-and-modified network is passed to the solver exactly as
in a single-step call.  Any optimisation problem or custom solver can be used.

**Extract** — after the solve, `StepState` records the solved float values of
all *tracked* attributes, making them available as constants in the next step's
equation set.

---

## TimeseriesData

`TimeseriesData` maps each `(component, attribute)` pair to a list of values,
one value per timestep.

```
component type   lookup method          internal storage
─────────────────────────────────────────────────────────
node             id                     _node_id_to_series
child            id  /  name            _child_id_to_series / _child_name_to_series
branch           id  /  name            _branch_id_to_series / _branch_name_to_series
compound         id  /  name            _compound_id_to_series / _compound_name_to_series
```

All series registered on a single `TimeseriesData` object must have the same
length.  The length is inferred from the first registration and validated on
every subsequent `add_*_series` call.  Mismatches raise `ValueError` early —
before the simulation loop — so failures are visible at setup time.

### Application order

Within a step, attributes are applied in this order:

1. Node model attributes (id-based)
2. Child model attributes (id-based, then name-based)
3. Branch model attributes (id-based, then name-based)
4. Compound model attributes (id-based, then name-based)

If both an id-series and a name-series exist for the same component and
attribute, the name-series is applied second and wins.

### Merging

Two `TimeseriesData` objects are merged with `extend()` or `+`.  The merge
is **attribute-level**: for each `(component_id, attribute)` pair the
receiver's value wins on conflict, and only attributes absent from the receiver
are taken from the argument.

---

## StepResult and TimeseriesResult

`run()` returns a `TimeseriesResult` that wraps a `list[StepResult]`:

```
TimeseriesResult
├── step_results: list[StepResult]
│   ├── StepResult(step=0, result=SolverResult, failed=False)
│   ├── StepResult(step=1, result=None, failed=True, error=RuntimeError(...))
│   └── StepResult(step=2, result=SolverResult, failed=False)
├── failed_steps → [1]
└── raw → [SolverResult(step 0), SolverResult(step 2)]
```

`get_result_for(ModelClass, attribute)` builds a pandas DataFrame by
assembling the per-step `SolverResult.dataframes`.  Only **successful** steps
contribute rows; failed steps are silently excluded.

`get_result_for_id(component_id, attribute)` searches all type DataFrames in
each successful step for the given component id and returns a `Series` with
one value per successful step.

---

## Inter-step coupling: StepState

By default each step is fully independent — a component's solved output at
step *k* has no influence on step *k+1*.  `StepState` bridges this gap.

`StepState` is a `dict[(component_id, attribute) → float]` that:

* Is populated after each step's solve by scanning the network for tracked
  attributes.
* Is passed to every model's `inter_step_equations()` method before the next
  step's equation set is built.
* Stores only plain Python floats — solver-library objects are never carried
  across steps.

Because `StepState` is keyed by `(component_id, attribute)`, values survive
the network copy: the copied component retains the same id as the base
component.

---

## Declaring tracked variables

### `tracked` Var (recommended)

Replace `Var` with `tracked` in the model's `__init__`:

```python
from monee.model import tracked

class StorageModel(ChildModel):
    def __init__(self, soc_init):
        self.soc = tracked(soc_init, min=0.0, max=1.0)
        self.charge_mw = Var(0.0, min=0.0, max=100.0)
```

During solver injection the framework scans the model's `__dict__` for
`tracked` instances before they are replaced by solver variables.  Their
attribute names are stored in `model._inter_step_attrs`.  After the solve and
withdrawal phase `_extract_step_state` reads those names from `_inter_step_attrs`
and records the solved values in `StepState`.

No `inter_step_vars()` method is needed.

### `inter_step_vars()` (legacy)

For models that pre-date `tracked`, declaring the method is still supported:

```python
def inter_step_vars(self):
    return ["soc"]
```

Both protocols can coexist; duplicates are de-duplicated automatically.

---

## Writing inter-step equations

Implement `inter_step_equations` on any model or formulation to add
constraints that link the current solve to the previous step:

```python
def inter_step_equations(self, prev_state, component_id, **kwargs):
    prev_soc = prev_state.get(component_id, "soc")
    if prev_soc is None:
        return []   # first timestep — no history yet
    dt = 1.0        # hours
    return [
        self.soc == prev_soc + (self.charge_mw - self.discharge_mw) * dt / self.capacity_mwh
    ]
```

`prev_state.get(component_id, attr)` returns `None` on the first timestep
(nothing has been recorded yet) and a plain float on subsequent ones.  The
returned expressions are added to the solver's equation set alongside the
model's regular equations.

The method signature accepts `**kwargs` because formulations receive additional
keyword arguments (grid, network, …).

---

## StepHook execution points

```
─── step k ──────────────────────────────────────────────────────────────────
 net_copy = net.copy()
 timeseries_data.apply_to_network(net_copy, k)
 → StepHook.pre_run(net_copy, net, k, step_state)
 result = solve(net_copy, ..., step_state=step_state)
 _extract_step_state(step_state, result.network)
 → StepHook.post_run(net_copy, net, k, step_state, step_result)
─────────────────────────────────────────────────────────────────────────────
```

`pre_run` sees the network after timeseries data has been applied but before
the solve.  `post_run` sees both the solved network copy and the `StepResult`
(including the `SolverResult` with its dataframes).  Both callbacks receive the
live `StepState`, so a hook can read or write inter-step values.

Plain callables registered as hooks are invoked only in the `post_run`
position with the signature `(net_copy, base_net, step)`.

---

## Solver warm-starting (note)

The current architecture does not automatically feed the previous step's solved
values back as initial variable guesses for the next step.  Each step starts
from the model's original `Var.value` defaults.  For slowly-varying profiles
this is usually fine; for highly nonlinear networks with rapidly changing
setpoints you may want to implement warm-starting logic inside a `StepHook`
by manually updating `Var.value` on the base network after each step.

---

## Complexity and scalability

| Factor | Impact |
|---|---|
| Steps | Linear — each step is one independent solve |
| Network size | Same as single-step; memory proportional to `steps × result_size` |
| Inter-step equations | Adds O(tracked vars) scalar constraints per step; negligible overhead |
| Failed steps (`on_step_error='skip'`) | Does not affect subsequent steps — `StepState` is not updated on failure |
