
# Tutorials

Step-by-step walkthroughs that take you from a blank script to a fully working
simulation or optimisation. Each tutorial builds on the {doc}`../quickstart`.

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} 01 · Optimisation basics
:link: 01_optimization_basics
:link-type: doc
:shadow: sm

Formulate a **load-shedding** optimisation problem from scratch: define
controllables, voltage bounds, an objective, and constraints — then run and
inspect the results.
:::

:::{grid-item-card} 02 · Time-series simulation
:link: 02_timeseries_simulation
:link-type: doc
:shadow: sm

Drive a network through a sequence of time steps with varying load profiles,
collect per-step results, and inject custom logic with step hooks.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

01_optimization_basics
02_timeseries_simulation
```
