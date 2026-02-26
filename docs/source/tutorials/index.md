
# Tutorials

Step-by-step walkthroughs that take you from a blank script to a fully working
simulation or optimisation. Each tutorial builds on the {doc}`../quickstart`.

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} 01 · Minimum-cost load curtailment
:link: 01_optimization_basics
:link-type: doc
:shadow: sm

A feeder serves a factory and a warehouse.  An upstream fault caps supply at
0.6 MW.  Define controllables, a differentiated-cost objective, and a power
constraint — the optimiser sheds the cheapest load first.
:::

:::{grid-item-card} 02 · Solar feeder — day-ahead
:link: 02_timeseries_simulation
:link-type: doc
:shadow: sm

Simulate a residential bus with rooftop PV across eight three-hour slots.
Track the grid import "duck curve", query per-step voltages, and monitor
under-voltage events with a step hook.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

01_optimization_basics
02_timeseries_simulation
```
