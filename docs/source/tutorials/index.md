
# Tutorials

Step-by-step walkthroughs that take you from a blank script to a fully working
simulation or optimisation. Each tutorial builds on the {doc}`../quickstart`
and stands on its own, except {doc}`mes_storage_and_ramps`,
{doc}`mes_network_limits` and {doc}`mes_local_optima`, which continue the
town of
{doc}`the multi-energy economic dispatch tutorial <04_mes_economic_dispatch>`;
read that one first, and {doc}`mes_network_limits` before
{doc}`mes_local_optima`.

::::{grid} 1 1 2 2
:gutter: 4

:::{grid-item-card} Minimum-cost load curtailment
:link: 01_optimization_basics
:link-type: doc
:shadow: sm

A feeder serves a factory and a warehouse.  An upstream fault caps supply at
0.6 MW.  Define controllables, a differentiated-cost objective, and a power
constraint, then let the optimiser shed the cheapest load first.
:::

:::{grid-item-card} Solar feeder: day-ahead
:link: 02_timeseries_simulation
:link-type: doc
:shadow: sm

Simulate a residential bus with rooftop PV across eight three-hour slots.
Track the grid import "duck curve", query per-step voltages, and monitor
under-voltage events with a step hook.
:::

:::{grid-item-card} Coupled network: CHP dispatch
:link: 03_coupled_network
:link-type: doc
:shadow: sm

Wire electricity, gas and heat into one network with a CHP and a power to
heat unit, solve the coupled energy flow, then dispatch both couplers under
a capped grid connection, with the sign, wiring and formulation conventions
spelled out at each step.
:::

:::{grid-item-card} Multi-energy economic dispatch
:link: 04_mes_economic_dispatch
:link-type: doc
:shadow: sm

Dispatch a CHP, an electric boiler and an electrolyser against hourly power
prices, then check the result: the merit order against hand computed
breakeven prices, every energy balance, and the saving against two
baselines.
:::

:::{grid-item-card} Storage and ramp limits
:link: mes_storage_and_ramps
:link-type: doc
:shadow: sm

Continue the multi-energy economic dispatch town with a battery and ramp
limits. Both couple the hours, so the day is solved jointly with
`run_multi_period`; compare it with the hour by hour dispatch and check the
battery's bookkeeping, the ramp limits and the value of foresight against a
copper plate model.
:::

:::{grid-item-card} Network limits and congestion
:link: mes_network_limits
:link-type: doc
:shadow: sm

Continue the multi-energy economic dispatch town: rate the cable to its plant
site and narrow its gas pipe until both bind. See in which hours they bind,
why the dispatch leaves the merit order, what each limit costs and what one
more MW through each bottleneck is worth, all checked against hand
calculations.
:::

:::{grid-item-card} Local optima and global solvers
:link: mes_local_optima
:link-type: doc
:shadow: sm

Continue the network limits town: compare the local solvers APOPT and IPOPT
one to one with the global solver SCIP, see them stop at dispatches that are
not optimal while reporting success, trace the stop to the zero-flow corner
of a feed-in exchanger, and get the proven optimum from SCIP on the exact
quadratic formulation, with a lower bound from a relaxation.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

01_optimization_basics
02_timeseries_simulation
03_coupled_network
04_mes_economic_dispatch
mes_storage_and_ramps
mes_network_limits
mes_local_optima
```
