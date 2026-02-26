
# Concepts

These pages explain the ideas and architecture behind monee — from how
networks are represented in memory to the physical equations used for each
energy carrier and the solver interfaces that bring it all together.

::::{grid} 1 2 2 3
:gutter: 4

:::{grid-item-card} Data model
:link: data_model
:link-type: doc
:shadow: sm

How monee represents networks as directed graphs: nodes, branches, children,
grids, variables, and parameters — and how to build custom components.
:::

:::{grid-item-card} Physical models
:link: domains
:link-type: doc
:shadow: sm

The steady-state equations for **electricity** (AC power flow), **gas**
(Weymouth), and **water / heat** (Darcy–Weisbach) networks.
:::

:::{grid-item-card} Multi-energy coupling
:link: multi_energy
:link-type: doc
:shadow: sm

All built-in coupling components — P2H, P2G, G2P, G2H, CHP, and heat
exchanger — and how to dispatch them in an optimisation.
:::

:::{grid-item-card} Formulations
:link: formulations
:link-type: doc
:shadow: sm

The formulation layer: how equation sets are mapped to model types, what
built-in formulations ship with monee, and how to write a custom one.
:::

:::{grid-item-card} Solvers
:link: solvers
:link-type: doc
:shadow: sm

GEKKO vs Pyomo: capabilities, limitations, and guidance on choosing the
right back-end for each type of problem.
:::

:::{grid-item-card} Islanding
:link: islanding
:link-type: doc
:shadow: sm

Solve networks with multiple disconnected islands: the connectivity-flow
MIP formulation, grid-forming nodes, and per-carrier physical constraints.
:::

:::{grid-item-card} Timeseries simulation
:link: timeseries
:link-type: doc
:shadow: sm

Sequential solve architecture, `TimeseriesData`, `StepState`, inter-step
coupling with `tracked` Vars, and ramp constraints.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

data_model
domains
multi_energy
formulations
solvers
islanding
timeseries
```
