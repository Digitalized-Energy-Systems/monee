
# Benchmark & Validation

Two questions: how fast is monee, and can you trust its answers. Both are
backed by reproducible scripts in
[`benchmarks/`](https://github.com/Digitalized-Energy-Systems/monee/tree/development/benchmarks),
and every figure below is regenerated from a saved CSV. Only the solve call is
timed (networks are built before the timer starts). Every case carries a
cross-tool or cross-backend agreement check, so the timings always compare
identical solutions.

## Results at a glance

monee's nonlinear models reproduce the established reference tools on
like-for-like networks:

| Problem | vs reference | Agreement | Speed |
|---|---|---|---|
| AC power flow | pandapower `runpp` | voltage ≤ 4e-3 pu, slack power ≤ 5e-4 MW | ~1.4 to 4× slower (specialised NR wins) |
| AC OPF / dispatch | pandapower `runopp` | voltage ≤ 3e-8 pu, cost ≤ 4e-7 rel. | often *faster* |
| Gas flow | pandapipes `pipeflow` | pressure within 1.2 % of drop | ~1.3 to 7× slower |
| Water / heat flow | pandapipes `pipeflow` | pressure ~0.1 %, temperature ≤ 8.6e-3 K | ~1.4 to 23× slower |
| Coupled MES (P2G, G2P, CHP) | pandapipes `multinet` | voltage ~1e-11 pu, CHP ΔT ≤ 0.07 K | *1.4 to 5× faster* |

The pattern: on a plain single-sector power flow, a dedicated Newton or
hydraulic solver beats a general NLP, as expected. But once the problem becomes
an optimisation or a multi-energy coupling, monee's generality starts to pay
off. It solves one simultaneous NLP instead of an iterative inter-sector loop.

For a given formulation, the in-process backends win:

| Formulation class | Fastest backend | vs alternative | Agreement |
|---|---|---|---|
| Smooth NLP (`*_NLP`, `SMOOTH_NLP`) | CasADi / IPOPT | 3.5 to 18× faster than GEKKO | ≤ 1e-8 |
| (MI)QCQP / MISOCP / MILP | native gurobipy | 2 to 7× faster than Pyomo + Gurobi | ≤ 1e-4 |

## Dive deeper

::::{grid} 1 2 2 2
:gutter: 4

:::{grid-item-card} Validation
:link: validation
:link-type: doc
:shadow: sm

The full pandapower and pandapipes comparison: per-case voltage, pressure,
temperature and power agreement for power flow, OPF and coupled multi-energy,
with the figures behind the table above.
:::

:::{grid-item-card} Choosing a backend
:link: backend_selection
:link-type: doc
:shadow: sm

Which numerical backend is fastest for each formulation class, the head-to-head
timings behind the recommendation, and the one keyword that selects each one.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

validation
backend_selection
```
