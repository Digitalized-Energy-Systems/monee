# Choosing a solver backend

monee keeps the formulation (the equations) separate from the backend (the
numerical engine that solves them). The same network and formulation can be
handed to several backends, so the practical question is which backend is
fastest for the formulation class you picked. This page answers that with a
head-to-head benchmark.

The script
[`benchmarks/backend_comparison.py`](https://github.com/Digitalized-Energy-Systems/monee/blob/development/benchmarks/backend_comparison.py)
runs two shoot-outs across representative cases (power flow, OPF, multi-sector,
timeseries with inter-step coupling). Only the `run_energy_flow`,
`run_energy_flow_optimization` or `run_timeseries` call is timed; every network
is built before the timer starts. Each case is solved on both backends, and a
cross-backend agreement metric confirms they reach the same solution before the
timings are compared.

````{only} html
```{raw} html
<iframe src="../_static/interactive/benchmark_backend.html" width="100%" height="780" style="border:none;" loading="lazy" title="monee backend performance comparison: GEKKO vs CasADi on smooth NLP and Pyomo/Gurobi vs native gurobipy on (MI)QCQP"></iframe>
```
````

````{only} latex
```{figure} /_static/interactive/benchmark_backend.png
:alt: monee backend performance comparison, GEKKO vs CasADi on smooth NLP and Pyomo/Gurobi vs native gurobipy on (MI)QCQP
:width: 100%

Backend shoot-outs. Top (Group A): GEKKO vs the in-process CasADi backend on the
smooth-NLP formulations. Bottom (Group B): Pyomo/Gurobi vs the native gurobipy
backend on the (MI)QCQP and MISOCP formulations. Left: solve time per case (log
scale); right: speedup of the native backend. Across every case the two backends
agree to ≤ 1e-4 (most to ≤ 1e-8), so the speedup is a like-for-like comparison.
```
````

## The two families

Backends split along the same line the formulations do. Smooth nonlinear
programs go to an interior-point NLP solver; everything with integer variables
or conic/quadratic structure goes to a branch-and-bound (MI)QCQP solver.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Smooth NLP, use CasADi / IPOPT
:shadow: sm

For the pure-NLP formulations (`EL_NLP`, `GAS_NLP`, `HEAT_NLP`, `SMOOTH_NLP`).
The in-process CasADi backend solves the same NLP as GEKKO but 3.5 to 18 times
faster (Group A), because it builds the model and calls IPOPT in-process with no
subprocess or file I/O. It is the default for IPOPT requests.

GEKKO remains a fine choice. It ships its own IPOPT/APOPT binaries (zero extra
install) and APOPT handles MINLP, but it is consistently slower here.
:::

:::{grid-item-card} (MI)QCQP / MISOCP / MILP, use Gurobi
:shadow: sm

For the relaxation and exact-quadratic formulations (`EL_MISOCP`,
`*_CONVEX_MIQCQP`, `*_NONCONVEX_MIQCQP`, `HEAT_CONVEX_MILP`, the gas/heat PWL
variants). The native gurobipy backend is 2 to 7 times faster than driving the
same Gurobi through Pyomo (Group B), as it skips Pyomo's model-translation layer.

The Pyomo backend trades that speed for solver flexibility. It drives any
Pyomo-registered solver (Gurobi, SCIP, CBC, GLPK) and supports
lexicographic objectives, so prefer it when you need a specific or open-source
solver, or a global solver (SCIP) for the non-convex models.
:::

::::

## Selecting a backend

The default solve uses IPOPT, which auto-routes to the CasADi backend when it is
installed (falling back to GEKKO's bundled IPOPT), so the smooth-NLP fast path
needs no configuration. Everything else is one keyword:

```python
import monee

# Smooth NLP: CasADi by default (nothing to set)
monee.run_energy_flow(net, formulation="smooth_nlp")

# (MI)QCQP / MISOCP: pick the solver. A plain name routes to Pyomo
monee.run_energy_flow_optimization(net, problem, solver="gurobi")  # Pyomo + Gurobi
monee.run_energy_flow_optimization(net, problem, solver="scip")  # Pyomo + SCIP (global)

# The faster native Gurobi backend is opt-in via backend=
monee.run_energy_flow_optimization(net, problem, backend="gurobipy")
```

## Formulation class to recommended backend

| Formulation class | Examples | First choice | Alternative |
|---|---|---|---|
| Smooth NLP (single sector) | `EL_NLP`, `GAS_NLP`, `HEAT_NLP` | CasADi (IPOPT) | GEKKO (IPOPT / APOPT) |
| Smooth NLP (full MES) | `SMOOTH_NLP` | CasADi (IPOPT) | GEKKO (IPOPT / APOPT) |
| Second-order cone | `EL_MISOCP` | gurobipy | Pyomo + Gurobi / SCIP |
| Convex MIQCQP | `GAS_CONVEX_MIQCQP`, `CONVEX_MIQCQP` | gurobipy | Pyomo + Gurobi / SCIP |
| Non-convex MIQCQP | `*_NONCONVEX_MIQCQP` | Pyomo + SCIP (global) | gurobipy / Pyomo + Gurobi |
| LP / MILP heat & PWL | `HEAT_CONVEX_MILP`, `make_gas_milp_pwl_formulation()` | gurobipy / Pyomo + SCIP | Pyomo + Gurobi |

```{tip}
When in doubt, take the default. A plain `run_energy_flow` already uses the fast
in-process CasADi/IPOPT path for the smooth NLPs. For the (MI)QCQP and MISOCP
models, name a solver (`solver="gurobi"` or `"scip"` routes to Pyomo)
and add `backend="gurobipy"` when you want the fastest native Gurobi path. See
{doc}`../concepts/solvers` for the full routing rules and
{doc}`../concepts/formulations` for what each formulation class models.
```

## What the numbers also show

- Correctness first. Every case agrees across backends to ≤ 1e-4, and the
  electricity NLP and storage cases to ≤ 1e-8, so the speedups compare identical
  solutions, not different ones.
- The win grows with structure. CasADi's lead is largest on the timeseries
  cases (up to 18 times on the 12-step EL series), where GEKKO pays its per-solve
  subprocess cost once per step.
- Coupling is where native backends shine. On the coupled multi-sector OPF the
  native gurobipy backend is 4 times faster than Pyomo/Gurobi; the translation
  overhead scales with model size.

Reproduce locally:

```bash
python benchmarks/backend_comparison.py            # run + plot
python benchmarks/backend_comparison.py --plot-only
```

## How this compares to a dedicated power-flow tool

A recurring question is how a monee power flow compares to pandapower's
Newton-Raphson on the same grid. The honest answer has two halves, and the
first one is that the popular single-number ratio is close to meaningless.

On CIGRE MV (15 buses) a warm `pp.runpp` and a warm `monee.run_energy_flow`
both land in the tens of milliseconds. Measured as five batches of 15 repeats
after three discarded warm-ups, each batch in a fresh process, pandapower's
median ran 12.7 to 18.1 ms and monee's 19.3 to 32.2 ms, so the ratio came out
anywhere between 1.1 and 2.5 times across batches. Within one batch the spread
is tight (standard deviation under 10 percent); it is the batch-to-batch
variation that dominates. At this size both numbers are per-call fixed cost, so
a ratio measured once tells you nothing.

The second half is that the ratio is not a constant, because the two tools
scale differently over this range. On mv_oberrhein (179 buses) pandapower's
median barely moves, 11.8 ms, while monee's goes to 341 ms, a factor of 29.
pandapower's time at these sizes is mostly its own pandas bookkeeping;
monee's grows with the model it assembles.

What drives monee's time is not the numerics. Splitting a CIGRE MV call with
the CasADi back end's own timers (`last_build_s`, `last_solve_s`):

| Phase | CIGRE MV, 15 buses | Share |
|---|---|---|
| Python-side work: network copy, symbolic equation assembly, result frames | 15.4 ms | 78 percent |
| CasADi `nlpsol` construction (graph to derivative functions) | 3.5 ms | 18 percent |
| The IPOPT solve itself | 0.8 ms | 4 percent |

So a monee power flow spends about 4 percent of its wall clock in the solver.
The rest is building a fresh symbolic model, which monee does on every
`run_energy_flow` call because each call may see a changed network.

That points at the fix, which is already in the library: when you solve the
same topology repeatedly, use {doc}`../how-to/timeseries` instead of a loop of
`run_energy_flow`. The CasADi timeseries path builds the `nlpsol` once and
re-solves it with the time-varying inputs as parameters. On CIGRE MV, 12 steps
took 378 ms as 12 separate calls (31.5 ms per step) against 87 ms as one
`run_timeseries` (7.2 ms per step), a 4.4 times speedup from amortising the
build alone.

```{note}
monee is a multi-energy modelling and optimisation framework, not a
power-flow-only engine. A single-sector power flow is the case where it carries
the most overhead it does not need, and it is also the case a dedicated tool
already solves well. Reach for monee when the network is coupled, the model is
an optimisation, or the run is a series; use `run_timeseries` when it is a
series of solves on one topology.
```
