
# How-to guides

Short, task-focused guides for common operations. Each guide assumes you are
already familiar with the {doc}`../quickstart`.

::::{grid} 1 2 2 2
:gutter: 4

:::{grid-item-card} Load shedding
:link: load_shedding
:link-type: doc
:shadow: sm

Find the minimum demand curtailment needed to keep a network feasible under
voltage, pressure, and temperature bounds - one call or fully customised.
:::

:::{grid-item-card} Import MATPOWER files
:link: matpower_io
:link-type: doc
:shadow: sm

Read standard IEEE test cases or any `.mat` MATPOWER file, and persist
networks to/from the native OMEF JSON format.
:::

:::{grid-item-card} Convert from pandapower
:link: convert_from_pandapower
:link-type: doc
:shadow: sm

Import an existing pandapower network into monee with a single function
call. *(Experimental - complex elements may not convert correctly.)*
:::

:::{grid-item-card} Use the Pyomo solver
:link: use_pyomo_solver
:link-type: doc
:shadow: sm

Plug in HiGHS, Gurobi, GLPK, or CBC as the solver back-end - required for
MILP / MIQCP problems such as the MISOCP optimal power flow.
:::

:::{grid-item-card} Solve islanded networks
:link: islanding
:link-type: doc
:shadow: sm

Enable multi-island solves with `enable_islanding()`, add grid-forming
generators or sources, and retrieve per-island results.
:::

:::{grid-item-card} Timeseries simulation
:link: timeseries
:link-type: doc
:shadow: sm

Drive a multi-energy network through hundreds or thousands of timesteps with
time-varying load profiles, ramp constraints, and rich result queries.
:::

:::{grid-item-card} Multi-period optimization
:link: multi_period
:link-type: doc
:shadow: sm

Jointly optimize storage dispatch, CHP scheduling, and linepack usage over a
full planning horizon - including rolling-horizon MPC.
:::

:::{grid-item-card} Storage dispatch
:link: storage
:link-type: doc
:shadow: sm

Attach electric, gas, and thermal storage to a network.  Prescribe a charge
schedule via `TimeseriesData` or let the optimizer choose the dispatch.
:::

:::{grid-item-card} Externally paced simulation
:link: conductor
:link-type: doc
:shadow: sm

Drive a network step by step from an external co-simulation framework with
`Conductor` - variable step sizes, data overrides, and persistent state.
:::

:::{grid-item-card} Diagnose infeasibility
:link: diagnose_infeasibility
:link-type: doc
:shadow: sm

Find out *why* a solve failed: bound-violation reports, Pyomo
`InfeasibilityReport`, and GEKKO APM diagnostics.
:::

:::{grid-item-card} Bulk topology builders
:link: express_structures
:link-type: doc
:shadow: sm

Build lines, rings, stars, and paired supply/return district-heating
structures with the `monee.express` structure builders.
:::

:::{grid-item-card} Generate synthetic MES
:link: generate_mes
:link-type: doc
:shadow: sm

Overlay gas and district-heating networks plus coupling points on any power
grid to generate reproducible multi-energy test systems.
:::

:::{grid-item-card} Import CIM / ESDL models
:link: import_cim_esdl
:link-type: doc
:shadow: sm

Import CIM/CGMES grid models and ESDL energy-system descriptions.
*(Experimental - see the per-import transparency reports.)*
:::

:::{grid-item-card} Reference networks
:link: reference_networks
:link-type: doc
:shadow: sm

Load ready-made multi-energy test cases (urban district, industrial hub,
regional MES) for benchmarking, tutorials, and quick experiments.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

load_shedding
matpower_io
convert_from_pandapower
import_cim_esdl
use_pyomo_solver
diagnose_infeasibility
islanding
timeseries
conductor
multi_period
storage
express_structures
generate_mes
reference_networks
```
