
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
voltage, pressure, and temperature bounds — one call or fully customised.
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
call. *(Experimental — complex elements may not convert correctly.)*
:::

:::{grid-item-card} Use the Pyomo solver
:link: use_pyomo_solver
:link-type: doc
:shadow: sm

Plug in HiGHS, Gurobi, GLPK, or CBC as the solver back-end — required for
MILP / MIQCP problems such as the MISOCP optimal power flow.
:::

::::

```{toctree}
:maxdepth: 1
:hidden:

load_shedding
matpower_io
convert_from_pandapower
use_pyomo_solver
```
