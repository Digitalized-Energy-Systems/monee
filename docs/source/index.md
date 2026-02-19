
```{toctree}
:maxdepth: 2
:hidden:

install
quickstart
concepts/index
tutorials/index
how-to/index
api/index
```

# monee

:::::{div} sd-text-center sd-py-4

```{image} _static/monee-logo.drawio.svg
:width: 220px
:class: sd-pb-4
```

**Modular Network-based Energy Grid Optimization**

Model, simulate, and optimize interconnected electricity, gas, and heat networks —
in one unified Python framework.


::::{grid} 2

:::{grid-item}

```{button-ref} quickstart
:color: primary
:shadow:
Get started
```
:::

:::{grid-item}

```{button-link} https://github.com/Digitalized-Energy-Systems/monee
:color: secondary
:outline:
GitHub
```
:::

::::

```{code-block} bash
pip install monee
```

:::::

---

## Features

::::{grid} 1 2 3 3
:gutter: 4

:::{grid-item-card}
:shadow: sm

**Multi-energy networks**
^^^
Electricity, gas, and water/heat in **one model**. Networks are represented
as directed graphs — any topology is supported.
:::

:::{grid-item-card}
:shadow: sm

**Energy-carrier coupling**
^^^
Connect carriers with built-in units:
**P2H** (power-to-heat), **G2P** (gas-to-power), **P2G**, **G2H**, and
**CHP**. Bidirectional flows are handled automatically.
:::

:::{grid-item-card}
:shadow: sm

**Steady-state simulation**
^^^
Run energy-flow calculations across all carriers simultaneously.
Results come back as typed dataframes — one row per component.
:::

:::{grid-item-card}
:shadow: sm

**Optimisation**
^^^
Swap {func}`~monee.run_energy_flow` for
{func}`~monee.run_energy_flow_optimization` and pass a problem
formulation. Built-in: **load shedding**. Supports custom objectives and
constraints.
:::

:::{grid-item-card}
:shadow: sm

**Flexible solver back-ends**
^^^
Ships with **GEKKO** (IPOPT, default) and **Pyomo** (HiGHS · Gurobi · GLPK).
Switch back-ends without changing model code. MISOCP relaxations available
for convex OPF.
:::

:::{grid-item-card}
:shadow: sm

**Import / Export**
^^^
Round-trip networks in **MATPOWER**, **pandapower**, and **SimBench**
formats. One function call in each direction.
:::

::::

---

## Quick look

Build a multi-energy network coupling an electricity grid to a district
heating loop — and solve it — in under 25 lines:

```{code-block} python
from monee import mx, run_energy_flow

net = mx.create_multi_energy_network()

# ── Electricity grid ──────────────────────────────────────────────────
bus_0 = mx.create_bus(net)
bus_1 = mx.create_bus(net)
mx.create_line(net, bus_0, bus_1, length_m=100,
               r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
mx.create_ext_power_grid(net, bus_0)
mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

# ── District heating grid ─────────────────────────────────────────────
j_supply = mx.create_water_junction(net)
j_mid    = mx.create_water_junction(net)
j_return = mx.create_water_junction(net)
mx.create_ext_hydr_grid(net, j_supply)
mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.12, length_m=100)
mx.create_sink(net, j_return, mass_flow=1)

# ── Couple: electric bus drives a heat pump feeding the heating loop ──
mx.create_p2h(net, bus_1, j_mid, j_return,
              heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9)

result = run_energy_flow(net)
print(result.dataframes["Bus"][["id", "vm_pu", "va_degree"]])
```

---

## Where to go next

::::{grid} 1 2 2 4
:gutter: 3

:::{grid-item-card} Quickstart
:link: quickstart
:link-type: doc
:text-align: center
:shadow: sm

A five-minute guided tour of the core workflow.
:::

:::{grid-item-card} Concepts
:link: concepts/index
:link-type: doc
:text-align: center
:shadow: sm

Physical equations, formulations, solvers, and the data model.
:::

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc
:text-align: center
:shadow: sm

End-to-end worked examples including optimisation and time-series simulation.
:::

:::{grid-item-card} API Reference
:link: api/index
:link-type: doc
:text-align: center
:shadow: sm

Complete auto-generated reference for every public function and class.
:::

::::
