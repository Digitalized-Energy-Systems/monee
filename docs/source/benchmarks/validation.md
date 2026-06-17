# Validation against pandapower and pandapipes

To trust a general multi-energy NLP, check it against established single-purpose
reference tools. monee's nonlinear models are validated against
[pandapower](https://www.pandapower.org) (electricity) and
[pandapipes](https://www.pandapipes.org) (gas, water/heat, and sequential
multi-energy coupling) on like-for-like networks.

The headline:

- Single-sector power flow reproduces the reference solution almost always:
  voltages to ≤ 5e-3 pu, pressures to ~1 % of the pipe drop, temperatures to
  ≤ 1e-2 K. On raw power-flow speed the specialised tools win, since a dedicated
  Newton or hydraulic Jacobian beats a general NLP, as expected.
- Optimization (AC OPF, economic dispatch) reaches pandapower's optimum to
  machine precision and is frequently faster. A general NLP is no longer at a
  structural disadvantage once the reference tool also has to solve a nonlinear
  program.
- Multi-energy coupling (P2G, G2P, CHP) is monee's home turf. It solves every
  carrier in one simultaneous NLP, matching pandapipes' iterative `multinet` to
  ~1e-11 pu on voltage while solving the coupled problem faster.

## Electricity, against pandapower (identical model)

The electrical model is handed to both engines through the neutral MATPOWER
`mpc` exchange, so any difference is the solver, not the model. monee uses its
in-process CasADi/IPOPT backend; pandapower uses its native Newton-Raphson
`runpp` and the PYPOWER AC-OPF `runopp`.

````{only} html
```{raw} html
<iframe src="../_static/interactive/benchmark_pandapower.html" width="100%" height="600" style="border:none;" loading="lazy" title="monee vs pandapower: solve time, bus-voltage agreement, and slack-power agreement for AC power flow and OPF"></iframe>
```
````

````{only} latex
```{figure} /_static/interactive/benchmark_pandapower.png
:alt: monee vs pandapower, solve time, bus-voltage agreement, and slack-power agreement for AC power flow and OPF
:width: 100%

monee (electric NLP) vs pandapower on identical grids. Left: solve time (log
scale). Middle: bus-voltage agreement `|Δvm|` (pu, log). Right: slack
active-power agreement `|ΔP|` (MW, log). Top: AC power flow; middle and bottom:
economic-dispatch OPF without and with a binding line-loading limit.
```
````

| Regime | Voltage agreement | Power agreement | Speed vs pandapower |
|---|---|---|---|
| Power flow (CIGRE MV / MV+DER / LV) | ≤ 5e-3 pu | ≤ 6e-4 MW | ~1.4 to 4× slower (NR wins) |
| OPF, economic dispatch | ≤ 3e-8 pu | ≤ 3e-5 MW | often faster than `runopp` |
| OPF + line-loading limit | ≤ 8e-7 pu | ≤ 2e-4 MW | often faster than `runopp` |

For OPF the cost objective matches pandapower to a relative error ≤ 4e-7, so the
two optimisers find the same economic optimum. The line-limited case has the cap
binding on both tools, so monee's current and loading constraints are exercised
too.

```{note}
monee imports every generator as a fixed-PQ injection with the external grid as
the voltage reference, a distribution PQ-with-slack scope. MATPOWER transmission
cases with PV (voltage-controlled) generators are out of scope and are not part
of this validation set. This is a modelling-scope choice, not a solver-accuracy
result.
```

## Gas, heat, and coupled MES, against pandapipes

Gas and heat have no neutral exchange format. monee solves a smooth Weymouth
(gas) or Darcy-Weisbach plus temperature-transport (heat) NLP; pandapipes solves
its own detailed hydraulic and thermal Newton system. So like-for-like networks
are built independently in each tool from one spec, and the comparison is
agreement within tolerance.

````{only} html
```{raw} html
<iframe src="../_static/interactive/benchmark_pandapipes.html" width="100%" height="750" style="border:none;" loading="lazy" title="monee vs pandapipes: solve time, pressure agreement, and temperature agreement for gas, heat, and coupled MES flow"></iframe>
```
````

````{only} latex
```{figure} /_static/interactive/benchmark_pandapipes.png
:alt: monee vs pandapipes, solve time, pressure agreement, and temperature agreement for gas, heat, and coupled MES flow
:width: 100%

monee (multi-energy NLP) vs pandapipes. Left: solve time (log scale). Middle:
pressure agreement (% of pipe drop). Right: temperature agreement (K), shown only
where heat participates (the HEAT suite and the CHP MES cases); the isothermal
gas rows are blank by design.
```
````

| Suite | Agreement | Speed vs pandapipes |
|---|---|---|
| GAS hydraulics | pressure within 1.2 % of the pipe drop | ~1.3 to 7× slower (`pipeflow` wins) |
| HEAT hydraulics + thermal | pressure ~0.1 %, temperature ≤ 8.7e-3 K | ~1.4 to 23× slower (`pipeflow` wins) |
| MES P2G / G2P / CHP (coupled) | bus voltage to ~1e-11 pu, CHP heat-exchanger ΔT within 0.07 K | 1.4 to 5× faster |

The MES result is the point of the whole exercise. pandapipes couples a
pandapower net and pandapipes gas/water nets through `multinet` controllers that
iterate to a fixed point between sectors; monee writes all carriers into a single
NLP and solves them simultaneously. The simultaneous solve agrees to machine
precision on the shared electrical quantities and is faster than the iterative
coupling.

## Reading the story

The two "slower on single-sector power flow" results are not a weakness; they are
the expected cost of generality. A power-flow Jacobian or a hydraulic Newton step
is a specialised routine. A general interior-point NLP does strictly more work to
offer the same model the freedom to become an OPF, gain a custom constraint, or
couple to another carrier. Once the problem stops being a plain single-sector
power flow and becomes an optimisation or a multi-energy coupling, that
generality stops costing and starts paying.

Reproduce locally:

```bash
python benchmarks/pandapower_comparison.py      # electricity
python benchmarks/pandapipes_comparison.py     # gas / heat / MES
# add --plot-only to re-render the figures from the saved CSVs
```

```{seealso}
{doc}`backend_selection`: once the model is validated, which backend solves it
fastest for each formulation class.
```
