
# Install

## Requirements

- Python 3.10 or newer
- A fresh virtual environment is strongly recommended.

---

## Install monee

::::{tab-set}

:::{tab-item} From PyPI
:selected:

```bash
pip install monee
```

Installs the core package with all required dependencies, all of them
third-party packages: CasADi, GEKKO, Pyomo, pyscipopt, NumPy, SciPy, pandas,
NetworkX, Plotly, and geopy.
:::

:::{tab-item} From source

```bash
git clone https://github.com/Digitalized-Energy-Systems/monee.git
cd monee
pip install -e .
```

Editable install: changes to the source are reflected immediately without
reinstalling.
:::

::::

---

## Optional extras

| Extra | Install command | What it adds |
|---|---|---|
| `simbench` | `pip install monee[simbench]` | Import networks from [SimBench](https://simbench.de/en/) and convert [pandapower](https://www.pandapower.org/) networks |

### Optional dependencies

Some features pull in extra packages on demand. monee imports them lazily and
raises an instructive `ImportError` when they are missing, so install them only
when you need the feature:

| Package | Install command | Needed for |
|---|---|---|
| `cimpy` | `pip install cimpy` | CIM/CGMES import (`monee.io.from_cim`) |
| `pyESDL` | `pip install pyESDL` | ESDL import (`monee.io.from_esdl`) |
| `simbench` + `pandapower` | `pip install monee[simbench]` | SimBench and pandapower grid import |
| `pygraphviz` (plus a system [Graphviz](https://graphviz.org/)) | `pip install pygraphviz` | Automatic graph layout in `plot_network` / `plot_result` |
| `geopy` | installed with monee | Geodesic branch lengths in the MES generators |

```{note}
SciPy (MATPOWER import), Plotly (visualization) and geopy are core
dependencies of monee, so pip installs them together with it and those
features need no extra step.
```

---

## Solver back-ends

monee does not contain a solver of its own. It builds the mathematical model
and hands it to third-party solvers through the packages below, which pip
installs with monee as core dependencies. By default monee hands the model to
IPOPT, an open-source solver, and runs it in-process through CasADi (no
subprocess, typically much faster); if CasADi cannot be imported, monee runs
IPOPT through GEKKO instead, using the IPOPT binary that the GEKKO package
bundles. For mixed-integer or conic problems you switch to Pyomo, which passes
the model on to SCIP through pyscipopt, or to another solver you install.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} CasADi / GEKKO {bdg-success}`default`
:shadow: sm

[CasADi](https://web.casadi.org) builds the NLP as an in-memory expression
graph and calls IPOPT in-process. It covers the smooth formulations, including
temporal and multi-period coupling. If CasADi cannot be imported, monee uses
the [GEKKO](https://gekko.readthedocs.io) suite instead, which bundles its own
APOPT, BPOPT and IPOPT binaries. Both are third-party packages that pip
installs with monee.

Best for: nonlinear energy-flow simulation and NLP optimisation.
:::

:::{grid-item-card} Pyomo {bdg-secondary}`MILP and MIQCP`
:shadow: sm

Translates the network model to a [Pyomo](https://www.pyomo.org)
`ConcreteModel` and passes it to a solver. pyscipopt, the Python interface to
the open-source SCIP solver, is a core dependency, so SCIP works without an
extra step; other solvers such as Gurobi are installed separately.

Best for: MILP and MIQCP problems, for example MISOCP optimal power flow.

See {doc}`how-to/use_pyomo_solver` for a full walk-through.
:::

::::

### Selection of available solver binaries for Pyomo

| Solver | Licence | Problem types | Install |
|---|---|---|---|
| [SCIP](https://scipopt.org/) | Open-source | LP · MILP · MIQCP · MINLP | installed with monee (pyscipopt is a core dependency) |
| [GLPK](https://www.gnu.org/software/glpk/) | Open-source | LP · MILP | `conda install -c conda-forge glpk` |
| [CBC](https://github.com/coin-or/Cbc) | Open-source | LP · MILP | `conda install -c conda-forge coincbc` |
| [Gurobi](https://www.gurobi.com/) | Commercial | LP · MILP · MIQCP | requires licence |

Pass the solver by name, e.g. `run_energy_flow(net, solver="scip")`. The name
chooses the back-end: `ipopt` routes to CasADi when installed (otherwise
GEKKO), `apopt` and `bpopt` route to GEKKO, and every other name is forwarded
to Pyomo. Override the back-end explicitly with `backend=...` if needed.
