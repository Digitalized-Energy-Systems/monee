
# Install

## Requirements

- **Python ≥ 3.10**
- A fresh virtual environment is strongly recommended.

---

## Install monee

::::{tab-set}

:::{tab-item} From PyPI
:selected:

```bash
pip install monee
```

Installs the core package with all required dependencies: GEKKO, Pyomo,
NumPy, SciPy, pandas, NetworkX, Plotly, and geopy.
:::

:::{tab-item} From source

```bash
git clone https://github.com/Digitalized-Energy-Systems/monee.git
cd monee
pip install -e .
```

Editable install - changes to the source are reflected immediately without
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
| `geopy` | already included | Geodesic branch lengths in the MES generators |

```{note}
SciPy (MATPOWER import), Plotly (visualization), and geopy ship with the core
install - no extra step needed for those features.
```

---

## Solver back-ends

monee ships two solver interfaces. Both are installed automatically; you only
need to add a solver binary for the Pyomo back-end.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GEKKO {bdg-success}`default`
:shadow: sm

Wraps the [GEKKO](https://gekko.readthedocs.io) optimisation suite, which
bundles its own APOPT, BPOPT and IPOPT binaries. **No extra installation
needed.**

**Best for:** nonlinear energy-flow simulation and NLP optimisation.

```bash
# already included - nothing to do
pip install monee
```
:::

:::{grid-item-card} Pyomo {bdg-secondary}`bring your own solver`
:shadow: sm

Translates the network model to a [Pyomo](https://www.pyomo.org)
`ConcreteModel`. You must separately install at least one solver binary.

**Best for:** MILP / MIQCP problems (e.g. MISOCP optimal power flow).

```bash
# SCIP - recommended non-commerical solver for MIQCP
conda install -y pyscipopt
```

See {doc}`how-to/use_pyomo_solver` for a full walk-through.
:::

::::

### Selection of available solver binaries for Pyomo

| Solver | Licence | Problem types | Install |
|---|---|---|---|
| [SCIP](https://scipopt.org/) | Open-source | LP · MILP · MIQCP · MINLP | `conda install -y pyscipopt` or `pip install pyscipopt` |
| [HiGHS](https://highs.dev/) | Open-source | LP · MILP · QP | `pip install highspy` |
| [GLPK](https://www.gnu.org/software/glpk/) | Open-source | LP · MILP | `conda install -c conda-forge glpk` |
| [CBC](https://github.com/coin-or/Cbc) | Open-source | LP · MILP | `conda install -c conda-forge coincbc` |
| [Gurobi](https://www.gurobi.com/) | Commercial | LP · MILP · MIQCP | requires licence |

Pass the solver by name, e.g. `run_energy_flow(net, solver="scip")` - GEKKO
names (`apopt`, `bpopt`, `ipopt`) route to the GEKKO back-end, every other
name is forwarded to Pyomo.
