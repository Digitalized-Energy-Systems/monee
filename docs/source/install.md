
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

Editable install — changes to the source are reflected immediately without
reinstalling.
:::

::::

---

## Optional extras

| Extra | Install command | What it adds |
|---|---|---|
| `simbench` | `pip install monee[simbench]` | Import networks from [SimBench](https://simbench.de/en/) and convert [pandapower](https://www.pandapower.org/) networks |

---

## Solver back-ends

monee ships two solver interfaces. Both are installed automatically; you only
need to add a solver binary for the Pyomo back-end.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} GEKKO {bdg-success}`default`
:shadow: sm

Wraps the [GEKKO](https://gekko.readthedocs.io) optimisation suite, which
bundles its own IPOPT binaries. **No extra installation needed.**

**Best for:** nonlinear energy-flow simulation and NLP optimisation.

```bash
# already included — nothing to do
pip install monee
```
:::

:::{grid-item-card} Pyomo {bdg-secondary}`bring your own solver`
:shadow: sm

Translates the network model to a [Pyomo](https://www.pyomo.org)
`ConcreteModel`. You must separately install at least one solver binary.

**Best for:** MILP / MIQCP problems (e.g. MISOCP optimal power flow).

```bash
# SCIP — recommended non-commerical solver for MIQCP
conda install -y pyscipopt
```

See {doc}`how-to/use_pyomo_solver` for a full walk-through.
:::

::::

### Selection of available solver binaries for Pyomo

| Solver | Licence | Problem types | Install |
|---|---|---|---|
| [HiGHS](https://scipopt.org//) | Open-source | LP · MILP · MIQCP | `conda install -y pyscipopt` |
| [GLPK](https://www.gnu.org/software/glpk/) | Open-source | LP · MILP | `conda install -c conda-forge glpk` |
| [CBC](https://github.com/coin-or/Cbc) | Open-source | LP · MILP | `conda install -c conda-forge coincbc` |
| [Gurobi](https://www.gurobi.com/) | Commercial | LP · MILP · MIQCP | requires licence |
