
# Islanding

Islanding models a grid that has split into several disconnected islands, each
anchored by a different grid-forming source. Without it, a standard energy-flow
solver requires the grid to be fully connected to a single slack node, and any
bus unreachable from that node is dropped from the solve as ignored. With
islanding enabled, every connected island is solved at once, and each island's
own slack node absorbs the local supply and demand imbalance.

monee adds islanding as a set of mixed-integer constraints on top of the
carrier's normal steady-state equations. The per-node energisation variables
(`e_el`, `e_gas`, `e_water`) are binary, so islanding works best with a
mixed-integer (MIP) capable back-end such as `GEKKOSolver`, `GurobipySolver`,
or `PyomoSolver` with a MIP solver like SCIP or Gurobi. `CasADiSolver` also
reads the islanding config, but IPOPT relaxes the integers, so it does not
return a true binary solution.

---

## Grid-forming nodes

A grid-forming node is a junction that can act as the slack bus of its island.
A junction is grid-forming for a carrier when at least one of its children
inherits from `GridFormingMixin`.

| Carrier | Grid-forming child | Role |
|---|---|---|
| Electricity | `ExtPowerGrid` | External grid connection (always) |
| Electricity | `GridFormingGenerator` | Islanded generator (islanding only) |
| Gas / Water | `ExtHydrGrid` | External hydraulic connection (always) |
| Gas / Water | `GridFormingSource` | Islanded source (islanding only) |

`GridFormingMixin` is a marker mixin: it carries no equations. It only lets
`find_ignored_nodes` identify which nodes anchor each island.

---

## Connectivity-flow formulation

A single-commodity virtual flow model enforces reachability from a grid-forming
node. A virtual super-source node "0" injects flow into every grid-forming
junction. The flow propagates through the carrier topology until each energised
junction has received exactly one unit.

The binary variable $e_i \in \{0,1\}$ indicates whether junction $i$ is
energised ($e_i = 1$) or de-energised ($e_i = 0$).

### Variables

| Variable | Domain | Meaning |
|---|---|---|
| $e_i$ | $\{0,1\}$ | Junction $i$ energised |
| $c_{ij}^+$ | $\mathbb{R}_{\geq 0}$ | Connectivity flow on branch $(i,j)$, forward direction |
| $c_{ij}^-$ | $\mathbb{R}_{\geq 0}$ | Connectivity flow on branch $(i,j)$, reverse direction |
| $c_i^{\text{src}}$ | $\mathbb{R}_{\geq 0}$ | Super-source arc into grid-forming junction $i$ |

### Constraints

Grid-forming junctions are always energised:

$$e_k = 1 \qquad \forall k \in \text{GF}$$

Arc capacity (physical branches), controlled by the branch on/off variable $x_{ij}$:

$$c_{ij}^+,\; c_{ij}^- \;\leq\; M \cdot x_{ij}$$

Super-source arc capacity (always open for GF nodes, so $x = 1$):

$$c_k^{\text{src}} \;\leq\; M \qquad \forall k \in \text{GF}$$

Per-node flow balance, net inflow equals the energisation indicator:

$$\left(\sum_{\text{in}} c\right) - \left(\sum_{\text{out}} c\right) = e_i$$

For grid-forming junctions the super-source arc is counted as additional inflow.

Global super-source supply equals total demand:

$$\sum_{k \in \text{GF}} c_k^{\text{src}} = \sum_i e_i$$

The big-M constant $M$ must be at least as large as the number of carrier
nodes. monee sizes it automatically as ten times the total node count
(`len(network.nodes) * 10`), so manual tuning is normally unnecessary.

```{note}
The islanding modes accept a `big_m_conn` parameter. When passed explicitly it
is used as the connectivity big-M; the default (`None`) falls back to the
automatic `len(network.nodes) * 10` sizing described above.
```

---

## Per-carrier physical constraints

Beyond the connectivity-flow integers, each carrier mode adds physical
constraints so the state of de-energised nodes stays well-defined for the
solver.

### Electricity

| Nodes | Constraint | Why |
|---|---|---|
| Grid-forming | $\theta_k = 0$ | Angle reference for each island |
| Regular | $-M_\theta\, e_i \leq \theta_i \leq M_\theta\, e_i$ forces $\theta_i = 0$ when $e_i = 0$ | Numerical stability: prevents the angle from floating freely on de-energised buses |

$M_\theta$ is the mode's `angle_bound` parameter (default 3.15 rad).

The angle reference is required for multi-island DC flow. Without it, the
power-flow equations become singular for every island beyond the first, since
no angle reference is defined there.

The mode owns angle handling, not the grid-forming child.
`ElectricityIslandingMode.prepare()` sets the `_islanding_angle_managed` flag
on every active bus, and `GridFormingGenerator.overwrite()` pins only `vm_pu`,
not `va_radians`, so the mode's $\theta_k = 0$ constraint stays the single
angle reference per island.

### Gas and water

| Nodes | Constraint | Why |
|---|---|---|
| Regular | $p_i \leq 2\, e_i$ | Forces $p_i = 0$ when $e_i = 0$ |

`GridFormingSource.overwrite()` (or `ExtHydrGrid.overwrite()`) already pins the
grid-forming junction's pressure to a setpoint, so GF nodes need no extra
constraint.

The pressure bound matters most when switchable pipes are present, where a
de-energised island with floating pressure can confuse NLP solvers. Without
switchable pipes it is only a nice-to-have.

---

## Solver integration

Islanding constraints are a `NetworkAspect`, the same interface the formulation
layer uses, so every solver back-end picks them up without modification.

**Phase 1, `prepare(network)`:** runs before the solver injects variables. Adds
`Var` placeholders (`e_el`, `e_gas`, `e_water`, the connectivity flows, and the
super-source arcs) to the node and branch models. The standard
variable-injection loops pick these up automatically.

**Phase 2, `equations(network, ignored_nodes)`:** runs after variable
injection. Returns the connectivity-flow and physical constraint expressions,
which the solver appends to its normal energy-flow equations.

**`find_ignored_nodes` interaction:** each solver looks up the registered
`NetworkIslandingConfig` and passes it to `find_ignored_nodes`. With a config
present, that function uses the complete network topology (all branches,
including open ones) and keeps any island that contains a grid-forming child.
This stops a second island's buses from being pruned before the solve starts.

**Persistence:** like all network extensions, the islanding config registered
by `enable_islanding` (or `add_extension`) is not serialized by the native JSON
IO (`monee.io.native`). After loading a network from disk, call
`enable_islanding` again before solving.

---

## Class hierarchy

```{mermaid}
classDiagram
    class NetworkAspect {
        <<abstract>>
    }
    class IslandingMode {
        <<abstract>>
    }
    class GridFormingMixin {
        <<mixin>>
    }

    NetworkAspect <|-- NetworkIslandingConfig
    NetworkAspect <|-- IslandingMode
    IslandingMode <|-- ElectricityIslandingMode
    IslandingMode <|-- GasIslandingMode
    IslandingMode <|-- WaterIslandingMode
    GridFormingMixin <|-- GridFormingGenerator
    GridFormingMixin <|-- GridFormingSource
```
