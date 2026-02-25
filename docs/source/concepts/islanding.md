
# Islanding

**Islanding** is the ability to model a grid that has split into several
disconnected islands, each anchored by a different grid-forming source.
Without islanding support, a standard energy-flow solver requires the grid to
be fully connected to a single slack node; any bus unreachable from that node
is dropped from the solve as *ignored*.  With islanding enabled, every
connected island is solved simultaneously, with each island's own slack node
absorbing the local supply–demand imbalance.

monee implements islanding as a set of **mixed-integer** (MIP) constraints
that are added on top of the carrier's normal steady-state equations.  Any
solver back-end that supports integer variables can be used
(`GEKKOSolver` or `PyomoSolver`).

---

## Grid-forming nodes

A **grid-forming node** is a junction that can act as the slack bus of its
island.  Any junction whose child list contains at least one component that
inherits from `GridFormingMixin` is classified as grid-forming for that
carrier.

| Carrier | Grid-forming child | Role |
|---|---|---|
| Electricity | `ExtPowerGrid` | External grid connection (always) |
| Electricity | `GridFormingGenerator` | Islanded generator (islanding only) |
| Gas / Water | `ExtHydrGrid` | External hydraulic connection (always) |
| Gas / Water | `GridFormingSource` | Islanded compressor / source (islanding only) |

`GridFormingMixin` is a pure marker mixin — it carries no equations.  Its sole
purpose is to let `find_ignored_nodes` identify which nodes anchor each island.

---

## Connectivity-flow formulation

Reachability from a grid-forming node is enforced by a
**single-commodity virtual flow** model.  A virtual super-source node "0"
injects flow into every grid-forming junction; the flow propagates through the
carrier topology until each energised junction has received exactly one unit.

The binary variable $e_i \in \{0,1\}$ indicates whether junction $i$ is
**energised** ($e_i = 1$) or de-energised ($e_i = 0$).

### Variables

| Variable | Domain | Meaning |
|---|---|---|
| $e_i$ | $\{0,1\}$ | Junction $i$ energised |
| $c_{ij}^+$ | $\mathbb{R}_{\geq 0}$ | Connectivity flow on branch $(i,j)$, forward direction |
| $c_{ij}^-$ | $\mathbb{R}_{\geq 0}$ | Connectivity flow on branch $(i,j)$, reverse direction |
| $c_i^{\text{src}}$ | $\mathbb{R}_{\geq 0}$ | Super-source arc into grid-forming junction $i$ |

### Constraints

**Grid-forming junctions are always energised:**

$$e_k = 1 \qquad \forall k \in \text{GF}$$

**Arc capacity (physical branches):** controlled by the branch on/off variable $x_{ij}$:

$$c_{ij}^+,\; c_{ij}^- \;\leq\; M \cdot x_{ij}$$

**Super-source arc capacity** (always open for GF nodes, so $x = 1$):

$$c_k^{\text{src}} \;\leq\; M \qquad \forall k \in \text{GF}$$

**Per-node flow balance:** net inflow equals the energisation indicator:

$$\left(\sum_{\text{in}} c\right) - \left(\sum_{\text{out}} c\right) = e_i$$

For grid-forming junctions the super-source arc is counted as additional inflow.

**Global super-source supply equals total demand:**

$$\sum_{k \in \text{GF}} c_k^{\text{src}} = \sum_i e_i$$

The big-M constant $M$ must be at least as large as the number of carrier
nodes.  The default is 200; override via `big_m_conn` when your network is
larger.

---

## Per-carrier physical constraints

Beyond the connectivity-flow integers, each carrier mode adds
**carrier-specific physical constraints** to ensure the physical state of
de-energised nodes is well-defined for the solver.

### Electricity

| Nodes | Constraint | Why |
|---|---|---|
| Grid-forming | $\theta_k = 0$ | Angle reference for each island |
| Regular | $-M_\theta (1 - e_i) \leq \theta_i \leq M_\theta (1 - e_i)$ would force $\theta_i = 0$ when $e_i = 0$ | Numerical stability — prevents the angle from floating freely on de-energised buses |

The angle reference constraint is **strictly necessary** for multi-island DC
flow: without it the power-flow equations become singular for each island
beyond the first (no angle reference defined).

### Gas and water

| Nodes | Constraint | Why |
|---|---|---|
| Regular | $p_i \leq 2\, e_i$ | Forces $p_i = 0$ when $e_i = 0$ |

The grid-forming junction's pressure is already pinned to a setpoint by
`GridFormingSource.overwrite()` (or `ExtHydrGrid.overwrite()`), so no
additional constraint is needed at GF nodes.

The pressure bound is primarily useful when switchable pipes are present: a
de-energised island with floating pressure can confuse NLP solvers.  For
networks without switchable pipes the bound is a nice-to-have.

---

## Solver integration

Islanding constraints are implemented as a `NetworkConstraint` — the same
interface used by the formulation layer — so they integrate with both solver
back-ends without modification.

**Phase 1 — `prepare(network)`:**
Called before the solver injects variables.  Adds binary `Var` placeholders
(`e_el`, `e_gas`, `e_water`, etc.) to the node and branch model objects.
These are picked up automatically by the standard variable-injection loops.

**Phase 2 — `equations(network, ignored_nodes)`:**
Called after variable injection.  Returns the full list of connectivity-flow
and physical constraint expressions.  The solver appends these to its equation
set alongside the normal energy-flow equations.

**`find_ignored_nodes` interaction:**
When islanding is enabled, `find_ignored_nodes` uses the **complete** network
topology (all branches, including open ones) and classifies a junction as
non-ignored if any of its children is a `GridFormingMixin`.  This prevents
island B's buses from being pruned before the MIP solve even starts.

---

## Class hierarchy

```
NetworkConstraint
└── NetworkIslandingConfig        ← bundles all per-carrier modes
    registered via network.add_extension()

NetworkConstraint (ABC)
└── IslandingMode                 ← per-carrier base class
    ├── ElectricityIslandingMode  ← PowerGrid; angle pinning
    ├── GasIslandingMode          ← GasGrid; pressure bounds
    └── WaterIslandingMode        ← WaterGrid; pressure bounds

ChildModel + GridFormingMixin     ← marker mixin
├── GridFormingGenerator          ← electricity slack generator
└── GridFormingSource             ← gas / water slack source
```
