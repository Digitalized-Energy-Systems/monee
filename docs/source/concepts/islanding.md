
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

### Island leadership

An `ExtPowerGrid` / `ExtHydrGrid` always leads its connected component. A
`GridFormingGenerator` / `GridFormingSource` leads only when its component
(over the real topology: active branches with `on_off` not fixed to 0) has
lost its ext grid, and then exactly one deterministic leader is chosen per
island. `IslandingMode.prepare()` stamps `_gf_leading` on every grid-forming
(non-ext) child accordingly, and the reference pins below apply only to
leaders: an unconditional `vm_pu`/pressure pin at every grid-forming junction
of an intact ext-led grid over-constrains the flow equations (no voltage or
pressure gradient means no transport) and makes the solve infeasible.

Only a registered islanding configuration stamps leadership, so the class
default is "not leading". Without `enable_islanding` for the carrier, a
`GridFormingGenerator` or `GridFormingSource` pins nothing and cannot form an
island; it logs a warning when the solver assembles its equations. This matters
after a round trip through the native JSON IO, which does not persist the
configuration (see Persistence below).

While a unit does not lead, its exchange with the grid is held at a nominal
setpoint rather than left free:

| Component | Setpoint parameters | Default |
|---|---|---|
| `GridFormingGenerator` | `nominal_p_mw`, `nominal_q_mvar` | `0.0`, the unit is idle |
| `GridFormingSource` | `nominal_mass_flow_kgs` | `None`, the flow stays free |

Pass `None` for a parameter to keep that injection a free variable. Beware that
a free injection on a non-leading unit is priced by nothing: plain energy flow
carries no objective, so the split between the unit and its island's reference
is an arbitrary vertex of the feasible box and differs from back-end to
back-end. Set a nominal, or make the unit controllable in an optimization
problem that prices its output.

### Electricity

| Nodes | Constraint | Why |
|---|---|---|
| Island-leading | $\theta_k = 0$ | Angle reference for each island |
| Other | $-M_\theta\, e_i \leq \theta_i \leq M_\theta\, e_i$ forces $\theta_i = 0$ when $e_i = 0$ | Numerical stability: prevents the angle from floating freely on de-energised buses |
| Other, in a gated component | $1 - (1 - V^{\min})\, e_i \leq v_i \leq 1 + (V^{\max} - 1)\, e_i$ forces $v_i = 1$ when $e_i = 0$ | A detached bus has no equation left that fixes its voltage; pinning it to nominal keeps the model non-singular and the report deterministic |

$M_\theta$ is the mode's `angle_bound` parameter (default 3.15 rad).
$V^{\min}$ and $V^{\max}$ are the grid's voltage bounds (0.5 to 1.5 pu by
default). The voltage pin applies only where the branch gate below detaches
buses; with $e_i = 1$ both sides collapse to the ordinary model bounds. The
same pair is written on `vm_pu_squared` where that is the decision variable
(branch-flow MISOCP).

The angle reference is required for multi-island DC flow. Without it, the
power-flow equations become singular for every island beyond the first, since
no angle reference is defined there.

The mode owns angle handling, not the grid-forming child.
`ElectricityIslandingMode.prepare()` sets the `_islanding_angle_managed` flag
on every active bus, and `GridFormingGenerator.overwrite()` pins only `vm_pu`
(and only when leading), not `va_radians`, so the mode's $\theta_k = 0$
constraint stays the single angle reference per island.

### Gas and water

| Nodes | Constraint | Why |
|---|---|---|
| Regular | $p_i \leq 2\, e_i$ | Forces $p_i = 0$ when $e_i = 0$ |

`GridFormingSource.overwrite()` (when leading, see above) or
`ExtHydrGrid.overwrite()` already pins the grid-forming junction's pressure to
a setpoint, so GF nodes need no extra constraint.

The pressure bound matters most when switchable pipes are present, where a
de-energised island with floating pressure can confuse NLP solvers. Without
switchable pipes it is only a nice-to-have.

### Fixed-injection gating (plain flow solves)

For solves without an optimization problem, each mode replaces fixed numeric
child injections (`p_mw`/`q_mvar` for electricity, `mass_flow_kgs` for gas and
water) with variables tied to the host node's energisation binary
($x = \text{setpoint} \cdot e_i$), and registers a network objective that
minimises $\sum_i (1 - e_i)$. An island whose grid-forming capacity cannot
cover its fixed demand then de-energises nodes instead of rendering the whole
solve infeasible, while the objective keeps blackout a last resort, never a
free alternative to serving reachable load. Optimization problems are
untouched: they bring their own shedding variables and are applied after
`prepare()`.

### Branch energisation gating (electricity)

De-energising a bus only detaches it if the branches at that bus go out of
service with it. `ElectricityIslandingMode` therefore promotes the branch
on/off state $x_{ij}$ to a binary decision variable on the gated branches and
ties it to the energisation of both endpoints:

$$x_{ij} \leq e_i, \qquad x_{ij} \leq e_j, \qquad x_{ij} \geq e_i + e_j - 1$$

That is exactly $x_{ij} = e_i \wedge e_j$, so the branch state adds no freedom
of its own: it is determined by the two energisation binaries. No power big-M is
involved. Because the branch flow equations and the nodal balance already
multiply by $x_{ij}$, a branch with a dark endpoint carries zero flow, and the
dark bus is genuinely removed from the island rather than held at the reference
angle. A grid-forming unit can now serve as many neighbours along a connected
feeder as its capacity covers, and sheds the rest.

The gate turns the AC branch equations into products of a binary with a
non-convex term, so a gated network is a MIQCQP and needs a global mixed-integer
back-end (SCIP or Gurobi). The `branch_gating` parameter of
`ElectricityIslandingMode` decides which branches pay that price:

| `branch_gating` | Gated branches |
|---|---|
| `"auto"` (default) | Only branches in a component anchored solely by capacity-limited grid-forming units, that is a component with no ext grid whose formers carry a finite injection bound |
| `"all"` | Every active electricity branch |
| `"off"` | None; the pre-gating formulation, kept for comparison and for back-ends that cannot take the extra binaries |

Under `"auto"` a network reached by an ext grid is untouched: no branch becomes
a decision, no voltage pin is written, and the model class and solve time are
what they were before the gate existed. Nothing has to be shed there, because
the ext grid absorbs any imbalance.

A branch whose `on_off` is already a decision, and a branch marked
`backup=True`, get only the two upper bounds. Whether to close such a branch
stays the user's decision, made by
{meth}`~monee.problem.core.OptimizationProblem.controllable_backup_lines`,
which promotes `on_off` on every `backup=True` branch after `prepare()` has
run. The AND lower bound would take that decision away by forcing the branch
closed between two live buses. A branch pinned out of service (`on_off == 0`)
is left alone.

```{note}
The gate is what makes $e_i$ physical, so it also changes what an infeasible
model means: an island that cannot be balanced now sheds buses along the feeder
instead of failing, and load beyond a capacity-limited unit is dropped bus by
bus rather than all at once.
```

### De-energisation under an optimization problem

An optimization problem brings its own shedding variables, so islanding skips
the injection gating and the energisation objective there. What stays is one
constraint per controllable child on an islanded carrier:

$$\text{regulation}_c \leq e_i \qquad \forall c \text{ at node } i$$

Both sides are in $[0, 1]$ and $e_i$ is binary, so the constraint is linear.
It gives de-energisation its cost: a node at $e_i = 0$ serves none of its
children, and the objective pays the full shed for each of them. Without it
$e_i$ is tied to nothing electrical and the solve can report a node as both
de-energised and fully served. Grid-forming children are exempt: a reference
unit is not sheddable demand.

The gate can turn a model infeasible that previously reported a
serve-it-and-blackout-it solution. If an optimization problem becomes
infeasible after enabling islanding, the usual cause is a priority floor (or
another minimum-service constraint) on a load whose bus the connectivity flow
cannot energise. Check `e_el` / `e_gas` / `e_water` on the affected buses,
confirm that every island holds a grid-forming child, and either relax the
floor or give the island a reference unit with enough capacity.

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
present, that function keeps any real-topology island that contains a
grid-forming child, so a second island's buses are not pruned before the
solve starts. Islands with neither an ext grid nor a grid-forming child are
still pruned: their connectivity arcs are capped at zero, so they can never
be energised, and keeping them would leave unservable loads (for example an
optimization problem's priority floor) that make the model infeasible.

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
