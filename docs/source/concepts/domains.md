
# Physical models

monee uses steady-state models for each energy carrier. Select a tab to see
the governing equations, key variables and modelling conventions for that
domain. How to *select* between the formulations described here (default
MISOCP-shaped, piecewise-linear, smooth NLP) is covered in
{doc}`formulations`.

::::{tab-set}

:::{tab-item} Electricity

## AC power flow

The standard nonlinear AC power-flow equations are used, with **voltage
magnitude** *V* and **voltage angle** *δ* as the node decision variables.

**Active and reactive power injected at bus *i*:**

$$P_i = \sum_j V_i V_j \left( G_{ij} \cos(\delta_i - \delta_j) + B_{ij} \sin(\delta_i - \delta_j) \right)$$

$$Q_i = \sum_j V_i V_j \left( G_{ij} \sin(\delta_i - \delta_j) - B_{ij} \cos(\delta_i - \delta_j) \right)$$

where *G* and *B* are the conductance and susceptance entries from the
nodal admittance matrix **Y**.

| Variable | Symbol | Unit |
|---|---|---|
| Voltage magnitude | *V* | per unit |
| Voltage angle | *δ* | radians |
| Active power injection | *P* | MW |
| Reactive power injection | *Q* | Mvar |

## MISOCP Branch-Flow relaxation

The **MISOCP relaxation** (`MISOCP_NETWORK_FORMULATION`) uses the Branch-Flow
Model: the bilinear voltage products are replaced by the lifted variables
*W* = *V*² (squared voltage) and *ℓ* = |*I*|² (squared current), and the
nonlinear coupling is relaxed into the **rotated second-order cone**

$$P_{ij}^2 + Q_{ij}^2 \le W_i \, \ell_{ij}$$

yielding a convex problem solvable globally by a MIQCP solver such as Gurobi
or HiGHS. Two cone encodings are available: the standard rotated cone above,
and a **Lorentz-cone** form — with $s = (W_i + \ell_{ij})/2$ and
$d = (W_i - \ell_{ij})/2$ the same constraint becomes
$P_{ij}^2 + Q_{ij}^2 + d^2 \le s^2$ — which survives inside otherwise
non-convex MIQCPs. A **gap expression**
$W_i \, \ell_{ij} - (P_{ij}^2 + Q_{ij}^2)$ is provided for diagnostics: a
near-zero gap at the optimum certifies that the relaxation is exact.

:::

:::{tab-item} Gas

## Weymouth pipe flow

Pressure drops along a gas pipe are governed by the **Weymouth equation**.
Pressure is modelled as *pressure-squared* p² to keep the equation quadratic:

$$\left(p_i^2 - p_j^2\right) \cdot C^2 = f \left( \dot{m}_{ij}^2 - \dot{m}_{ji}^2 \right)$$

with *f* the Darcy friction factor (see **Friction models** in the
Water / Heat tab — the same three modes apply to gas pipes). The **pipe
constant** *C* encodes the physical properties of the pipe and gas:

$$C^2 = \frac{\pi^2 D^5}{128 \, L \, R \, T_\text{gas} \, Z}$$

where *D* is the inner diameter, *L* is the pipe length, *R* is the specific
gas constant, *T*\_gas is the gas temperature, and *Z* is the compressibility
factor. **Gas density** along each pipe is computed from the ideal-gas law
using the average pressure between the two endpoints.

| Variable | Symbol | Unit |
|---|---|---|
| Pressure squared | *p²* | per unit (of *p*²\_ref) |
| Mass flow | *ṁ* | kg/s |

> **Note:** The pressure state is the **squared per-unit pressure**
> (`pressure_squared_pu`, scaled by the grid's reference pressure). Using *p²*
> as the primary variable avoids the square root that would otherwise appear
> in a direct pressure formulation, keeping the problem structure quadratic
> and easier for NLP solvers; the SI value is recovered after the solve as a
> report-only quantity.

In the **MISOCP-shaped default** (`NL_WEYMOUTH_NETWORK_FORMULATION`) the flow
is split into two non-negative variables $\dot{m}_{ij}$ / $\dot{m}_{ji}$
(`mass_flow_pos` / `mass_flow_neg`) gated by a **direction binary**; by
convention `direction == 0` means *forward* flow, carried by
`mass_flow_neg`. The squared flows enter through a convex epigraph relaxation
kept tight by a small objective term, so MIQCP solvers recognise the problem
as a MISOCP.

Each pipe's flow is capped by the tighter of the grid-wide limit and a
velocity cap,

$$\dot{m}_\text{max} = \min\!\left( f_\text{max},\; \tfrac{\pi}{4} D^2 \, \rho \, v_\text{max} \right)$$

where the gas $v_\text{max}$ defaults to 20 m/s — the erosional velocity cap
for gas pipelines. This per-pipe bound tightens every big-M constraint.

## Smooth (binary-free) hydraulics

The direction binary and the pos/neg split do not survive a pure NLP
relaxation. The **smooth Weymouth variant**
(`SMOOTH_WEYMOUTH_NETWORK_FORMULATION`) therefore uses a single *signed* mass
flow *ṁ* and the smooth identity

$$\dot{m}_{ij}^2 - \dot{m}_{ji}^2 = \dot{m} \, |\dot{m}|, \qquad |\dot{m}| \approx \sqrt{\dot{m}^2 + \varepsilon^2}$$

with the smoothing scale ε (`smoothing_eps`) defaulting to 10⁻³ kg/s. The
pressure drop becomes C∞, monotonic and odd in the flow — no binaries, no
epigraph. The equation is additionally **row-normalised** by the per-pipe
coefficient $C^2 \, p_\text{ref}^2$:

$$\left(\tilde{p}_i^2 - \tilde{p}_j^2\right) = -\frac{f \, \dot{m} \, |\dot{m}|}{C^2 \, p_\text{ref}^2}$$

Since $C^2 \propto D^5$ spans roughly six orders of magnitude across a
wide-diameter network, dividing through gives every pipe a unit coefficient
on the (dimensionless) squared-pressure difference — this materially improves
IPOPT's scaling and conditioning while leaving the solution unchanged. See
{doc}`formulations` for how to apply the smooth formulations.

:::

:::{tab-item} Water / Heat

## Darcy–Weisbach hydraulic flow

Hydraulic pressure drops along water pipes follow the **Darcy–Weisbach
equation**:

$$p_i - p_j = R_m \left( \dot{m}_{ij}^2 - \dot{m}_{ji}^2 \right)$$

where the **hydraulic resistance** *R*\_m is:

$$R_m = f \cdot \frac{L}{D} \cdot \frac{1}{2 \rho A^2}$$

with *f* the Darcy friction factor, *L* the pipe length, *D* the inner
diameter, *ρ* the fluid density, and *A* the cross-sectional area.

As in the gas domain, the default formulation splits the flow into
`mass_flow_pos` / `mass_flow_neg` gated by a **direction binary**
(`direction == 0` ⇒ forward flow via `mass_flow_neg`), and each pipe's flow
is capped by $\min\!\left( f_\text{max},\; \tfrac{\pi}{4} D^2 \rho \, v_\text{max} \right)$
using the water grid's velocity limit `v_max_mps`.

## Friction models

The friction factor *f* is selectable via the `friction_model` option of the
gas and water/heat formulations. Three modes exist:

| Mode | Friction factor | When to use |
|---|---|---|
| `constant` (default) | Per-pipe constant: the turbulent Swamee–Jain asymptote $f = 0.25 / \log_{10}\!\big(\tfrac{\epsilon}{3.7 D}\big)^2$ | Turbulent networks (Re ≳ 10⁴), where it is accurate to within a few percent |
| `pwl` | Piecewise function of the flow: an SOS2 / cubic-spline interpolation of the full drop term over log-spaced Reynolds breakpoints | Laminar-heavy networks needing a MIP-compatible variable friction |
| `nonlinear` | Smooth sigmoid blend of the laminar law and Swamee–Jain (smooth formulations only) | Laminar-heavy networks solved as pure NLPs |

The **constant** mode depends only on relative roughness ε/D, so friction,
Reynolds number and the friction–flow bilinear all drop out of the solver
model. The price is accuracy at low flow: in the laminar regime (Re < 2 300)
it under-estimates the pressure drop by a factor of 5–50.

The **pwl** mode restores variable friction as a piecewise interpolation of
$\varphi(\dot{m}) = f\!\big(\mathrm{Re}(\dot{m})\big) \cdot \dot{m}^2$ (one
odd spline $\psi(\dot{m}) = f \cdot \dot{m} |\dot{m}|$ in the smooth
variants) over log-spaced breakpoints, with a zero-anchor so a no-flow branch
stays feasible and collinear breakpoints removed automatically. It is the
opt-in choice for laminar-heavy networks; around 12 breakpoints suffice.

The **nonlinear** mode replaces the non-smooth `if Re < 2300` regime switch
with a single continuously differentiable blend of the laminar law 64/Re and
the Swamee–Jain correlation:

$$f = (1 - w) \, \frac{64}{\mathrm{Re}} + w \, f_\text{SJ}(\mathrm{Re}),
\qquad w = \frac{1}{1 + e^{-(\mathrm{Re} - \mathrm{Re}_\text{crit})/s}}$$

with $\mathrm{Re}_\text{crit} = 2300$ and sharpness $s = 400$, plus a
Reynolds closure equation per pipe. It is available in the smooth NLP
formulations only.

> **Note:** Reynolds-number variables are stored **scaled by 10⁻⁶**
> (`REYNOLDS_SCALE`), so breakpoints and bounds sit in [0, 10] rather than
> [0, 10⁷] — keeping the coefficient range of the solver matrix manageable.

## Smooth (binary-free) hydraulics

As in the gas domain, the **smooth Darcy–Weisbach variant**
(`SMOOTH_DARCY_WEISBACH_NETWORK_FORMULATION`) replaces the pos/neg split and
the direction binary with one signed flow *ṁ* and the smooth identity

$$\dot{m}_{ij}^2 - \dot{m}_{ji}^2 = \dot{m} \, |\dot{m}|, \qquad |\dot{m}| \approx \sqrt{\dot{m}^2 + \varepsilon^2}$$

(ε = `smoothing_eps`, default 10⁻³ kg/s), so the pressure drop
$p_i - p_j = -R_m \, \dot{m} |\dot{m}|$ is C∞, monotonic and odd. Temperature
upwinding is likewise expressed through the smooth flow split instead of the
direction binary, keeping the heat model a pure NLP. See
{doc}`formulations` for how to select the smooth formulations.

## Temperature propagation

Temperatures are modelled **per-unit**, scaled by the grid's reference
temperature *T*\_ref. The **outlet temperature** accounts for insulation
losses and the ambient temperature:

$$T_\text{out} = T_\text{ext} + \alpha \left( T_\text{in} - T_\text{ext} \right)$$

The **attenuation factor** α depends on the UA product of the insulation
layer and the specific heat capacity of water:

$$\alpha = \frac{\dot{m} \, c_p}{\dot{m} \, c_p + UA},
\qquad UA = \frac{2 \pi \lambda L}{\ln(r_o / r_i)}$$

with λ the insulation conductivity and *r*\_o, *r*\_i the outer and inner
insulation radii. This rational form is the first-order equivalent of the
exponential decay law and stays well-defined at zero flow.

| Variable | Symbol | Unit |
|---|---|---|
| Node pressure | *p* | per unit (of *p*\_ref) |
| Mass flow | *ṁ* | kg/s |
| Inlet / outlet temperature | *T*\_in, *T*\_out | per unit (of *T*\_ref) |
| External temperature | *T*\_ext | K |

:::

::::

---

## Scaling conventions at a glance

Several states are deliberately stored in scaled form; raw SI values are
recovered after the solve.

| Quantity | Stored as | Scale |
|---|---|---|
| Gas pressure | Squared per-unit (`pressure_squared_pu`) | *p*²\_ref |
| Water pressure | Per-unit (`pressure_pu`) | *p*\_ref |
| Heat temperatures | Per-unit (`t_pu`, `t_in_pu`, `t_out_pu`) | *T*\_ref |
| Reynolds number | Re / 10⁶ (`REYNOLDS_SCALE`) | 10⁶ |
| Voltage | Per-unit (*V*, or *W* = *V*² in the MISOCP) | base voltage |
| Pipe flow cap | $\min\!\left( f_\text{max},\; \tfrac{\pi}{4} D^2 \rho \, v_\text{max} \right)$ | gas: *v*\_max = 20 m/s (erosional); water: grid `v_max_mps` |
