
# Physical models

monee uses steady-state models for each energy carrier. Select a tab to see
the governing equations and key variables for that domain.

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

The **MISOCP relaxation** (`MISOCP_NETWORK_FORMULATION`) replaces the bilinear
voltage products with lifted variables and second-order cone constraints,
yielding a convex problem solvable globally by a MIQCP solver such as Gurobi
or HiGHS.

:::

:::{tab-item} Gas

## Weymouth pipe flow

Pressure drops along a gas pipe are governed by the **Weymouth equation**.
Pressure is modelled as *pressure-squared* p² to keep the equation quadratic:

$$\left(p_i^2 - p_j^2\right) \cdot C^2 = \dot{m}_{ij}^2 - \dot{m}_{ji}^2$$

The **pipe constant** *C* encodes the physical properties of the pipe and gas:

$$C = f\!\left(D,\, L,\, T_\text{gas},\, Z\right)$$

where *D* is the inner diameter, *L* is the pipe length, *T*\_gas is the
gas temperature, and *Z* is the compressibility factor.

The **Swamee–Jain** approximation is used for the Darcy–Weisbach friction
factor within the Weymouth constant. **Gas density** along each pipe is
computed from the ideal-gas law using the average pressure between the two
endpoints.

| Variable | Symbol | Unit |
|---|---|---|
| Pressure squared | *p²* | Pa² |
| Mass flow | *ṁ* | kg/s |

> **Note:** Using *p²* as the primary variable avoids the square root that
> would otherwise appear in a direct pressure formulation, keeping the problem
> structure quadratic and easier for NLP solvers.

:::

:::{tab-item} Water / Heat

## Darcy–Weisbach hydraulic flow

Hydraulic pressure drops along water pipes follow the **Darcy–Weisbach
equation**:

$$p_i - p_j = R_m \left( \dot{m}_{ij}^2 - \dot{m}_{ji}^2 \right)$$

where the **hydraulic resistance** *R*\_m is:

$$R_m = f \cdot \frac{L}{D} \cdot \frac{1}{2 \rho A^2}$$

with *f* the friction factor (laminar approximation *f* = 64 / Re), *L*
the pipe length, *D* the inner diameter, *ρ* the fluid density, and *A*
the cross-sectional area.

## Temperature propagation

The **outlet temperature** accounts for insulation losses and the ambient
temperature:

$$T_\text{out} = T_\text{ext} + \alpha \left( T_\text{in} - T_\text{ext} \right)$$

The **attenuation factor** α depends on the UA product of the insulation
layer and the specific heat capacity of water:

$$\alpha = \exp\!\left( -\frac{UA}{\dot{m}\, c_p} \right)$$

| Variable | Symbol | Unit |
|---|---|---|
| Node pressure | *p* | Pa |
| Mass flow | *ṁ* | kg/s |
| Inlet / outlet temperature | *T*\_in, *T*\_out | K |
| External temperature | *T*\_ext | K |

:::

::::
