=================
Economic dispatch
=================

:func:`~monee.problem.create_economic_dispatch_problem` builds an optimal
power flow: it chooses the active power of every
:class:`~monee.model.child.PowerGenerator` so that the total generation cost,
plus the cost of importing from the external grid when that is priced, is
minimal while the AC power flow holds and voltages and line loadings stay in
their limits. It works with the default AC formulation and IPOPT, the
third-party solver monee uses by default, without further setup; for the
convex MISOCP variant and mixed-integer extensions see
:doc:`../concepts/formulations` and :doc:`../how-to/use_pyomo_solver`.

The same builder dispatches gas and heat. ``gas_cost_default`` adds the gas
sources and the gas external grids, ``heat_cost_default`` the heat units and
the water external grids, and once the carriers of a coupling unit (CHP,
power to heat, gas to heat, power to gas, gas to power) are dispatched its
``regulation`` becomes a decision too. Every price is in currency per MW and
period, so a gas price is per MW of higher heating value and a heat price per
MW of heat. :ref:`problems/economic_dispatch:Gas and heat` explains how each
carrier enters.

Mathematical formulation
========================

Sets and parameters
-------------------

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Symbol
     - Meaning
     - Set by
   * - :math:`\mathcal{N}`
     - Buses of the electricity grid.
     - The network.
   * - :math:`\mathcal{L}`
     - Power branches (lines, transformers and generic branches), with
       from-bus :math:`f(\ell)` and to-bus :math:`t(\ell)`.
     - The network.
   * - :math:`\mathcal{G}`
     - Active, not ignored power generators.
     - The network.
   * - :math:`\mathcal{E}`
     - External grids (slack connections).
     - The network.
   * - :math:`\mathcal{K}(i)`
     - All children attached to bus :math:`i`: loads, generators, external
       grids, shunts, storages.
     - The network.
   * - :math:`c_g`
     - Marginal cost of generator :math:`g` in currency per MW and period, so
       :math:`c_g P_g` is the cost of one period. An hourly price in currency
       per MWh enters as :math:`c_g = \text{price} \cdot \Delta t` with the
       step length :math:`\Delta t` in hours (``dt_h``, 1 by default).
     - The generator's ``cost``, else ``gen_cost_default``.
   * - :math:`\overline{P}_g`
     - Capacity of generator :math:`g` in MW.
     - The ``p_mw`` magnitude passed at creation.
   * - :math:`c_e`
     - Price of the exchange with external grid :math:`e`, in currency per MW
       and period like :math:`c_g`.
     - The grid's ``cost``, else ``ext_grid_cost_default``.
   * - :math:`V^\text{ref}_e, \delta^\text{ref}_e`
     - Voltage magnitude in per unit and angle in radians held by external
       grid :math:`e` at its bus; the magnitude only when the grid regulates
       voltage.
     - The grid's ``vm_pu``, ``va_degree`` (converted to radians) and
       ``regulate_vm``.
   * - :math:`\underline{V}, \overline{V}`
     - Voltage band in per unit.
     - ``bounds_vm``
   * - :math:`\lambda`
     - Line loading limit as a fraction of the rating.
     - ``bounds_lp[1]``
   * - :math:`\overline{S}_\ell, \overline{I}_\ell`
     - Apparent-power rating in MVA and current rating in kA of branch
       :math:`\ell`.
     - The branch's ``max_s_mva`` and ``max_i_ka``.
   * - :math:`\underline{p}_e, \overline{p}_e`
     - Exchange bounds of the external grid in MW, load convention.
     - ``ext_grid_bounds``
   * - :math:`\overline{m}^\text{imp}_e, \overline{m}^\text{exp}_e`
     - Import and export caps of external grid :math:`e` in MW, as positive
       magnitudes; unbounded when not set.
     - The grid's ``max_import_mw`` and ``max_export_mw``.
   * - :math:`R`
     - Ramp limit in MW per period.
     - ``ramp_limit``
   * - :math:`S_\text{base}`
     - Apparent-power base of the grid in MVA.
     - The grid's ``sn_mva``.
   * - :math:`\mathcal{S}`
     - Active gas sources and gas external grids (gas supply). Only with
       ``gas_cost_default``.
     - The network.
   * - :math:`c_s`
     - Price of gas supply :math:`s` in currency per MW of higher heating
       value and period.
     - The component's ``cost``, else ``gas_cost_default``.
   * - :math:`k_s`
     - MW of higher heating value carried by 1 kg/s of the gas on the grid of
       :math:`s`, :math:`3.6 \cdot \text{HHV}` with the HHV in kWh/kg (42.44
       for the default L gas, 55.08 for methane).
     - The grid's ``higher_heating_value_kwh_per_kg``.
   * - :math:`\overline{m}_s`
     - Capacity of gas source :math:`s` in kg/s.
     - The ``mass_flow_kgs`` magnitude passed at creation.
   * - :math:`\mathcal{U}`
     - Heat units: active heat generators and generating heat exchangers
       (active or passive). Only with ``heat_cost_default``.
     - The network.
   * - :math:`c_u, \overline{Q}_u`
     - Price in currency per MW heat and period, and capacity in MW, of heat
       unit :math:`u`.
     - The unit's ``cost``, else ``heat_cost_default``; the ``q_mw``
       magnitude passed at creation.
   * - :math:`\mathcal{W}, I(w)`
     - Active water external grids, and the heat island of :math:`w`: the
       water junctions and coupler control nodes connected to its junction.
     - The network.
   * - :math:`c_w`
     - Price of the heat that water external grid :math:`w` supplies, in
       currency per MW heat and period.
     - The grid's ``cost``, else ``heat_cost_default``.
   * - :math:`\underline{h}_w, \overline{h}_w`
     - Bounds on the heat exchange of :math:`w` in MW, load convention.
     - ``bounds_ext_heat_mw``, default ``(None, 0.0)``.
   * - :math:`\underline{T}, \overline{T}`
     - Water temperature band in K.
     - ``bounds_t_k``, default 278.15 to 423.15.
   * - :math:`\mathcal{C}, \overline{r}_c`
     - Coupling units whose carriers are all dispatched, and the
       ``regulation`` of unit :math:`c` when the problem is applied (1
       unless derated).
     - The network, ``dispatch_coupling_points``.

Decision variables
------------------

.. list-table::
   :header-rows: 1
   :widths: 18 52 30

   * - Symbol
     - Meaning
     - Result column
   * - :math:`P_g`
     - Active power produced by generator :math:`g`, in MW, :math:`P_g \ge 0`.
     - ``PowerGenerator.p_mw`` is :math:`-P_g`.
   * - :math:`p_e`
     - Exchange with external grid :math:`e` in MW, negative for import.
     - ``ExtPowerGrid.p_mw``
   * - :math:`q_e`
     - Reactive exchange with external grid :math:`e` in Mvar.
     - ``ExtPowerGrid.q_mvar``
   * - :math:`q_v`
     - Reactive power of voltage-controlled generator :math:`v` in Mvar, load
       convention; free so that its bus holds its voltage setpoint.
     - ``VoltageControlledGenerator.q_mvar``
   * - :math:`V_i, \delta_i`
     - Voltage magnitude in per unit and angle in radians at bus :math:`i`.
     - ``Bus.vm_pu``, ``Bus.va_radians``
   * - :math:`P_\ell^{s}, Q_\ell^{s}`
     - Active and reactive power entering branch :math:`\ell` at end
       :math:`s \in \{f, t\}`, in MW and Mvar.
     - ``p_from_mw``, ``q_from_mvar``, ``p_to_mw``, ``q_to_mvar``
   * - :math:`m_s`
     - Gas injected by gas supply :math:`s` in kg/s; for an external grid
       negative when it takes gas back.
     - ``Source.mass_flow_kgs`` and ``ExtHydrGrid.mass_flow_kgs`` are
       :math:`-m_s`.
   * - :math:`Q_u`
     - Heat produced by heat unit :math:`u` in MW, :math:`Q_u \ge 0`.
     - ``HeatGenerator.q_mw_heat`` and the exchanger's ``q_mw_set`` are
       :math:`-Q_u`.
   * - :math:`r_c`
     - Operating point of coupling unit :math:`c` as a fraction of its
       setpoint.
     - ``regulation`` of the control node (CHP, both CHP variants, power to
       heat, gas to heat) or of the branch (the power to heat and gas to
       heat HG variants, gas to power, power to gas).
   * - :math:`H_w`
     - Heat supplied by water external grid :math:`w` in MW, negative when it
       absorbs a surplus. Not a variable of its own but an expression over
       the island's temperatures and flows, see
       :ref:`problems/economic_dispatch:How heat is priced`.
     - None; the exchange is :math:`-H_w` in the load convention.

With gas and heat off, and ``include_storages`` left ``False``, the builder
promotes no other setpoint to a decision. Loads, storages, coupling units and
the reactive power of :class:`~monee.model.child.PowerGenerator` units then
stay at their setpoints. Gas and heat dispatch add :math:`m_s`, :math:`Q_u` and
:math:`r_c`; storages become decisions only with ``include_storages`` over a
horizon, and loads, :class:`~monee.model.storage.ThermalStorage` and every
other setpoint stay fixed in every mode. A
:class:`~monee.model.child.PowerShunt` :math:`k` at bus :math:`i` enters the
balance with :math:`p_k = g^\text{sh}_k V_i^2` and
:math:`q_k = -b^\text{sh}_k V_i^2`, where :math:`g^\text{sh}_k` (``gs_mw``) is
the active power in MW it draws at 1 pu and :math:`b^\text{sh}_k`
(``bs_mvar``) the reactive power in Mvar it injects at 1 pu. A
:class:`~monee.model.child.VoltageControlledGenerator` keeps its active power
fixed and uncosted and pins :math:`V_i` at its bus to its ``vm_pu`` setpoint;
its reactive power :math:`q_v` is a free network variable that the balance
determines.

Problem
-------

.. math::

   \begin{aligned}
   \min \quad & \sum_{g \in \mathcal{G}} c_g \, r_g P_g \;+\; \sum_{e \in \mathcal{E}} c_e \, (-p_e)
     \;+\; \sum_{s \in \mathcal{S}} c_s k_s\, m_s \;+\; \sum_{u \in \mathcal{U}} c_u Q_u
     \;+\; \sum_{w \in \mathcal{W}} c_w H_w \;+\; A(x) \\
   \text{s.t.} \quad
   & \sum_{\ell:\, f(\ell) = i} S_\ell^{f} + \sum_{\ell:\, t(\ell) = i} S_\ell^{t} + \sum_{k \in \mathcal{K}(i)} S_k = 0
     && \forall i \in \mathcal{N} && (1),\,(2) \\
   & S_\ell^{s} = S_\text{base}\, \kappa_\ell\, \tilde{V}_{s(\ell)} \big( Y_\ell^{sf}\, \tilde{V}_{f(\ell)} + Y_\ell^{st}\, \tilde{V}_{t(\ell)} \big)^{*}
     && \forall \ell \in \mathcal{L},\, s \in \{f, t\} && (3) \\
   & V_i = V^\text{ref}_e, \;\; \delta_i = \delta^\text{ref}_e
     && \text{at the bus of each } e \in \mathcal{E} && (4) \\
   & 0 \le P_g \le \overline{P}_g
     && \forall g \in \mathcal{G} && (5) \\
   & \underline{V} \le V_i \le \overline{V}
     && \forall i \in \mathcal{N} \text{ whose magnitude is not pinned} && (6) \\
   & \text{loading}_\ell^{s} \le \lambda
     && \forall \ell \in \mathcal{L},\, s \in \{f, t\} && (7) \\
   & \underline{p}_e \le p_e \le \overline{p}_e
     && \forall e \in \mathcal{E} && (8) \\
   & -\overline{m}^\text{imp}_e \le p_e \le \overline{m}^\text{exp}_e
     && \forall e \in \mathcal{E} \text{ unless (8) is set} && (9) \\
   & 0 \le m_s \le \overline{m}_s
     && \forall \text{ gas sources } s \in \mathcal{S} && (10) \\
   & 0 \le Q_u \le \overline{Q}_u
     && \forall u \in \mathcal{U} && (11) \\
   & 0 \le r_c \le \overline{r}_c
     && \forall c \in \mathcal{C} && (12) \\
   & q^\text{delivered}_x = q_x
     && \forall \text{ heat exchangers } x && (13) \\
   & \underline{h}_w \le -H_w \le \overline{h}_w
     && \forall w \in \mathcal{W} && (14) \\
   & \underline{T} \le T \le \overline{T}
     && \forall \text{ water temperatures } T && (15)
   \end{aligned}

Rows (10) to (15) and the gas and heat terms of the objective exist only for
the carriers that are dispatched: (10) with ``gas_cost_default``; (11), (13),
(14) and (15) with ``heat_cost_default``; and (12) once all carriers of a
coupling unit are. :math:`r_g` is the ``regulation`` of generator :math:`g`
(1 by default), so a generator is priced on the power the balance actually
sees.

Power balance (1), (2). Here :math:`S_\ell^{s} = P_\ell^{s} + j Q_\ell^{s}`
and :math:`S_k = p_k + j q_k` is the active and reactive injection of child
:math:`k`. The real part of the balance is (1), the imaginary part (2). The
children enter in the load convention: a load :math:`k` contributes its demand
setpoint :math:`p_k + j q_k` (positive for consumption), generator :math:`g`
contributes :math:`-P_g` plus its fixed reactive setpoint, the external grid
its exchange :math:`p_e + j q_e`, and every other child its fixed setpoint or,
for shunts, the voltage-dependent draw above. In general each child enters as
``regulation`` times its value; economic dispatch keeps every child's
``regulation`` at its setpoint and only makes that of a coupling unit a
decision (12).

Branch flows (3) are the standard :math:`\pi`-model of MATPOWER. With the
series admittance :math:`y_\ell = 1 / (r_\ell + j x_\ell)` (set to 0 when
:math:`r_\ell = x_\ell = 0`, so a zero-impedance branch does not couple its two
buses and carries only its shunt flows, like an open branch rather than a
closed switch), the shunt admittances :math:`y^{f}_\ell` and :math:`y^{t}_\ell`
at the two ends, the complex tap :math:`T_\ell = \tau_\ell e^{j\theta_\ell}`
with the phase shift :math:`\theta_\ell` in radians, the complex bus voltages
:math:`\tilde{V}_i = V_i e^{j\delta_i}` and the in-service flag
:math:`\kappa_\ell \in \{0, 1\}` (``on_off``), the branch admittance matrix is

.. math::

   \begin{pmatrix} Y_\ell^{ff} & Y_\ell^{ft} \\ Y_\ell^{tf} & Y_\ell^{tt} \end{pmatrix}
   = \begin{pmatrix} (y_\ell + y^{f}_\ell) / \tau_\ell^2 & -y_\ell / T_\ell^{*} \\
     -y_\ell / T_\ell & y_\ell + y^{t}_\ell \end{pmatrix},

all impedances in per unit of the grid base.
:ref:`PowerLine <components/electricity:PowerLine>` and
:ref:`Trafo <components/electricity:Trafo>` derive :math:`r_\ell, x_\ell` from
their physical parameters,
:ref:`GenericPowerBranch <components/electricity:GenericPowerBranch>` takes
them directly.

Slack (4). Each external grid pins the voltage magnitude and the angle of its
bus (``vm_pu``, ``va_degree``) and absorbs the power balance through
:math:`p_e, q_e`. A pinned magnitude is not subject to (6), so a slack
setpoint outside the voltage band is kept, not clipped. With
``regulate_vm=False`` only the angle is pinned and the magnitude is limited by
(6) instead. A :class:`~monee.model.child.VoltageControlledGenerator` pins its
bus magnitude to its ``vm_pu`` setpoint in the same way, so that bus is not
subject to (6) either and a setpoint outside the band is kept. Under an
islanding extension the angle reference is managed by the extension. Buses
whose magnitude is not pinned also keep the bus variable's own box of 0.5 to
1.5 pu, which matters only when ``check_vm`` is off, and every angle not
pinned by (4) keeps the box :math:`-\pi \le \delta_i \le \pi`.

Line loading (7). When a branch carries an MVA rating the limit is the
apparent power at each end,

.. math::

   \frac{(P_\ell^{s})^2 + (Q_\ell^{s})^2}{(\lambda\, \overline{S}_\ell)^2} \le 1,

otherwise it is the current. The from-end current in kA is
:math:`I_\ell^{f} = \sqrt{(P_\ell^{f})^2 + (Q_\ell^{f})^2 + \varepsilon^2} / (\sqrt{3}\, V_f\, U_f)`
with the bus base voltage :math:`U_f` in kV and a smoothing constant
:math:`\varepsilon = 10^{-4}` MVA, and the loading is
:math:`I_\ell^{f} / \overline{I}_\ell`. The to-end current is referred to the
from-side base,
:math:`\text{loading}_\ell^{t} = I_\ell^{t} U_t / (U_f \overline{I}_\ell)`.
A branch without ``max_s_mva`` is limited by its current rating, which
defaults to 3.19 kA for every line, transformer and generic branch. The limit
is dropped only when the branch model's ``max_i_ka`` is at least 999 kA, so
pass ``max_i_ka=999`` to leave a line unlimited. Do not use ``None``: passing
it to ``mx.create_line`` keeps the 3.19 kA default, and setting it on the
model makes the default AC formulation fail with a ``TypeError`` because it
divides by the rating to report ``loading_*_pu``.
``mm.Trafo`` and ``mx.create_trafo`` take no rating argument; set
``max_i_ka`` on the transformer model after creating it. Under an MVA rating
the result columns ``loading_from_pu`` and ``loading_to_pu`` still report the
current against ``max_i_ka``, so they do not show the limit that was enforced;
check the apparent power directly.

Exchange price and bounds (8), (9). The external-grid term of the objective
and the problem bounds (8) exist only when ``ext_grid_cost_default`` is set.
The grid's own caps (9) are the bounds of the exchange variable itself and
apply whether or not the exchange is priced, except when
``ext_grid_cost_default`` and ``ext_grid_bounds`` are both given. (8) is then
written onto the same variable and replaces both bounds of (9) instead of
tightening them, so a wider side of ``ext_grid_bounds`` lifts the
corresponding ``max_import_mw`` or ``max_export_mw`` cap. Both sides must be
numbers; a ``None`` side raises a ``TypeError`` while the constraints are
built. Keep
``ext_grid_bounds`` inside the caps, or leave it unset, when the caps must
hold. Import makes :math:`p_e` negative, so :math:`c_e (-p_e)` charges an
import at :math:`c_e` and credits an export at the same price.

Auxiliary terms. :math:`A(x)` collects the formulation's auxiliary terms (see
:doc:`index`). The default AC formulation adds none for the electricity grid,
so on a pure electricity network ``result.objective`` equals
``result.user_objective``. On a multi-energy network the gas and water models
can add terms. Without heat dispatch this holds even on the default CasADi
backend, where heat exchangers add a pull that closes their duty inequality;
with ``heat_cost_default`` the duties are exact (13) and that pull is gone.
On a mixed-integer backend the default gas and water pipe models also add
small epigraph terms, and under MISOCP the branch losses
:math:`\sum_\ell r_\ell\, \ell_\ell` enter :math:`A`. Compare
``result.user_objective`` in those cases.

The balances and flows (1) to (3) are the electricity rows of the network
equations :math:`F(x) = 0` of :doc:`index`. On a multi-energy network the gas,
water and heat equations and the coupling equations are rows of :math:`F` as
well, with every setpoint that is not a decision fixed, so their own variable
bounds can make the dispatch infeasible. Exact exchanger duties (13) make
heat demand hard: a heat network that cannot serve its load at any dispatch
is infeasible rather than short, so use :doc:`load_shedding` to curtail it.

Gas and heat
------------

Gas supply. Each gas source and gas external grid is priced on the energy it
supplies, :math:`c_s k_s m_s`, with :math:`k_s` read from its own grid, so an
L gas and a methane network convert with their own heating values. Gas
sources become decisions between zero and their setpoint (10). The gas
external grid follows the load convention like the power slack: import is
charged and export is credited at the same price, so a source cheaper than
the grid runs flat out and sells its surplus. Cap the export with the grid's
``max_export_kgs`` when that is not intended. Gas pressures keep the band of
the gas junction model; tighten it with ``problem.bounds`` on
``pressure_squared_pu`` as shown in :ref:`problems/economic_dispatch:Recipes`.

Heat supply. Heat generators and generating exchangers are priced on the
heat they deliver, :math:`c_u Q_u`, and become decisions between zero and
their setpoint (11). For an active exchanger the decision is ``q_mw_set``:
the duty changes at the design mass flow, so the temperature spread moves and
the hydraulics stay put. Every heat exchanger, the consuming ones included,
delivers exactly its duty (13), and (15) keeps every water temperature in a
physical band. Unlike ``bounds_t`` of :doc:`load_shedding`, ``bounds_t_k`` is
in K, not in per unit. A network designed for hotter water leaves the band
and is infeasible under the default: the CHP branch of
:func:`~monee.network.mes.create_monee_benchmark_net` reaches about 515 K, so
widen ``bounds_t_k`` or pass ``None`` there.

Coupling units. A coupling unit carries no price of its own. Its fuel is paid
where it enters the network: the gas of a CHP at the gas source or gas
external grid, the electricity of a power to heat unit at the generator or
power external grid. Its ``regulation`` becomes a decision (12) once every
non-electric carrier it touches is dispatched: CHP and gas to heat need gas
and heat, power to heat needs heat, gas to power and power to gas need gas.
The upper bound is the regulation the unit had when the problem was applied,
so a derated unit, or one a timeseries sets to part load, keeps that limit.
With ``ext_grid_cost_default`` unset, power from the external grid is free,
which makes the merit order of every unit with an electric side degenerate;
the builder warns in that case. ``dispatch_coupling_points=False`` keeps
every unit at its setpoint.

How heat is priced
------------------

A water external grid is the hydraulic slack of its island and, through its
pinned feed temperature, an unlimited heat plant. Its heat is implicit: the
slack's own nodal heat balance is the redundant one the solver drops. The
builder therefore writes :math:`H_w` as the sum over the island
:math:`I(w)` of every nodal heat balance without the mass flow children:

.. math::

   H_w = \frac{c_p\, T_\text{ref}}{10^6} \sum_{n \in I(w)} \text{heat terms}_n
       = \text{heat loads} + \text{exchanger duties} + \text{pipe losses}
       - \text{other heat injection}.

In a closed loop this is the heat the slack adds to the returning water. At a
supply and return pair built with ``attach_heat_plant`` it is
:math:`\dot m\, c_p\, (T_\text{supply} - T_\text{return})`. On an open
island, water that leaves through a :class:`~monee.model.child.Sink` or a
:class:`~monee.model.child.ConsumeHydrGrid` counts as returned to the plant
at its own temperature, and the enthalpy of a water
:class:`~monee.model.child.Source` counts as slack heat.

A negative :math:`H_w` means the slack absorbs a surplus, which a district
heating plant usually cannot sell. The default ``bounds_ext_heat_mw=(None,
0.0)`` therefore lets the slack supply heat but not absorb any; pass ``None``
to allow absorption, which is then credited at :math:`c_w`. With the default,
a unit whose heat cannot be absorbed runs only as far as the island needs
it, and a fixed heat source larger than the demand makes the dispatch
infeasible. On an open island this includes heat carried out by a sink: a
unit that heats water leaving through a :class:`~monee.model.child.Sink` is
held back under the default, so pass ``bounds_ext_heat_mw=None`` when that
outflow is meant to be served.

Limits of the heat pricing:

- An island may hold one active water external grid. A second one raises a
  ``ValueError``, so networks built with ``heat_plant_mode="two_port"`` are
  not supported.
- The McCormick heat formulation (``formulation="heat_convex_milp"``) and the
  LTC extension replace the nodal heat balance and raise a ``ValueError``.
- An island without a water external grid has no :math:`H_w`; its heat
  balance must close through the heat units alone.
- The gas :class:`~monee.model.extension.islanding.gas.GridFormingSource` of
  the islanding extension is not priced, like the power
  ``GridFormingGenerator``.

Multi-period dispatch
---------------------

Over the periods :math:`h \in \mathcal{T}` the problem is solved jointly, the
objective becomes the sum of the per-period objectives (without a time-step
weight), every price may vary per period, and the ramp limit couples
consecutive periods. Written for the generators (the gas and heat rows follow
below):

.. math::

   \begin{aligned}
   \min \quad & \sum_{h \in \mathcal{T}} \Big( \sum_{g \in \mathcal{G}} c_{g,h} \, r_{g,h} P_{g,h}
     + \sum_{e \in \mathcal{E}} c_{e,h} \, (-p_{e,h}) + A(x_h) \Big) \\
   \text{s.t.} \quad & (1) \text{ to } (15) \quad \forall h \in \mathcal{T} \\
   & |r_{g,h} P_{g,h} - r_{g,h-1} P_{g,h-1}| \le R \quad \forall g \in \mathcal{G},\; h \in \mathcal{T} \text{ with } P_{g,h-1} \text{ known}
   \end{aligned}

The ramp rows only exist when ``ramp_limit`` is set. For the first period,
:math:`P_{g,0}` is read from the run's ``initial_state`` entry for the
generator's ``p_mw``, taken in the component's load convention
(:math:`P_{g,0} = -p_\text{mw}`, so 5 MW of generation is given as ``-5``).
Without that entry the first period is free.

A float ``ramp_limit`` applies to every dispatched carrier; a dict such as
``{"power": 1.0, "heat": 0.5}`` sets one limit per carrier. Every limit is in
MW per period and applies to the served quantity, the setpoint times that
period's ``regulation``, so a regulation series (for example a PV
availability) is ramp limited on the power the network actually receives. A
period without a known regulation reuses the current one. A gas source ramps
:math:`k_s |m_{s,h} - m_{s,h-1}| \le R`,
seeded by the ``initial_state`` entry for its ``mass_flow_kgs``; a heat
generator ramps on ``q_mw_heat`` and a generating exchanger on ``q_mw_set``.

``ramp_limit`` does not reach the coupling units.
``problem.constraints.regulation_ramp(L)`` adds
:math:`|r_{c,h} - r_{c,h-1}| \le L` for every component whose ``regulation``
is a decision, in the dispatch the operating points :math:`r_c` of the
coupling units (12); a regulation that is data, such as an availability
series, is skipped. The first period is free unless ``initial_state`` gives
the previous ``(id, "regulation")``.
:doc:`../tutorials/mes_storage_and_ramps` uses it.

With ``include_storages=True`` every :class:`~monee.model.ElectricStorage`,
and every :class:`~monee.model.GasStorage` when gas is dispatched, is
dispatched with its state of charge carried across the periods; a
``terminal_state`` entry fixes the final charge. Outside a horizon the
storages keep their setpoints and the builder warns.

What the problem does not model
-------------------------------

Only the active power of the generators, the gas and heat units, the
coupling units and, over a horizon, the storages is dispatched. The reactive
power of the generators stays at the ``q_mvar`` setpoint, so voltage support
comes from the external
grids (whose magnitude the optimiser chooses when ``regulate_vm=False``),
shunts and :class:`~monee.model.child.VoltageControlledGenerator` units. There
is no minimum stable generation and no commitment decision: a generator can
run anywhere between zero and its capacity. Loads, gas sinks and heat
demand are served in full; to curtail demand use :doc:`load_shedding`.
Coupling units have no operating cost of their own and import and export
share one price per external grid; both can be added as user objective terms,
see :ref:`problems/economic_dispatch:Recipes`.

How costs enter
===============

Each generator contributes ``cost * dispatched MW`` to the objective, the
dispatched MW being ``-p_mw`` times its ``regulation``. Set the
price (currency per MW and period, that is the hourly price times ``dt_h``)
when creating the component
(``mx.create_power_generator(..., cost=...)``), assign it to the model
afterwards, or vary it per period via
:meth:`~monee.simulation.timeseries.TimeseriesData.add_objective_data`. Generators
without a cost fall back to ``gen_cost_default``.

The ``p_mw`` magnitude passed at creation is the capacity: the problem makes
``p_mw`` a decision variable ranging from zero to that magnitude, so a
generator created with ``p_mw=0.6`` can dispatch anywhere in 0 to 0.6 MW.

Gas and heat units take ``cost`` the same way:
``mx.create_gas_source(..., cost=...)`` and
``mx.create_gas_ext_grid(..., cost=...)`` in currency per MW of higher
heating value, ``mx.create_heat_generator(..., cost=...)``,
``mx.create_heat_exchanger(..., cost=...)`` (generators only) and
``mx.create_water_ext_grid(..., cost=...)`` in currency per MW heat. Over a
horizon, ``add_objective_data`` varies the price of a child and
``add_branch_series(hx_id, "cost", [...])`` that of a heat exchanger. A unit
without its own ``cost`` falls back to the carrier default, like the slack it
competes with, so the two tie and their split is arbitrary; give one of them
its own price. A ``cost`` on a component whose carrier is not dispatched, on
a consuming exchanger or on a water source is never read, and the builder
warns about it.

Signs and the load convention
=============================

monee uses the load convention everywhere (positive means consumption), and
the dispatch objective is written against it:

- A solved generator reports a negative ``p_mw``; the objective charges
  ``cost * (-p_mw) * regulation``, i.e. the produced MW.
- ``ExtPowerGrid.p_mw`` is negative when the network
  imports and positive when it exports (see
  :ref:`data-model-slack-sign`). With ``ext_grid_cost_default`` set, import
  is charged at that price and export is credited at the same price. An
  external price above a generator's cost therefore makes exporting
  profitable, and the optimiser will run that generator at full capacity to
  sell; bound the exchange with ``ext_grid_bounds`` if that is not intended.
- ``ext_grid_bounds`` is stated in the same convention: to cap the import at
  ``X`` MW while forbidding export, pass ``(-X, 0.0)``. Writing it with the
  signs flipped, ``(0.0, X)``, does not fail: it silently forbids importing
  and allows an export of up to ``X`` MW.
- The gas external grid reports ``mass_flow_kgs`` in the same convention and
  is priced on :math:`c_s k_s (-\text{mass\_flow\_kgs})`, so gas import is
  charged and export credited.
- ``bounds_ext_heat_mw`` is stated in the load convention as well: the heat
  exchange of a water external grid is :math:`-H_w`, negative while the slack
  supplies heat. ``(-X, 0.0)`` lets it supply at most ``X`` MW and absorb
  nothing.

By default (``ext_grid_cost_default=None``) the external grid is left out of
the objective entirely, and a ``cost`` set on an individual external grid is
ignored as well. Slack import is then free, so the optimum degenerates to
importing as much as the network limits allow and dispatching only the local
generation those limits force. Pass a price
whenever the network has an external grid and you want a meaningful dispatch.

Keyword reference
=================

.. list-table::
   :header-rows: 1
   :widths: 26 12 12 50

   * - Keyword
     - Symbol
     - Default
     - Meaning
   * - ``gen_cost_default``
     - :math:`c_g`
     - ``1.0``
     - Price (currency per MW and period) for generators without their own
       ``cost``.
   * - ``ext_grid_cost_default``
     - :math:`c_e`
     - ``None``
     - Price for the external grid exchange. ``None`` excludes the slack
       from the objective (free import, see above) and disables
       ``ext_grid_bounds``; a number prices import and export symmetrically.
       An external grid's own ``cost`` overrides it.
   * - ``bounds_vm``
     - :math:`\underline{V}, \overline{V}`
     - ``(0.9, 1.1)``
     - Voltage magnitude bounds (6), applied to the actual voltage decision
       variable of the formulation (``vm_pu`` under the NLP,
       ``vm_pu_squared`` under MISOCP, where the bounds are squared).
   * - ``bounds_lp``
     - :math:`\lambda`
     - ``(0, 1.0)``
     - Line loading limit (7), enforced on both branch ends. The lower bound
       must be ``0``; a non-zero minimum loading would force flow onto every
       line and raises a ``ValueError``.
   * - ``ext_grid_bounds``
     - :math:`\underline{p}_e, \overline{p}_e`
     - ``None``
     - ``(min, max)`` bounds (8) on ``ExtPowerGrid.p_mw`` in the load
       convention; only used together with ``ext_grid_cost_default``. When
       set, replaces the grid's own ``max_import_mw`` and ``max_export_mw``
       caps (9) instead of tightening them.
   * - ``ramp_limit``
     - :math:`R`
     - ``None``
     - Maximum MW change of each dispatched unit between consecutive
       periods; a temporal constraint that only binds in timeseries and
       multi-period runs. A float applies to every dispatched carrier, a
       dict with keys among ``"power"``, ``"gas"`` and ``"heat"`` sets one
       limit per carrier.
   * - ``check_vm`` / ``check_lp``
     -
     - ``True``
     - ``False`` drops constraint (6) or (7). The bus variable's own box of
       0.5 to 1.5 pu still applies without (6).
   * - ``debug``
     -
     - ``False``
     - Verbose problem construction.
   * - ``gas_cost_default``
     - :math:`c_s`
     - ``None``
     - Keyword only. Price per MW of higher heating value for gas supply
       without its own ``cost``. ``None`` leaves gas out of the dispatch.
   * - ``heat_cost_default``
     - :math:`c_u, c_w`
     - ``None``
     - Keyword only. Price per MW heat for heat units and water external
       grids without their own ``cost``. ``None`` leaves heat out of the
       dispatch.
   * - ``bounds_ext_heat_mw``
     - :math:`\underline{h}_w, \overline{h}_w`
     - ``(None, 0.0)``
     - Keyword only, heat only. Bounds (14) on the heat exchange of each water
       external grid, load convention; ``None`` on a side leaves it open,
       ``None`` as a whole removes (14).
   * - ``bounds_t_k``
     - :math:`\underline{T}, \overline{T}`
     - ``(278.15, 423.15)``
     - Keyword only, heat only. Water temperature band (15) in K; ``None``
       removes it. It only tightens existing bounds, and a temperature pinned
       outside the band keeps its pin.
   * - ``dispatch_coupling_points``
     -
     - ``True``
     - Keyword only. Make the ``regulation`` of each coupling unit whose
       carriers are dispatched a decision (12).
   * - ``include_storages``
     -
     - ``False``
     - Keyword only. Dispatch electric storages, and gas storages when gas is
       dispatched, over a horizon.

Worked example
==============

Two generators and a priced grid connection. The cheap unit runs at capacity,
the expensive one stays off because importing at 25 is cheaper than
generating at 40, and the remainder of the demand (plus losses) is imported:

.. testcode::

    import monee
    import monee.express as mx
    import monee.model as mm
    from monee.problem import create_economic_dispatch_problem

    net = mx.create_multi_energy_network()
    b0, b1, b2 = mx.create_bus(net), mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, 200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_line(net, b1, b2, 200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_generator(net, b1, p_mw=0.6, q_mvar=0.0, cost=10, name="cheap")
    mx.create_power_generator(net, b2, p_mw=0.6, q_mvar=0.0, cost=40, name="pricey")
    mx.create_power_load(net, b2, p_mw=0.8, q_mvar=0.0, name="demand")

    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=25,
        ext_grid_bounds=(-0.5, 0.0),
    )
    result = monee.run_energy_flow_optimization(net, problem)

    gens = result.get(mm.PowerGenerator)
    for name, p in zip(gens["name"], gens["p_mw"]):
        print(f"{name}: {abs(p):.2f} MW dispatched")
    ext_p = result.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
    print(f"grid exchange: {ext_p:.2f} MW (negative = import)")

.. testoutput::

    cheap: 0.60 MW dispatched
    pricey: 0.00 MW dispatched
    grid exchange: -0.21 MW (negative = import)

Evaluating the objective of the formulation on the solved values reproduces
the number the solver minimised:

.. testcode::

    by_hand = sum(c * -p for c, p in zip(gens["cost"], gens["p_mw"])) + 25 * -ext_p
    print(f"sum c_g P_g + c_e (-p_e) = {by_hand:.3f}")
    print(f"result.user_objective    = {result.user_objective:.3f}")

.. testoutput::

    sum c_g P_g + c_e (-p_e) = 11.357
    result.user_objective    = 11.357

The import of about 0.214 MW covers the 0.2 MW gap and the losses of the two
lines, and neither the import cap nor any voltage or loading limit binds.
The load convention explains the signs printed above.

Gas merit order
---------------

A gas source at 10 per MW competes with a grid connection at 30 per MW, both
per MW of higher heating value. The source runs at its 0.1 kg/s capacity and
the grid covers the rest of the 0.15 kg/s demand; ``max_export_kgs=0`` keeps
the source from selling gas back to the grid:

.. testcode::

    gas_net = mm.Network(gas_model=mm.create_gas_grid("gas"))
    j0, j1, j2 = (mx.create_gas_junction(gas_net) for _ in range(3))
    mx.create_gas_pipe(gas_net, j0, j1, diameter_m=0.15, length_m=400)
    mx.create_gas_pipe(gas_net, j1, j2, diameter_m=0.15, length_m=400)
    mx.create_gas_ext_grid(gas_net, j0, max_export_kgs=0)
    mx.create_gas_sink(gas_net, j1, mass_flow_kgs=0.15)
    mx.create_gas_source(gas_net, j2, mass_flow_kgs=0.1, cost=10)

    gas_result = monee.run_energy_flow_optimization(
        gas_net, create_economic_dispatch_problem(gas_cost_default=30)
    )
    source = gas_result.get(mm.Source)["mass_flow_kgs"].iloc[0]
    ext_m = gas_result.get(mm.ExtHydrGrid)["mass_flow_kgs"].iloc[0]
    print(f"source: {-source:.3f} kg/s")
    print(f"import: {-ext_m:.3f} kg/s")
    print(f"objective: {gas_result.user_objective:.3f}")

.. testoutput::

    source: 0.100 kg/s
    import: 0.050 kg/s
    objective: 106.111

The default L gas carries 42.44 MW per kg/s, so the objective is
:math:`42.44 \cdot (10 \cdot 0.1 + 30 \cdot 0.05)`.

District heating plant against a heat generator
------------------------------------------------

A two-consumer district heating line is fed by a plant at its supply head
(the water external grid) and by a 0.2 MW heat generator at 30 per MW. With
the plant's heat at 50 per MW the generator runs at capacity; at 20 per MW it
stays off and the plant carries the whole load:

.. testcode::

    for plant_price in (50, 20):
        heat_net = mm.Network()
        dhs = mx.dhs_structure(heat_net, diameter_m=0.15, length_m=200)
        seg = dhs.line(2, heat_exchanger_q_mw=[0.15, 0.2])
        dhs.attach_heat_plant(seg.supply.first, seg.return_.last, t_k=358.0)
        mx.create_heat_generator(heat_net, seg.supply.last, q_mw=0.2, cost=30)

        heat_result = monee.run_energy_flow_optimization(
            heat_net, create_economic_dispatch_problem(heat_cost_default=plant_price)
        )
        q_gen = heat_result.get(mm.HeatGenerator)["q_mw_heat"].iloc[0]
        print(f"plant at {plant_price}: heat generator {abs(q_gen):.3f} MW, "
              f"objective {heat_result.user_objective:.3f}")

.. testoutput::

    plant at 50: heat generator 0.200 MW, objective 13.613
    plant at 20: heat generator 0.000 MW, objective 7.045

The plant's share of the objective is its supply heat
:math:`\dot m\, c_p\, (358\text{ K} - T_\text{return})` times its price, which
covers the two exchanger duties and the pipe losses of supply and return.

A coupled day
-------------

The tutorial :doc:`../tutorials/04_mes_economic_dispatch` dispatches a town
with all three carriers over eight hourly prices: a CHP, an electric boiler
and an electrolyser compete with the grid connections and a boiler plant.
Their breakeven power prices follow from the efficiencies (the electric boiler
runs below 38, the electrolyser below 22.75, the CHP above 42.86), and the
tutorial checks that the dispatch switches exactly there, that every carrier
balances, that the objective matches the cost recomputed from the result
tables, and that the saving against a boiler only and a must run CHP baseline
equals the sum of the units' hourly margins. Its script is
``examples/mes_economic_dispatch.py``.

A second tutorial, :doc:`../tutorials/mes_storage_and_ramps`, adds a battery
and ramp limits on the coupling units (``regulation_ramp``) to the same town
and solves the day with ``run_multi_period``; its script is
``examples/mes_storage_and_ramps.py``.

A third, :doc:`../tutorials/mes_network_limits`, rates the cable to the
town's plant site and narrows its gas pipe until the line loading limit (7)
and the gas pressure floor bind, and prices both; its script is
``examples/mes_network_limits.py``.

Recipes
=======

Gas pressure band. In an optimisation every gas junction keeps its squared
pressure within the grid's ``pressure_squared_pu_min`` and
``pressure_squared_pu_max`` (0.7 to 1.3 by default). Bound
``pressure_squared_pu``, the decision variable, to tighten it per junction;
under the gas formulations ``pressure_pu`` is only a reported value, so a
bound on it has no effect::

    problem = create_economic_dispatch_problem(gas_cost_default=30)
    problem.bounds(
        (0.9**2, 1.05**2),
        lambda m, g: type(m) is mm.Junction and isinstance(g, mm.GasGrid),
        ["pressure_squared_pu"],
    )

Gas on a mixed-integer backend. The default convex gas pipe relaxation lets a
price push more gas through a pipe than its pressure drop allows once the
pressure band binds. Pass ``formulation="gas_nonconvex_miqcqp"`` to the
solve, see :doc:`../concepts/formulations`.

Coupling unit ramps. ``ramp_limit`` covers generators, gas sources and heat
units; limit the change of a coupling unit's operating point with
``problem.constraints.regulation_ramp(0.2)``.

Coupling unit operating cost. Add a term per unit of output, here 2 per MW of
CHP electricity::

    problem.objectives.select(
        lambda m, c: isinstance(m, mm.CHPHGControlNode), optional=True
    ).calculate(lambda models: sum(2.0 * -m.el_mw for m in models))


Running it over a horizon
=========================

:func:`~monee.problem.create_multi_period_economic_dispatch_problem` is an
alias of the same builder; the problem becomes multi-period simply by running
it through :func:`~monee.simulation.multi_period.run_multi_period`. Two keywords matter
then: ``ramp_limit`` bounds the MW change of each generator between
consecutive steps, and per-period prices are registered on the
:class:`~monee.simulation.timeseries.TimeseriesData` with
``add_objective_data(component_id, "cost", [...])``, which sets each
generator's ``cost`` before the period's objective is assembled. The same call
on an external grid's id varies :math:`c_{e,h}`, on a gas source or heat
generator its gas or heat price, and ``add_branch_series`` does the same for
a heat exchanger. ``include_storages=True`` dispatches the storages across
the horizon; give the final charge with ``terminal_state``, for example
``terminal_state={(storage_id, "m_stored_kg"): 0.0}`` for a gas storage. See
:doc:`../how-to/multi_period` for the surrounding workflow.
