=============
Load shedding
=============

Load shedding finds the least harmful way to curtail demand when a network
cannot serve all of it within its operational limits: after a contingency, a
capped grid connection or a loss of supply on one carrier. monee ships a
ready-made formulation for multi-energy networks: a one-call interface for the
common case, and a builder function when you need to customise it.

The optimiser gives every demand, generator and coupling unit a service level
``regulation`` between 0 and 1 and minimises the weighted power that is not
served, in MW, across electricity, gas and heat at once, subject to the full network
physics and to bounds on voltage, pressure, temperature and line loading.

Mathematical formulation
========================

Sets
----

Except for :math:`\mathcal{B}`, the sets contain only components that are active
and not ignored.

.. list-table::
   :header-rows: 1
   :widths: 14 86

   * - Symbol
     - Members
   * - :math:`\mathcal{D}`
     - Demands: :class:`~monee.model.child.PowerLoad`,
       :class:`~monee.model.child.HeatLoad`, gas-grid
       :class:`~monee.model.child.Sink`,
       :class:`~monee.model.branch.HeatExchangerLoad`,
       :class:`~monee.model.branch.PassiveHeatExchangerLoad` and every other
       heat exchanger whose setpoint consumes.
   * - :math:`\mathcal{G}`
     - Generators: :class:`~monee.model.child.PowerGenerator`,
       :class:`~monee.model.child.HeatGenerator`,
       :class:`~monee.model.child.Source` on gas and water grids,
       :class:`~monee.model.branch.HeatExchangerGenerator`,
       :class:`~monee.model.branch.PassiveHeatExchangerGenerator` and every
       other heat exchanger whose setpoint generates.
   * - :math:`\mathcal{C}`
     - Coupling points: the control nodes of CHP, CHPHG, G2H and P2H, and the
       branch couplers P2G, G2P, P2H-HG and G2H-HG
       (:doc:`../components/coupling`).
   * - :math:`\mathcal{B}`
     - Branches built with ``backup=True``, whether active or not.
   * - :math:`\mathcal{L}`
     - Power branches with a rating, as in :doc:`economic_dispatch`.
   * - :math:`\mathcal{K}(i)`
     - Children attached to node :math:`i`.
   * - :math:`\mathcal{E}`
     - External grids: :class:`~monee.model.child.ExtPowerGrid` and
       :class:`~monee.model.child.ExtHydrGrid` on gas and water grids.
   * - :math:`\mathcal{S}`
     - Storages, only with ``include_storages=True``.
   * - :math:`\mathcal{N}^\text{el}, \mathcal{N}^\text{gas}, \mathcal{N}^\text{water}`
     - Buses, plain junctions of gas grids, plain junctions of water grids.
       Coupler control nodes are not part of these sets.

Decision variables
------------------

.. list-table::
   :header-rows: 1
   :widths: 22 50 28

   * - Symbol
     - Meaning
     - Result column
   * - :math:`r_k \in [0, 1]`
     - Service level of :math:`k \in \mathcal{D} \cup \mathcal{G} \cup \mathcal{C}`:
       1 is full service, 0 fully shed. The component acts on the network with
       :math:`r_k` times its setpoint.
     - ``regulation``
   * - :math:`\kappa_\ell \in \{0, 1\}`
     - Switch of a reserve branch :math:`\ell \in \mathcal{B}`.
     - ``on_off``
   * - :math:`p_e`
     - Exchange of external grid :math:`e`: MW for electricity, kg/s for gas
       and water, load convention (import negative).
     - ``p_mw``, ``mass_flow_kgs``
   * - :math:`p_z, E_z`
     - Dispatch of storage :math:`z \in \mathcal{S}` in the load convention
       (positive charging), and its state of charge.
     - ``p_mw`` or ``mass_flow_kgs``, and ``e_mwh`` or ``m_stored_kg``
   * - :math:`x`
     - Network state of all carriers: voltages, angles, branch flows,
       pressures, temperatures, coupler internals.
     - The node and branch tables.

Power not served
----------------

Every component in the objective is a power in MW, with gas mass flow
converted to its MW equivalent through its heating value, so the three
carriers can be summed. :math:`\sigma_k` is unserved power; it becomes energy
in MWh only when multiplied by the step length, which the objective does not
do. Let :math:`y_k` be the setpoint of component :math:`k` in the load
convention: :math:`p_k` in MW for power, :math:`\dot{Q}_k` in MW for heat,
:math:`\dot{m}_k` in kg/s for gas. The power not served is

.. math::

   \sigma_k =
   \begin{cases}
   \varsigma_k\, c_k\, y_k \,(1 - r_k) & \text{power and heat loads and generators, gas sinks and sources} \\
   |\dot{Q}_k| - |\dot{Q}^\text{del}_k| & \text{heat exchangers} \\
   \hat{P}^\text{in}_k \,(1 - r_k) & \text{coupling points} \\
   0 & \text{water-grid sources}
   \end{cases}
   \qquad
   \varsigma_k = \begin{cases} +1 & k \in \mathcal{D} \\ -1 & k \in \mathcal{G} \end{cases}
   \qquad
   c_k = \begin{cases} 3.6\, H & \text{gas} \\ 1 & \text{power and heat} \end{cases}

:math:`H` is the ``higher_heating_value_kwh_per_kg`` of the gas grid, so
:math:`3.6\,H` converts kg/s into MW (11.79 kWh/kg for the default lgas grid).
The sign :math:`\varsigma_k` is applied to the setpoint, not to its magnitude:
for an ordinary load or generator :math:`\sigma_k \ge 0`, but a load built
with a negative setpoint yields a negative :math:`\sigma_k` and is rewarded for
being shed.

A heat exchanger is measured by the gap between its setpoint
:math:`\dot{Q}_k` and the duty it actually delivered,
:math:`\dot{Q}^\text{del}_k` (``q_mw_delivered``), which also catches a
shortfall caused by a too narrow temperature spread; a formulation without a
delivered-duty variable falls back to :math:`|\dot{Q}_k|(1 - r_k)`.

A coupling point is measured at its rated input :math:`\hat{P}^\text{in}_k` in
MW on the input carrier: the electric setpoint for P2H and P2H-HG, the gas mass
flow setpoint times :math:`3.6\,H` for CHP, CHPHG and G2H, the rated output
divided by the efficiency :math:`\eta_k` for G2P, and the gas output setpoint
times :math:`3.6\,H` divided by :math:`\eta_k` for P2G. A setpoint that is a
variable is read at its upper bound. A network with a G2H-HG coupler currently
fails with an ``AttributeError`` when ``include_coupling_points=True``.

Water-grid sources have :math:`\sigma_k = 0`: mass flow in a heating loop is a
transport quantity, not demand, and pricing it with a heating value would add a
phantom penalty. Their service level is still a decision variable, so the
optimiser may curtail them at no cost. Water-grid sinks are not controllable
and always run at their setpoint.

Problem
-------

.. math::

   \begin{aligned}
   \min \quad & \sum_{k \in \mathcal{D} \cup \mathcal{G} \cup \mathcal{C}} W_k\, \sigma_k \;+\; A(x) \\
   \text{s.t.} \quad
   & F(x, r, \kappa, p, E) = 0 && && (1) \\
   & 0 \le r_k \le 1 && \forall k \in \mathcal{D} \cup \mathcal{G} \cup \mathcal{C} && (2) \\
   & \kappa_\ell \in \{0, 1\} && \forall \ell \in \mathcal{B} && (3) \\
   & \underline{V} \le V_i \le \overline{V} && \forall i \in \mathcal{N}^\text{el} && (4) \\
   & \underline{\pi} \le \pi_j \le \overline{\pi} && \forall j \in \mathcal{N}^\text{gas} \cup \mathcal{N}^\text{water} && (5) \\
   & \underline{T} \le T_j \le \overline{T} && \forall j \in \mathcal{N}^\text{water} && (6) \\
   & \text{loading}_\ell^{s} \le \lambda && \forall \ell \in \mathcal{L},\, s \in \{f, t\} && (7) \\
   & \underline{p}_e \le p_e \le \overline{p}_e && \forall e \in \mathcal{E} && (8) \\
   & -\overline{p}_z \le p_z \le \overline{p}_z, \;\; 0 \le E_z \le \overline{E}_z && \forall z \in \mathcal{S} && (9)
   \end{aligned}

Here :math:`A(x)` collects the auxiliary terms the formulations add to the
objective (see :doc:`index`). They are small but not always positive. By
default the auto priority floor described under
:ref:`problems/load_shedding:Weights` lifts all weights so that these terms do
not compete with shed power; :ref:`objective-semantics` shows how to read them
separately from the result.

The coupling-point terms are present only with
``include_coupling_points=True``; otherwise the coupling points' service
levels are free. Constraints (4) to (7) are present only when their
``check_*`` switch is on; (8) only with ``include_ext_grids=True`` and a bound
given for that carrier; (9) only with ``include_storages=True``. Constraint
(3) holds only on a mixed-integer backend. The default CasADi and IPOPT
backend relaxes it to :math:`0 \le \kappa_\ell \le 1` without a message, so a
reserve branch can come back partly closed; see :doc:`index` and
:doc:`../how-to/use_pyomo_solver`.

Network equations (1). These are the equations an energy flow solves, for
every carrier at once, with each child entering its node balance at
:math:`r_k` times its setpoint and each reserve branch's flows multiplied by
:math:`\kappa_\ell`:

- Electricity: the active and reactive power balances at every bus and the AC
  branch flows, exactly as in constraints (1) to (4) of
  :doc:`economic_dispatch`, with :math:`\sum_{k \in \mathcal{K}(i)} r_k\, p_k`
  and :math:`\sum_{k \in \mathcal{K}(i)} r_k\, q_k` in place of the fixed
  injections, :math:`q_k` being the reactive setpoint (``q_mvar``) of child
  :math:`k`. Shedding a load therefore also drops its reactive demand; the
  couplers listed below are the exception.
- Gas and water: the mass balance at every junction,
  :math:`\sum_\text{branches} \dot{m} + \sum_{k} r_k\, \dot{m}_k = 0`, the
  Weymouth relation on gas pipes, the Darcy-Weisbach relation on water pipes
  and the fixed ratio of compressors.
- Heat: the enthalpy balance at every water junction, with heat loads and
  generators entering as :math:`r_k\, \dot{Q}_k`, and the temperature
  propagation along pipes and exchangers.
- Coupling points: input and output are tied by the unit's efficiency, so
  :math:`r_k` scales both sides of the active-power conversion together. The
  reactive setpoints of CHP, CHPHG, G2P, P2G and P2H-HG are not scaled by
  :math:`r_k`.

:doc:`../concepts/domains` states the pipe, compressor and heat equations and
:doc:`../components/coupling` the coupling equations.

Operational limits (4) to (9). :math:`V_i` is the bus voltage magnitude,
:math:`\pi_j` the junction pressure in per unit of the grid's
``pressure_ref_pa`` (10 bar absolute by default; the bound is squared where a
formulation solves the squared pressure), :math:`T_j` the water temperature in
per unit of the grid's ``t_ref_k`` (356 K by default, so the default band 0.9
to 1.1 is about 320 K to 392 K). The loading rows (7) are the same as in
:doc:`economic_dispatch`. The exchange bounds (8) are stated in the load
convention, in MW for electricity and kg/s for gas and heat. They replace any
import or export cap set on the external grid itself (``max_import_mw``,
``max_export_mw``, ``max_import_kgs``, ``max_export_kgs``) instead of
tightening it, so a wider ``bounds_ext_*`` lifts such a cap; without
``bounds_ext_*`` for a carrier, the grid's own caps stay in force.

Storage (9). :math:`p_z` is the storage dispatch bounded by its power or flow
limit and :math:`E_z` its state of charge bounded by the capacity; a storage
with charge and discharge efficiencies also carries separate charge and
discharge variables, each between zero and the limit. The state of charge is
linked to the dispatch only across periods, through the storage's
inter-temporal equation (see :ref:`problems/load_shedding:Over a horizon`),
which is added in :func:`~monee.simulation.multi_period.run_multi_period` and
in timeseries runs. In a single-period solve :math:`E_z` only has to stay
within its bounds and is not linked to :math:`p_z`. With
``include_storages=True`` a storage therefore acts as a source or sink limited
only by its power or flow limit, even when it is empty, and its reported
``e_mwh`` or ``m_stored_kg`` has no meaning.

Weights
-------

The weights put a price on each MW not served:

.. math::

   W_k = \gamma\, \omega_k, \qquad
   \omega_k =
   \begin{cases}
   \text{callback}(k) & k \in \mathcal{D} \text{ consumes and the callback returns a number} \\
   w_d & k \in \mathcal{D} \text{ otherwise, or } k \in \mathcal{C} \\
   w_g & k \in \mathcal{G}
   \end{cases}

with :math:`w_d` = ``demand_weight`` (default :math:`10^3`) and :math:`w_g` =
``generator_weight`` (default :math:`0.1`). An exception raised inside the
callback is treated like ``None``. Curtailing a generator costs four orders of
magnitude less than shedding a demand: the optimiser only uses it to resolve
an infeasibility that shedding cannot fix, such as an over-voltage caused by
too much generation.

The scale :math:`\gamma \ge 1` is the auto priority floor. It keeps shed power
dominant over the auxiliary terms :math:`A(x)` however large the network is:

.. math::

   \gamma = \max\!\Big(1,\; \frac{\alpha\, A_\text{max}}{\omega_\text{min}}\Big)

with :math:`\alpha` = ``priority_safety_factor`` (default 10),
:math:`A_\text{max}` an estimate of the largest :math:`|A(x)|` computed from the
network data before the solve, and :math:`\omega_\text{min}` the smallest
demand weight (:math:`w_d` without a callback). The same :math:`\gamma`
multiplies every weight, so the ratios you chose are preserved. With
``auto_priority_floor=False``, :math:`\gamma = 1`.

The floor works per MW. When it lifts the weights, one MW of shed at the
smallest weight costs at least :math:`\alpha\,A_\text{max}`, so a shed of
:math:`1/\alpha` MW or more (0.1 MW at the default) costs at least
:math:`A_\text{max} \ge |A(x)|`. Smaller sheds carry no such guarantee. With
a callback and ``include_coupling_points=True``, coupling points are weighted
:math:`\gamma w_d`, which is below this floor when
:math:`w_d < \omega_\text{min}`.

.. note::

   Implementation details of the floor. :math:`A_\text{max}` sums over all
   components, active or not:

   .. math::

      A_\text{max} = \sum_{\ell:\, R_\ell > 0} R_\ell \min\!\Big(\frac{4\,\overline{V}^2}{R_\ell^2 + X_\ell^2},\; \lambda^2 \Big(\frac{\overline{I}_\ell}{I^\text{base}_\ell}\Big)^2\Big) + \sum_{k} |\dot{Q}_k| + 10^{-4} \sum_{p} \big(\overline{\dot{m}}_{p,+}^2 + \overline{\dot{m}}_{p,-}^2\big) + 2 \cdot 10^{-6}\, n_T

   The first sum runs over power branches, with the series resistance and
   reactance :math:`R_\ell, X_\ell` in per
   unit of the grid base ``sn_mva``, :math:`\overline{V}` = ``vm_pu_max`` of
   the grid (not ``bounds_vm``), :math:`\overline{I}_\ell` = ``max_i_ka`` and
   :math:`\lambda` = ``max_line_loading``. :math:`I^\text{base}_\ell` is the
   smaller of :math:`S_\text{base} / (\sqrt{3}\, U_f\, \tau_\ell)` and
   :math:`S_\text{base} / (\sqrt{3}\, U_t)`. The loading term enters only with
   ``check_lp``. The second sum runs over models with a fixed duty in MW, the
   third over hydraulic branches with squared mass flow bounds in
   :math:`(\text{kg/s})^2`, and :math:`n_T` counts the models with a
   temperature variable (junctions of both carriers and coupler control
   nodes). The terms carry different units; :math:`A_\text{max}` bounds an
   objective value, not a physical quantity. The estimate is a true upper
   bound for the heat exchanger terms, and for the electrical terms of
   branches whose loading is limited by current; for a branch that also
   carries ``max_s_mva`` (MATPOWER imports) the loading term is not enforced
   by a constraint and the estimate can fall short. It assumes at most
   100 kg/s for any hydraulic flow that has no upper bound of its own.

   With a callback, :math:`\omega_\text{min}` is taken over every consuming
   demand model in the network, including inactive and ignored loads and
   water-grid sinks, so such a component with a low weight can lift
   :math:`\gamma`; if that minimum is zero or negative, :math:`\gamma = 1`. In
   the callback case generating heat exchangers receive :math:`\gamma^2 w_g`
   instead of :math:`\gamma\, w_g`. Without a callback and :math:`w_d \le 0`
   the formula does not apply: the demand, coupling-point and consuming heat
   exchanger weights are set directly to :math:`\alpha A_\text{max}`, and
   :math:`w_g` stays unscaled.

Lexicographic variant
---------------------

With ``lex_objectives=True`` on the Pyomo or the native Gurobi backend, in a
single-period solve, the weighting of shed power against :math:`A` disappears;
the weights between loads and generators still matter. Phase one solves
:math:`\Sigma^\star = \min \sum_k W_k \sigma_k` subject to (1) to (9). Phase two
minimises :math:`A(x)` subject to (1) to (9) and
:math:`\sum_k W_k \sigma_k \le \Sigma^\star + \epsilon`, with
:math:`\epsilon = 10^{-9} + \max(\mu, 10^{-6})\,\max(1, |\Sigma^\star|)` and
:math:`\mu` the relative MIP gap in effect for the solve: by default
:math:`10^{-4}` for Gurobi and SCIP through Pyomo and :math:`10^{-3}` on the
native Gurobi backend, overridable through the solver options, and zero for a
solver without a gap option.
When both tiers are linear, the native Gurobi backend does not run two
phases; it hands both tiers to Gurobi's hierarchical multi-objective solve,
with an absolute tolerance of :math:`10^{-9}` on the first tier. The GEKKO
backend, the default CasADi backend and
:func:`~monee.simulation.multi_period.run_multi_period` solve the summed
objective without a message.

Over a horizon
--------------

Run through :func:`~monee.simulation.multi_period.run_multi_period` over the
periods :math:`h \in \mathcal{T}`, the objective is the sum of the per-period
objectives (without a time-step
weight), constraints (1) to (9) hold in every period, and each storage carries
its state of charge from one period to the next,

.. math::

   E_{z,h} = (1 - \beta_z \Delta t_h)\, E_{z,h-1} + r_z\, \Delta t \big(\eta^c_z\, p^+_{z,h} - p^-_{z,h} / \eta^d_z\big),
   \qquad p_{z,h} = p^+_{z,h} - p^-_{z,h},

where :math:`E_{z,0}` is the run's ``initial_state`` entry for the state of
charge, or the storage's initial state of charge without one, :math:`r_z` its
``regulation``, :math:`\Delta t_h` is ``dt_h`` in hours, and :math:`\Delta t`
is :math:`\Delta t_h` for electric storage (MW into MWh) and
:math:`3600\, \Delta t_h` in seconds for gas and thermal storage (kg/s into
kg). The split into :math:`p^+_z, p^-_z \ge 0` exists only for a storage with
efficiencies different from one; otherwise the bracket is :math:`p_{z,h}`. The
self-discharge rate :math:`\beta_z` (``loss_factor_per_h``) is nonzero only
for thermal storage, which has no efficiencies. ``regulation_ramp_limit``
:math:`\rho` adds

.. math::

   |r_{k,h} - r_{k,h-1}| \le \rho \qquad \forall k \text{ with a regulation variable},\; h \in \mathcal{T} \text{ with } r_{k,h-1} \text{ known}.

For the first period :math:`r_{k,0}` is read from the run's ``initial_state``
entry for the component's ``regulation``; without it the first period is not
ramp-limited. The auto priority floor is evaluated once per period while the
periods are prepared, and the value of the last period is used for all of
them.

Worked example
==============

A hospital and a warehouse share a feeder whose substation import is capped at
0.5 MW, below their combined 0.8 MW. The hospital carries a weight of
:math:`10^5`, the warehouse keeps the default :math:`w_d = 10^3`, so the
optimiser sheds the warehouse first:

.. testcode::

    import monee
    import monee.express as mx
    import monee.model as mm
    from monee.problem import (
        calc_general_resilience_performance,
        create_min_load_shedding_problem,
    )

    net = mx.create_multi_energy_network()
    sub, b1, b2 = mx.create_bus(net), mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, sub, b1, 500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_line(net, b1, b2, 500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_ext_power_grid(net, sub)
    hospital = mx.create_power_load(net, b1, p_mw=0.4, q_mvar=0.0, name="hospital")
    mx.create_power_load(net, b2, p_mw=0.4, q_mvar=0.0, name="warehouse")
    net.child_by_id(hospital).model._priority_weight = 1e5

    problem = create_min_load_shedding_problem(
        bounds_ext_el=(-0.5, 0.0),
        weight_for_load=lambda model: getattr(model, "_priority_weight", None),
        check_pressure=False,
        check_t=False,
    )
    result = monee.run_energy_flow_optimization(net, problem)

    loads = result.get(mm.PowerLoad)
    for name, reg in zip(loads["name"], loads["regulation"]):
        print(f"{name}: r = {reg:.2f}")
    shed_mw, _, _ = calc_general_resilience_performance(result.network)
    print(f"sigma = {shed_mw:.3f} MW")
    print(f"objective / w_d = {result.user_objective / 1e3:.3f}")

.. testoutput::

    hospital: r = 1.00
    warehouse: r = 0.22
    sigma = 0.313 MW
    objective / w_d = 0.313

The warehouse keeps :math:`r_\text{wh} \approx 0.218`, so
:math:`\sigma_\text{wh} = 0.4\,(1 - 0.218) \approx 0.313` MW: the 0.3 MW gap
plus the line losses. The hospital is not shed, so the objective is exactly
:math:`w_d\,\sigma_\text{wh}`, which shows that the auto priority floor left
:math:`\gamma = 1` on this small network.

----

One-call interface
==================

.. tip::

   Use :func:`monee.solve_load_shedding_problem` for the simplest path: it
   builds the problem with the same keywords and defaults as
   :func:`~monee.problem.create_min_load_shedding_problem` and solves it in
   one call. The example below overrides some of them.

.. code-block:: python

    from monee import solve_load_shedding_problem

    result = solve_load_shedding_problem(
        network,
        bounds_vm=(0.9, 1.1),
        bounds_t=(0.8, 1.2),
        bounds_pressure=(0.8, 1.2),
        bounds_ext_el=(-0.5, 0.5),
        bounds_ext_gas=(-2.0, 2.0),
    )

    print(f"Solver objective: {result.objective:.4f}")

.. note::

   ``result.objective`` is the solver's objective value, which is dominated by
   the weighted shed power but is not a shed-power meter. See
   :ref:`objective-semantics` below before comparing it across scenarios.

The result is a :class:`~monee.solver.core.SolverResult` with the solved
network and DataFrames for each model type. The bound and check arguments are
keyword-only and mirror :func:`~monee.problem.create_min_load_shedding_problem`
one-to-one.

----

Custom problem
==============

For finer control (different weights per carrier, extra constraints, or
different bounds) use
:func:`~monee.problem.create_min_load_shedding_problem` directly:

.. testcode::

    import monee.express as mx
    from monee import run_energy_flow_optimization
    from monee.problem import create_min_load_shedding_problem

    # Build a small test network
    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.4, q_mvar=0.0, name="campus_load")

    # Create and customise the load-shedding problem
    problem = create_min_load_shedding_problem(
        demand_weight=20,            # penalty per MW of shed demand
        bounds_vm=(0.92, 1.08),      # tighter voltage bounds
        check_pressure=False,        # no gas grid present
        check_t=False,               # no heat grid present
        debug=False,
    )

    result = run_energy_flow_optimization(
        net, problem, exclude_unconnected_nodes=True
    )

.. note::

   Pass ``exclude_unconnected_nodes=True`` when the network may contain buses
   or junctions that are topologically disconnected (e.g. because a line is
   deactivated). Without this flag such nodes enter the solve and can cause
   infeasibility. With it, disconnected components are excluded and their
   ``regulation`` is reported as ``0.0`` (fully shed) in the result.

Keyword reference
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 56

   * - Parameter
     - Symbol
     - Effect
   * - ``demand_weight`` / ``generator_weight``
     - :math:`w_d`, :math:`w_g`
     - Penalty per MW of shed demand and of curtailed generation. Defaults:
       ``1e3`` / ``0.1``.
   * - ``weight_for_load``
     - :math:`\omega_k`
     - Callback ``(model) -> float | None`` returning a per-load shed weight
       for priority ordering. ``None`` falls back to ``demand_weight``.
       Default: ``None``. See
       :ref:`problems/load_shedding:Prioritising individual loads`.
   * - ``bounds_vm``
     - :math:`\underline{V}, \overline{V}`
     - ``(min, max)`` voltage bounds (4) in per unit. Default: ``(0.9, 1.1)``.
   * - ``bounds_pressure``
     - :math:`\underline{\pi}, \overline{\pi}`
     - ``(min, max)`` pressure bounds (5) in per unit on the junctions of gas
       and water grids. Default: ``(0.9, 1.1)``.
   * - ``bounds_t``
     - :math:`\underline{T}, \overline{T}`
     - ``(min, max)`` temperature bounds (6) in per unit on water junctions.
       Default: ``(0.9, 1.1)``.
   * - ``max_line_loading``
     - :math:`\lambda`
     - Maximum electrical line loading (7). Default: ``1.5`` (150 %). Built by
       :func:`~monee.problem.utils.line_loading_limit`: apparent power against
       ``max_s_mva`` when set, otherwise current against ``max_i_ka``. Under
       the MISOCP formulation the current form caps the squared series
       current, so line-charging current is not included.
   * - ``check_vm`` / ``check_pressure`` / ``check_t`` / ``check_lp``
     -
     - Drop constraint (4), (5), (6) or (7), for example for carriers not
       present in the network.
   * - ``bounds_ext_el`` / ``bounds_ext_gas`` / ``bounds_ext_heat``
     - :math:`\underline{p}_e, \overline{p}_e`
     - Exchange range (8) at external grids, in the load convention (import
       negative): MW for electricity, kg/s for gas and heat. Capping the
       import at X is ``(-X, 0)``. A non-negative lower bound with a positive
       upper one forbids import and sheds every load local supply cannot
       cover; for ``bounds_ext_el`` this raises a warning, while ``(0, 0)``,
       the deliberate island case, does not. Replaces the grid's own
       ``max_import_*`` / ``max_export_*`` caps. Applied only with
       ``include_ext_grids=True``. Default: ``None`` (unconstrained), since any
       bound tighter than the network's intact import forces shedding.
   * - ``include_ext_grids``
     -
     - Apply the exchange bounds (8). External grids stay the slack either way
       and carry no cost in the objective; their exchange is limited only by
       (8) or, without it, by their own import and export caps. Default:
       ``True``.
   * - ``include_storages``
     - :math:`\mathcal{S}`, :math:`p_z, E_z`
     - Make storage units dispatchable during shedding, adding (9). In a
       single-period solve the state of charge does not limit the dispatch,
       see :ref:`problems/load_shedding:Mathematical formulation`. Default:
       ``False``.
   * - ``include_coupling_points``
     - :math:`\mathcal{C}`
     - Penalise curtailment of coupling points like loads on their input
       carrier. Default: ``False``. See
       :ref:`problems/load_shedding:Shedding coupling points`.
   * - ``regulation_ramp_limit``
     - :math:`\rho`
     - Limit on the change of each ``regulation`` between consecutive
       periods in timeseries or multi-period runs. Default: ``None`` (no
       limit); ignored in single-period solves.
   * - ``lex_objectives``
     -
     - Two-phase lexicographic solve on the Pyomo and native Gurobi backends
       that removes the weighting of shed power against the auxiliary terms.
       Other backends and multi-period runs solve the summed objective.
       Default: ``False``. See
       :ref:`problems/load_shedding:Lexicographic variant`.
   * - ``auto_priority_floor`` / ``priority_safety_factor``
     - :math:`\gamma`, :math:`\alpha`
     - Lift the weights so that one MW of shed at the smallest weight costs at
       least :math:`\alpha\,A_\text{max}`. Defaults: ``True`` / ``10.0``. See
       :ref:`problems/load_shedding:Weights`.
   * - ``debug``
     -
     - Verbose problem construction; also logs the priority floor. Default:
       ``False``.

Prioritising individual loads
-----------------------------

By default every demand is shed at the same cost per MW. To establish a
priority ordering (keep the hospital, shed the warehouse first), pass
``weight_for_load``, a callback that receives a load *model* and returns its
shed weight, or ``None`` to fall back to ``demand_weight``. A convenient
pattern is to tag the models with a custom attribute when building the
network, as the :ref:`worked example <problems/load_shedding:Worked example>`
does:

.. code-block:: python

    import monee.express as mx
    from monee.problem import create_min_load_shedding_problem

    hospital_id = mx.create_power_load(net, bus_1, p_mw=0.4, q_mvar=0.0)
    net.child_by_id(hospital_id).model._priority_weight = 1e5  # shed last

    problem = create_min_load_shedding_problem(
        # None -> fall back to demand_weight for untagged loads
        weight_for_load=lambda model: getattr(model, "_priority_weight", None),
    )

The callback is consulted only for members of :math:`\mathcal{D}`; generators,
including generating heat exchangers, are not passed to it and use
:math:`w_g`. The auto priority floor multiplies every weight by the same
:math:`\gamma`, so the ratios you choose are preserved, with the one exception
listed under :ref:`problems/load_shedding:Weights`.

Shedding coupling points
------------------------

By default, coupling units between carriers are controllable but their
curtailment is free: only the downstream service loss is penalised. With
``include_coupling_points=True``, the coupling points of :math:`\mathcal{C}`
(the CHP, CHPHG, G2H, and P2H control nodes as well as the P2G, G2P, P2H-HG,
and G2H-HG branches) enter the objective with
:math:`W_k\,\sigma_k = \gamma\, w_d\, \hat{P}^\text{in}_k\,(1 - r_k)`: each
coupling point is treated as a load on its input carrier (power or gas), with
gas input converted to MW via :math:`3.6\,H`. A network with a G2H-HG coupler
currently fails with this option.
Use this when curtailing a conversion unit should count as lost demand in its
own right.

.. _objective-semantics:

What ``result.objective`` is, and is not
----------------------------------------

``result.objective`` is the value the solver reported for its own objective,
not a shed-power meter. Two things separate the two.

Inactive and ignored components are left out of the objective entirely. A
component switched off with ``deactivate_by_id``, and every component dropped
by ``exclude_unconnected_nodes=True`` after a contingency islands it, has no
Vars registered with the backend, so its unserved demand contributes nothing.
The paradox is that the worse the contingency, the lower the objective can go:
a fault that islands a whole feeder scores better than one that merely strains
it, because the islanded demand has left the sum. The ``regulation`` values in
the result dataframes report those loads as ``0.0``, which is the honest
picture, but the scalar does not.

The formulation's own tightening terms :math:`A(x)` are added to the same
solver objective. They are tiny compared with shed power, but they are not
always positive, so a network at zero curtailment can report a small nonzero
or slightly negative objective rather than an exact ``0.0``.
``result.user_objective`` excludes them; ``result.aux_objective`` reports them
alone.

Both effects make the scalar unsafe for ranking scenarios against each other.
For that, compute the shed power from the solved network instead:

.. code-block:: python

    from monee.problem import calc_general_resilience_performance

    shed_power, shed_heat, shed_gas = calc_general_resilience_performance(
        result.network
    )

:func:`~monee.problem.calc_general_resilience_performance` walks the solved
network, so it counts a deactivated or excluded load at its full rating and
returns MW-equivalents per carrier that are comparable across contingencies.
Within a single solve the objective is still the right thing to watch, since
it is what the optimiser minimised.

The shed-headroom check
-----------------------

After every successful single-period load-shedding solve monee compares, per
carrier and per connected island, the demand that was shed with the import
headroom of the external grids in that island. Loads shedding less than
:math:`10^{-4}` MW (0.1 kW, gas converted with :math:`3.6\,H`) each are
ignored as solver noise. When the remaining shed is positive while an external
grid of the same carrier could still import more (it is unbounded, or its
exchange sits more than :math:`10^{-6}` MW for electricity or
:math:`10^{-6}` kg/s for gas and water away from its import bound), a
``shed_headroom`` entry is added to ``result.warnings``. It names the largest shed loads and the
slack with room to spare.

The warning points at a possible local optimum: the non-convex AC and gas
formulations can stop at a point that sheds more than needed, and re-solving,
or cross-checking with a mixed-integer backend, settles it. It can also be
legitimate. A voltage, pressure, temperature or loading limit can bind inside
the island while the slack still has room, in which case more import would not
help; inspect the node and branch tables for the binding limit. The check stays
silent for islands without an external grid of that carrier and for loads
that are inactive, ignored or excluded, and it does not run for
:func:`~monee.simulation.multi_period.run_multi_period`.

----

Interpreting the result
=======================

After solving, the ``regulation`` attribute of each controllable component
shows how much of its nominal setpoint was served. A value of ``1.0`` means
fully served; ``0.0`` means completely shed. Read the values from the result
dataframes, which contain plain floats:

.. testcode::

    import monee.model as mm

    loads = result.get(mm.PowerLoad)
    for name, reg in zip(loads["name"], loads["regulation"]):
        print(f"{name}: regulation = {reg:.2f}")

.. testoutput::

    campus_load: regulation = 1.00

On the solved network itself (``result.network``), every attribute the
problem promoted to a decision variable stays a
:class:`~monee.model.core.Var` object, which does not format or cast as a
number (``f"{reg:.2f}"`` and ``float(reg)`` both raise ``TypeError``);
attributes that were never promoted remain plain floats. Read the solved
number from the ``.value`` attribute, or call ``monee.model.value``,
which unwraps ``Var`` and passes plain floats through unchanged; see
:doc:`../concepts/data_model` for the full contract.

.. testcode::

    for child in result.network.childs:
        reg = getattr(child.model, "regulation", None)
        if reg is not None:
            print(f"{child.name or child.id}: regulation = {mm.value(reg):.2f}")

.. testoutput::

    0: regulation = 1.00
    campus_load: regulation = 1.00

.. note::

   A ``regulation`` value between 0 and 1 indicates partial load curtailment.
   Inspect the ``dataframes`` dict for per-component voltage, pressure, and
   temperature to understand why curtailment was needed.

In optimisation mode, heat exchangers with a fixed ``regulation`` are a
special case, because their duty is stated as an inequality
(``q_mw_delivered <= q_mw``) that only an objective term pulls tight. A user
objective can outweigh that pull and leave an exchanger delivering less than
its setpoint while the solve still reports success. In a simulation the duty
is a hard equation, so a shortfall there means the solve stopped short of
feasibility. Such a shortfall is listed in ``result.violations`` under the key
``<Type>.<id>.q_mw_delivered_shortfall``, together with a warning on the
``monee.solver.core`` logger:

.. code-block:: python

    for key, magnitude in result.violations.items():
        if key.endswith("q_mw_delivered_shortfall"):
            print(f"{key}: {magnitude:.4f} MW unmet")

The magnitude is the unmet duty in MW. If you see one, either add
``q_mw_delivered == q_mw_set`` as an explicit constraint or scale your
objective coefficients so the duty pull is not outbid. Exchangers whose
``regulation`` is a decision variable are not reported: there the shortfall is
the shedding decision you asked for.

The check also covers the internal ``SubHE`` exchanger of coupling compounds
(CHP, GasToHeat, PowerToHeat) in optimisation mode. Its duty is not a fixed
setpoint but the solved ``q_mw`` that the control node's coupling equation
prescribes, so a compound heat side that delivers less than that value appears
in ``result.violations`` as ``SubHE.<id>.q_mw_delivered_shortfall``. In a
simulation the compound's duty is an equation and cannot fall short.

----

Measuring curtailment
=====================

To quantify how much demand was actually curtailed, use
:class:`~monee.problem.GeneralResiliencePerformanceMetric` on the solved
network: it returns curtailed MW per carrier as a
``(power, heat, gas)`` tuple:

.. code-block:: python

    from monee.problem import GeneralResiliencePerformanceMetric

    power_mw, heat_mw, gas_mw = GeneralResiliencePerformanceMetric().calc(
        result.network,
        include_ext_grid=False,
        include_coupling_points=False,
    )

or, equivalently, via the convenience wrapper
:func:`~monee.problem.calc_general_resilience_performance`:

.. code-block:: python

    from monee.problem import calc_general_resilience_performance

    power_mw, heat_mw, gas_mw = calc_general_resilience_performance(result.network)

Ignored or inactive loads (e.g. excluded by ``exclude_unconnected_nodes``)
count at their full rating; regulated loads count at
``upper - value * regulation``; a heat exchanger counts at the gap between its
setpoint and the duty it reached. Gas curtailment is converted to MW with the
same ``3.6 * higher_heating_value_kwh_per_kg`` convention as the objective. Pass
``include_coupling_points=True`` to also account coupling-point curtailment
on the input carrier, mirroring the corresponding problem option.

``include_ext_grid`` (False by default) adds a fourth term that is easy to
misread: every external grid that feeds the network is counted as unserved
demand, at its full import. That is the islanding view, where import stands for
the load an islanded network would have had to shed, so it is what you want
when ranking islanding scenarios. On a grid-connected network it inflates the
power component by the whole substation import, and the number no longer equals
the sum of the per-load shed.
