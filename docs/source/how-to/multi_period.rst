==========================
Multi-period optimization
==========================

Worked examples for multi-period solves, storage dispatch, and
rolling-horizon MPC. For the architectural background see
:doc:`../concepts/multi_period`.

----

Battery dispatch: 6-hour horizon
==================================

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData

   # Network
   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1,
                  length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net, bus1, p_mw=0.0, q_mvar=0.0, name="load")

   storage = mm.ElectricStorage(
       e_mwh_initial=2.0,
       e_mwh_max=4.0,
       p_max_mw=1.0,
   )
   bat = mx.create_el_child(net, storage, node_id=bus1, name="battery")

   # Load profile
   td = TimeseriesData()
   td.add_child_series_by_name("load", "p_mw",
                                [0.4, 0.5, 1.4, 1.8, 1.5, 0.4])

Solve and inspect:

.. testcode::

   from monee.simulation import run_multi_period
   from monee.problem.core import OptimizationProblem

   prob = OptimizationProblem()
   prob.controllable_storages()

   result = run_multi_period(
       net, td,
       optimization_problem=prob,
       dt_h=1.0,
       terminal_state={(bat, "e_mwh"): 2.0},
   )

   soc  = result.get_result_for_id(bat, "e_mwh")
   disp = result.get_result_for_id(bat, "p_mw")
   print("SoC  [MWh]:", soc.round(2).tolist())
   print("Disp [MW]:", disp.round(2).tolist())

.. testoutput::
   :options: +SKIP

   SoC  [MWh]: [...]
   Disp [MW]: [...]

The solver shifts charging to off-peak hours to serve the midday peak.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/howto_multi_period_1.html" width="100%" height="660" style="border:none;" loading="lazy" title="Battery optimal dispatch over 6-hour horizon"></iframe>

.. only:: latex

   .. image:: /_static/interactive/howto_multi_period_1.png
      :width: 100%

----

Cyclical operation
==================

Force the battery to return to its starting state (day-ahead planning):

.. code-block:: python

   result = run_multi_period(
       net, td,
       optimization_problem=prob,
       dt_h=1.0,
       initial_state ={(bat, "e_mwh"): 2.0},
       terminal_state={(bat, "e_mwh"): 2.0},
   )

----

Variable step sizes
===================

Mix 15-minute resolution during the morning peak with hourly resolution
for the rest of the day:

.. code-block:: python

   # 4 × 15-min + 6 × 1-hour = 10 periods
   td_mixed = TimeseriesData()
   td_mixed.add_child_series_by_name("load", "p_mw",
       [0.5, 0.7, 1.2, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5, 0.4])

   result = run_multi_period(
       net, td_mixed,
       dt_h=[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
   )

Or derive durations from a ``DatetimeIndex``:

.. code-block:: python

   import pandas as pd

   idx = pd.date_range("2024-01-01 00:00", periods=10, freq="15min")
   idx = idx[:4].append(pd.date_range("2024-01-01 01:00", periods=6, freq="h"))
   result = run_multi_period(net, td_mixed, datetime_index=idx)

----

Multi-energy: CHP dispatch
===========================

Jointly optimize a CHP unit serving both electrical and heat demand:

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData

   net_mes = mx.create_multi_energy_network()

   # Electricity
   bus_slack = mx.create_bus(net_mes)
   bus_load  = mx.create_bus(net_mes)
   mx.create_ext_power_grid(net_mes, bus_slack)
   mx.create_line(net_mes, bus_slack, bus_load,
                  length_m=200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
   mx.create_power_load(net_mes, bus_load, p_mw=0.0, q_mvar=0.0,
                        name="el_load")

   # Gas + heat
   j_gas    = mx.create_gas_junction(net_mes)
   j_supply = mx.create_water_junction(net_mes)
   j_return = mx.create_water_junction(net_mes)
   mx.create_gas_ext_grid(net_mes, j_gas)
   mx.create_ext_hydr_grid(net_mes, j_supply)
   mx.create_water_sink(net_mes, j_return, mass_flow_kgs=0.0, name="heat_load")

   mx.create_chp(net_mes,
                 power_node_id=bus_load,
                 gas_node_id=j_gas,
                 heat_node_id=j_supply,
                 heat_return_node_id=j_return,
                 diameter_m=0.1,
                 efficiency_power=0.35,
                 efficiency_heat=0.45,
                 mass_flow_setpoint_kgs=0.1)

   el_prof   = [0.8, 1.0, 1.4, 1.8, 1.6, 1.2]
   heat_prof = [0.4, 0.5, 0.7, 0.9, 0.8, 0.6]

   td_mes = TimeseriesData()
   td_mes.add_child_series_by_name("el_load",   "p_mw",     el_prof)
   td_mes.add_child_series_by_name("heat_load", "mass_flow_kgs", heat_prof)

Solve and query CHP dispatch:

.. testcode::

   import monee.problem as mp
   from monee.simulation import run_multi_period

   prob = mp.OptimizationProblem()
   prob.controllable_cps(["regulation"])

   result_mes = run_multi_period(net_mes, td_mes, dt_h=1.0,
                                 optimization_problem=prob)
   chp_reg = result_mes.get_result_for(mm.CHPControlNode, "regulation")
   print(chp_reg.shape)

.. testoutput::

   (6, 1)

Regulation tracks the combined electrical and heat demand.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/howto_multi_period_chp.html" width="100%" height="660" style="border:none;" loading="lazy" title="CHP multi-period dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/howto_multi_period_chp.png
      :width: 100%

----

Gas linepack with multi-period
================================

The optimizer uses pipeline storage to buffer the demand peak, reducing the
required feed-source capacity:

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.model import GasLinepack
   from monee.simulation import TimeseriesData

   net_lp = mx.create_multi_energy_network()

   j0 = mx.create_gas_junction(net_lp)
   j1 = mx.create_gas_junction(net_lp)
   j2 = mx.create_gas_junction(net_lp)
   mx.create_gas_ext_grid(net_lp, j0)
   mx.create_gas_sink(net_lp, j2, mass_flow_kgs=0.3, name="consumer")

   pipe_id = mx.create_gas_pipe(net_lp, j0, j1,
                                diameter_m=0.5, length_m=50_000)
   mx.create_gas_pipe(net_lp, j1, j2,
                      diameter_m=0.3, length_m=10_000)

   net_lp.add_extension(GasLinepack())

   td_lp = TimeseriesData()
   td_lp.add_child_series_by_name("consumer", "mass_flow_kgs",
                                   [0.3, 0.3, 0.6, 0.9, 0.8, 0.4])

Stored mass rises at low demand and drains during the peak.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/howto_multi_period_linepack.html" width="100%" height="500" style="border:none;" loading="lazy" title="Gas linepack buffers demand peak"></iframe>

.. only:: latex

   .. image:: /_static/interactive/howto_multi_period_linepack.png
      :width: 100%

----

Rolling-horizon MPC
====================

Re-solve every step with a 6-step lookahead to model real-time dispatch:

.. code-block:: python

   from monee.simulation import run_mpc
   from monee.problem.core import OptimizationProblem

   prob = OptimizationProblem()
   prob.controllable_storages()

   result = run_mpc(
       net, td_24h,
       total_steps=24,
       horizon=6,
       execution_steps=1,
       optimization_problem=prob,
       dt_h=1.0,
   )

   print(f"Total periods: {result.T}")   # 24
   soc = result.get_result_for_id(bat, "e_mwh")

.. note::

   The ``objective`` returned by ``run_mpc`` is a sum over the rolling windows
   and overcounts when ``execution_steps`` is less than ``horizon``. A
   ``terminal_state`` passed to ``run_mpc`` anchors only the final window (the
   global horizon end), not every window.

.. tip::

   Increase ``execution_steps`` to reduce computation. ``execution_steps=1``
   (re-solve every step) gives the closest approximation to a true online MPC
   but is *T/execution_steps* times more expensive than a full-horizon solve.

----

Time-varying pricing
====================

Register per-period prices with ``add_objective_data``. The objective
lambda reads ``model.price``, which ``TimeseriesData`` sets before each
period's equations are assembled.

.. code-block:: python

   from monee.problem.core import OptimizationProblem, Objectives
   from monee.simulation import TimeseriesData, run_multi_period
   import monee.model as mm

   td = TimeseriesData()
   td.add_child_series_by_name("load", "p_mw",
                                [0.4, 0.5, 1.4, 1.8, 1.5, 0.4])
   # Time-of-use price: cheap off-peak, expensive mid-day
   td.add_objective_data(ext_grid_id, "price", [30, 35, 80, 90, 70, 30])

   prob = OptimizationProblem()
   prob.controllable_storages()

   obj = Objectives()
   obj.select(
       lambda m: isinstance(m, mm.ExtPowerGrid)
   ).calculate(
       lambda models: sum(m.price * m.p_mw for m in models)
   )
   prob.objectives = obj

   result = run_multi_period(net, td, optimization_problem=prob, dt_h=1.0)

----

Per-period constraints
======================

Use ``when_period`` to restrict a constraint to specific periods:

.. code-block:: python

   from monee.problem.core import Constraints

   cons = Constraints()

   # Force generator offline during periods 0 and 1 (maintenance)
   cons.select_types(mm.PowerGenerator).equation(
       lambda m: m.p_mw == 0
   ).when_period(lambda t: t <= 1)

   prob.constraints = cons

``when_period`` accepts a callable ``(t: int) -> bool`` or a collection of
period indices (set, list, range). In single-period solves the filter has
no effect.

----

Cross-period constraints (temporal_equation)
=============================================

Use ``temporal_equation`` to couple variables across periods: ramp rates,
custom storage dynamics, look-ahead limits, and so on.

The lambda receives ``(model, component_id, temporal_state)``. Read variables
from other periods with ``temporal_state.get(component_id, attribute)``.

.. code-block:: python

   from monee.problem.core import Constraints

   cons = Constraints()

   # Ramp-rate limit: ext-grid power can change by at most 1 MW per period
   def ramp_limit(model, cid, ts):
       prev_p = ts.get(cid, "p_mw")
       if prev_p is None:
           return []  # no previous period (t=0)
       return [
           model.p_mw - prev_p <= 1.0,   # ramp up
           prev_p - model.p_mw <= 1.0,   # ramp down
       ]

   cons.select_types(mm.ExtPowerGrid).temporal_equation(ramp_limit)
   prob.constraints = cons

   result = run_multi_period(net, td, steps=6, optimization_problem=prob)

.. note::

   Before the horizon start (t < 0) ``temporal_state.get()`` returns the
   ``initial_state`` value for that attribute when one was seeded, or ``None``
   otherwise. Guard for ``None`` and return an empty list to skip the constraint
   at the first period when no initial value is available.

``temporal_equation`` composes with ``when_period``:

.. code-block:: python

   # Ramp limit only during peak hours (periods 2 to 4)
   cons.select_types(mm.ExtPowerGrid).temporal_equation(
       ramp_limit
   ).when_period(range(2, 5))

These constraints are evaluated during inter-period equation assembly,
alongside the built-in storage and thermal-mass coupling. They are
silently skipped in single-period solves.

----

Solver selection
================

The default is CasADi when installed, with GEKKO as the automatic fallback.

.. tab-set::

   .. tab-item:: CasADi (default, in-process NLP)

      .. code-block:: python

         # CasADi is selected automatically when installed; force it with backend=
         result = run_multi_period(net, td, backend="casadi")

   .. tab-item:: GEKKO (continuous NLP, IPOPT fallback)

      .. code-block:: python

         from monee.simulation import GekkoMultiPeriodSolver

         result = run_multi_period(
             net, td, solver=GekkoMultiPeriodSolver()
         )

   .. tab-item:: Pyomo (integers / islanding)

      .. code-block:: python

         from monee.simulation import PyomoMultiPeriodSolver

         result = run_multi_period(
             net, td, solver=PyomoMultiPeriodSolver()
         )

----

API quick reference
====================

Runner functions
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Description
   * - ``run_multi_period(net, td, ...)``
     - Solve multi-period optimization
   * - ``run_multi_period(..., dt_h=1.0)``
     - Set timestep duration in hours (default 1.0)
   * - ``run_multi_period(..., terminal_state={...})``
     - Anchor final-step variable values, e.g. ``{(bat_id, "e_mwh"): 2.0}``
   * - ``run_multi_period(..., initial_state={...})``
     - Override t=-1 values used by ``inter_temporal_equations``
   * - ``run_multi_period(..., optimization_problem=prob)``
     - Pass an ``OptimizationProblem`` to declare free variables and objectives
   * - ``run_mpc(net, td, total_steps, horizon, execution_steps, ...)``
     - Rolling-horizon MPC

Results
-------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Description
   * - ``MultiPeriodResult.T``
     - Number of periods
   * - ``MultiPeriodResult.get_result_for(Type, attr)``
     - DataFrame: periods × components
   * - ``MultiPeriodResult.get_result_for_id(id, attr)``
     - Series: one float per period
   * - ``MultiPeriodResult[comp_id]``
     - DataFrame: all attributes over all periods
   * - ``MultiPeriodResult.get_period_result(t)``
     - ``SolverResult`` for one period
   * - ``MultiPeriodResult.objective``
     - Global objective value. For run_mpc this is a sum over rolling windows
       and overcounts when execution_steps is less than horizon
   * - ``GekkoMultiPeriodSolver``
     - GEKKO / IPOPT backend (fallback when CasADi is unavailable)
   * - ``CasADiMultiPeriodSolver``
     - In-process CasADi / IPOPT backend (default when CasADi is installed)
   * - ``PyomoMultiPeriodSolver``
     - Pyomo backend (use for MIP / islanding)

OptimizationProblem
-------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Description
   * - ``OptimizationProblem()``
     - Create a problem descriptor
   * - ``prob.controllable_storages()``
     - Let the solver freely dispatch all ``ElectricStorage``, ``GasStorage``,
       and ``ThermalStorage`` components
   * - ``prob.controllable_generators(["p_mw"])``
     - Let the solver freely dispatch ``PowerGenerator`` and ``Source`` outputs
   * - ``prob.controllable_demands(["p_mw"])``
     - Let the solver freely curtail loads (load shedding / demand response)
   * - ``prob.controllable_cps(["regulation"])``
     - Let the solver freely modulate CHP, P2H, P2G coupling points
   * - ``prob.controllable(attrs, component_condition=...)``
     - General-purpose: free any attribute on matching components
   * - ``prob.bounds((lo, hi), component_condition, attributes)``
     - Override min/max bounds of specific ``Var`` attributes
   * - ``prob.objectives``
     - Set / get the ``Objectives`` object for the solver objective function
   * - ``prob.constraints``
     - Set / get the ``Constraints`` object for additional constraints

PeriodState
-----------

``PeriodState`` is passed to ``inter_temporal_equations`` in multi-period solves.
Unlike ``StepState`` (which returns floats), ``PeriodState.get()`` returns
live solver variables, so equations that read from it become algebraic
cross-period constraints inside the joint solve.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Description
   * - ``state.get(component_id, attr, step=-1)``
     - Solver variable at a given period. Negative ``step`` is relative to the
       current period; non-negative is an absolute index (``0..T-1``).
   * - ``state.has(component_id, attr)``
     - ``True`` if a non-``None`` value exists
   * - ``state.dt_h``
     - Duration of the current period in hours
   * - ``state.current_t``
     - Zero-based index of the period currently being assembled
   * - ``state.T``
     - Total number of periods in the horizon

.. note::

   Under :func:`~monee.simulation.run_mpc`, absolute (non-negative) ``step``
   indices are window-local: ``step=0`` is the first period of the current
   rolling window, not the global start of the run. Prefer negative relative
   lookbacks in ``inter_temporal_equations`` for behaviour that does not depend
   on the MPC window. Look-ahead (a non-negative index past ``current_t``)
   couples to a future period only for injected solver variables; a value that
   is only derived during that future period is not yet available when an
   earlier period is assembled.
