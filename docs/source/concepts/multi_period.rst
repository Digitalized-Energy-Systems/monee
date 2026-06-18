==========================
Multi-period optimization
==========================

Multi-period optimization builds one large problem that optimizes over *T*
time periods at once.  All periods share the same solver model, so the optimizer
trades off actions across time. It can charge a battery cheaply now to discharge
it at a more expensive future hour.

.. note::

   For forward simulation with known profiles, see :doc:`timeseries`.
   For rolling real-time control, see the :ref:`mpc` section below.

----

Timeseries vs. multi-period
============================

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Timeseries simulation
     - Multi-period optimization
   * - Solve structure
     - One NLP per step
     - One large NLP for all T periods
   * - Cross-time coupling
     - Constants (previous step's floats)
     - Coupled solver variables
   * - Optimality
     - Greedy: locally optimal per step
     - Global across all T periods
   * - Scalability
     - Linear in T
     - Super-linear in T
   * - Typical use
     - Profile replay, forward simulation
     - Storage dispatch, ramp scheduling, linepack management

----

PeriodState
===========

:class:`~monee.simulation.step_state.PeriodState` is the coupling mechanism that
lets the same ``inter_temporal_equations`` implementation work in both contexts.

In a timeseries solve, ``StepState.get(id, "e_mwh")`` returns a float.

In a multi-period solve, ``PeriodState.get(id, "e_mwh")`` returns a live
solver variable from the previous period's network copy.  The constraint
``e_mwh[t] == prev_e + dt * p_mw[t]`` then ties two solver variables together,
creating a genuine coupling constraint.

.. mermaid::

   flowchart LR
       subgraph prev["period t-1"]
           E0["e_mwh"]
       end
       subgraph cur["period t"]
           P["p_mw"] --> E1["e_mwh"]
       end
       E0 -->|"coupling constraint"| E1

Both ``StepState`` and ``PeriodState`` implement the
:class:`~monee.simulation.step_state.InterStepState` interface, so a model's
``inter_temporal_equations`` works unmodified in either context.

----

Quick start: battery dispatch
================================

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData

   # Network with a battery attached to a loaded bus
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

   # Varying load: low off-peak, high during peak hours
   td = TimeseriesData()
   td.add_child_series_by_name("load",    "p_mw",
                                [0.4, 0.5, 1.4, 1.8, 1.5, 0.4])

To solve with the optimizer choosing when to charge/discharge:

.. code-block:: python

   from monee.simulation import run_multi_period
   from monee.problem.core import OptimizationProblem

   prob = OptimizationProblem()
   prob.controllable_storages()   # promotes p_mw from float → Var

   result = run_multi_period(
       net, td,
       optimization_problem=prob,
       dt_h=1.0,
       terminal_state={(bat, "e_mwh"): 2.0},  # return to initial SoC
   )
   print(result)

Battery optimal dispatch: the solver charges during off-peak hours and discharges during the midday peak.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/concepts_multi_period.html" width="100%" height="660" style="border:none;" loading="lazy" title="Multi-period battery dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/concepts_multi_period.png
      :width: 100%

----

Result API
==========

``run_multi_period`` returns a :class:`~monee.simulation.multi_period.MultiPeriodResult`
containing one solved network copy per period.

.. tab-set::

   .. tab-item:: By model class

      .. code-block:: python

         # DataFrame: rows = periods, columns = component ids
         vm  = result.get_result_for(mm.Bus, "vm_pu")
         soc = result.get_result_for(mm.ElectricStorage, "e_mwh")

   .. tab-item:: By component id

      .. code-block:: python

         # Series: one float per period
         soc  = result.get_result_for_id(bat, "e_mwh")
         disp = result.get_result_for_id(bat, "p_mw")

   .. tab-item:: All attributes

      .. code-block:: python

         # DataFrame: rows = periods, columns = all attributes
         df = result[bat]
         print(df["e_mwh"])   # SoC trajectory
         print(df["p_mw"])    # dispatch trajectory

   .. tab-item:: Single period

      .. code-block:: python

         # Inspect one period with the same API as a single-step solve
         sr = result.get_period_result(3)
         buses = sr.get(mm.Bus)

   .. tab-item:: Datetime index

      .. code-block:: python

         import pandas as pd

         idx = pd.date_range("2024-06-01 00:00", periods=6, freq="h")
         result = run_multi_period(net, td, dt_h=1.0, datetime_index=idx)

         soc = result.get_result_for_id(bat, "e_mwh")
         print(soc.index)   # DatetimeIndex

----

Initial and terminal state
===========================

Initial state
-------------

Override the model's built-in initial value with an explicit dict:

.. code-block:: python

   result = run_multi_period(
       net, td,
       initial_state={(bat, "e_mwh"): 1.5},
   )

This is how :func:`~monee.simulation.run_mpc` seeds each horizon window from
the terminal state of the previous one.

Terminal state
--------------

Force the optimizer to reach a target value at the last period:

.. code-block:: python

   result = run_multi_period(
       net, td,
       initial_state ={(bat, "e_mwh"): 2.0},
       terminal_state={(bat, "e_mwh"): 2.0},  # cyclic: return to start
   )

----

Variable step sizes
====================

Steps do not need to be uniform.  Pass a list or derive durations from a
``DatetimeIndex``:

.. code-block:: python

   # 4 × 15-min steps followed by 4 × 1-hour steps
   result = run_multi_period(
       net, td,
       dt_h=[0.25, 0.25, 0.25, 0.25, 1.0, 1.0, 1.0, 1.0],
   )

   # Or let monee derive durations from the index
   import pandas as pd
   idx = pd.DatetimeIndex([
       "2024-01-01 00:00", "2024-01-01 00:15",
       "2024-01-01 01:00", "2024-01-01 03:00",
   ])
   result = run_multi_period(net, td, datetime_index=idx)

----

.. _mpc:

Rolling-horizon MPC
===================

:func:`~monee.simulation.run_mpc` solves on a sliding window, executing only
the first few steps before re-solving with an updated forecast:

.. code-block:: text

   total_steps = 12 │ horizon = 6 │ execution_steps = 2

   Solve window:  [0 … 5] [2 … 7] [4 … 9] [6 … 11] [8 … 11] [10 … 11]
   Execute:        0, 1    2, 3    4, 5    6, 7     8, 9     10, 11

.. code-block:: python

   from monee.simulation import run_mpc

   result = run_mpc(
       net, td,
       total_steps=24,
       horizon=6,
       execution_steps=1,
       optimization_problem=prob,
       dt_h=1.0,
   )
   print(f"Executed {result.T} periods")

``run_mpc`` returns a ``MultiPeriodResult`` of length ``total_steps``.  The
terminal state of each window automatically becomes the ``initial_state`` of
the next, ensuring SoC continuity across window boundaries.

----

Solver backends
===============

.. tab-set::

   .. tab-item:: GEKKO

      .. code-block:: python

         from monee.simulation import GekkoMultiPeriodSolver, run_multi_period

         result = run_multi_period(net, td, solver=GekkoMultiPeriodSolver())

      Best for smooth NLP problems without integer variables.  Ships with
      its own IPOPT binaries, no extra installation needed.

   .. tab-item:: Pyomo

      .. code-block:: python

         from monee.simulation import PyomoMultiPeriodSolver, run_multi_period

         result = run_multi_period(net, td, solver=PyomoMultiPeriodSolver())

      Required for MILP / MIQCP problems: on/off switching, islanding, or
      networks with binary variables.  Requires an external solver such as
      SCIP, Gurobi, or CBC.

----

Two-pass assembly
=================

Both GEKKO and Pyomo multi-period solvers use a two-pass approach to build
the joint problem efficiently:

.. code-block:: text

   Pass 1 - variable injection
     for t in 0 … T-1:
       ext.prepare(net_t)          # inject Var placeholders
       inject_vars(solver, net_t)  # register with the shared model

   Pass 2 - equation assembly
     for t in 0 … T-1:
       period_state = PeriodState(all_nets, current_t=t, ...)
       equations(net_t, ...)
       inter_temporal_equations(net_t, ..., period_state)
       inter_period_equations(net_t, ..., period_state)

At pass-2 time, ``PeriodState`` has access to all T period networks, so
``get()`` can reference any period (past or future) without any special
handling in the model code.

----

See also
========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Timeseries simulation
      :link: timeseries
      :link-type: doc
      :shadow: sm

      Sequential solve architecture, ``StepState``, and inter-step hooks.

   .. grid-item-card:: Temporal extensions
      :link: temporal_extensions
      :link-type: doc
      :shadow: sm

      ``LumpedThermalCapacitance`` and ``GasLinepack``: step-by-step
      walkthroughs for both timeseries and multi-period use.

   .. grid-item-card:: NetworkAspect
      :link: network_aspects
      :link-type: doc
      :shadow: sm

      The general extension mechanism behind LTC, linepack, and islanding.
