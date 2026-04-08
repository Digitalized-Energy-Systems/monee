=====================
Timeseries simulation
=====================

The timeseries runner drives a network through a sequence of timesteps, solving
each independently while passing scalar state values between steps.  It is the
right choice when you want to **replay known profiles** or when inter-step
coupling is one-directional (past → future).

.. note::

   For globally-optimal dispatch — where the solver must look ahead, e.g.
   to decide when to charge a battery — see :doc:`multi_period`.

----

Sequential pipeline
===================

Each timestep follows a fixed four-phase cycle:

.. code-block:: text

   ┌─────────────────────────────────────────────────────────────┐
   │  for step k in 0 … T-1:                                     │
   │                                                             │
   │  1. net_copy = net.copy()          (base net never touched) │
   │  2. timeseries_data.apply(net_copy, k)  (inject values)     │
   │  3. result = solve(net_copy, step_state=state)              │
   │  4. state.push(result.network)     (record solved values)   │
   └─────────────────────────────────────────────────────────────┘

**Copy** — a fresh ``Network.copy()`` is taken each step so solved attributes
from step *k* never bleed into step *k+1*.

**Inject** — ``TimeseriesData`` writes per-step scalar values directly onto
model attributes before the solver sees them.  It is equivalent to setting the
attribute by hand.

**Solve** — the copied-and-patched network is passed to the solver exactly like
a single-step call.  Any solver or optimisation problem works here.

**Record** — :class:`~monee.simulation.step_state.StepState` stores the solved
float value of every ``Var`` attribute, making it available as a constant in
the next step's inter-step constraint.

----

Quick start
===========

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData, run_timeseries

   # 1. Build network
   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1,
                  length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   load = mx.create_power_load(net, bus1, p_mw=1.0, q_mvar=0.0, name="demand")

   # 2. Define a 6-step load profile
   td = TimeseriesData()
   td.add_child_series_by_name("demand", "p_mw",
                                [0.4, 0.8, 1.2, 1.0, 0.6, 0.3])

   # 3. Run — step count inferred from series length
   result = run_timeseries(net, td)
   print(f"{len(result.raw)} successful steps, "
         f"{len(result.failed_steps)} failures")

.. testoutput::

   6 successful steps, 0 failures

----

TimeseriesData
==============

``TimeseriesData`` maps ``(component, attribute)`` pairs to per-step value
lists.  All registered series must have the same length — mismatches raise
``ValueError`` at registration time.

Registering series
------------------

.. tab-set::

   .. tab-item:: By name

      Components added with a ``name=`` keyword can be referenced directly:

      .. testcode::

         from monee.simulation import TimeseriesData
         td = TimeseriesData()
         td.add_child_series_by_name("demand", "p_mw",
                                      [0.4, 0.8, 1.2, 1.0, 0.6, 0.3])
         td.add_branch_series_by_name("main_pipe", "on_off",
                                       [1, 1, 1, 0, 1, 1])

   .. tab-item:: By id

      Use the integer id returned when adding a component:

      .. testcode::

         import monee.model as mm
         import monee.express as mx
         from monee.simulation import TimeseriesData

         net2 = mx.create_multi_energy_network()
         bus  = mx.create_bus(net2)
         mx.create_ext_power_grid(net2, bus)
         load2 = mx.create_power_load(net2, bus, p_mw=1.0, q_mvar=0.0)

         td2 = TimeseriesData()
         td2.add_child_series(load2, "p_mw", [0.5, 0.9, 1.3])

   .. tab-item:: From DataFrame

      .. code-block:: python

         import pandas as pd
         from monee.simulation import TimeseriesData

         df = pd.read_csv("load_profile.csv")  # columns: p_mw, q_mvar
         td = TimeseriesData.from_dataframe(
             df, component_type="child", component_name="demand"
         )

Merging
-------

Two ``TimeseriesData`` objects combine with ``+`` or ``extend()``.  For
duplicate ``(component, attribute)`` pairs the **receiver wins**:

.. testcode::

   from monee.simulation import TimeseriesData

   td_loads = TimeseriesData()
   td_loads.add_child_series_by_name("demand", "p_mw",
                                      [0.4, 0.8, 1.2])

   td_pipes = TimeseriesData()
   td_pipes.add_branch_series_by_name("main_pipe", "on_off", [1, 0, 1])

   combined = td_loads + td_pipes
   print(combined.length)

.. testoutput::

   3

----

Querying results
================

``run_timeseries`` returns a :class:`~monee.simulation.timeseries.TimeseriesResult`:

.. tab-set::

   .. tab-item:: By model class

      .. code-block:: python

         # DataFrame: rows = successful steps, columns = component ids
         vm = result.get_result_for(mm.Bus, "vm_pu")
         p  = result.get_result_for(mm.PowerLoad, "p_mw")

   .. tab-item:: By component id

      .. code-block:: python

         # Series: one value per successful step
         p_load = result.get_result_for_id(load.id, "p_mw")

   .. tab-item:: Datetime index

      .. code-block:: python

         import pandas as pd

         idx = pd.date_range("2024-01-01", periods=6, freq="h")
         result = run_timeseries(net, td, datetime_index=idx)

         vm = result.get_result_for(mm.Bus, "vm_pu")
         print(vm.index)   # DatetimeIndex

   .. tab-item:: Error handling

      .. code-block:: python

         result = run_timeseries(net, td, on_step_error="skip")

         print("Failed steps:", result.failed_steps)
         for sr in result.step_results:
             if sr.failed:
                 print(f"  step {sr.step}: {sr.error}")

----

Inter-step coupling
===================

By default every step is independent.  :class:`~monee.simulation.step_state.StepState`
bridges consecutive steps by recording the previous step's solved values and
making them available as constants in the next step's equations.

Implementing dynamics
---------------------

Implement ``inter_temporal_equations`` on any model to add coupling constraints.
The method receives a ``temporal_state`` object whose ``.get()`` returns the
previous step's solved float (or ``None`` on the first step):

.. code-block:: python

   from monee.model.core import ChildModel, Var, model

   @model
   class Battery(ChildModel):
       def __init__(self, e_init, e_max, p_max):
           super().__init__()
           self.e_mwh = Var(e_init, min=0, max=e_max, name="e_mwh")
           self.p_mw  = 0.0           # fixed dispatch by default
           self._e_init = e_init

       def inter_temporal_equations(self, temporal_state, component_id, **kwargs):
           prev_e = temporal_state.get(component_id, "e_mwh")
           if prev_e is None:
               prev_e = self._e_init  # first step: use initial condition
           return [
               self.e_mwh == prev_e + temporal_state.dt_h * self.p_mw
           ]

.. note::

   ``inter_temporal_equations`` works identically in timeseries simulation
   and multi-period optimization.  Use ``inter_step_equations`` only when
   you need timeseries-specific behaviour that should **not** activate in a
   multi-period solve.

Accessing earlier steps
-----------------------

The ``step`` argument allows look-back beyond the immediately previous step:

.. code-block:: python

   prev_1 = temporal_state.get(component_id, "e_mwh")        # step t-1 (default)
   prev_2 = temporal_state.get(component_id, "e_mwh", step=-2)  # step t-2
   step_0 = temporal_state.get(component_id, "e_mwh", step=0)   # absolute index

Three temporal hooks
--------------------

.. list-table::
   :header-rows: 1
   :widths: 35 25 40

   * - Method
     - Invoked by
     - Use
   * - ``inter_temporal_equations``
     - timeseries **and** multi-period
     - Storage SoC, linepack, thermal mass
   * - ``inter_step_equations``
     - timeseries only
     - Controller clamps, step-specific corrections
   * - ``inter_period_equations``
     - multi-period only
     - Look-ahead constraints across the full horizon

----

Step hooks
==========

Hooks let you inspect or modify the network before and after each step:

.. code-block:: python

   from monee.simulation import StepHook

   class MyHook(StepHook):
       def pre_run(self, net, step, step_state):
           """Called before the solve — net is the base network."""
           print(f"Step {step}: starting")

       def post_run(self, net, step, step_state, step_result):
           """Called after the solve — net is the solved copy."""
           if step_result.failed:
               print(f"Step {step}: FAILED — {step_result.error}")

   result = run_timeseries(net, td, step_hooks=[MyHook()])

The ``step_state`` argument is the live :class:`~monee.simulation.step_state.StepState`
— hooks can read or write inter-step values directly.

----

Temporal network extensions
============================

For time-coupled physics that spans the entire network — thermal inertia,
pipeline linepack — use a :class:`~monee.model.formulation.core.NetworkAspect`
extension rather than modifying individual model classes:

.. code-block:: python

   from monee.model import LumpedThermalCapacitance, GasLinepack

   net.add_extension(LumpedThermalCapacitance())
   net.add_extension(GasLinepack(overrides={pipe_id: dict(
       linepack_kg_initial=500, linepack_kg_max=2_000
   )}))

See :doc:`network_aspects` and :doc:`temporal_extensions` for the full
walkthrough.

----

Scalability
===========

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Factor
     - Impact
   * - Steps
     - Linear — each step is one independent solve
   * - Network size
     - Same as single-step; memory ∝ steps × result size
   * - Inter-step constraints
     - O(coupled vars) extra constraints per step; negligible overhead
   * - Failed steps (``on_step_error='skip'``)
     - Skipped steps do not update ``StepState``; subsequent steps
       continue from the last successful state
