=============================================
Externally paced co-simulation (Conductor)
=============================================

Use a **Conductor** when an *external* program decides when and how far the
simulation advances - typically a **co-simulation framework** such as
`mosaik <https://mosaik.offis.de/>`_, where monee is one simulator among many
and the orchestrator dictates the clock.  Unlike
:func:`~monee.run_timeseries`, which loops over a fixed number of steps on its
own, a :class:`~monee.simulation.Conductor` exposes a single
:meth:`~monee.simulation.Conductor.step` method that you call whenever your
framework wants the network to advance - by one hour, by fifteen minutes, or
by any other positive duration, varying freely from call to call.

Between calls the Conductor keeps a persistent
:class:`~monee.simulation.StepState`, so storage state of charge, gas
linepack, and LTC tap positions carry over from one step to the next exactly
as they do in a regular timeseries run.

For the architectural background on sequential simulation see
:doc:`../concepts/timeseries`.

----

When to use which driver
========================

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - ``Conductor``
     - :func:`~monee.run_timeseries`
     - :func:`~monee.run_multi_period`
   * - Who paces the loop
     - The **caller** (co-simulation framework)
     - monee (internal loop)
     - monee (single joint solve)
   * - Step size
     - Any ``dt_h > 0``, may vary per call
     - Fixed, or derived from a ``datetime_index``
     - Scalar, list, or ``datetime_index``
   * - Lookahead
     - None (myopic, one step at a time)
     - None (myopic)
     - Full horizon (anticipative)
   * - Inter-step coupling
     - ``StepState`` (floats from prior solves)
     - ``StepState`` (floats from prior solves)
     - ``PeriodState`` (live solver variables)

----

Quick start
===========

Build a small electricity network, hand it to a Conductor, and step it
forward at whatever pace you like:

.. testcode::

   from monee import Conductor
   import monee.express as mx
   from monee.simulation import TimeseriesData

   # ── Electricity grid ──────────────────────────────────────────────────
   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1,
                  length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   load_id = mx.create_power_load(net, bus1, p_mw=0.1, q_mvar=0.0)

   # Optional: a profile the Conductor can pull rows from
   td = TimeseriesData()
   td.add_child_series(load_id, "p_mw", [0.1, 0.2, 0.15, 0.1])

   c = Conductor(net, timeseries_data=td)

   # The caller decides when and how far to step:
   r = c.step(dt_h=1.0)                                       # one hour
   r2 = c.step(0.25, data_overrides={(load_id, "p_mw"): 0.3})  # 15 minutes
   print(c.t_h)

.. testoutput::
   :options: +SKIP

   1.25

Each :meth:`~monee.simulation.Conductor.step` call returns a
:class:`~monee.simulation.StepResult` whose ``result`` attribute is the
familiar single-snapshot ``SolverResult``.

----

Constructing a Conductor
========================

.. code-block:: python

   Conductor(
       net,
       *,
       solver=None,                  # name string, SolverInterface, or None
       backend=None,                 # "gekko" or "pyomo"
       optimization_problem=None,    # applied at every step
       timeseries_data=None,         # source for ts_index slices
       initial_state=None,           # {(component_id, attr): float}
       on_step_error="raise",        # or "skip"
       **solver_kwargs,              # forwarded to every solver.solve(...)
   )

The constructor mirrors :func:`~monee.run_timeseries`:

* ``solver`` / ``backend`` are resolved once, exactly as in
  :func:`~monee.solve` - a solver name string such as ``"ipopt"`` or
  ``"scip"``, a solver instance, or ``None`` for the GEKKO/IPOPT default.
* ``optimization_problem`` lets every step solve an optimization instead of a
  plain energy flow.
* ``initial_state`` seeds the :class:`~monee.simulation.StepState` with
  fallback values - for example a starting storage state of charge
  ``{(bat_id, "e_mwh"): 2.0}`` - used until the attribute has been written by
  a prior solve.

The Conductor never mutates the base network: every step works on a fresh
``net.copy()``.  After a successful solve the result network is pushed into
the persistent ``StepState``, which is how storage, linepack, and LTC state
carry across calls.

----

Stepping
========

.. code-block:: python

   step(dt_h, *, data_overrides=None, ts_index=None) -> StepResult

``step`` advances the simulation clock by ``dt_h`` hours and solves one
snapshot.  Inputs for the step come from two optional sources, applied in
this order:

1. ``ts_index`` - apply row ``ts_index`` of the constructor's
   :class:`~monee.simulation.TimeseriesData` to the network copy.  Because
   the *caller* paces the simulation, the row index is explicit rather than
   an internal counter - you may revisit, skip, or interpolate rows as your
   framework requires.
2. ``data_overrides`` - a mapping ``{(component_id, attribute): value}``
   applied *after* the timeseries slice, so overrides win on conflicts.
   This is the natural channel for values arriving live from other
   simulators (e.g. setpoints computed by an agent system).

.. code-block:: python

   r = c.step(
       0.25,
       ts_index=3,                                  # profile row first …
       data_overrides={(load_id, "p_mw"): 0.3},     # … then live inputs win
   )

Two validation rules are deliberate:

* ``dt_h <= 0`` raises ``ValueError`` - time must move forward.
* An unknown component id in ``data_overrides`` raises ``KeyError``; a known
  id whose models lack the attribute raises ``AttributeError``.  In a
  co-simulation, a misspelled id or attribute is a **wiring bug**, not a
  transient condition, so it always raises regardless of ``on_step_error``.

.. note::

   Like :class:`~monee.simulation.TimeseriesData`, overrides that target a
   solver ``Var`` pin its ``value``/``min``/``max`` to the given value - the
   quantity stays a variable, so it remains visible to ``StepState`` and the
   result tables.

----

Error handling
==============

With the default ``on_step_error="raise"``, a solver failure propagates to
the caller and neither the history nor the clock changes.

With ``on_step_error="skip"``, a failure is logged, a *failed*
:class:`~monee.simulation.StepResult` (``failed=True``, ``error`` set) is
appended to the history, and **time still advances** by ``dt_h`` - the
external clock and the Conductor's clock stay in sync even across failed
steps:

.. code-block:: python

   c = Conductor(net, on_step_error="skip")
   r = c.step(1.0)
   if r.failed:
       print("step failed:", r.error)

Failed steps are excluded from the DataFrame queries of
:meth:`~monee.simulation.Conductor.to_timeseries_result` but remain
inspectable via the result's ``failed_steps`` property.

----

State, history, and the clock
=============================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Symbol
     - Description
   * - ``c.state``
     - The persistent :class:`~monee.simulation.StepState`;
       ``c.state.get(comp_id, attr)`` reads values from prior solves
   * - ``c.history``
     - Copied list of all :class:`~monee.simulation.StepResult` objects
   * - ``c.step_count``
     - Number of ``step()`` calls so far
   * - ``c.t_h``
     - Accumulated simulated time in hours (sum of all ``dt_h``)
   * - ``c.reset(initial_state=...)``
     - Clear history and clock, recreate the ``StepState`` (optionally with
       a new ``initial_state``)
   * - ``with Conductor(net) as c: ...``
     - Context-manager support for tidy scoping in framework adapters

``reset`` is handy when a co-simulation scenario is re-run: the same
Conductor (and resolved solver) can be reused without rebuilding the
network.

----

Converting results
==================

After the run, wrap the accumulated history in the standard
:class:`~monee.simulation.TimeseriesResult` API:

.. code-block:: python

   import monee.model as mm

   ts_result = c.to_timeseries_result()
   vm = ts_result.get_result_for(mm.Bus, "vm_pu")     # rows=steps, cols=ids
   p  = ts_result.get_result_for_id(load_id, "p_mw")  # one Series

   # Optionally label rows with real timestamps:
   import pandas as pd
   idx = pd.date_range("2024-01-01", periods=c.step_count, freq="h")
   ts_result = c.to_timeseries_result(datetime_index=idx)

Everything documented in :doc:`timeseries` for result querying -
:meth:`~monee.simulation.TimeseriesResult.get_result_for`,
:meth:`~monee.simulation.TimeseriesResult.get_result_for_id`,
``ts_result[component_id]``, ``failed_steps``, the notebook HTML repr -
works unchanged.

.. tip::

   Because the Conductor records every step in order, you can call
   ``to_timeseries_result()`` at any point mid-run to inspect partial
   results without disturbing the simulation.

----

See also
========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Timeseries concept
      :link: ../concepts/timeseries
      :link-type: doc
      :shadow: sm

      The sequential solve architecture, ``StepState``, and inter-step
      coupling that the Conductor reuses.

   .. grid-item-card:: Timeseries how-to
      :link: timeseries
      :link-type: doc
      :shadow: sm

      Internally paced runs with :func:`~monee.run_timeseries`, hooks, and
      ``TimeseriesData`` recipes.

   .. grid-item-card:: Multi-period optimization
      :link: multi_period
      :link-type: doc
      :shadow: sm

      Anticipative dispatch over a full horizon with
      :func:`~monee.run_multi_period` and MPC.
