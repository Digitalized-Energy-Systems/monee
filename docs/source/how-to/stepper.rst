=============================================
Externally paced co-simulation (Stepper)
=============================================

Use a Stepper when an external program decides when and how far the
simulation advances. A typical case is a co-simulation framework such as
`mosaik <https://mosaik.offis.de/>`_, where monee is one simulator among many
and the orchestrator owns the clock. Unlike
:func:`~monee.run_timeseries`, which loops over a fixed number of steps on its
own, a :class:`~monee.simulation.Stepper` exposes a single
:meth:`~monee.simulation.Stepper.step` method. You call it whenever your
framework wants the network to advance: by one hour, by fifteen minutes, or
by any other positive duration, varying freely from call to call.

Between calls the Stepper keeps a persistent
:class:`~monee.simulation.StepState`, so storage state of charge, gas
linepack, and LTC tap positions carry over from one step to the next, exactly
as in a regular timeseries run.

For the architectural background on sequential simulation see
:doc:`../concepts/timeseries`.

----

When to use which driver
========================

.. list-table::
   :header-rows: 1
   :widths: 22 26 26 26

   * -
     - ``Stepper``
     - :func:`~monee.run_timeseries`
     - :func:`~monee.run_multi_period`
   * - Who paces the loop
     - The caller (co-simulation framework)
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

Build a small electricity network, hand it to a Stepper, and step it
forward at whatever pace you like:

.. testcode::

   from monee import Stepper
   import monee.express as mx
   from monee.simulation import TimeseriesData

   # Electricity grid
   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1,
                  length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   load_id = mx.create_power_load(net, bus1, p_mw=0.1, q_mvar=0.0)

   # Optional: a profile the Stepper can pull rows from
   td = TimeseriesData()
   td.add_child_series(load_id, "p_mw", [0.1, 0.2, 0.15, 0.1])

   c = Stepper(net, timeseries_data=td)

   # The caller decides when and how far to step:
   r = c.step(dt_h=1.0)                                       # one hour
   r2 = c.step(0.25, data_overrides={(load_id, "p_mw"): 0.3})  # 15 minutes
   print(c.t_h)

.. testoutput::

   1.25

Each :meth:`~monee.simulation.Stepper.step` call returns a
:class:`~monee.simulation.StepResult` whose ``result`` attribute is the
familiar single-snapshot ``SolverResult``.

----

Constructing a Stepper
========================

.. code-block:: python

   Stepper(
       net,
       *,
       solver=None,                  # name string, SolverInterface, or None
       backend=None,                 # "casadi", "gekko", "pyomo", or "gurobipy"
       optimization_problem=None,    # applied at every step
       timeseries_data=None,         # source for ts_index slices
       initial_state=None,           # {(component_id, attr): float}
       on_step_error="raise",        # or "skip"
       max_history=None,             # retain only the last N steps
       record_changes=True,          # record topology change events
       max_changes=None,             # retain only the last N change events
       copy_base=True,               # step on a copy, not the caller's net
       warm_start=None,              # None follows the backend
       carry_failed_dt=True,         # roll a skipped step's dt into the next
       **solver_kwargs,              # forwarded to every solver.solve(...)
   )

The constructor mirrors :func:`~monee.run_timeseries`:

* ``solver`` / ``backend`` are resolved once, exactly as in
  :func:`~monee.solve`: a solver name string such as ``"ipopt"`` or
  ``"scip"``, a solver instance, or ``None`` for the default (IPOPT on CasADi
  when installed, otherwise GEKKO's bundled IPOPT).
* ``optimization_problem`` lets every step solve an optimization instead of a
  plain energy flow.
* ``initial_state`` seeds the :class:`~monee.simulation.StepState` with
  fallback values (for example a starting storage state of charge
  ``{(bat_id, "e_mwh"): 2.0}``) used until the attribute has been written by
  a prior solve.
* ``max_history`` bounds memory for open-ended runs. Every step retains a
  full solved network copy (in the ``StepState`` and in the history), so an
  unbounded co-simulation grows memory step by step. Set ``max_history`` to
  the longest lookback your inter-step physics needs, then old steps are
  dropped as new ones arrive. The built-in storage, linepack, and LTC
  couplings only read the previous step, so a small value such as
  ``max_history=8`` is generous. ``step_count`` and ``t_h`` keep counting
  across dropped steps; :meth:`~monee.simulation.Stepper.to_timeseries_result`
  then covers only the retained window.
* ``record_changes`` (default ``True``) keeps a log of the structural changes
  seen at each step: components added, removed, activated or deactivated,
  whether by an explicit Stepper mutation, by an applied input network, or by
  islanding decided during the solve. Read it as
  :attr:`~monee.simulation.Stepper.changes` (a list of
  :class:`~monee.simulation.NetworkChange` records with ``step``, ``t_h``,
  ``kind``, ``component_type``, ``component_id``, ``name`` and ``source``) or
  as a DataFrame via :meth:`~monee.simulation.Stepper.changes_df`. Pass
  ``record_changes=False`` for a run that never changes topology and does not
  want the bookkeeping.
* ``max_changes`` caps that log the way ``max_history`` caps the step history,
  dropping the oldest events first. It is independent of ``max_history``:
  changes survive the trimming of the steps they happened in.
* ``copy_base`` and ``carry_failed_dt`` are covered below: ``copy_base=False``
  steps on the caller's live network instead of a copy, and
  ``carry_failed_dt`` decides whether a skipped step's interval is rolled into
  the next successful one.

The Stepper never mutates the base network: every step works on a fresh
``net.copy()``. After a successful solve the result network is pushed into
the persistent ``StepState``, which is how storage, linepack, and LTC state
carry across calls.

By default each successful step's solved values are also carried into the
Stepper's working copy, so the next solve starts from the previous solution
instead of the base network's initial values. On the CasADi backend such
solves additionally run with IPOPT warm start options (a small initial
barrier), which typically cuts the iteration count by half or more on
repeated, similar steps. Islanded components keep their last solved values,
so a component that rejoins later starts from a sane point. Pass
``warm_start=False`` to solve every step from the base network's initial
values instead.

The default follows the backend: warm starting is on for CasADi, GEKKO and
Pyomo, but off for the gurobipy backend, where seeding ``Var.Start`` from
the previous solution measured neutral (Gurobi's per step cost is presolve
and branching, which a start vector does not touch). Pass
``warm_start=True`` explicitly to enable it there anyway.

Note that with ``copy_base=False`` the working copy is the caller's live
network, so the default warm start writes solved values into it after every
successful step; pass ``warm_start=False`` there if the network must stay
strictly caller-owned.

On large networks using the smooth gas and heat NLP formulations (several
hundred nodes, such as the simbench based MES from the test suite) the
default IPOPT warm start options can backfire: iteration counts roughly
double compared to cold starts, because the previous solution sits on the
degenerate bounds of the pos/neg flow splits. For such networks either pass
``warm_start=False``, or keep the value carry-over but soften the warm
options per solver instance:

.. code-block:: python

   from monee.solver import CasADiSolver

   solver = CasADiSolver(solver_options={
       "ipopt.warm_start_init_point": "no",
       "ipopt.mu_init": 1e-5,
   })
   c = Stepper(net, solver=solver)

This measured about 1.4x faster than cold starts on the simbench MES, at
the cost of occasional failed steps (each self-heals with a flat start
retry) and warm results only being feasible to the warm tolerance rather
than converging as tightly as cold ones.

----

Stepping
========

.. code-block:: python

   step(dt_h, *, data_overrides=None, ts_index=None) -> StepResult

``step`` advances the simulation clock by ``dt_h`` hours and solves one
snapshot. Inputs for the step come from two optional sources, applied in
this order:

1. ``ts_index``: apply row ``ts_index`` of the constructor's
   :class:`~monee.simulation.TimeseriesData` to the network copy. Because the
   caller paces the simulation, the row index is explicit rather than an
   internal counter, so you may revisit, skip, or interpolate rows as your
   framework requires.
2. ``data_overrides``: a mapping ``{(component_id, attribute): value}``
   applied after the timeseries slice, so overrides win on conflicts. This is
   the natural channel for values arriving live from other simulators (for
   example setpoints computed by an agent system).

.. code-block:: python

   r = c.step(
       0.25,
       ts_index=3,                                  # profile row first,
       data_overrides={(load_id, "p_mw"): 0.3},     # then live inputs win
   )

Two validation rules are deliberate:

* ``dt_h <= 0`` raises ``ValueError``: time must move forward.
* An unknown component id in ``data_overrides`` raises ``KeyError``; a known
  id whose models lack the attribute raises ``AttributeError``. In a
  co-simulation, a misspelled id or attribute is a wiring bug, not a
  transient condition, so it always raises regardless of ``on_step_error``.

.. note::

   Like :class:`~monee.simulation.TimeseriesData`, overrides that target a
   solver ``Var`` pin its ``value``/``min``/``max`` to the given value: the
   quantity stays a variable, so it remains visible to ``StepState`` and the
   result tables.

----

Reading values back
===================

A co-simulation adapter needs the full set, step, get contract:
``data_overrides`` is the set side, :meth:`~monee.simulation.Stepper.step` the
step side, and :meth:`~monee.simulation.Stepper.get` the get side. Use ``get``
for the values you publish back to the orchestrator after each step:

.. code-block:: python

   r = c.step(0.25, data_overrides={(load_id, "p_mw"): 0.3})   # set + step
   p = c.get(load_id, "p_mw")                                  # get (latest)
   p_first = c.get(load_id, "p_mw", step=0)                    # absolute index

``get`` reads from the persistent ``StepState``: by default the most recent
successful solve (``step=-1``); negative values index relative to the latest
solve, non-negative values are absolute step indices. It returns ``None``
(or the ``initial_state`` fallback) when no solve has written the attribute
yet, or when the requested step has been dropped under ``max_history``.

``get`` accepts the same id forms as ``data_overrides`` keys: a bare id, a
``(type, id)`` tuple, or a bare id plus ``component_type=``. Node and child
ids are independent counters, so a bare id can be shared by a bus and a
child. Resolution is attribute aware: only components whose model carries
the requested attribute are considered, so a shared bare id resolves to the
bus for ``vm_pu`` and to the load for ``p_mw`` without further hints. When
several matching components carry the attribute, the child wins over the
node; the nodal aggregate is then returned only when the node is addressed
explicitly, for example ``c.get(1, "p_mw", component_type=mm.Bus)``. An id
that matches no component raises ``KeyError``; an id whose matching models
all lack the attribute raises ``AttributeError``, mirroring the
``data_overrides`` validation, so a misspelled attribute surfaces as a
wiring bug instead of a silent ``None``; and an id that stays ambiguous
(for example a child and a compound sharing it, both carrying the
attribute) raises ``ValueError`` asking for a disambiguated form.

A compound coupler's realized operating point (for example a CHP's
``el_mw`` and ``heat_mw``) lives on its control node model, not on the
compound model itself, which only carries the build inputs (setpoints,
efficiencies, ``regulation``). Read it from the control node's own id:

.. code-block:: python

   chp = net.compound_by_id(chp_id)
   cn = next(
       n for n in chp.subcomponents if isinstance(n.model, mm.CHPControlNode)
   )
   el = c.get(cn.id, "el_mw", component_type=mm.CHPControlNode)

or query whole runs by the control node model class, for example
``c.to_timeseries_result().get_result_for(mm.CHPControlNode, "el_mw")``.

For tabular post-processing of whole runs, prefer the ``StepResult`` returned
by each ``step`` call or :meth:`~monee.simulation.Stepper.to_timeseries_result`.

----

Error handling
==============

With the default ``on_step_error="raise"``, a solver failure propagates to
the caller and neither the history nor the clock changes.

With ``on_step_error="skip"``, a failure is logged, a failed
:class:`~monee.simulation.StepResult` (``failed=True``, ``error`` set) is
appended to the history, and time still advances by ``dt_h``. The external
clock and the Stepper's clock stay in sync even across failed steps:

.. code-block:: python

   c = Stepper(net, on_step_error="skip")
   r = c.step(1.0)
   if r.failed:
       print("step failed:", r.error)

Failed steps are excluded from the DataFrame queries of
:meth:`~monee.simulation.Stepper.to_timeseries_result` but remain
inspectable via the result's ``failed_steps`` property.

The interval of a failed step is not thrown away: by default it is carried
into the next successful step, which then integrates over
``dt_h + backlog`` so that no time goes missing from a storage state of
charge or a linepack.  The step is logged at WARNING level and reports both
intervals:

.. code-block:: python

   c = Stepper(net, on_step_error="skip")
   c.step(1.0)                       # fails
   r = c.step(1.0)                   # integrates 2 h
   r.dt_h, r.effective_dt_h, r.t_h   # (1.0, 2.0, 2.0)

Two consequences to plan for.  The carried interval is integrated against
the setpoints of the new step, which were never requested for the failed
one, so a battery told to charge at 0.3 MW for 1 h gains 0.6 MWh.  And the
longer interval multiplies the setpoint against the storage bounds, so after
a streak of failures a step at rated power can turn infeasible on its own
capacity limit.  A host that owns the clock and wants every step integrated
over exactly the ``dt_h`` it passed constructs the Stepper with
``carry_failed_dt=False``; ``t_h`` advances by the raw ``dt_h`` either way.

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
   * - ``with Stepper(net) as c: ...``
     - Context-manager support for tidy scoping in framework adapters

``reset`` is handy when a co-simulation scenario is re-run: the same
Stepper (and resolved solver) can be reused without rebuilding the
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
   p  = ts_result.get_result_for_id(load_id, "p_mw", mm.PowerLoad)

   # Optionally label rows with real timestamps:
   import pandas as pd
   idx = pd.date_range("2024-01-01", periods=c.step_count, freq="h")
   ts_result = c.to_timeseries_result(datetime_index=idx)

The Stepper's headline feature is a variable step size, so the index is
rarely a fixed-frequency range. Build it from the per-step ``dt_h`` values
instead, following the same convention as :func:`~monee.run_timeseries`:
a label marks the end of the interval it integrates over (and the first
step's interval is the first difference). With a start time ``t0`` the label
of step ``k`` is ``t0`` plus the cumulative sum of ``dt_h`` up to and
including step ``k``:

.. code-block:: python

   import numpy as np

   dts = [0.25] * 16 + [1.0] * 16          # the dt_h passed to each step()
   # or reconstruct from the retained history:
   # dts = [sr.dt_h for sr in c.history]
   t0 = pd.Timestamp("2024-01-01")
   idx = t0 + pd.to_timedelta(np.cumsum(dts), unit="h")
   ts_result = c.to_timeseries_result(datetime_index=idx)

Under ``max_history`` the result covers only the retained window, and
``datetime_index`` may take either of two lengths: one entry per taken step
(``c.step_count``, absolute labels; entries for dropped steps are simply
never used) or one entry per retained step (``len(c.history)``, labels for
the window, like the ``dts`` reconstructed from ``c.history`` above). Any
other length raises a ``ValueError`` naming both accepted forms.

Everything documented in :doc:`timeseries` for result querying
(:meth:`~monee.simulation.TimeseriesResult.get_result_for`,
:meth:`~monee.simulation.TimeseriesResult.get_result_for_id`,
``ts_result[component_id]``, ``failed_steps``, the notebook HTML repr)
works unchanged.

.. tip::

   Because the Stepper records every step in order, you can call
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
      coupling that the Stepper reuses.

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
