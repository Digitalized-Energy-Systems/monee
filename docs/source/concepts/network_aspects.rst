================
Network aspects
================

A NetworkAspect is a solver-agnostic plug-in that adds variables and constraints
to a network without modifying any existing model class. It works at the network
level rather than the component level. Reach for one whenever a feature spans
multiple components or needs coordination across the whole topology.

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Islanding
      :shadow: sm

      Binary connectivity variables and flow constraints that restore
      feasibility when the network splits into electrically isolated islands.

   .. grid-item-card:: Lumped Thermal Capacitance
      :shadow: sm

      Thermal inertia in district heating junctions: slows temperature
      propagation between consecutive timesteps.

   .. grid-item-card:: Gas Linepack
      :shadow: sm

      Stored gas mass in pipeline segments: enables the pipeline itself to
      act as a short-term storage buffer.

   .. grid-item-card:: Your extension
      :shadow: sm

      Subclass :class:`~monee.model.extension.core.NetworkAspect`, implement
      the phases you need, and call ``network.add_extension()``.

----

Why not a formulation?
======================

Formulations (:class:`~monee.model.formulation.core.BranchFormulation`,
:class:`~monee.model.formulation.core.NodeFormulation`, …) define how a single
component type behaves: the equations for a gas pipe, the variables for a bus.
A ``NetworkAspect`` is different:

* It touches many components at once (for example, all water junctions).
* It injects variables onto existing models that were not designed for the
  feature, such as adding a ``linepack_kg`` variable to every ``GasPipe`` that
  carries linepack.
* It can activate or suppress behaviour conditionally. Gas linepack, for
  instance, pins its net mass change to zero in a steady-state solve and lets a
  dynamic balance govern it only when a timeseries solve is running.

----

The four phases
===============

Every ``NetworkAspect`` participates in up to four phases of the solver pipeline:

.. mermaid::

   flowchart LR
       P0["Phase 0<br/>prepare"]
       P1a["Phase 1a<br/>activate timeseries"]
       P1b["Phase 1b<br/>equations"]
       P1c["Phase 1c<br/>time coupling"]
       P0 --> P1a --> P1b --> P1c

Only implement the phases you need; every phase has a no-op default.

----

Minimal example
===============

A one-file aspect that adds a pressure-spread variable to every gas junction and
bounds it against a reference junction:

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.model.extension.core import NetworkAspect
   from monee.model.core import Var

   class PressureSpreadAspect(NetworkAspect):
       """Adds a ``p_spread_pu`` Var bounded against a reference junction."""

       def prepare(self, network) -> None:
           for node in network.nodes:
               if isinstance(node.model, mm.Junction):
                   node.model.p_spread_pu = Var(0.0, min=0.0, name="p_spread_pu")

       def equations(self, network, ignored_nodes: set) -> list:
           eqs = []
           junctions = [n.model for n in network.nodes
                        if isinstance(n.model, mm.Junction)
                        and n.id not in ignored_nodes]
           if len(junctions) >= 2:
               ref = junctions[0]
               for junc in junctions[1:]:
                   eqs.append(junc.p_spread_pu >= ref.pressure_squared_pu
                                                 - junc.pressure_squared_pu)
           return eqs

   # Register and use
   net = mx.create_multi_energy_network()
   net.add_extension(PressureSpreadAspect())

----

Temporal phases in detail
=========================

All three temporal methods share the same ``InterStepState`` interface but run
in different contexts:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Method
     - Called by
     - Typical use
   * - ``inter_temporal_equations``
     - timeseries and multi-period
     - Storage SoC, linepack balance, LTC: anything that should work in both
   * - ``inter_step_equations``
     - timeseries only
     - Logic that only makes sense step-by-step (e.g. clamp from a controller)
   * - ``inter_period_equations``
     - multi-period only
     - Look-ahead constraints (e.g. require non-decreasing SoC over horizon)

The ``temporal_state`` argument implements
:class:`~monee.simulation.step_state.InterStepState`:

.. code-block:: python

   # In a timeseries solve: temporal_state is StepState
   #   → .get() returns a plain float
   #
   # In a multi-period solve: temporal_state is PeriodState
   #   → .get() returns a live solver variable (GEKKO GKVariable / Pyomo Var)
   #
   # Both expose the same .get() and .dt_h API, so your code is identical.

   def inter_temporal_equations(self, network, ignored_nodes, temporal_state):
       eqs = []
       dt_s = temporal_state.dt_h * 3600.0
       for branch in network.branches:
           if branch.id not in self._tracked:
               continue
           bm = branch.model
           prev = temporal_state.get(branch.id, "stored_kg")
           if prev is None:
               prev = self._initial[branch.id]
           eqs.append(bm.stored_kg == prev + dt_s * bm.net_inflow)
       return eqs

----

Registering an aspect
=====================

Register aspects on the network before the first solve. They persist across all
later single-step, timeseries, and multi-period solves:

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.model import LumpedThermalCapacitance, GasLinepack

   net = mx.create_multi_energy_network()

   # Water side
   js = mx.create_water_junction(net)
   jl = mx.create_water_junction(net)
   mx.create_ext_hydr_grid(net, js)
   mx.create_water_pipe(net, js, jl, diameter_m=0.3, length_m=500)

   # Gas side
   jg0 = mx.create_gas_junction(net)
   jg1 = mx.create_gas_junction(net)
   mx.create_gas_ext_grid(net, jg0)
   pipe_id = mx.create_gas_pipe(net, jg0, jg1, diameter_m=0.3, length_m=10_000)

   # Attach both aspects
   net.add_extension(LumpedThermalCapacitance())
   net.add_extension(GasLinepack())

   print(len(net.extensions))

.. testoutput::

   2

Multiple aspects compose without conflict; each operates on its own
variable subset.

----

See also
========

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Temporal extensions
      :link: temporal_extensions
      :link-type: doc
      :shadow: sm

      Step-by-step walkthroughs for ``LumpedThermalCapacitance`` and
      ``GasLinepack``, including physics background and visualisation code.

   .. grid-item-card:: Islanding
      :link: islanding
      :link-type: doc
      :shadow: sm

      The built-in islanding system is implemented as a ``NetworkAspect``.
      A detailed breakdown of its phases is in the islanding concept page.

   .. grid-item-card:: Timeseries simulation
      :link: timeseries
      :link-type: doc
      :shadow: sm

      How the solver pipeline calls temporal aspect methods and how
      ``StepState`` bridges consecutive steps.

   .. grid-item-card:: Multi-period optimization
      :link: multi_period
      :link-type: doc
      :shadow: sm

      How ``PeriodState`` replaces ``StepState`` in a single-shot joint
      solve, and why existing aspect code works unchanged.
