====================
Temporal extensions
====================

Two built-in :doc:`network_aspects` model time-dependent physics that a
steady-state solve cannot capture:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: LumpedThermalCapacitance
      :shadow: sm

      Thermal inertia of the water mass in district heating junctions.
      Smooths temperature propagation: a supply-temperature step-change
      arrives at downstream junctions with a delay and a rounded wavefront.

   .. grid-item-card:: GasLinepack
      :shadow: sm

      Gas stored in the volume of a pipeline segment.  Lets the pipeline
      itself act as a short-term storage buffer, absorbing demand peaks
      without requiring immediate upstream response.

Both extensions work identically in :doc:`timeseries` simulation and
:doc:`multi_period` optimization: the same ``net.add_extension()`` call
activates them for any solve mode.

Under the hood both are plain :class:`~monee.model.extension.core.NetworkAspect`
subclasses and rely only on the standard temporal hooks (explained in full in
:doc:`network_aspects`):

* ``prepare``: inject ``Var`` placeholders before solver variable injection.
* ``activate_timeseries``: flag the coupled solve and warm-start variables
  from the previous step's ``step_state``.
* ``inter_temporal_equations``: the actual time-coupling constraint, applied
  in both timeseries and multi-period solves.
* ``inter_step_equations`` / ``inter_period_equations``: mode-specific
  variants; not used by the built-in extensions but available to yours.

.. note::

   Extensions registered via ``network.add_extension()`` (including
   ``LumpedThermalCapacitance``, ``GasLinepack``, and the islanding
   configuration from :doc:`islanding`) are not serialized by the native
   JSON format (:mod:`monee.io.native`).  After
   :func:`~monee.io.native.load_to_network` you must re-register every
   extension before solving.

----

LumpedThermalCapacitance
========================

Physical background
-------------------

Each water junction stores thermal energy in proportion to the water mass of
the adjacent half-pipes:

.. math::

   \rho \cdot V_\text{node}
   \cdot \frac{T(t) - T(t-1)}{\Delta t}
   \;=\;
   \sum_\text{in} \dot{m}_\text{in} \cdot T_\text{in}
   \;-\;
   \sum_\text{out} \dot{m}_\text{out} \cdot T_\text{node}

where :math:`V_\text{node} = \sum_\text{pipes} \frac{\pi}{4} d^2 L / 2`.

Without this term, junction temperatures jump instantaneously.  With it,
large-diameter or long pipes create appreciable thermal lag.

The extension applies only to water-grid junctions without a
:class:`~monee.model.child.GridFormingMixin` child: fixed-temperature
supplies such as ``ExtHydrGrid`` or a grid-forming source are excluded
automatically, because their temperature is driven externally.

Step-by-step walkthrough
------------------------

Build the network

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.model import LumpedThermalCapacitance
   from monee.simulation import TimeseriesData

   net = mx.create_multi_energy_network()

   j_supply = mx.create_water_junction(net)
   j_mid    = mx.create_water_junction(net)
   j_load   = mx.create_water_junction(net)

   mx.create_ext_hydr_grid(net, j_supply)
   mx.create_water_sink(net, j_load, mass_flow_kgs=0.5)
   mx.create_water_pipe(net, j_supply, j_mid,
                        diameter_m=0.3, length_m=500)
   mx.create_water_pipe(net, j_mid, j_load,
                        diameter_m=0.2, length_m=300)

Attach the extension (one line)

.. testcode::

   net.add_extension(LumpedThermalCapacitance())

No per-junction parameters are required.  The extension scans the network
topology, computes nodal volumes, and patches each eligible junction's
``t_pu`` into a proper solver variable.

Define a supply-temperature step-change

.. testcode::

   # Supply temperature drops from nominal (1.0 pu) to 0.8 pu at step 4
   td = TimeseriesData()
   td.add_node_series(j_supply, "t_pu",
                      [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8])

Comparison: with vs. without LTC

The plot below runs the same network twice (with and without the extension)
and overlays the junction temperatures at ``j_mid`` to show the inertia
effect.

LTC thermal inertia, supply step-change at step 4.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/temporal_extensions_1.html" width="100%" height="500" style="border:none;" loading="lazy" title="LTC thermal inertia"></iframe>

.. only:: latex

   .. image:: /_static/interactive/temporal_extensions_1.png
      :width: 100%

Without ``LumpedThermalCapacitance`` all three temperatures jump to 0.8
simultaneously at step 4.  With it, ``j_mid`` and ``j_load`` respond
gradually, reflecting the thermal mass of the water stored in the pipes.

.. tip::

   The extension is a no-op in single-step solves: its static
   ``equations()`` hook returns an empty list, so results are identical
   whether or not it is attached.  Thermal inertia only enters through
   ``inter_temporal_equations``, i.e. inside ``run_timeseries`` or
   ``run_multi_period``.

First-step anchoring
--------------------

The inertia equation needs a previous temperature :math:`T(t-1)`.  On the very
first step there is none, so the extension anchors each junction to an initial
value.  All three constructor arguments control this anchor:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Argument
     - Effect
   * - ``default_t_init``
     - First-step temperature anchor (in ``t_pu``) for all LTC junctions.
   * - ``t_init_overrides``
     - ``{node_id: t_pu}``: per-junction anchors that take precedence over
       ``default_t_init``.
   * - ``first_step_steady_state``
     - If ``True``, the first step emits ``net_heat == 0`` (a steady-state
       heat balance) instead of anchoring.  MIP backends only.

The anchor for each junction is resolved with this precedence:

1. ``t_init_overrides[node_id]``: explicit per-junction value,
2. ``default_t_init``: network-wide fallback,
3. the junction's own ``t_pu`` initializer.

.. testcode::

   net.add_extension(LumpedThermalCapacitance(
       default_t_init=0.95,             # anchor all junctions at 0.95 pu
       t_init_overrides={j_mid: 0.97},  # ... except j_mid
   ))

.. tip::

   Set ``default_t_init`` close to the expected operating mean.  Otherwise
   the first steps of the simulation are spent heating (or cooling) the
   thermal mass from the anchor towards the operating point, a warm-up
   transient that distorts early results.

.. note::

   The default anchored mode is required for the NLP solvers
   (GEKKO/IPOPT).  ``first_step_steady_state=True`` is supported on MIP
   backends only; use it exclusively with a MIP-capable Pyomo solver.

----

GasLinepack
===========

Linepack physics
----------------

A pipeline of volume *V* and gas density *ρ* stores mass:

.. math::

   \text{linepack\_kg} \;=\; V \cdot \rho
   \;=\; \frac{\pi}{4} d^2 L \;\cdot\; \rho(p_\text{avg}, T)

The density *ρ* is derived from the average of the two endpoint pressures via
the ideal-gas equation of state.  ``GasLinepack`` constrains ``linepack_kg``
to equal this product at every step, so stored mass automatically follows
pipeline pressure as demand varies over a timeseries or multi-period run.

Single-step behaviour and temporal coupling
-------------------------------------------

The extension injects two variables on every ``GasPipe``: ``linepack_kg``
(stored mass) and ``net_pack_kgs`` (charging rate, positive = mass flowing
*into* storage).  Their behaviour depends on the solve mode:

* Single-step solves pin ``net_pack_kgs == 0``: the pipe neither charges
  nor discharges, and the solve is a pure steady state.  ``linepack_kg``
  still reports the stored mass implied by the solved pressures.
* Timeseries and multi-period solves call ``activate_timeseries``, which
  lifts the pin and warm-starts ``linepack_kg`` from the previous step's
  state.  The temporal coupling then enforces

  .. math::

     \text{net\_pack\_kgs}(t) \cdot \Delta t
     \;=\; \text{linepack\_kg}(t) - \text{linepack\_kg}(t-1)

  with :math:`\Delta t` in seconds; the first step measures against the
  initial linepack.

On the node side, each endpoint junction's mass balance receives
``0.5 * net_pack_kgs`` (outflow-positive): the charged mass is split equally
between the two ends of the pipe, so charging draws gas from both endpoint
junctions and discharging releases it to both.

Linepack walkthrough
--------------------

Build the gas network

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.model import GasLinepack
   from monee.simulation import TimeseriesData

   net_gas = mx.create_multi_energy_network()

   j0 = mx.create_gas_junction(net_gas)   # feed end
   j1 = mx.create_gas_junction(net_gas)   # mid
   j2 = mx.create_gas_junction(net_gas)   # demand end

   mx.create_gas_ext_grid(net_gas, j0)
   mx.create_gas_sink(net_gas, j2, mass_flow_kgs=0.3, name="industry")

   # Long transmission pipe - large volume = significant linepack
   long_pipe_id = mx.create_gas_pipe(net_gas, j0, j1,
                                     diameter_m=0.5, length_m=50_000)
   mx.create_gas_pipe(net_gas, j1, j2,
                      diameter_m=0.3, length_m=10_000)

Attach GasLinepack (one line for all pipes)

.. testcode::

   net_gas.add_extension(GasLinepack())

``GasLinepack()`` with no arguments applies to all ``GasPipe`` branches.
Capacities (initial and maximum stored mass) are auto-computed from each pipe's
geometry and the gas-grid thermodynamics:

* Initial = ``V_pipe × ρ`` at nominal pressure
* Maximum = ``V_pipe × ρ`` at the grid's maximum pressure bound

To override per-pipe values pass an ``overrides`` dict:

.. testcode::

   net_gas.add_extension(GasLinepack(overrides={
       long_pipe_id: dict(
           linepack_kg_initial=65_000,  # override initial stored mass  [kg]
           linepack_kg_max=75_000,      # override maximum linepack     [kg]
       )
   }))

Omitted keys fall back to the auto-computed values; other pipes not listed
in ``overrides`` always use auto-computed values.

Define a demand profile

.. testcode::

   # Off-peak → peak → off-peak
   demand = [0.20, 0.20, 0.35, 0.45, 0.50, 0.45, 0.25, 0.20]
   td_gas = TimeseriesData()
   td_gas.add_child_series_by_name("industry", "mass_flow_kgs", demand)

Timeseries: replay the profile

.. code-block:: python

   from monee.simulation import run_timeseries

   result = run_timeseries(net_gas, td_gas)
   lp = result.get_result_for_id(long_pipe_id, "linepack_kg")

Multi-period: let the optimizer exploit the buffer

.. code-block:: python

   from monee.simulation import run_multi_period

   result = run_multi_period(net_gas, td_gas, dt_h=1.0)
   lp = result.get_result_for_id(long_pipe_id, "linepack_kg")

Plot: linepack as a buffer, with vs. without

Linepack decouples the source from instant demand changes.  Without it the
source must match demand exactly at every step.  With it the pipeline absorbs
or releases the difference, so the source responds more gradually.

The plot below runs the same network twice (with and without the extension)
and overlays the source feed rate alongside the consumer demand profile.

Gas linepack, source is buffered from demand peaks.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/temporal_extensions_2.html" width="100%" height="660" style="border:none;" loading="lazy" title="Gas linepack buffering"></iframe>

.. only:: latex

   .. image:: /_static/interactive/temporal_extensions_2.png
      :width: 100%

The middle panel is the key comparison: without linepack the source feed (red
dashed) tracks demand exactly.  With linepack (blue) the source responds more
gradually because the pipe absorbs the shortfall during ramp-up and releases
it during the subsequent decline.  The bottom panel shows the stored mass
draining during the peak and recovering afterwards.

Auto-computed capacity
----------------------

``GasLinepack`` derives per-pipe capacities from the grid's thermodynamic
parameters using the ideal-gas equation of state:

.. math::

   \rho(p, T) \;=\; \frac{p \cdot M}{R \cdot T}

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Quantity
     - Formula
   * - ``linepack_kg_initial``
     - ``V_pipe × ρ(p_ref × p_nominal_pu, T)``
   * - ``linepack_kg_max``
     - ``V_pipe × ρ(p_ref × √pressure_squared_pu_max, T)``

Both values can be overridden per pipe via the ``overrides`` argument to
``GasLinepack``.

Override reference
------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Override key
     - Description
   * - ``linepack_kg_initial``
     - Override initial stored mass in kg; seeds the first-step constraint and
       the solver's starting value.
   * - ``linepack_kg_max``
     - Override upper bound on stored mass.  Values below
       ``1.05 × linepack_kg_initial`` are raised to that floor automatically.

----

Combining extensions
====================

Multiple aspects compose without conflict:

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.model import LumpedThermalCapacitance, GasLinepack

   net_mes = mx.create_multi_energy_network()

   # Water side
   jw0 = mx.create_water_junction(net_mes)
   jw1 = mx.create_water_junction(net_mes)
   mx.create_ext_hydr_grid(net_mes, jw0)
   mx.create_water_pipe(net_mes, jw0, jw1, diameter_m=0.3, length_m=500)

   # Gas side
   jg0 = mx.create_gas_junction(net_mes)
   jg1 = mx.create_gas_junction(net_mes)
   mx.create_gas_ext_grid(net_mes, jg0)
   gp_id = mx.create_gas_pipe(net_mes, jg0, jg1,
                               diameter_m=0.4, length_m=20_000)

   net_mes.add_extension(LumpedThermalCapacitance())
   net_mes.add_extension(GasLinepack())

   print(len(net_mes.extensions))

.. testoutput::

   2

Each extension operates on its own variable subset and their equation sets
are concatenated.  There is no cross-extension interaction unless the network
physics couples them (e.g. a CHP unit connected to both carriers).

----

See also
========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: NetworkAspect
      :link: network_aspects
      :link-type: doc
      :shadow: sm

      The general extension mechanism: four phases, how to write your own.

   .. grid-item-card:: Timeseries simulation
      :link: timeseries
      :link-type: doc
      :shadow: sm

      Sequential solver pipeline, ``StepState``, inter-step hooks.

   .. grid-item-card:: Multi-period optimization
      :link: multi_period
      :link-type: doc
      :shadow: sm

      Single-shot joint solve, ``PeriodState``, MPC.
