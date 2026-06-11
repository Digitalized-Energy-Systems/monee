====================
Temporal extensions
====================

Two built-in :doc:`network_aspects` model time-dependent physical phenomena
that cannot be captured in a steady-state solve:

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: LumpedThermalCapacitance
      :shadow: sm

      Thermal inertia of the water mass in district heating junctions.
      Smooths temperature propagation - a supply-temperature step-change
      arrives at downstream junctions with a delay and a rounded wavefront.

   .. grid-item-card:: GasLinepack
      :shadow: sm

      Gas stored in the volume of a pipeline segment.  Lets the pipeline
      itself act as a short-term storage buffer, absorbing demand peaks
      without requiring immediate upstream response.

Both extensions work identically in :doc:`timeseries` simulation and
:doc:`multi_period` optimization - the same ``net.add_extension()`` call
activates them for any solve mode.

Under the hood both are plain :class:`~monee.model.formulation.core.NetworkAspect`
subclasses and rely only on the standard temporal hooks (explained in detail in
:doc:`network_aspects` - they are not re-explained here):

* ``prepare`` - inject ``Var`` placeholders before solver variable injection.
* ``activate_timeseries`` - flag the coupled solve and warm-start variables
  from the previous step's ``step_state``.
* ``inter_temporal_equations`` - the actual time-coupling constraint, applied
  in **both** timeseries and multi-period solves.
* ``inter_step_equations`` / ``inter_period_equations`` - mode-specific
  variants; not used by the built-in extensions but available to yours.

.. note::

   Extensions registered via ``network.add_extension()`` - including
   ``LumpedThermalCapacitance``, ``GasLinepack``, and the islanding
   configuration from :doc:`islanding` - are **not serialized** by the native
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

The extension applies only to water-grid junctions **without** a
:class:`~monee.model.child.GridFormingMixin` child - fixed-temperature
supplies such as ``ExtHydrGrid`` or a grid-forming source are excluded
automatically, because their temperature is driven externally.

Step-by-step walkthrough
------------------------

**Build the network**

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
   mx.create_water_sink(net, j_load, mass_flow=0.5)
   mx.create_water_pipe(net, j_supply, j_mid,
                        diameter_m=0.3, length_m=500)
   mx.create_water_pipe(net, j_mid, j_load,
                        diameter_m=0.2, length_m=300)

**Attach the extension - one line**

.. testcode::

   net.add_extension(LumpedThermalCapacitance())

No per-junction parameters are required.  The extension scans the network
topology, computes nodal volumes, and patches each eligible junction's
``t_pu`` into a proper solver variable.

**Define a supply-temperature step-change**

.. testcode::

   # Supply temperature drops from nominal (1.0 pu) to 0.8 pu at step 4
   td = TimeseriesData()
   td.add_node_series(j_supply, "t_pu",
                      [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8])

**Comparison: with vs. without LTC**

The plot below runs the same network twice - with and without the extension
- and overlays the junction temperatures at ``j_mid`` to show the inertia
effect.

.. plot::
   :caption: LTC thermal inertia - supply step-change at step 4

   import monee.model as mm
   import monee.express as mx
   from monee.model import LumpedThermalCapacitance
   from monee.simulation import TimeseriesData, run_timeseries
   import matplotlib.pyplot as plt
   import matplotlib.patches as mpatches

   supply_temp = [1.0, 1.0, 1.0, 1.0, 0.8, 0.8, 0.8, 0.8]

   def build_net(with_ltc):
       net = mx.create_multi_energy_network()
       j_supply = mx.create_water_junction(net)
       j_mid    = mx.create_water_junction(net)
       j_load   = mx.create_water_junction(net)
       mx.create_ext_hydr_grid(net, j_supply)
       mx.create_water_sink(net, j_load, mass_flow=0.5)
       mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.3, length_m=500)
       mx.create_water_pipe(net, j_mid, j_load,   diameter_m=0.2, length_m=300)
       if with_ltc:
           net.add_extension(LumpedThermalCapacitance())
       return net, j_supply, j_mid, j_load

   fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
   colors = {"supply": "#2c7bb6", "mid": "#d7191c", "load": "#1a9641"}

   for ax, with_ltc in zip(axes, [False, True]):
       net, j_supply, j_mid, j_load = build_net(with_ltc)
       td = TimeseriesData()
       td.add_node_series(j_supply, "t_pu", supply_temp)
       result = run_timeseries(net, td)

       t_supply_s = result.get_result_for_id(j_supply, "t_pu")
       t_mid_s    = result.get_result_for_id(j_mid,    "t_pu")
       t_load_s   = result.get_result_for_id(j_load,   "t_pu")
       steps = range(len(t_supply_s))

       ax.step(steps, t_supply_s.values, where="post", lw=2,
               color=colors["supply"], label="supply (j₀)")
       ax.step(steps, t_mid_s.values,    where="post", lw=2,
               color=colors["mid"], label="mid (j₁)")
       ax.step(steps, t_load_s.values,   where="post", lw=2,
               color=colors["load"], label="load (j₂)")
       ax.axvline(3.5, color="grey", linestyle="--", alpha=0.5, lw=1)
       ax.set_xlabel("Timestep  [h]")
       ax.set_ylim(0.75, 1.05)
       ax.set_title("With LTC" if with_ltc else "Without LTC")
       ax.grid(True, alpha=0.25)
       ax.legend(fontsize=8)

   axes[0].set_ylabel("Temperature  [pu]")
   fig.suptitle("Thermal inertia - supply step-change at t = 4",
                fontsize=11, fontweight="bold")
   plt.tight_layout()

Without ``LumpedThermalCapacitance`` all three temperatures jump to 0.8
simultaneously at step 4.  With it, ``j_mid`` and ``j_load`` respond
gradually, reflecting the thermal mass of the water stored in the pipes.

.. tip::

   The extension is a **no-op in single-step solves** - its static
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
     - First-step temperature anchor (in ``t_pu``) for **all** LTC junctions.
   * - ``t_init_overrides``
     - ``{node_id: t_pu}`` - per-junction anchors that take precedence over
       ``default_t_init``.
   * - ``first_step_steady_state``
     - If ``True``, the first step emits ``net_heat == 0`` (a steady-state
       heat balance) instead of anchoring.  **MIP backends only.**

The anchor for each junction is resolved with this precedence:

1. ``t_init_overrides[node_id]`` - explicit per-junction value,
2. ``default_t_init`` - network-wide fallback,
3. the junction's own ``t_pu`` initializer.

.. testcode::

   net.add_extension(LumpedThermalCapacitance(
       default_t_init=0.95,             # anchor all junctions at 0.95 pu
       t_init_overrides={j_mid: 0.97},  # ... except j_mid
   ))

.. tip::

   Set ``default_t_init`` close to the expected operating mean.  Otherwise
   the first steps of the simulation are spent heating (or cooling) the
   thermal mass from the anchor towards the operating point - a warm-up
   transient that distorts early results.

.. note::

   The default anchored mode is **required** for the NLP solvers
   (GEKKO/IPOPT).  ``first_step_steady_state=True`` is supported on MIP
   backends only - use it exclusively with a MIP-capable Pyomo solver.

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
to equal this product at every step - so stored mass automatically follows
pipeline pressure as demand varies over a timeseries or multi-period run.

Single-step behaviour and temporal coupling
-------------------------------------------

The extension injects two variables on every ``GasPipe``: ``linepack_kg``
(stored mass) and ``net_pack_kgs`` (charging rate, positive = mass flowing
*into* storage).  Their behaviour depends on the solve mode:

* **Single-step solves** pin ``net_pack_kgs == 0`` - the pipe neither charges
  nor discharges, and the solve is a pure steady state.  ``linepack_kg``
  still reports the stored mass implied by the solved pressures.
* **Timeseries and multi-period solves** call ``activate_timeseries``, which
  lifts the pin and warm-starts ``linepack_kg`` from the previous step's
  state.  The temporal coupling then enforces

  .. math::

     \text{net\_pack\_kgs}(t) \cdot \Delta t
     \;=\; \text{linepack\_kg}(t) - \text{linepack\_kg}(t-1)

  with :math:`\Delta t` in seconds; the first step measures against the
  initial linepack.

On the node side, each endpoint junction's mass balance receives
``0.5 * net_pack_kgs`` (outflow-positive) - the charged mass is split equally
between the two ends of the pipe, so charging draws gas from both endpoint
junctions and discharging releases it to both.

Linepack walkthrough
--------------------

**Build the gas network**

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
   mx.create_gas_sink(net_gas, j2, mass_flow=0.3, name="industry")

   # Long transmission pipe - large volume = significant linepack
   long_pipe_id = mx.create_gas_pipe(net_gas, j0, j1,
                                     diameter_m=0.5, length_m=50_000)
   mx.create_gas_pipe(net_gas, j1, j2,
                      diameter_m=0.3, length_m=10_000)

**Attach GasLinepack - one line for all pipes**

.. testcode::

   net_gas.add_extension(GasLinepack())

``GasLinepack()`` with no arguments applies to **all** ``GasPipe`` branches.
Capacities (initial and maximum stored mass) are auto-computed from each pipe's
geometry and the gas-grid thermodynamics:

* **Initial** = ``V_pipe × ρ`` at nominal pressure
* **Maximum** = ``V_pipe × ρ`` at the grid's maximum pressure bound

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

**Define a demand profile**

.. testcode::

   # Off-peak → peak → off-peak
   demand = [0.20, 0.20, 0.35, 0.45, 0.50, 0.45, 0.25, 0.20]
   td_gas = TimeseriesData()
   td_gas.add_child_series_by_name("industry", "mass_flow", demand)

**Timeseries: replay the profile**

.. code-block:: python

   from monee.simulation import run_timeseries

   result = run_timeseries(net_gas, td_gas)
   lp = result.get_result_for_id(long_pipe_id, "linepack_kg")

**Multi-period: let the optimizer exploit the buffer**

.. code-block:: python

   from monee.simulation import run_multi_period

   result = run_multi_period(net_gas, td_gas, dt_h=1.0)
   lp = result.get_result_for_id(long_pipe_id, "linepack_kg")

**Plot: linepack as a buffer - with vs. without**

The key insight is that linepack *decouples* the source from instant demand
changes.  Without linepack the source must match demand exactly at every step.
With linepack the pipeline absorbs or releases the difference, so the source
responds more gradually.

The plot below runs the same network twice - with and without the extension
- and overlays the source feed rate alongside the consumer demand profile.

.. plot::
   :caption: Gas linepack - source is buffered from demand peaks

   import monee.model as mm
   import monee.express as mx
   from monee.model import GasLinepack
   from monee.simulation import TimeseriesData, run_timeseries
   import matplotlib.pyplot as plt

   # Demand: low baseline → morning ramp → midday peak → evening decline
   DEMAND = [0.15, 0.16, 0.17, 0.14, 0.15, 0.10, 0.12, 0.04]   # kg/s

   def build_and_run(with_linepack):
       net = mx.create_multi_energy_network()
       j0  = mx.create_gas_junction(net)
       j1  = mx.create_gas_junction(net)
       j2  = mx.create_gas_junction(net)
       src_id  = mx.create_gas_ext_grid(net, j0)
       mx.create_gas_sink(net, j2, mass_flow=0.20, name="consumer")
       # Long high-pressure pipe - significant stored volume
       pipe_id = mx.create_gas_pipe(net, j0, j1, diameter_m=0.6, length_m=40_000)
       mx.create_gas_pipe(net, j1, j2, diameter_m=0.3, length_m=8_000)

       if with_linepack:
           net.add_extension(GasLinepack())

       td = TimeseriesData()
       td.add_child_series_by_name("consumer", "mass_flow", DEMAND)
       result = run_timeseries(net, td)
       return result, pipe_id, j0

   result_lp,  pipe_id, j0 = build_and_run(with_linepack=True)
   result_nolp, _,       _  = build_and_run(with_linepack=False)

   # Source outflow is the mass_flow of ExtHydrGrid at j0 (negative = injection)
   src_lp   = result_lp.get_result_for_id(j0, "mass_flow")
   src_nolp = result_nolp.get_result_for_id(j0, "mass_flow")
   lp_kg    = result_lp.get_result_for_id(pipe_id, "linepack_kg")
   lp0      = lp_kg.values[0]

   steps = range(len(DEMAND))

   fig, axes = plt.subplots(3, 1, figsize=(9, 8),
                             gridspec_kw={"hspace": 0.55})

   C_DEM  = "#f4a261"
   C_LP   = "#2c7bb6"
   C_NOLP = "#d7191c"
   C_PACK = "#1a9641"

   # ── Top: demand profile ───────────────────────────────────────────────
   ax_d = axes[0]
   ax_d.step(steps, DEMAND, where="post", lw=2.5, color=C_DEM, label="demand")
   ax_d.fill_between(steps, 0, DEMAND, step="post", color=C_DEM, alpha=0.15)
   ax_d.set_ylabel("Flow  [kg/s]")
   ax_d.set_title("Consumer demand", fontsize=10)
   ax_d.set_xticks(list(steps))
   ax_d.grid(axis="y", alpha=0.3)

   # ── Middle: source feed rate ──────────────────────────────────────────
   ax_s = axes[1]
   # ExtHydrGrid mass_flow sign: negative = injection into network → negate for "feed"
   feed_lp   = [-v for v in src_lp.values]
   feed_nolp = [-v for v in src_nolp.values]

   ax_s.step(steps, feed_nolp, where="post", lw=2, color=C_NOLP,
             linestyle="--", label="without linepack")
   ax_s.step(steps, feed_lp,   where="post", lw=2, color=C_LP,
             label="with linepack")
   ax_s.step(steps, DEMAND, where="post", lw=1, color=C_DEM,
             linestyle=":", alpha=0.6, label="demand (ref.)")
   ax_s.set_ylabel("Feed rate  [kg/s]")
   ax_s.set_title("Source feed rate", fontsize=10)
   ax_s.set_xticks(list(steps))
   ax_s.legend(fontsize=8)
   ax_s.grid(axis="y", alpha=0.3)

   # ── Bottom: linepack state ────────────────────────────────────────────
   ax_lp = axes[2]
   delta = [v - lp0 for v in lp_kg.values]
   ax_lp.step(steps, delta, where="post", lw=2, color=C_PACK)
   ax_lp.fill_between(steps, 0, delta, step="post",
                       where=[v < 0 for v in delta],
                       color=C_NOLP, alpha=0.20, label="discharging")
   ax_lp.fill_between(steps, 0, delta, step="post",
                       where=[v >= 0 for v in delta],
                       color=C_PACK, alpha=0.20, label="charging")
   ax_lp.axhline(0, color="grey", linestyle="--", alpha=0.5,
                  label=f"initial  ({lp0:,.0f} kg)")
   ax_lp.set_ylabel("Δ stored mass  [kg]")
   ax_lp.set_title("Linepack - stored mass deviation from initial", fontsize=10)
   ax_lp.set_xlabel("Hour")
   ax_lp.set_xticks(list(steps))
   ax_lp.legend(fontsize=8)
   ax_lp.grid(axis="y", alpha=0.3)

   fig.suptitle("Gas linepack buffers source from demand variation",
                fontsize=12, fontweight="bold")
   plt.tight_layout()

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
     - ``V_pipe × ρ(p_ref × √p_squared_pu_max, T)``

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

      The general extension mechanism - four phases, how to write your own.

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
