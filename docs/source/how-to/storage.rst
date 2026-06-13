================
Storage dispatch
================

Attach battery, gas-storage, or thermal-storage components to a network and
drive them either by an externally-prescribed schedule or by letting the
optimizer choose the dispatch.

For the physics background see :doc:`../concepts/timeseries` and
:doc:`../concepts/multi_period`.

----

Storage models at a glance
===========================

.. list-table::
   :header-rows: 1
   :widths: 25 20 20 35

   * - Class
     - Network
     - State variable
     - Key constructor args
   * - ``ElectricStorage``
     - Power (Bus)
     - ``e_mwh``
     - ``e_mwh_initial``, ``e_mwh_max``, ``p_max_mw``
   * - ``GasStorage``
     - Gas (Junction)
     - ``m_stored_kg``
     - ``m_stored_kg_initial``, ``m_stored_kg_max``, ``flow_max_kgs``
   * - ``ThermalStorage``
     - Water/heat (Junction)
     - ``m_stored_kg``
     - ``m_stored_kg_initial``, ``m_stored_kg_max``, ``flow_max_kgs``

All three follow the **load convention**: positive dispatch = charging
(consuming from the network), negative dispatch = discharging (injecting
into the network).

By default the dispatch attribute (``p_mw`` or ``mass_flow_kgs``) is a plain
Python float - fixed at zero - so the model acts as an idle element in a
plain energy-flow solve.  Activate it in one of two ways:

* **Prescribed dispatch** - register the dispatch series via
  ``TimeseriesData`` and call :func:`~monee.simulation.run_timeseries`.
* **Optimised dispatch** - call
  :meth:`~monee.problem.core.OptimizationProblem.controllable_storages`
  and pass the problem to :func:`~monee.simulation.run_multi_period`.

----

Electric storage
================

Prescribed dispatch in timeseries
----------------------------------

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData, run_timeseries

   # ── Build a simple two-bus power network ─────────────────────────────────
   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1,
                  length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net, bus1, p_mw=0.8, q_mvar=0.0)

   # ── Attach a 10-MWh / 2-MW battery at bus1 ───────────────────────────────
   storage = mm.ElectricStorage(
       e_mwh_initial=5.0,     # start at 50 % SoC
       e_mwh_max=10.0,        # usable capacity
       p_max_mw=2.0,          # charge/discharge limit
   )
   bat_id = mx.create_el_child(net, storage, node_id=bus1, name="battery")

   # ── Schedule charge (+) / discharge (-) ──────────────────────────────────
   td = TimeseriesData()
   td.add_child_series(bat_id, "p_mw", [1.0, 0.5, -1.0, -1.5, 0.0, 0.5])

   result = run_timeseries(net, td)
   soc = result.get_result_for_id(bat_id, "e_mwh")
   print("SoC [MWh]:", soc.round(2).tolist())

.. testoutput::

   SoC [MWh]: [6.0, 6.5, 5.5, 4.0, 4.0, 4.5]

.. plot::
   :caption: Battery SoC and dispatch - prescribed schedule

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData, run_timeseries
   import matplotlib.pyplot as plt

   DISPATCH = [1.0, 0.5, -1.0, -1.5, 0.0, 0.5]

   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1, length_m=500,
                  r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net, bus1, p_mw=0.8, q_mvar=0.0)
   storage = mm.ElectricStorage(e_mwh_initial=5.0, e_mwh_max=10.0, p_max_mw=2.0)
   bat_id = mx.create_el_child(net, storage, node_id=bus1, name="battery")

   td = TimeseriesData()
   td.add_child_series(bat_id, "p_mw", DISPATCH)
   result = run_timeseries(net, td)
   soc = result.get_result_for_id(bat_id, "e_mwh")

   steps = range(len(DISPATCH))
   fig, (ax_d, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                     gridspec_kw={"hspace": 0.35})

   C_CHG = "#2c7bb6"
   C_DIS = "#d7191c"

   colors = [C_CHG if v >= 0 else C_DIS for v in DISPATCH]
   ax_d.bar(steps, DISPATCH, color=colors, alpha=0.8, width=0.6)
   ax_d.axhline(0, color="grey", lw=0.8)
   ax_d.set_ylabel("Dispatch  [MW]\n+ charge  /  - discharge")
   ax_d.set_title("Prescribed battery dispatch", fontsize=10)
   ax_d.grid(axis="y", alpha=0.3)

   ax_s.plot(steps, soc.values, marker="o", lw=2, color="#1a9641")
   ax_s.fill_between(steps, 0, soc.values, alpha=0.12, color="#1a9641")
   ax_s.axhline(10.0, color="grey", ls="--", alpha=0.5, label="capacity (10 MWh)")
   ax_s.set_ylabel("State of charge  [MWh]")
   ax_s.set_xlabel("Hour")
   ax_s.set_ylim(0, 11)
   ax_s.set_xticks(list(steps))
   ax_s.legend(fontsize=8)
   ax_s.grid(axis="y", alpha=0.3)

   fig.suptitle("Electric storage - prescribed dispatch", fontsize=12, fontweight="bold")
   plt.tight_layout()

Optimised dispatch
------------------

Pass ``OptimizationProblem.controllable_storages()`` to
:func:`~monee.simulation.run_multi_period` and the solver freely chooses
charge/discharge at every period:

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.problem.core import OptimizationProblem
   from monee.simulation import TimeseriesData

   net2 = mx.create_multi_energy_network()
   b0 = mx.create_bus(net2)
   b1 = mx.create_bus(net2)
   mx.create_ext_power_grid(net2, b0)
   mx.create_line(net2, b0, b1,
                  length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net2, b1, p_mw=0.0, q_mvar=0.0, name="load")

   bat2 = mm.ElectricStorage(
       e_mwh_initial=2.0,
       e_mwh_max=4.0,
       p_max_mw=1.0,
   )
   bat2_id = mx.create_el_child(net2, bat2, node_id=b1, name="battery2")

   td2 = TimeseriesData()
   td2.add_child_series_by_name("load", "p_mw", [0.4, 0.5, 1.4, 1.8, 1.5, 0.4])

   prob = OptimizationProblem()
   prob.controllable_storages()

.. tip::

   Use the ``terminal_state`` argument to anchor the final state of charge
   and prevent the optimizer from draining the battery at the end of the
   horizon:

   .. code-block:: python

      from monee.simulation import run_multi_period

      result = run_multi_period(
          net2, td2,
          optimization_problem=prob,
          dt_h=1.0,
          terminal_state={(bat2_id, "e_mwh"): 2.0},
      )

.. plot::
   :caption: Optimised battery dispatch - the solver charges during cheap off-peak hours and discharges during the peak

   import monee.model as mm
   import monee.express as mx
   from monee.problem.core import OptimizationProblem
   from monee.simulation import TimeseriesData, run_multi_period
   import matplotlib.pyplot as plt

   LOAD = [0.4, 0.5, 1.4, 1.8, 1.5, 0.4]

   net2 = mx.create_multi_energy_network()
   b0 = mx.create_bus(net2)
   b1 = mx.create_bus(net2)
   mx.create_ext_power_grid(net2, b0)
   mx.create_line(net2, b0, b1, length_m=500,
                  r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net2, b1, p_mw=0.0, q_mvar=0.0, name="load")
   bat2 = mm.ElectricStorage(e_mwh_initial=2.0, e_mwh_max=4.0, p_max_mw=1.0)
   bat2_id = mx.create_el_child(net2, bat2, node_id=b1, name="battery2")

   td2 = TimeseriesData()
   td2.add_child_series_by_name("load", "p_mw", LOAD)

   prob = OptimizationProblem()
   prob.controllable_storages()
   result = run_multi_period(net2, td2, optimization_problem=prob, dt_h=1.0,
                             terminal_state={(bat2_id, "e_mwh"): 2.0})

   soc  = result.get_result_for_id(bat2_id, "e_mwh")
   disp = result.get_result_for_id(bat2_id, "p_mw")
   steps = range(len(LOAD))

   fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 6),
                             gridspec_kw={"hspace": 0.4})

   C_LOAD = "#f4a261"
   C_CHG  = "#2c7bb6"
   C_DIS  = "#d7191c"
   C_SOC  = "#1a9641"

   axes[0].step(steps, LOAD, where="post", lw=2, color=C_LOAD)
   axes[0].fill_between(steps, 0, LOAD, step="post", color=C_LOAD, alpha=0.15)
   axes[0].set_ylabel("Load  [MW]")
   axes[0].set_title("Consumer demand", fontsize=10)
   axes[0].grid(axis="y", alpha=0.3)

   disp_vals = list(disp.values)
   bar_colors = [C_CHG if v >= 0 else C_DIS for v in disp_vals]
   axes[1].bar(steps, disp_vals, color=bar_colors, alpha=0.8, width=0.6)
   axes[1].axhline(0, color="grey", lw=0.8)
   axes[1].set_ylabel("Battery  [MW]\n+ charge  /  - discharge")
   axes[1].set_title("Optimised dispatch", fontsize=10)
   axes[1].grid(axis="y", alpha=0.3)

   axes[2].plot(steps, soc.values, marker="o", lw=2, color=C_SOC)
   axes[2].fill_between(steps, 0, soc.values, alpha=0.12, color=C_SOC)
   axes[2].axhline(4.0, color="grey", ls="--", alpha=0.5, label="capacity (4 MWh)")
   axes[2].set_ylabel("SoC  [MWh]")
   axes[2].set_xlabel("Hour")
   axes[2].set_ylim(0, 4.5)
   axes[2].set_xticks(list(steps))
   axes[2].legend(fontsize=8)
   axes[2].grid(axis="y", alpha=0.3)

   fig.suptitle("Electric storage - optimised dispatch", fontsize=12, fontweight="bold")
   plt.tight_layout()

Round-trip efficiency
---------------------

Pass ``efficiency_charge`` and ``efficiency_discharge`` (both 0–1) to model
realistic round-trip losses.  The model introduces separate charge and
discharge variables so the efficiency is applied in the correct direction:

.. code-block:: python

   lossy_bat = mm.ElectricStorage(
       e_mwh_initial=5.0,
       e_mwh_max=10.0,
       p_max_mw=2.0,
       efficiency_charge=0.95,      # 5 % loss on the way in
       efficiency_discharge=0.95,   # 5 % loss on the way out
   )

When ``p_mw`` is prescribed (plain energy flow), the efficiency is applied
based on the sign of the fixed dispatch value - no extra variables are created.

----

Gas storage
===========

``GasStorage`` attaches to a gas junction.  The state variable is
``m_stored_kg`` (stored gas mass in kg).

Prescribed discharge
--------------------

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData, run_timeseries

   net_g = mx.create_multi_energy_network()
   j0 = mx.create_gas_junction(net_g)
   j1 = mx.create_gas_junction(net_g)
   mx.create_gas_ext_grid(net_g, j0)
   mx.create_gas_pipe(net_g, j0, j1, diameter_m=0.3, length_m=5000)
   mx.create_gas_sink(net_g, j1, mass_flow_kgs=0.05)

   tank = mm.GasStorage(
       m_stored_kg_initial=1000.0,   # start with 1 tonne of gas
       m_stored_kg_max=5000.0,       # capacity 5 tonnes
       flow_max_kgs=0.2,             # max charge/discharge rate
   )
   tank_id = mx.create_gas_child(net_g, tank, node_id=j1, name="tank")

   td_g = TimeseriesData()
   # Discharge 0.1 kg/s at each step (negative = inject into network)
   td_g.add_child_series(tank_id, "mass_flow_kgs", [-0.1, -0.1, -0.1, -0.1])

   result_g = run_timeseries(net_g, td_g)
   stored = result_g.get_result_for_id(tank_id, "m_stored_kg")
   print("Stored [kg]:", stored.round(1).tolist())

.. testoutput::

   Stored [kg]: [640.0, 280.0, -80.0, -440.0]

.. note::

   The SoC update is ``m_stored_kg(t) = m_stored_kg(t-1) + dt_s * mass_flow_kgs(t)``
   where ``dt_s = dt_h * 3600``.  At 1 h per step:
   1000 - 0.1 × 3600 = 640 kg after step 1.

.. plot::
   :caption: Gas storage - charge and discharge cycle over 8 hours

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData, run_timeseries
   import matplotlib.pyplot as plt

   # Realistic 8-step cycle: charge 2 h, hold 2 h, discharge 2 h, hold 2 h
   DISPATCH = [0.05, 0.05, 0.0, 0.0, -0.05, -0.05, 0.0, 0.0]  # kg/s

   net_g = mx.create_multi_energy_network()
   j0 = mx.create_gas_junction(net_g)
   j1 = mx.create_gas_junction(net_g)
   mx.create_gas_ext_grid(net_g, j0)
   mx.create_gas_pipe(net_g, j0, j1, diameter_m=0.3, length_m=5000)
   mx.create_gas_sink(net_g, j1, mass_flow_kgs=0.05)
   tank = mm.GasStorage(m_stored_kg_initial=1000.0, m_stored_kg_max=5000.0,
                        flow_max_kgs=0.2)
   tank_id = mx.create_gas_child(net_g, tank, node_id=j1, name="tank")

   td_g = TimeseriesData()
   td_g.add_child_series(tank_id, "mass_flow_kgs", DISPATCH)
   result_g = run_timeseries(net_g, td_g)
   stored = result_g.get_result_for_id(tank_id, "m_stored_kg")

   steps = range(len(DISPATCH))
   fig, (ax_d, ax_s) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                     gridspec_kw={"hspace": 0.35})

   C_CHG = "#2c7bb6"
   C_DIS = "#d7191c"
   colors = [C_CHG if v > 0 else (C_DIS if v < 0 else "lightgrey") for v in DISPATCH]
   ax_d.bar(steps, DISPATCH, color=colors, alpha=0.8, width=0.6)
   ax_d.axhline(0, color="grey", lw=0.8)
   ax_d.set_ylabel("Net flow  [kg/s]\n+ charge  /  - discharge")
   ax_d.set_title("Prescribed gas storage dispatch", fontsize=10)
   ax_d.grid(axis="y", alpha=0.3)

   ax_s.plot(steps, stored.values, marker="o", lw=2, color="#1a9641")
   ax_s.fill_between(steps, 0, stored.values, alpha=0.12, color="#1a9641")
   ax_s.axhline(1000.0, color="grey", ls="--", alpha=0.5, label="initial (1 000 kg)")
   ax_s.axhline(5000.0, color="#d7191c", ls=":", alpha=0.4, label="capacity (5 000 kg)")
   ax_s.set_ylabel("Stored mass  [kg]")
   ax_s.set_xlabel("Hour")
   ax_s.set_xticks(list(steps))
   ax_s.legend(fontsize=8)
   ax_s.grid(axis="y", alpha=0.3)

   fig.suptitle("Gas storage - charge / discharge cycle", fontsize=12, fontweight="bold")
   plt.tight_layout()

Optimised gas dispatch
----------------------

.. code-block:: python

   from monee.problem.core import OptimizationProblem
   from monee.simulation import run_multi_period

   prob_g = OptimizationProblem()
   prob_g.controllable_storages()

   result_g_opt = run_multi_period(net_g, td_g,
                                   optimization_problem=prob_g, dt_h=1.0)

----

Thermal storage
===============

``ThermalStorage`` attaches to a water/heat junction.  An optional
``loss_factor_per_h`` models standing heat losses (e.g. tank insulation):

.. testcode::

   import monee.model as mm
   import monee.express as mx

   net_th = mx.create_multi_energy_network()
   jw0 = mx.create_water_junction(net_th)
   jw1 = mx.create_water_junction(net_th)
   mx.create_ext_hydr_grid(net_th, jw0)
   mx.create_water_pipe(net_th, jw0, jw1, diameter_m=0.3, length_m=200)

   tank_th = mm.ThermalStorage(
       m_stored_kg_initial=2000.0,    # 2 tonnes of hot water
       m_stored_kg_max=10_000.0,
       flow_max_kgs=1.0,
       loss_factor_per_h=0.005,       # 0.5 % standing loss per hour
   )
   th_id = mx.create_water_child(net_th, tank_th, node_id=jw1, name="hot_tank")

The SoC update with standing losses is:

.. math::

   m(t) \;=\; m(t-1) \;\times\; (1 - \lambda \cdot \Delta t_h)
             \;+\; \Delta t_s \cdot \dot{m}(t)

where :math:`\lambda` is ``loss_factor_per_h``.

To optimise thermal dispatch, call ``controllable_storages()`` as for
electric or gas storage.

----

API reference
=============

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Description
   * - ``mm.ElectricStorage(e_mwh_initial, e_mwh_max, p_max_mw, ...)``
     - Battery attached to a power bus.  State: ``e_mwh``.
   * - ``mm.GasStorage(m_stored_kg_initial, m_stored_kg_max, flow_max_kgs, ...)``
     - Gas tank/cavern attached to a gas junction.  State: ``m_stored_kg``.
   * - ``mm.ThermalStorage(m_stored_kg_initial, m_stored_kg_max, flow_max_kgs, ...)``
     - Thermal tank attached to a water junction.  State: ``m_stored_kg``.
   * - ``ElectricStorage.make_controllable()``
     - Convert ``p_mw`` into a ``Var`` for optimisation.  Called automatically
       by ``OptimizationProblem.controllable_storages()``.
   * - ``GasStorage.make_controllable()``
     - Convert ``mass_flow_kgs`` into a ``Var`` for optimisation.
   * - ``ThermalStorage.make_controllable()``
     - Convert ``mass_flow_kgs`` into a ``Var`` for optimisation.
   * - ``mx.create_el_child(net, model, node_id, name=...)``
     - Attach any electric child model (incl. ``ElectricStorage``) to a bus.
   * - ``mx.create_gas_child(net, model, node_id, name=...)``
     - Attach any gas child model (incl. ``GasStorage``) to a junction.
   * - ``mx.create_water_child(net, model, node_id, name=...)``
     - Attach any water child model (incl. ``ThermalStorage``) to a junction.
   * - ``OptimizationProblem.controllable_storages()``
     - Make all storage components controllable before a multi-period solve.
   * - ``result.get_result_for_id(storage_id, "e_mwh")``
     - Time series of state of charge for ``ElectricStorage``.
   * - ``result.get_result_for_id(storage_id, "m_stored_kg")``
     - Time series of stored mass for ``GasStorage`` / ``ThermalStorage``.

.. seealso::

   * :doc:`timeseries` - timeseries simulation workflow
   * :doc:`multi_period` - multi-period optimisation with storage dispatch
   * :doc:`../concepts/temporal_extensions` - GasLinepack and LTC extensions
