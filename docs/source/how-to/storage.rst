================
Storage dispatch
================

Attach a battery, gas storage, or thermal storage to a network, then drive it
either with a prescribed schedule or by letting the optimizer choose the
dispatch.

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

All three follow the load convention: positive dispatch = charging
(consuming from the network), negative dispatch = discharging (injecting
into the network).

By default the dispatch attribute (``p_mw`` or ``mass_flow_kgs``) is a plain
Python float fixed at zero, so the model sits idle in an energy-flow solve.
Activate it in one of two ways:

* Prescribed dispatch: register the dispatch series via
  ``TimeseriesData`` and call :func:`~monee.simulation.run_timeseries`.
* Optimised dispatch: call
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

   # Build a simple two-bus power network
   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1,
                  length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net, bus1, p_mw=0.8, q_mvar=0.0)

   # Attach a 10-MWh / 2-MW battery at bus1
   storage = mm.ElectricStorage(
       e_mwh_initial=5.0,     # start at 50 % SoC
       e_mwh_max=10.0,        # usable capacity
       p_max_mw=2.0,          # charge/discharge limit
   )
   bat_id = mx.create_el_child(net, storage, node_id=bus1, name="battery")

   # Schedule charge (+) / discharge (-)
   td = TimeseriesData()
   td.add_child_series(bat_id, "p_mw", [1.0, 0.5, -1.0, -1.5, 0.0, 0.5])

   result = run_timeseries(net, td)
   soc = result.get_result_for_id(bat_id, "e_mwh")
   print("SoC [MWh]:", soc.round(2).tolist())

.. testoutput::

   SoC [MWh]: [6.0, 6.5, 5.5, 4.0, 4.0, 4.5]

Battery SoC and dispatch (prescribed schedule).

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/storage_prescribed.html" width="100%" height="500" style="border:none;" loading="lazy" title="Electric storage prescribed dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/storage_prescribed.png
      :width: 100%

Optimised dispatch
------------------

Pass ``OptimizationProblem.controllable_storages()`` to
:func:`~monee.simulation.run_multi_period` and the solver chooses
charge or discharge at every period:

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

Optimised dispatch against a price signal: the battery charges in cheap hours and discharges into the expensive peak.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/storage_dispatch.html" width="100%" height="660" style="border:none;" loading="lazy" title="Optimised battery dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/storage_dispatch.png
      :width: 100%

Round-trip efficiency
---------------------

Pass ``efficiency_charge`` and ``efficiency_discharge`` (both 0 to 1) to model
round-trip losses. In optimised dispatch the model splits dispatch into
separate charge and discharge variables, so each efficiency is applied in the
correct direction:

.. code-block:: python

   lossy_bat = mm.ElectricStorage(
       e_mwh_initial=5.0,
       e_mwh_max=10.0,
       p_max_mw=2.0,
       efficiency_charge=0.95,      # 5 % loss on the way in
       efficiency_discharge=0.95,   # 5 % loss on the way out
   )

With a prescribed ``p_mw``, the efficiency is chosen from the sign of the fixed
dispatch value and no extra variables are created.

----

Gas storage
===========

``GasStorage`` attaches to a gas junction. Its state variable is
``m_stored_kg``, the stored gas mass in kg.

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
       m_stored_kg_initial=2000.0,   # start with 2 tonnes of gas
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

   Stored [kg]: [1640.0, 1280.0, 920.0, 560.0]

.. note::

   The SoC update is ``m_stored_kg(t) = m_stored_kg(t-1) + dt_s * mass_flow_kgs(t)``
   where ``dt_s = dt_h * 3600``.  At 1 h per step:
   1000 - 0.1 × 3600 = 640 kg after step 1.

Gas storage: charge and discharge cycle over 8 hours.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/storage_gas.html" width="100%" height="500" style="border:none;" loading="lazy" title="Gas storage charge/discharge cycle"></iframe>

.. only:: latex

   .. image:: /_static/interactive/storage_gas.png
      :width: 100%

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

``ThermalStorage`` attaches to a water junction. An optional
``loss_factor_per_h`` models standing heat losses, for example imperfect tank
insulation:

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
     - Battery attached to a power bus. State: ``e_mwh``.
   * - ``mm.GasStorage(m_stored_kg_initial, m_stored_kg_max, flow_max_kgs, ...)``
     - Gas tank or cavern attached to a gas junction. State: ``m_stored_kg``.
   * - ``mm.ThermalStorage(m_stored_kg_initial, m_stored_kg_max, flow_max_kgs, ...)``
     - Thermal tank attached to a water junction. State: ``m_stored_kg``.
   * - ``ElectricStorage.make_controllable()``
     - Convert ``p_mw`` into a ``Var`` for optimisation. Called automatically
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

   * :doc:`timeseries`: timeseries simulation workflow
   * :doc:`multi_period`: multi-period optimisation with storage dispatch
   * :doc:`../concepts/temporal_extensions`: GasLinepack and LTC extensions
