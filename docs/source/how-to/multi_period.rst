==========================
Multi-period optimization
==========================

Task-focused examples for multi-period solves, storage dispatch, and
rolling-horizon MPC.  For the architectural background see
:doc:`../concepts/multi_period`.

----

Battery dispatch — 6-hour horizon
==================================

.. testcode::

   import monee.model as mm
   import monee.express as mx
   from monee.simulation import TimeseriesData

   # ── Network ───────────────────────────────────────────────────────────
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

   # ── Load profile ──────────────────────────────────────────────────────
   td = TimeseriesData()
   td.add_child_series_by_name("load", "p_mw",
                                [0.4, 0.5, 1.4, 1.8, 1.5, 0.4])

Solve and inspect:

.. code-block:: python

   from monee.simulation import run_multi_period
   from monee.problem.core import OptimizationProblem

   prob = OptimizationProblem()
   prob.controllable_storages()

   result = run_multi_period(
       net, td,
       optimization_problem=prob,
       dt_h=1.0,
       terminal_state={(bat.id, "e_mwh"): 2.0},
   )

   soc  = result.get_result_for_id(bat.id, "e_mwh")
   disp = result.get_result_for_id(bat.id, "p_mw")
   print("SoC  [MWh]:", soc.round(2).tolist())
   print("Disp [MW]:", disp.round(2).tolist())

.. plot::
   :caption: Battery optimal dispatch — the solver shifts charge to off-peak hours to serve the midday peak

   import monee.model as mm
   import monee.express as mx
   from monee.problem.core import OptimizationProblem
   from monee.simulation import TimeseriesData, run_multi_period
   import matplotlib.pyplot as plt

   LOAD = [0.4, 0.5, 1.4, 1.8, 1.5, 0.4]

   net = mx.create_multi_energy_network()
   bus0 = mx.create_bus(net)
   bus1 = mx.create_bus(net)
   mx.create_ext_power_grid(net, bus0)
   mx.create_line(net, bus0, bus1, length_m=500,
                  r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_power_load(net, bus1, p_mw=0.0, q_mvar=0.0, name="load")
   storage = mm.ElectricStorage(e_mwh_initial=2.0, e_mwh_max=4.0, p_max_mw=1.0)
   bat = mx.create_el_child(net, storage, node_id=bus1, name="battery")

   td = TimeseriesData()
   td.add_child_series_by_name("load", "p_mw", LOAD)

   prob = OptimizationProblem()
   prob.controllable_storages()
   result = run_multi_period(net, td, optimization_problem=prob, dt_h=1.0,
                             terminal_state={(bat, "e_mwh"): 2.0})

   soc  = result.get_result_for_id(bat, "e_mwh")
   disp = result.get_result_for_id(bat, "p_mw")
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
   axes[1].set_ylabel("Battery  [MW]\n+ charge  /  − discharge")
   axes[1].set_title("Optimised dispatch", fontsize=10)
   axes[1].grid(axis="y", alpha=0.3)

   axes[2].plot(steps, soc.values, marker="o", lw=2, color=C_SOC)
   axes[2].fill_between(steps, 0, soc.values, alpha=0.12, color=C_SOC)
   axes[2].axhline(4.0, color="grey", ls="--", alpha=0.4, label="capacity (4 MWh)")
   axes[2].set_ylabel("SoC  [MWh]")
   axes[2].set_xlabel("Hour")
   axes[2].set_ylim(0, 4.5)
   axes[2].set_xticks(list(steps))
   axes[2].legend(fontsize=8)
   axes[2].grid(axis="y", alpha=0.3)

   fig.suptitle("Battery — optimal dispatch over 6-hour horizon",
                fontsize=12, fontweight="bold")
   plt.tight_layout()

----

Cyclical operation
==================

Force the battery to return to its starting state (day-ahead planning):

.. code-block:: python

   result = run_multi_period(
       net, td,
       optimization_problem=prob,
       dt_h=1.0,
       initial_state ={(bat.id, "e_mwh"): 2.0},
       terminal_state={(bat.id, "e_mwh"): 2.0},
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
   mx.create_water_sink(net_mes, j_return, mass_flow=0.0, name="heat_load")

   mx.create_chp(net_mes,
                 power_node_id=bus_load,
                 gas_node_id=j_gas,
                 heat_node_id=j_supply,
                 heat_return_node_id=j_return,
                 diameter_m=0.1,
                 efficiency_power=0.35,
                 efficiency_heat=0.45,
                 mass_flow_setpoint=0.1)

   el_prof   = [0.8, 1.0, 1.4, 1.8, 1.6, 1.2]
   heat_prof = [0.4, 0.5, 0.7, 0.9, 0.8, 0.6]

   td_mes = TimeseriesData()
   td_mes.add_child_series_by_name("el_load",   "p_mw",     el_prof)
   td_mes.add_child_series_by_name("heat_load", "mass_flow", heat_prof)

Solve and query CHP dispatch:

.. code-block:: python

   from monee.simulation import run_multi_period

   result = run_multi_period(net_mes, td_mes, dt_h=1.0)
   chp_p = result.get_result_for(mm.CHP, "regulation")
   print(chp_p)

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
   mx.create_gas_sink(net_lp, j2, mass_flow=0.3, name="consumer")

   pipe_id = mx.create_gas_pipe(net_lp, j0, j1,
                                diameter_m=0.5, length_m=50_000)
   mx.create_gas_pipe(net_lp, j1, j2,
                      diameter_m=0.3, length_m=10_000)

   net_lp.add_extension(GasLinepack(overrides={
       pipe_id: dict(linepack_kg_initial=5_000, linepack_kg_max=15_000)
   }))

   td_lp = TimeseriesData()
   td_lp.add_child_series_by_name("consumer", "mass_flow",
                                   [0.3, 0.3, 0.6, 0.9, 0.8, 0.4])

.. plot::
   :caption: Gas linepack buffers the demand peak — stored mass rises at low demand, drains during the peak

   import monee.model as mm
   import monee.express as mx
   from monee.model import GasLinepack
   from monee.simulation import TimeseriesData, run_multi_period
   import matplotlib.pyplot as plt

   DEMAND = [0.3, 0.3, 0.6, 0.9, 0.8, 0.4]

   net_lp = mx.create_multi_energy_network()
   j0 = mx.create_gas_junction(net_lp)
   j1 = mx.create_gas_junction(net_lp)
   j2 = mx.create_gas_junction(net_lp)
   mx.create_gas_ext_grid(net_lp, j0)
   mx.create_gas_sink(net_lp, j2, mass_flow=0.3, name="consumer")
   pipe_id = mx.create_gas_pipe(net_lp, j0, j1,
                                diameter_m=0.5, length_m=50_000)
   mx.create_gas_pipe(net_lp, j1, j2, diameter_m=0.3, length_m=10_000)
   net_lp.add_extension(GasLinepack(overrides={
       pipe_id: dict(linepack_kg_initial=5_000, linepack_kg_max=15_000)
   }))
   td_lp = TimeseriesData()
   td_lp.add_child_series_by_name("consumer", "mass_flow", DEMAND)

   result = run_multi_period(net_lp, td_lp, dt_h=1.0)
   lp = result.get_result_for_id(pipe_id, "linepack_kg")

   steps = range(len(DEMAND))
   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 5),
                                   gridspec_kw={"hspace": 0.35})

   C_LP  = "#2c7bb6"
   C_DEM = "#f4a261"

   lp_vals = list(lp.values)
   lp0 = 5_000
   ax1.plot(steps, lp_vals, marker="o", lw=2, color=C_LP)
   ax1.fill_between(steps, lp0, lp_vals,
                    where=[v < lp0 for v in lp_vals],
                    color="#d7191c", alpha=0.20, label="discharging")
   ax1.fill_between(steps, lp0, lp_vals,
                    where=[v >= lp0 for v in lp_vals],
                    color=C_LP, alpha=0.20, label="charging")
   ax1.axhline(lp0, color="grey", ls="--", alpha=0.6, label=f"initial ({lp0:,} kg)")
   ax1.set_ylabel("Linepack  [kg]")
   ax1.set_title("Pipeline stored mass", fontsize=10)
   ax1.set_xticks(list(steps))
   ax1.legend(fontsize=8)
   ax1.grid(axis="y", alpha=0.3)

   ax2.step(steps, DEMAND, where="post", lw=2, color=C_DEM)
   ax2.fill_between(steps, 0, DEMAND, step="post", color=C_DEM, alpha=0.15)
   ax2.set_ylabel("Demand  [kg/s]")
   ax2.set_title("Consumer demand", fontsize=10)
   ax2.set_xlabel("Hour")
   ax2.set_xticks(list(steps))
   ax2.grid(axis="y", alpha=0.3)

   fig.suptitle("Gas linepack buffers demand peak", fontsize=12, fontweight="bold")
   plt.tight_layout()

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
   soc = result.get_result_for_id(bat.id, "e_mwh")

.. tip::

   Increase ``execution_steps`` to reduce computation.  ``execution_steps=1``
   (re-solve every step) gives the closest approximation to a true online MPC
   but is *T/execution_steps* times more expensive than a full-horizon solve.

----

Solver selection
================

.. tab-set::

   .. tab-item:: GEKKO (default, continuous NLP)

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
     - Override t=−1 values used by ``inter_temporal_equations``
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
     - Global objective value
   * - ``GekkoMultiPeriodSolver``
     - GEKKO / IPOPT backend (default)
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
   * - ``prob.controllable(attrs, condition=...)``
     - General-purpose: free any attribute on matching components
   * - ``prob.bounds((lo, hi), condition, attrs)``
     - Override min/max bounds of specific ``Var`` attributes
   * - ``prob.objectives``
     - Set / get the ``Objectives`` object for the solver objective function
   * - ``prob.constraints``
     - Set / get the ``Constraints`` object for additional constraints

PeriodState
-----------

``PeriodState`` is passed to ``inter_temporal_equations`` in multi-period solves.
Unlike ``StepState`` (which returns floats), ``PeriodState.get()`` returns
**live solver variables** — so equations that read from it become algebraic
cross-period constraints inside the joint solve.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Symbol
     - Description
   * - ``state.get(component_id, attr, period=-1)``
     - Solver variable at a given period.  Negative = relative to current
       period; non-negative = absolute index (``0..T-1``).
   * - ``state.has(component_id, attr)``
     - ``True`` if a non-``None`` value exists
   * - ``state.dt_h``
     - Duration of the current period in hours
   * - ``state.current_t``
     - Zero-based index of the period currently being assembled
   * - ``state.T``
     - Total number of periods in the horizon
