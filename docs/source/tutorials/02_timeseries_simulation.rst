==============================
02 · Solar feeder - day-ahead
==============================

**Scenario.** A residential bus (Bus 1) has a rooftop PV system and a household
load.  An external grid (Bus 0) acts as the slack source.  We simulate **eight
three-hour slots** across a summer day, tracking how the bus voltage and the
net grid import change as the solar output rises, peaks, and falls.

During the afternoon the PV covers the full household demand, briefly pushing
the external grid import close to zero.  A monitoring hook raises a flag
whenever the bus voltage dips below a threshold.

Key features covered
--------------------

- Registering **time-varying profiles** for load and generation.
- Querying multi-step results with :meth:`~monee.simulation.TimeseriesResult.get_result_for`.
- Writing a :class:`~monee.simulation.StepHook` with the correct callback signatures.

----

Building the base network
==========================

The same network object is reused at every step - ``run_timeseries`` copies it
internally and never modifies the original.

.. testcode::

    import monee.express as mx
    import monee.model as mm
    from monee.simulation import TimeseriesData, run_timeseries

    net = mx.create_multi_energy_network()

    bus_grid = mx.create_bus(net)   # slack bus (external grid)
    bus_home = mx.create_bus(net)   # residential bus

    mx.create_line(net, bus_grid, bus_home, 500,
                   r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    mx.create_ext_power_grid(net, bus_grid)

    # Household load - initial value overwritten each step by the time series
    load_id = mx.create_power_load(net, bus_home, p_mw=0.30, q_mvar=0.0)

    # PV modelled as a load with negative p_mw (injection convention):
    # p_mw < 0 → power injected into the bus (generation)
    pv_id = mx.create_power_load(net, bus_home, p_mw=0.0, q_mvar=0.0)

----

Defining the time series
=========================

Eight steps represent three-hour slots from 00:00 to 21:00 on a summer day.

.. testcode::

    # Household demand (positive = consumption) - low overnight, peaks in evening
    load_profile = [0.10, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.25]  # MW

    # PV output (negative = generation) - zero at night, peak midday
    pv_profile   = [0.00, 0.00,-0.10,-0.30,-0.45,-0.30,-0.10, 0.00]  # MW

    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", load_profile)
    td.add_child_series(pv_id,   "p_mw", pv_profile)

----

Running the simulation
======================

.. testcode::

    ts_result = run_timeseries(net, td)
    print(f"Completed {len(ts_result.step_results)} steps")

.. testoutput::

    Completed 8 steps

----

Inspecting results
==================

:meth:`~monee.simulation.TimeseriesResult.get_result_for` returns a
:class:`pandas.DataFrame` with one row per successful step and one column per
component of the requested type.

**Bus voltage over the day:**

.. testcode::

    vm_df = ts_result.get_result_for(mm.Bus, "vm_pu")
    print(f"Voltage columns (one per bus): {vm_df.shape[1]}")
    print(f"Step rows:                     {vm_df.shape[0]}")

.. testoutput::

    Voltage columns (one per bus): 2
    Step rows:                     8

**Net import from the external grid:**

During the early-afternoon steps the PV covers or exceeds the household demand,
so the external grid import approaches zero or turns slightly negative (export).

.. testcode::

    ext_df = ts_result.get_result_for(mm.ExtPowerGrid, "p_mw")
    # Column 0 is the single ExtPowerGrid instance
    print(ext_df.iloc[:, 0].round(3).to_string())

.. testoutput::
   :options: +SKIP

    0    0.100
    1    0.100
    2    0.050
    3   -0.100
    4   -0.200
    5    0.050
    6    0.300
    7    0.250

Negative values indicate that excess PV is exported back to the grid.

.. plot::
   :caption: Solar feeder - bus voltage and grid import vary as rooftop PV ramps up and down across the day

   import monee.express as mx
   import monee.model as mm
   from monee.simulation import TimeseriesData, run_timeseries
   import matplotlib.pyplot as plt

   load_profile = [0.10, 0.10, 0.15, 0.20, 0.25, 0.35, 0.40, 0.25]
   pv_profile   = [0.00, 0.00,-0.10,-0.30,-0.45,-0.30,-0.10, 0.00]
   net_demand   = [l + p for l, p in zip(load_profile, pv_profile)]
   hours = [0, 3, 6, 9, 12, 15, 18, 21]

   net = mx.create_multi_energy_network()
   bus_grid = mx.create_bus(net)
   bus_home = mx.create_bus(net)
   mx.create_line(net, bus_grid, bus_home, 500,
                  r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
   mx.create_ext_power_grid(net, bus_grid)
   load_id = mx.create_power_load(net, bus_home, p_mw=0.30, q_mvar=0.0)
   pv_id   = mx.create_power_load(net, bus_home, p_mw=0.0,  q_mvar=0.0)

   td = TimeseriesData()
   td.add_child_series(load_id, "p_mw", load_profile)
   td.add_child_series(pv_id,   "p_mw", pv_profile)
   ts_result = run_timeseries(net, td)

   vm_df  = ts_result.get_result_for(mm.Bus, "vm_pu")
   ext_df = ts_result.get_result_for(mm.ExtPowerGrid, "p_mw")
   vm_home  = vm_df.iloc[:, 1].values   # bus_home column
   grid_imp = ext_df.iloc[:, 0].values

   fig, axes = plt.subplots(3, 1, sharex=True, figsize=(9, 7),
                             gridspec_kw={"hspace": 0.45})

   C_LOAD = "#f4a261"
   C_PV   = "#ffe066"
   C_VM   = "#2c7bb6"
   C_IMP  = "#d7191c"
   C_EXP  = "#1a9641"

   axes[0].step(hours, load_profile, where="post", lw=2, color=C_LOAD, label="Load")
   axes[0].step(hours, [-p for p in pv_profile], where="post", lw=2,
                color=C_PV, label="PV output")
   axes[0].fill_between(hours, 0, load_profile, step="post", color=C_LOAD, alpha=0.15)
   axes[0].fill_between(hours, 0, [-p for p in pv_profile], step="post",
                         color=C_PV, alpha=0.15)
   axes[0].set_ylabel("Power  [MW]")
   axes[0].set_title("Load and PV generation", fontsize=10)
   axes[0].legend(fontsize=8)
   axes[0].grid(axis="y", alpha=0.3)

   imp_colors = [C_IMP if v >= 0 else C_EXP for v in grid_imp]
   axes[1].bar(hours, grid_imp, width=2.0, color=imp_colors, alpha=0.8,
               align="edge")
   axes[1].axhline(0, color="grey", lw=0.8)
   axes[1].set_ylabel("Grid flow  [MW]\n+ import  /  - export")
   axes[1].set_title("External grid import / export", fontsize=10)
   axes[1].grid(axis="y", alpha=0.3)

   axes[2].plot(hours, vm_home, marker="o", lw=2, color=C_VM)
   axes[2].axhline(0.97, color="grey", ls="--", alpha=0.6,
                   label="undervoltage limit (0.97 pu)")
   axes[2].set_ylabel("Voltage  [pu]")
   axes[2].set_title("Residential bus voltage", fontsize=10)
   axes[2].set_xlabel("Hour of day")
   axes[2].set_xticks(hours)
   axes[2].set_xticklabels([f"{h:02d}:00" for h in hours])
   axes[2].legend(fontsize=8)
   axes[2].grid(axis="y", alpha=0.3)

   fig.suptitle("Solar feeder - day-ahead simulation", fontsize=12, fontweight="bold")
   plt.tight_layout()

.. tip::

   Pass a :class:`pandas.DatetimeIndex` to ``run_timeseries`` via the
   ``datetime_index`` argument to get human-readable timestamps as the row
   index instead of step numbers.

----

Monitoring with a step hook
============================

A :class:`~monee.simulation.StepHook` lets you inject logic before or after
each step.  Here a hook logs a warning whenever the voltage at the home bus
falls below 0.97 pu - a simple under-voltage alert.

.. testcode::

    from monee.simulation import StepHook

    VOLTAGE_THRESHOLD = 0.97  # pu

    class VoltageMonitor(StepHook):
        """Warn when the residential bus voltage dips below the threshold."""

        def post_run(self, net, step, step_state, step_result):
            if step_result.failed:
                return
            bus_df = step_result.result.get(mm.Bus)
            min_vm = bus_df["vm_pu"].min()
            if min_vm < VOLTAGE_THRESHOLD:
                print(f"  Step {step}: voltage dip - {min_vm:.4f} pu")

    ts_result2 = run_timeseries(net, td, step_hooks=[VoltageMonitor()])

.. testoutput::
   :options: +SKIP

      Step 6: voltage dip - 0.9687 pu

.. note::

   ``post_run`` receives the **solved** ``net`` (with variable values), the
   step index, the inter-step ``step_state``, and the
   :class:`~monee.simulation.StepResult` for this step.  Check
   ``step_result.failed`` before reading results if ``on_step_error='skip'``
   is set on the run.

----

Next steps
==========

- Combine a time-varying load with an optimisation problem by passing
  ``optimization_problem=...`` to :func:`~monee.simulation.run_timeseries`.
- Add ramp-rate constraints between steps using ``tracked`` variables - see the
  :doc:`../how-to/timeseries` how-to guide.
- Register multi-energy profiles (gas, heat) the same way and query results for
  :class:`~monee.model.Junction` or other component types.
