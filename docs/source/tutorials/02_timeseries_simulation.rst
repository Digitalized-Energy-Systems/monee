========================
Time-series simulation
========================

monee supports running a series of successive energy-flow calculations over
time by varying the inputs at each step. This is useful for studying how a
network responds to time-varying loads, generation profiles, or external
conditions.

Key components
==============

:class:`~monee.simulation.TimeseriesData`
    Holds the time series for individual components, keyed by component id
    or name and attribute name.

:func:`~monee.simulation.run_timeseries`
    Iterates over time steps, applies the series values to a copy of the
    network, solves the energy flow, and collects the results.

:class:`~monee.simulation.TimeseriesResult`
    Wraps the list of per-step results and provides a ``get_result_for``
    helper that assembles a :class:`pandas.DataFrame` for a chosen model
    type and attribute.

----

Basic example
=============

The following example builds a small electricity network, defines a load
profile with four time steps, and runs the simulation.

.. testcode::

    import monee.express as mx
    import monee.model as mm
    from monee.simulation import TimeseriesData, run_timeseries

    # ── Build the base network ──────────────────────────────────────────────
    net = mx.create_multi_energy_network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    load_id = mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    # ── Define a load profile (p_mw varies each step) ───────────────────────
    td = TimeseriesData()
    td.add_child_series(load_id, "p_mw", [0.05, 0.10, 0.15, 0.10])

    # ── Run 4 steps ─────────────────────────────────────────────────────────
    ts_result = run_timeseries(net, td, steps=4)

    print(f"Completed {len(ts_result.raw)} steps")

.. testoutput::

    Completed 4 steps

----

Inspecting results
==================

After the simulation, retrieve a :class:`pandas.DataFrame` for any model
type and attribute. Each row corresponds to one time step; each column
corresponds to one component of that type.

.. testcode::

    # One row per time step, one column per bus
    vm_df = ts_result.get_result_for(mm.Bus, "vm_pu")
    print(vm_df.shape[0])  # number of steps

.. testoutput::

    4

.. tip::

   The DataFrame returned by ``get_result_for`` can be plotted directly with
   ``vm_df.plot()`` or exported to CSV with ``vm_df.to_csv("results.csv")``.

----

Using step hooks
================

A :class:`~monee.simulation.StepHook` lets you inject custom logic before or
after each solve — for example to update external boundary conditions, log
intermediate results, or implement a rolling-horizon control strategy.

.. testcode::

    from monee.simulation import StepHook

    class LoggingHook(StepHook):
        def pre_run(self, net, step):          # called before each solve
            pass

        def post_run(self, net, base_net, step):
            print(f"Step {step} done")

    ts_result2 = run_timeseries(
        net,
        td,
        steps=2,
        step_hooks=[LoggingHook()],
    )

.. testoutput::

    Step 0 done
    Step 1 done

.. note::

   ``post_run`` receives both the *solved* ``net`` (with variable values) and
   the *base* ``base_net`` (the unmodified original). Use ``base_net`` if you
   need to read nominal setpoints before the solver overwrites them.
