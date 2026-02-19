======================
Optimization basics
======================

This tutorial shows how to formulate an optimisation problem in monee and pass
it to the solver. The example solves a simplified **load-shedding** problem:
given a network, the solver must find the minimum amount of demand that can
remain served while respecting voltage constraints.

.. tip::

   For a ready-made one-call load-shedding interface, see
   :doc:`../how-to/load_shedding`. This tutorial builds the problem from
   scratch to show the full API.

----

Building the network
====================

First, create a small electricity grid:

.. testcode::

    import monee.express as mx
    from monee import run_energy_flow_optimization

    net = mx.create_multi_energy_network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.5, q_mvar=0.0)

----

Defining the optimisation problem
==================================

An :class:`~monee.problem.core.OptimizationProblem` specifies:

- **Controllables** — which model attributes the solver may vary.
- **Bounds** — operational limits enforced as constraints.
- **Objective** — the scalar value to minimise.

.. testcode::

    import monee.problem as mp
    import monee.model as mm

    problem = mp.OptimizationProblem(debug=True)

    # ── Controllables ─────────────────────────────────────────────────────
    # Allow the solver to scale each demand between 0 % and 100 % of its
    # nominal value using a 'regulation' multiplier.
    problem.controllable_demands([
        (
            "regulation",
            mp.AttributeParameter(
                min=lambda attr, val: 0,
                max=lambda attr, val: 1,
                val=lambda attr, val: 1,
            ),
        )
    ])

    # ── Bounds ────────────────────────────────────────────────────────────
    # Enforce voltage magnitude between 0.9 pu and 1.1 pu at every bus.
    problem.bounds((0.9, 1.1), lambda m, _: type(m) is mm.Bus, ["vm_pu"])

    # ── Objective ─────────────────────────────────────────────────────────
    # Minimise total served load (each shed MW costs 10 units of penalty).
    objectives = mp.Objectives()
    objectives.with_models(problem.controllables_link) \
              .data(lambda model: 10) \
              .calculate(lambda model_to_data: sum(model_to_data.values()))

    # ── Constraints ───────────────────────────────────────────────────────
    # Ensure the external grid only injects power (no curtailment).
    constraints = mp.Constraints()
    constraints.select_types(mm.ExtPowerGrid).equation(
        lambda model: model.p_mw >= 0
    )

    # Attach objectives and constraints to the problem
    problem.constraints = constraints
    problem.objectives = objectives

----

Running the optimisation
========================

Pass the problem to :func:`~monee.run_energy_flow_optimization`:

.. testcode::

    result = run_energy_flow_optimization(net, problem)

The ``result`` object is a :class:`~monee.solver.core.SolverResult` with:

- ``result.network`` — the solved network with variable values filled in.
- ``result.dataframes`` — per-model-type DataFrames for easy inspection.
- ``result.objective`` — the achieved objective value.

----

Inspecting results
==================

.. testcode::

    # Voltage magnitude at every bus
    bus_df = result.dataframes.get("Bus")
    if bus_df is not None:
        print(list(bus_df.columns))

.. testoutput::
   :options: +SKIP

    ['vm_pu', 'va_radians', ...]

.. note::

   The ``debug=True`` flag on :class:`~monee.problem.core.OptimizationProblem`
   enables verbose logging during solve, which is useful during development
   but should be disabled in production.

----

Next steps
==========

- See :doc:`02_timeseries_simulation` to run the optimisation over a time
  series.
- Learn about :doc:`../concepts/formulations` to pick the right equation
  set for your problem.
- See :doc:`../how-to/use_pyomo_solver` to use a MILP solver back-end.
