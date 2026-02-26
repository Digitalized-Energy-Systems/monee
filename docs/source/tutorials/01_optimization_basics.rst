==================================
01 · Minimum-cost load curtailment
==================================

**Scenario.** A radial feeder connects a substation (Bus 0) to two loads via
two line segments. Bus 1 serves a small **factory** (0.6 MW); Bus 2 serves a
**warehouse** (0.4 MW).  An upstream fault limits the substation connection to
at most 0.6 MW — far less than the combined 1.0 MW demand.  Some load must be
shed.

Interrupting the factory costs **30 monetary units/MW** (critical production
process); interrupting the warehouse costs only **5 units/MW** (deferrable
refrigeration).  The optimiser finds the cheapest curtailment plan.

.. tip::

   For a ready-made one-call load-shedding interface see
   :doc:`../how-to/load_shedding`.  This tutorial builds the problem from
   scratch to show the full API.

----

Building the network
====================

.. testcode::

    import monee.express as mx
    import monee.model as mm
    import monee.problem as mp
    from monee import run_energy_flow_optimization

    net = mx.create_multi_energy_network()

    # Three buses on a radial feeder: substation → factory → warehouse
    bus_sub = mx.create_bus(net)
    bus_fac = mx.create_bus(net)
    bus_wh  = mx.create_bus(net)

    mx.create_line(net, bus_sub, bus_fac, 1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_line(net, bus_fac, bus_wh,  1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    mx.create_ext_power_grid(net, bus_sub)

    fac_id = mx.create_power_load(net, bus_fac, p_mw=0.6, q_mvar=0.0)
    wh_id  = mx.create_power_load(net, bus_wh,  p_mw=0.4, q_mvar=0.0)

    # Attach penalty and nominal demand as plain attributes on the model objects.
    # These are read by the objective at solve time.
    fac_model = net.child_by_id(fac_id).model
    wh_model  = net.child_by_id(wh_id).model

    fac_model._penalty   = 30   # high: critical process
    fac_model._p_nominal = 0.6  # MW
    wh_model._penalty    = 5    # low: deferrable
    wh_model._p_nominal  = 0.4  # MW

----

Defining the optimisation problem
==================================

An :class:`~monee.problem.core.OptimizationProblem` has three building blocks:

- **Controllables** — which attributes the solver may vary (here: the
  ``regulation`` fraction of each load).
- **Constraints** — additional restrictions beyond the energy-flow equations
  (here: the 0.6 MW substation limit).
- **Objective** — the scalar to minimise (here: total curtailment cost).

.. testcode::

    problem = mp.OptimizationProblem()

    # ── Controllables ──────────────────────────────────────────────────────────
    # regulation ∈ [0, 1]: fraction of each load that remains served.
    # 1 = fully served, 0 = completely curtailed.
    problem.controllable_demands([
        (
            "regulation",
            mp.AttributeParameter(
                min=lambda attr, val: 0,
                max=lambda attr, val: 1,
                val=lambda attr, val: 1,   # initialise at full service
            ),
        )
    ])

    # ── Constraint ─────────────────────────────────────────────────────────────
    # The substation can inject at most 0.6 MW (upstream fault limit).
    constraints = mp.Constraints()
    constraints.select_types(mm.ExtPowerGrid).equation(
        lambda model: model.p_mw <= 0.6
    )
    problem.constraints = constraints

    # ── Objective ──────────────────────────────────────────────────────────────
    # Minimise total curtailment cost:
    #   cost = Σ (1 - regulation_i) × p_nominal_i × penalty_i
    #
    # The penalty attributes were stored on the model objects above.
    # At solve time model.regulation is a solver variable; the expression
    # (1 - model.regulation) is therefore part of the optimisation problem.
    objectives = mp.Objectives()
    objectives.select(
        lambda model: isinstance(model, mm.PowerLoad) and hasattr(model, "_penalty")
    ).data(
        lambda model: (1 - model.regulation) * model._p_nominal * model._penalty
    ).calculate(
        lambda model_to_data: sum(model_to_data.values())
    )
    problem.objectives = objectives

----

Running the optimisation
========================

.. testcode::

    result = run_energy_flow_optimization(net, problem)
    print(f"Objective (curtailment cost): {result.objective:.2f}")

.. testoutput::
   :options: +SKIP

    Objective (curtailment cost): 2.00

The objective value of **2.00** matches the expected optimum: curtail the entire
warehouse (0.4 MW × penalty 5 = 2.0 units), which is far cheaper than reducing
the factory.

----

Inspecting results
==================

:meth:`~monee.solver.core.SolverResult.get` returns the result DataFrame for a
given model type:

.. testcode::

    load_df = result.get(mm.PowerLoad)
    print(load_df[["p_mw", "regulation"]].round(3))

.. testoutput::
   :options: +SKIP

       p_mw  regulation
    0   0.6       1.000
    1   0.0       0.000

The factory (row 0) keeps its full 0.6 MW at ``regulation = 1.0``.  The
warehouse (row 1) is completely curtailed to ``regulation = 0.0``.  The
substation import equals exactly the 0.6 MW limit:

.. testcode::

    ext_df = result.get(mm.ExtPowerGrid)
    print(f"Substation import: {ext_df['p_mw'].sum():.2f} MW")

.. testoutput::
   :options: +SKIP

    Substation import: 0.60 MW

.. note::

   Removing ``debug=False`` (the default) from
   :class:`~monee.problem.core.OptimizationProblem` keeps the solver output
   quiet.  Pass ``debug=True`` while developing to see which attributes were
   made controllable.

----

Next steps
==========

- See :doc:`02_timeseries_simulation` to run energy-flow calculations over a
  time series with varying demand profiles.
- Explore :doc:`../how-to/load_shedding` for the ready-made one-call interface.
- Read :doc:`../how-to/use_pyomo_solver` to switch to a MILP solver back-end
  (HiGHS, Gurobi, etc.) for integer-programming formulations.
