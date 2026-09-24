==================================
01 · Minimum-cost load curtailment
==================================

**Scenario.** A radial feeder connects a substation (Bus 0) to two loads over
two line segments. Bus 1 serves a small factory (0.6 MW); Bus 2 serves a
warehouse (0.4 MW). An upstream fault caps the substation connection at
0.6 MW, far below the combined 1.0 MW demand, so some load must be shed.

Interrupting the factory costs 30 monetary units/MW (a critical production
process); interrupting the warehouse costs only 5 units/MW (deferrable
refrigeration). The optimiser finds the cheapest curtailment plan.

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

- **Controllables:** which attributes the solver may vary (here: the
  ``regulation`` fraction of each load).
- **Constraints:** additional restrictions beyond the energy-flow equations
  (here: the 0.6 MW substation limit).
- **Objective:** the scalar to minimise (here: total curtailment cost).

.. testcode::

    problem = mp.OptimizationProblem()

    # Controllables
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

    # Constraint
    # The substation can inject at most 0.6 MW (upstream fault limit).
    # ExtPowerGrid.p_mw follows the load convention: import is negative,
    # export positive, so a 0.6 MW import cap is a lower bound on p_mw.
    constraints = mp.Constraints()
    constraints.select_types(mm.ExtPowerGrid).equation(
        lambda model: model.p_mw >= -0.6
    )
    problem.constraints = constraints

    # Objective
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

.. note::

   The one argument selections above take different arguments.
   ``objectives.select`` is called with the model, as written here, while
   ``constraints.select`` and ``problem.controllable(component_condition=...)``
   are called with the ``Component`` container, so an ``isinstance`` test there
   reads ``lambda c: isinstance(c.model, mm.PowerLoad)``. A two argument
   predicate ``lambda model, component: ...`` behaves identically on all three
   and is the recommended form. The typed shorthand
   ``constraints.select_types(mm.ExtPowerGrid)`` used above avoids the
   question. A one argument predicate written for the wrong argument is
   applied under the other interpretation with a warning when its own matches
   nothing. See :doc:`../api/monee.problem` for the full convention.

----

Running the optimisation
========================

.. testcode::

    result = run_energy_flow_optimization(net, problem)
    print(f"Objective (curtailment cost): {result.objective:.2f}")

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    Objective (curtailment cost): 2.76

Curtailing the entire warehouse costs 0.4 MW × penalty 5 = 2.0 units, far less
than shedding the same amount at the factory. That alone is not enough: the two
line segments also draw losses from the 0.6 MW the substation may deliver, so a
small slice of the factory load is curtailed on top, which accounts for the
remaining 0.76 units.

----

Inspecting results
==================

:meth:`~monee.solver.core.SolverResult.get` returns the result DataFrame for a
given model type:

.. testcode::

    load_df = result.get(mm.PowerLoad)
    # clip: the interior-point solver may land a hair outside [0, 1]
    print(load_df[["p_mw", "regulation"]].clip(lower=0).round(2))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

       p_mw  regulation
    0   0.6        0.96
    1   0.4        0.00

The ``p_mw`` column is the setpoint you passed to
:func:`~monee.express.create_power_load`; it never moves during the solve. The
served power is ``p_mw * regulation``, so the factory (row 0) keeps about
96 percent of its 0.6 MW, roughly 0.57 MW, and the warehouse (row 1) is
curtailed completely at ``regulation = 0.0``. The same holds for every
curtailable component: the setpoint attribute stays put and ``regulation``
carries the decision. The node result frames report the served value directly,
since a node balances ``p_mw * regulation`` over its children.

The substation import equals exactly the 0.6 MW limit. Import is negative under
the load convention, so negate the column to read it as an import:

.. testcode::

    ext_df = result.get(mm.ExtPowerGrid)
    print(f"Substation import: {-ext_df['p_mw'].sum():.2f} MW")

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    Substation import: 0.60 MW

.. note::

   :class:`~monee.problem.core.OptimizationProblem` accepts ``debug=False`` by
   default, which keeps variable-promotion logging quiet.  Pass ``debug=True``
   while developing to log which attributes were promoted to solver variables.

----

Next steps
==========

- See :doc:`02_timeseries_simulation` to run energy-flow calculations over a
  time series with varying demand profiles.
- Explore :doc:`../how-to/load_shedding` for the ready-made one-call interface.
- Read :doc:`../how-to/use_pyomo_solver` to switch to a MILP solver back-end
  (SCIP, Gurobi, etc.) for integer-programming formulations.
