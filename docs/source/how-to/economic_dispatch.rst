=================
Economic dispatch
=================

:func:`~monee.problem.create_economic_dispatch_problem` builds an optimal
power flow that minimises the total generation cost over all
:class:`~monee.model.child.PowerGenerator` components and, optionally, a
priced exchange with the external grid. It works with the default AC NLP
formulation and IPOPT out of the box; for the convex MISOCP variant and
mixed-integer extensions see :doc:`../concepts/formulations` and
:doc:`use_pyomo_solver`.

How costs enter
===============

Each generator contributes ``cost * dispatched MW`` to the objective. Set the
price (currency per MW) when creating the component
(``mx.create_power_generator(..., cost=...)``), assign it to the model
afterwards, or vary it per period via
:meth:`~monee.simulation.TimeseriesData.add_objective_data`. Generators
without a cost fall back to ``gen_cost_default``.

The ``p_mw`` magnitude passed at creation is the capacity: the problem makes
``p_mw`` a decision variable ranging from zero to that magnitude, so a
generator created with ``p_mw=0.6`` can dispatch anywhere in 0 to 0.6 MW.

Signs and the load convention
=============================

monee uses the load convention everywhere (positive means consumption), and
the dispatch objective is written against it:

- A solved generator reports a negative ``p_mw``; the objective charges
  ``cost * (-p_mw)``, i.e. the produced MW.
- :attr:`~monee.model.child.ExtPowerGrid.p_mw` is negative when the network
  imports and positive when it exports (see
  :ref:`data-model-slack-sign`). With ``ext_grid_cost_default`` set, import
  is charged at that price and export is credited at the same price. An
  external price above a generator's cost therefore makes exporting
  profitable, and the optimiser will run that generator at full capacity to
  sell; bound the exchange with ``ext_grid_bounds`` if that is not intended.
- ``ext_grid_bounds`` is stated in the same convention: to cap the import at
  ``X`` MW while forbidding export, pass ``(-X, 0.0)``. Writing the tuple the
  other way round does not fail, it silently forbids importing.

By default (``ext_grid_cost_default=None``) the external grid is left out of
the objective entirely. Slack import is then free, so the optimum degenerates
to importing everything and dispatching no local generation. Pass a price
whenever the network has an external grid and you want a meaningful dispatch.

Keyword reference
=================

.. list-table::
   :header-rows: 1
   :widths: 28 14 58

   * - Keyword
     - Default
     - Meaning
   * - ``gen_cost_default``
     - ``1.0``
     - Price (currency/MW) for generators without their own ``cost``.
   * - ``ext_grid_cost_default``
     - ``None``
     - Price for the external grid exchange. ``None`` excludes the slack
       from the objective (free import, see above); a number makes the
       exchange controllable and prices import and export symmetrically.
   * - ``bounds_vm``
     - ``(0.9, 1.1)``
     - Voltage magnitude bounds, applied to the actual voltage decision
       variable of the formulation (``vm_pu`` under the NLP,
       ``vm_pu_squared`` under MISOCP, where the bounds are squared).
   * - ``bounds_lp``
     - ``(0, 1.0)``
     - Line loading limit, enforced as a constraint on both branch ends.
       The lower bound must be ``0``; a non-zero minimum loading would
       force flow onto every line and raises a ``ValueError``.
   * - ``ext_grid_bounds``
     - ``None``
     - ``(min, max)`` bounds on ``ExtPowerGrid.p_mw`` in the load
       convention; only used together with ``ext_grid_cost_default``.
   * - ``ramp_limit``
     - ``None``
     - Maximum MW change of each generator's dispatch between consecutive
       timesteps; a temporal constraint that only binds in timeseries and
       multi-period runs.
   * - ``check_vm`` / ``check_lp``
     - ``True``
     - Disable the voltage or line-loading limits entirely.
   * - ``debug``
     - ``False``
     - Verbose problem construction.

Worked example
==============

Two generators and a priced grid connection. The cheap unit runs at capacity,
the expensive one stays off because importing at 25 is cheaper than
generating at 40, and the remainder of the demand (plus losses) is imported:

.. testcode::

    import monee
    import monee.express as mx
    import monee.model as mm
    from monee.problem import create_economic_dispatch_problem

    net = mx.create_multi_energy_network()
    b0, b1, b2 = mx.create_bus(net), mx.create_bus(net), mx.create_bus(net)
    mx.create_line(net, b0, b1, 200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_line(net, b1, b2, 200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_generator(net, b1, p_mw=0.6, q_mvar=0.0, cost=10, name="cheap")
    mx.create_power_generator(net, b2, p_mw=0.6, q_mvar=0.0, cost=40, name="pricey")
    mx.create_power_load(net, b2, p_mw=0.8, q_mvar=0.0, name="demand")

    problem = create_economic_dispatch_problem(
        ext_grid_cost_default=25,
        ext_grid_bounds=(-0.5, 0.0),
    )
    result = monee.run_energy_flow_optimization(net, problem)

    gens = result.get(mm.PowerGenerator)
    for name, p in zip(gens["name"], gens["p_mw"]):
        print(f"{name}: {abs(p):.2f} MW dispatched")
    ext_p = result.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
    print(f"grid exchange: {ext_p:.2f} MW (negative = import)")

.. testoutput::

    cheap: 0.60 MW dispatched
    pricey: 0.00 MW dispatched
    grid exchange: -0.21 MW (negative = import)

``result.objective`` is the total cost in the chosen currency
(``0.6 * 10 + 0.21 * 25``, roughly ``11.4`` here). The dataframes contain
plain floats; the load convention explains the signs printed above.

Multi-period dispatch
=====================

:func:`~monee.problem.create_multi_period_economic_dispatch_problem` is an
alias of the same builder; the problem becomes multi-period simply by running
it through :func:`~monee.simulation.run_multi_period`. Two keywords matter
then: ``ramp_limit`` bounds the MW change of each generator between
consecutive steps, and per-period prices are registered on the
:class:`~monee.simulation.TimeseriesData` with
``add_objective_data(component_id, "cost", [...])``, which sets each
generator's ``cost`` before the period's objective is assembled. See
:doc:`multi_period` for the surrounding workflow.
