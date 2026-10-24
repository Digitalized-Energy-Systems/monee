Temperature floor at selected junctions
=======================================

A district heating operator often has to keep the supply temperature at some
substations above a contractual minimum, for example 65 °C for domestic hot
water. This guide adds such a floor to an optimization problem.

Constrain ``t_pu``, not ``t_k``
-------------------------------

A water junction reports its temperature twice: ``t_pu``, the solver's
decision variable in per unit of the grid's ``t_ref_k``, and ``t_k``, a
:class:`~monee.model.core.PostProcess` report that monee computes as
``t_pu * t_ref_k`` after the solve. A report is not part of the model the
solver sees, so a constraint can only use ``t_pu``; written with ``t_k`` it
raises a ``TypeError`` that names ``t_pu``.

Build a small district heating line with two consumers 2 km apart:

.. testcode::

    import monee
    import monee.express as mx
    import monee.model as mm
    import monee.problem as mp

    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=2000, unidirectional=False)
    seg = dhs.line(3, heat_exchanger_q_mw=[0, 0.3, 0.4])
    dhs.attach_heat_plant(seg.supply.first, seg.return_.first, t_k=358.0)

    t_ref_k = net.node_by_id(seg.supply.first).grid.t_ref_k
    consumers = seg.supply.nodes[1:]

Select the supply junctions of the consumers by id and bound their ``t_pu``
from below. The selector takes the model and the component, so it can read
the component id:

.. testcode::

    problem = mp.OptimizationProblem()
    problem.constraints = mp.Constraints()
    problem.constraints.select(
        lambda m, c: isinstance(m, mm.Junction) and c.id in consumers
    ).equation(lambda m: m.t_pu >= 350.0 / t_ref_k)

    result = monee.run_energy_flow_optimization(net, problem)
    t_k = result.get(mm.Junction, index="id").loc[consumers, "t_k"]
    print(result.success, t_k.round(2).tolist())

.. testoutput::

    True [357.37, 356.28]

A problem factory already has a constraint set, so add to it instead of
replacing it: ``ed = create_economic_dispatch_problem(...)`` followed by
``ed.constraints.select(...).equation(...)``. If no dispatch can hold the
floor, the solve is infeasible; drop the floor for that case and record it.

The same constraint written with the report fails before the solve starts:

.. testcode::

    bad = mp.OptimizationProblem()
    bad.constraints = mp.Constraints()
    bad.constraints.select(
        lambda m, c: isinstance(m, mm.Junction) and c.id in consumers
    ).equation(lambda m: m.t_k >= 350.0)
    try:
        monee.run_energy_flow_optimization(net, bad)
    except TypeError as exc:
        print(exc)

.. testoutput::
   :options: +ELLIPSIS

   A PostProcess report cannot appear in a constraint or objective: ... Use the decision variable it is computed from instead: t_pu ...

A floor on every junction of a grid
-----------------------------------

When the floor holds for every junction that a model and grid test can
select, :meth:`~monee.problem.OptimizationProblem.bounds` sets it on the
variable bounds instead. The condition receives the model and the grid, not
the component, so it cannot select by id. Return junctions are cold, so a
grid-wide floor has to sit below the return temperature:

.. testcode::

    grid_floor = mp.OptimizationProblem()
    grid_floor.bounds(
        (320.0 / t_ref_k, 2.0),
        lambda m, g: type(m) is mm.Junction and isinstance(g, mm.WaterGrid),
        ["t_pu"],
    )
    result = monee.run_energy_flow_optimization(net, grid_floor)
    print(result.success, round(result.get(mm.Junction)["t_k"].min(), 2))

.. testoutput::

    True 326.13

``bounds`` replaces both bounds of the variable, so pass the upper bound as
well. :func:`~monee.problem.create_economic_dispatch_problem` has its own
grid-wide band, ``bounds_t_k`` in K, and
:func:`~monee.problem.create_min_load_shedding_problem` has ``bounds_t`` in
per unit.
