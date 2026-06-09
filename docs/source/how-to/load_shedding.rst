=====================
Load shedding
=====================

Load shedding finds the minimum amount of demand that must be curtailed to
keep a network feasible under given operational limits. monee ships a
ready-made load-shedding formulation for multi-energy networks with a
one-call interface and a lower-level builder function for customisation.

How it works
============

The optimiser makes each demand, generator, and coupling unit *controllable*
via a ``regulation`` multiplier in [0, 1]. It then minimises the total
deviation from full service, subject to operational bounds on voltage
(electrical), pressure (gas), and temperature (heat):

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Bound argument
     - Controls
   * - ``bounds_vm``
     - Voltage magnitude at buses (per unit, e.g. ``(0.9, 1.1)``)
   * - ``bounds_t``
     - Normalised temperature at heating junctions (per unit)
   * - ``bounds_pressure``
     - Normalised pressure at gas junctions (per unit)
   * - ``bounds_ext_el`` / ``bounds_ext_gas``
     - Active power / mass-flow range at external grid connections

----

One-call interface
==================

.. tip::

   Use :func:`monee.solve_load_shedding_problem` for the simplest path — it
   picks sensible defaults and requires minimal configuration.

.. code-block:: python

    from monee import solve_load_shedding_problem

    result = solve_load_shedding_problem(
        network,
        bounds_vm=(0.9, 1.1),
        bounds_t=(0.8, 1.2),
        bounds_pressure=(0.8, 1.2),
        bounds_ext_el=(-0.5, 0.5),
        bounds_ext_gas=(-2.0, 2.0),
    )

    print(f"Objective (shed load cost): {result.objective_value:.4f}")

The result is a :class:`~monee.solver.core.SolverResult` with the solved
network and DataFrames for each model type.

----

Custom problem
==============

For finer control — different weights per carrier, extra constraints, or
different bounds — use
:func:`~monee.problem.create_min_load_shedding_problem` directly:

.. testcode::

    import monee.express as mx
    from monee import run_energy_flow_optimization
    from monee.problem import create_min_load_shedding_problem

    # Build a small test network
    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.4, q_mvar=0.0)

    # Create and customise the load-shedding problem
    problem = create_min_load_shedding_problem(
        demand_weight=20,            # penalty per MW of shed demand
        bounds_el=(0.92, 1.08),      # tighter voltage bounds
        check_pressure=False,        # no gas grid present
        check_temperature=False,     # no heat grid present
        debug=False,
    )

    result = run_energy_flow_optimization(
        net, problem, exclude_unconnected_nodes=True
    )

.. note::

   Pass ``exclude_unconnected_nodes=True`` when the network may contain buses
   or junctions that are topologically disconnected (e.g. because a line is
   deactivated).  Without this flag such nodes are included in the solve,
   which can cause infeasibility.  With it, disconnected components are
   automatically excluded and their ``regulation`` is reported as ``0.0``
   (fully shed) in the result.

Key parameters of ``create_min_load_shedding_problem``:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Parameter
     - Effect
   * - ``demand_weight`` / ``generator_weight``
     - Penalty factors applied to each unit of shed demand / curtailed
       generation. Higher demand weight pushes the solver to avoid load
       curtailment. Defaults: ``1e3`` / ``0.1``.
   * - ``bounds_el``
     - ``(min, max)`` voltage bounds in per unit. Default: ``(0.9, 1.1)``.
   * - ``bounds_heat``
     - ``(min, max)`` temperature bounds in per unit. Default: ``(0.9, 1.1)``.
   * - ``bounds_gas``
     - ``(min, max)`` pressure bounds in per unit. Default: ``(0.9, 1.1)``.
   * - ``check_vm`` / ``check_temperature`` / ``check_pressure`` / ``check_line_loading``
     - Disable individual bound checks for carriers not present in the network.
   * - ``include_ext_grids``
     - Make external grids controllable, bound their exchange, and add a
       quadratic slack nudging exchange toward zero. Default: ``True``.
   * - ``auto_priority_floor``
     - Scale ``demand_weight`` so shed cost dominates formulation-tightening
       terms regardless of network size. Default: ``True``.
   * - ``debug``
     - Enable verbose solver output. Default: ``False``.

----

Interpreting the result
=======================

After solving, the ``regulation`` attribute of each controllable component
shows how much of its nominal setpoint was served. A value of ``1.0`` means
fully served; ``0.0`` means completely shed.

.. code-block:: python

    for child in result.network.childs:
        reg = getattr(child.model, "regulation", None)
        if reg is not None:
            print(f"{child.name or child.id}: regulation = {reg:.2f}")

.. note::

   A ``regulation`` value between 0 and 1 indicates partial load curtailment.
   Inspect the ``dataframes`` dict for per-component voltage, pressure, and
   temperature to understand why curtailment was needed.
