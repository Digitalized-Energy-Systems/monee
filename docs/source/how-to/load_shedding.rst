=====================
Load shedding
=====================

Load shedding finds the minimum demand that must be curtailed to keep a
network feasible under given operational limits. monee ships a ready-made
load-shedding formulation for multi-energy networks: a one-call interface for
the common case, and a builder function when you need to customise it.

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

   Use :func:`monee.solve_load_shedding_problem` for the simplest path: it
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

    print(f"Objective (shed load cost): {result.objective:.4f}")

The result is a :class:`~monee.solver.core.SolverResult` with the solved
network and DataFrames for each model type. The bound and check arguments are
keyword-only and mirror :func:`~monee.problem.create_min_load_shedding_problem`
one-to-one.

----

Custom problem
==============

For finer control (different weights per carrier, extra constraints, or
different bounds) use
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
        bounds_vm=(0.92, 1.08),      # tighter voltage bounds
        check_pressure=False,        # no gas grid present
        check_t=False,               # no heat grid present
        debug=False,
    )

    result = run_energy_flow_optimization(
        net, problem, exclude_unconnected_nodes=True
    )

.. note::

   Pass ``exclude_unconnected_nodes=True`` when the network may contain buses
   or junctions that are topologically disconnected (e.g. because a line is
   deactivated). Without this flag such nodes enter the solve and can cause
   infeasibility. With it, disconnected components are excluded and their
   ``regulation`` is reported as ``0.0`` (fully shed) in the result.

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
   * - ``weight_for_load``
     - Callback ``(model) -> float | None`` returning a per-load shed weight
       for priority ordering. ``None`` falls back to ``demand_weight``.
       Default: ``None``. See
       :ref:`how-to/load_shedding:Prioritising individual loads`.
   * - ``bounds_vm``
     - ``(min, max)`` voltage bounds in per unit. Default: ``(0.9, 1.1)``.
   * - ``bounds_t``
     - ``(min, max)`` temperature bounds in per unit. Default: ``(0.9, 1.1)``.
   * - ``bounds_pressure``
     - ``(min, max)`` pressure bounds in per unit. Default: ``(0.9, 1.1)``.
   * - ``max_line_loading``
     - Maximum electrical line loading. Default: ``1.5`` (150 %). The
       constraint is built by :func:`~monee.problem.utils.line_loading_limit`,
       which works for both the AC formulation (loading percent) and the
       MISOCP formulation (squared-current scale).
   * - ``check_vm`` / ``check_t`` / ``check_pressure`` / ``check_lp``
     - Disable individual bound checks for carriers not present in the network.
   * - ``bounds_ext_el`` / ``bounds_ext_gas`` / ``bounds_ext_heat``
     - Exchange range at external grids: active power in MW for electricity,
       mass flow in kg/s for gas and heat. Defaults: ``(-3, 3)`` /
       ``(-10, 10)`` / ``(-10, 10)``. Only applied when
       ``include_ext_grids=True``.
   * - ``include_ext_grids``
     - Make external grids controllable, bound their exchange, and add a
       quadratic slack that nudges exchange toward zero, weighted at
       ``demand_weight * 0.1`` so ties prefer self-sufficiency.
       Default: ``True``.
   * - ``include_storages``
     - Make storage units controllable (dispatchable) during shedding.
       Default: ``False``.
   * - ``include_coupling_points``
     - Penalise curtailment of coupling points like ordinary loads on their
       input carrier. Default: ``False``. See
       :ref:`how-to/load_shedding:Shedding coupling points`.
   * - ``regulation_ramp_limit``
     - Temporal constraint: limits how much each ``regulation`` value may
       change between consecutive steps in timeseries or multi-period runs.
       Default: ``None`` (no limit); ignored in single-period solves.
   * - ``lex_objectives``
     - Two-phase lexicographic solve (Pyomo only) that removes weight tuning
       entirely. Default: ``False``. See
       :ref:`how-to/load_shedding:Lexicographic objectives`.
   * - ``auto_priority_floor`` / ``priority_safety_factor``
     - Lift the demand weights so shed cost always dominates
       formulation-tightening terms regardless of network size.
       Defaults: ``True`` / ``10.0``. See
       :ref:`how-to/load_shedding:Why the default weights just work`.
   * - ``debug``
     - Enable verbose solver output. Default: ``False``.

Prioritising individual loads
-----------------------------

By default every demand is shed at the same cost per MW. To establish a
priority ordering (keep the hospital, shed the warehouse first), pass
``weight_for_load``, a callback that receives a load *model* and returns its
shed weight, or ``None`` to fall back to ``demand_weight``. A convenient
pattern is to tag the models with a custom attribute when building the
network:

.. code-block:: python

    import monee.express as mx
    from monee.problem import create_min_load_shedding_problem

    hospital_id = mx.create_power_load(net, bus_1, p_mw=0.4, q_mvar=0.0)
    net.child_by_id(hospital_id).model._priority_weight = 1e5  # shed last

    problem = create_min_load_shedding_problem(
        # None -> fall back to demand_weight for untagged loads
        weight_for_load=lambda model: getattr(model, "_priority_weight", None),
    )

The callback applies to all demand-type models (``PowerLoad``, ``HeatLoad``,
gas ``Sink``) and to every heat-exchanger variant, load and generator side
alike. All per-load weights are multiplied by the shared auto-priority-floor
scale (see below), so the *ratios* you choose are preserved even when monee
lifts the weights.

Why the default weights just work
---------------------------------

Some network formulations (e.g. MISOCP electricity, the smooth nonlinear gas
and heat models) add small auxiliary objective terms that tighten their
relaxations. On a large network these terms accumulate, and a fixed
``demand_weight`` could be undercut: the solver would rather shed load than
pay the tightening cost. With ``auto_priority_floor=True`` (the default),
monee computes an upper bound of the total auxiliary objective at apply time
and lifts the demand weights to at least ``priority_safety_factor`` times
that bound, so shedding always dominates the auxiliary terms regardless of
network size. The generator weight and any ``weight_for_load`` weights are
scaled by the same factor, preserving the relative priorities you set.

Lexicographic objectives
------------------------

Instead of weighting shed cost against the auxiliary terms, you can separate
them entirely: with ``lex_objectives=True`` the problem is solved in two
phases. Phase one minimises shed load alone, phase two minimises the
auxiliary tightening terms with the phase-one optimum pinned. This removes
weight tuning altogether, at the cost of a second solve. It requires the
Pyomo backend (:class:`~monee.solver.PyomoSolver`); GEKKO falls back to a
single summed objective. See :doc:`../concepts/solvers` for choosing a
backend.

Shedding coupling points
------------------------

By default, coupling units between carriers are controllable but their
curtailment is free: only the downstream service loss is penalised. With
``include_coupling_points=True``, coupling points (the CHP, CHPHG, G2H, and
P2H control nodes as well as the P2G, G2P, P2H-HG, and G2H-HG branches) are
penalised at ``demand_weight * rated input MW * (1 - regulation)``: each
coupling point is treated as a load on its *input* carrier (power or gas),
with gas input converted to MW via ``3.6 * higher_heating_value_kwh_per_kg``. Use this
when curtailing a conversion unit should count as lost demand in its own
right.

What counts as shed energy
--------------------------

The objective sums weighted unserved energy in MW-equivalent across all
carriers. The conversion conventions are:

- Electrical and heat loads/generators contribute
  ``setpoint * (1 - regulation)`` directly in MW.
- Gas sinks and sources are mass flows in kg/s; their shed is converted
  to MW via ``3.6 * higher_heating_value_kwh_per_kg`` of the enclosing gas grid
  (fallback: the lgas heating value of about 11.79 kWh/kg if the grid defines
  none).
- Water-grid sinks and sources are deliberately excluded: mass flow in a
  heating loop is a transport quantity, not demand, and applying a heating
  value to it would produce a phantom MW-scale penalty.
- Heat exchangers are penalised by the under-delivery gap
  ``|q_mw_set - q_mw_delivered|``, which captures both regulation cuts and
  physical shortfall from a narrow temperature spread that a pure
  ``(1 - regulation)`` proxy would miss.

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

----

Measuring curtailment
=====================

To quantify how much demand was actually curtailed, use
:class:`~monee.problem.GeneralResiliencePerformanceMetric` on the solved
network: it returns curtailed MW per carrier as a
``(power, heat, gas)`` tuple:

.. code-block:: python

    from monee.problem import GeneralResiliencePerformanceMetric

    power_mw, heat_mw, gas_mw = GeneralResiliencePerformanceMetric().calc(
        result.network,
        include_ext_grid=True,
        include_coupling_points=False,
    )

or, equivalently, via the convenience wrapper
:func:`~monee.problem.calc_general_resilience_performance`:

.. code-block:: python

    from monee.problem import calc_general_resilience_performance

    power_mw, heat_mw, gas_mw = calc_general_resilience_performance(result.network)

Ignored or inactive loads (e.g. excluded by ``exclude_unconnected_nodes``)
count at their full rating; regulated loads count at
``upper - value * regulation``. Gas curtailment is converted to MW with the
same ``3.6 * higher_heating_value_kwh_per_kg`` convention as the objective. Pass
``include_coupling_points=True`` to also account coupling-point curtailment
on the input carrier, mirroring the corresponding problem option.
