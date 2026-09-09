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
     - Active power / mass-flow range at external grid connections, in the
       load convention: import is negative, export positive. To cap the
       import at X, pass ``(-X, 0)``. Default ``None``: unconstrained.

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

    print(f"Solver objective: {result.objective:.4f}")

.. note::

   ``result.objective`` is the solver's objective value, which is dominated by
   the weighted shed energy but is not a shed-energy meter. See
   :ref:`objective-semantics` below before comparing it across scenarios.

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
    mx.create_power_load(net, bus_1, p_mw=0.4, q_mvar=0.0, name="campus_load")

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
       mass flow in kg/s for gas and heat. Default for all three: ``None``,
       meaning the exchange is unconstrained. Only applied when
       ``include_ext_grids=True``. The bounds sit directly on
       ``ExtPowerGrid.p_mw`` and ``ExtHydrGrid.mass_flow_kgs``, which follow
       the load convention: a network importing 2 MW reports ``p_mw = -2.0``.
       So an import cap of X is the lower bound, ``bounds_ext_el=(-X, 0)``,
       and the mirror image ``(0, X)`` forbids importing at all. That mistake
       does not fail the solve, it sheds every load the local generation
       cannot cover, so the problem builder warns for any non-negative lower
       bound paired with a positive upper one. ``(0, 0)``, the deliberate
       "no exchange at all" island case, is left alone. Any explicit bound
       tighter than the network's intact import will force shedding, which
       is why the default constrains nothing.
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

.. _objective-semantics:

What ``result.objective`` is, and is not
----------------------------------------

``result.objective`` is the value the solver reported for its own objective,
not a shed-energy meter. Two things separate the two.

Inactive and ignored components are left out of the objective entirely. A
component switched off with ``deactivate_by_id``, and every component dropped
by ``exclude_unconnected_nodes=True`` after a contingency islands it, has no
Vars registered with the backend, so its unserved demand contributes nothing.
The paradox is that the worse the contingency, the lower the objective can go:
a fault that islands a whole feeder scores better than one that merely strains
it, because the islanded demand has left the sum. The ``regulation`` values in
the result dataframes report those loads as ``0.0``, which is the honest
picture, but the scalar does not.

The formulation's own tightening terms are added to the same solver objective.
The convex epigraph relaxations of the gas and heat formulations keep
themselves tight with small auxiliary objective terms, which the backend sums
into the reported value alongside the shed term. They are tiny compared with
shed energy, but they are not always positive, so a network at zero
curtailment can report a small nonzero or slightly negative objective rather
than an exact ``0.0``.

Both effects make the scalar unsafe for ranking scenarios against each other.
For that, compute the shed energy from the solved network instead:

.. code-block:: python

    from monee.problem import calc_general_resilience_performance

    shed_power, shed_heat, shed_gas = calc_general_resilience_performance(
        result.network
    )

:func:`~monee.problem.calc_general_resilience_performance` walks the solved
network, so it counts a deactivated or excluded load at its full rating and
returns MW-equivalents per carrier that are comparable across contingencies.
Within a single solve the objective is still the right thing to watch, since
it is what the optimiser minimised.

----

Interpreting the result
=======================

After solving, the ``regulation`` attribute of each controllable component
shows how much of its nominal setpoint was served. A value of ``1.0`` means
fully served; ``0.0`` means completely shed. Read the values from the result
dataframes, which contain plain floats:

.. testcode::

    import monee.model as mm

    loads = result.get(mm.PowerLoad)
    for name, reg in zip(loads["name"], loads["regulation"]):
        print(f"{name}: regulation = {reg:.2f}")

.. testoutput::

    campus_load: regulation = 1.00

On the solved network itself (``result.network``), every attribute the
problem promoted to a decision variable stays a
:class:`~monee.model.core.Var` object, which does not format or cast as a
number (``f"{reg:.2f}"`` and ``float(reg)`` both raise ``TypeError``);
attributes that were never promoted remain plain floats. Read the solved
number from the ``.value`` attribute, or call :func:`monee.model.value`,
which unwraps ``Var`` and passes plain floats through unchanged; see
:doc:`../concepts/data_model` for the full contract.

.. testcode::

    for child in result.network.childs:
        reg = getattr(child.model, "regulation", None)
        if reg is not None:
            print(f"{child.name or child.id}: regulation = {mm.value(reg):.2f}")

.. testoutput::

    0: regulation = 1.00
    campus_load: regulation = 1.00

.. note::

   A ``regulation`` value between 0 and 1 indicates partial load curtailment.
   Inspect the ``dataframes`` dict for per-component voltage, pressure, and
   temperature to understand why curtailment was needed.

Heat exchangers with a fixed ``regulation`` are a special case, because their
duty is stated as an inequality (``q_mw_delivered <= q_mw``) that only an
objective term pulls tight. A user objective can outweigh that pull and leave
an exchanger delivering less than its setpoint while the solve still reports
success. Such a shortfall is listed in ``result.violations`` under the key
``<Type>.<id>.q_mw_delivered_shortfall``, together with a warning on the
``monee.solver.core`` logger:

.. code-block:: python

    for key, magnitude in result.violations.items():
        if key.endswith("q_mw_delivered_shortfall"):
            print(f"{key}: {magnitude:.4f} MW unmet")

The magnitude is the unmet duty in MW. If you see one, either add
``q_mw_delivered == q_mw_set`` as an explicit constraint or scale your
objective coefficients so the duty pull is not outbid. Exchangers whose
``regulation`` is a decision variable are not reported: there the shortfall is
the shedding decision you asked for.

The check also covers the internal ``SubHE`` exchanger of coupling compounds
(CHP, GasToHeat, PowerToHeat). Its duty is not a fixed setpoint but the
solved ``q_mw`` that the control node's coupling equation prescribes, so a
compound heat side that delivers less than that value appears in
``result.violations`` as ``SubHE.<id>.q_mw_delivered_shortfall``.

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
        include_ext_grid=False,
        include_coupling_points=False,
    )

or, equivalently, via the convenience wrapper
:func:`~monee.problem.calc_general_resilience_performance`:

.. code-block:: python

    from monee.problem import calc_general_resilience_performance

    power_mw, heat_mw, gas_mw = calc_general_resilience_performance(result.network)

Ignored or inactive loads (e.g. excluded by ``exclude_unconnected_nodes``)
count at their full rating; regulated loads count at
``upper - value * regulation``; a heat exchanger counts at the gap between its
setpoint and the duty it reached. Gas curtailment is converted to MW with the
same ``3.6 * higher_heating_value_kwh_per_kg`` convention as the objective. Pass
``include_coupling_points=True`` to also account coupling-point curtailment
on the input carrier, mirroring the corresponding problem option.

``include_ext_grid`` (False by default) adds a fourth term that is easy to
misread: every external grid that feeds the network is counted as unserved
demand, at its full import. That is the islanding view, where import stands for
the load an islanded network would have had to shed, so it is what you want
when ranking islanding scenarios. On a grid-connected network it inflates the
power component by the whole substation import, and the number no longer equals
the sum of the per-load shed.
