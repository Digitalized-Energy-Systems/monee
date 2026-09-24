===========
Conventions
===========

One page for the signs, bases and defaults that every other page relies on.
Each section is a summary; the linked pages carry the derivations and the
worked examples. When a number surprises you, check here first, then see
:doc:`../how-to/troubleshooting` for the symptoms these conventions produce
in practice.

The load convention
===================

monee uses the load convention everywhere: positive = consumption, negative =
generation, seen from the network. Generator-type constructors take positive
magnitudes and apply the sign internally; passing a negative number raises a
``ValueError``. Details: :doc:`data_model` (Sign and unit conventions).

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Child class
     - Attribute
     - Sign of the stored / solved value
   * - :class:`~monee.model.child.PowerLoad`
     - ``p_mw``, ``q_mvar``
     - Positive = consumption.
   * - :class:`~monee.model.child.PowerGenerator`,
       :class:`~monee.model.child.VoltageControlledGenerator`
     - ``p_mw``
     - Constructor takes a positive magnitude, stored negative (generation).
   * - :class:`~monee.model.child.ExtPowerGrid`
     - ``p_mw`` (solved)
     - Positive = the network exports into the external grid; an import is
       negative. ``max_import_mw=X`` bounds ``p_mw >= -X``.
   * - :class:`~monee.model.child.Sink`
     - ``mass_flow_kgs``
     - Positive = consumption.
   * - :class:`~monee.model.child.Source`
     - ``mass_flow_kgs``
     - Constructor takes a positive magnitude, stored negative (injection).
   * - :class:`~monee.model.child.ExtHydrGrid`
     - ``mass_flow_kgs`` (solved)
     - Negative = injection into the network (import).
       ``max_import_kgs=X`` bounds it at ``-X`` from below.
   * - :class:`~monee.model.child.HeatLoad`
     - ``q_mw_heat``
     - Positive = consumption.
   * - :class:`~monee.model.child.HeatGenerator`
     - ``q_mw_heat``
     - Constructor takes a positive ``q_mw`` magnitude, stored negative.
   * - :class:`~monee.model.ElectricStorage`,
       :class:`~monee.model.GasStorage`,
       :class:`~monee.model.ThermalStorage`
     - ``p_mw`` / flow
     - Positive = charging (consuming from the network), negative =
       discharging.

Anything that bounds or prices a slack follows the same rule: an import cap is
a lower bound (``p_mw >= -cap``), and an import cost objective needs
``-price * p_mw``, not ``price * p_mw``. Writing either the other way round
does not fail, it silently forbids or maximises the import. See
:ref:`data-model-slack-sign` and the pricing example in
:doc:`../how-to/multi_period`.

Branch signs
============

Branch signs say which way the flow runs, and the two branch families do not
agree. The one line cheat for the hydraulic side: a negative ``mass_flow_kgs``
on a pipe is flow running from the from node to the to node, and a negative
``mass_flow_kgs`` on an :class:`~monee.model.child.ExtHydrGrid` is an import
into the network. Details: :doc:`data_model` (Sign on branches).

.. list-table::
   :header-rows: 1
   :widths: 24 32 44

   * - Family
     - Reported variables
     - Sign for flow from the from node to the to node
   * - Power branches (lines, transformers)
     - ``p_from_mw``, ``p_to_mw`` (and the ``q_*`` pair), each the power
       entering the branch at that end
     - ``p_from_mw`` positive, ``p_to_mw`` negative; the two add up to the
       branch losses.
   * - Hydraulic branches (gas and water pipes, pumps, valves, heat
       exchangers)
     - ``mass_flow_kgs = mass_flow_pos_kgs - mass_flow_neg_kgs``;
       ``velocity_mps`` shares the sign
     - Negative. Forward flow is carried by ``mass_flow_neg_kgs``, so a pipe
       feeding a downstream sink reports a negative mass flow. Take ``abs()``
       for throughput, read direction off the sign only.

Electricity
===========

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Base
     - Default
     - Meaning
   * - ``sn_mva`` (:class:`~monee.model.PowerGrid`)
     - 1
     - Power base; all ``_mw`` / ``_mvar`` equations are scaled by it.
   * - ``base_kv`` (per :class:`~monee.model.Bus`)
     - 1 (express factories)
     - Voltage base of the bus; ``vm_pu`` is relative to it, and the branch
       impedance base is ``base_kv**2 / sn_mva``.
   * - ``vm_pu`` bounds
     - 0.5 to 1.5 (grid level)
     - Wide operational envelope; optimisation problems usually tighten it via
       ``bounds_vm``.

Buses connected by a line must share their ``base_kv`` (transformers are the
place where bases change). A mixed-base network solves without complaint and
silently collapses voltages; give every bus its real ``base_kv`` explicitly,
including buses passed as ``start_from`` to the structure builders. Physics:
:doc:`domains`.

Gas
===

Defaults of :func:`~monee.model.create_gas_grid` (details: :doc:`data_model`,
Grids):

.. list-table::
   :header-rows: 1
   :widths: 34 24 42

   * - Field
     - Default
     - Meaning
   * - ``type``
     - ``"lgas"``
     - Fluid preset; ``"methane"`` is the alternative.
   * - ``pressure_ref_pa``
     - 1e6 (10 bar absolute)
     - Base for ``pressure_pu``; the solved state is
       ``pressure_squared_pu``, so a value of 1.21 means 1.1 pu, i.e. 11 bar.
   * - ``t_k`` / ``t_ref_k``
     - 300 K / 356 K
     - Operating and reference temperature.
   * - ``higher_heating_value_kwh_per_kg``
     - 11.79011 (lgas), 15.3 (methane)
     - Thermal power converts as ``MW = mass_flow_kgs * 3.6 * hhv``.
   * - ``pressure_squared_pu_min`` / ``..._max``
     - 0.7 to 1.3
     - Operational band in squared per unit, applied to junctions in
       optimisation runs only; a plain simulation keeps the wide 0 to 3
       range.

To run at a different operating pressure, construct the grid with that
pressure (``mm.create_gas_grid("mp", pressure_ref_pa=1.5e5)``) so the derived
compressibility follows.

Water and heat
==============

:class:`~monee.model.WaterGrid` defaults: ``t_ref_k=356`` (the base for
``t_pu``), ``pressure_ref_pa=1e6``, ``v_max_mps=5.0`` (caps each pipe's mass
flow via its cross section). :class:`~monee.model.child.ExtHydrGrid` pins the
pressure of its junction and, unless ``pin_temperature=False``, its
temperature at ``t_k`` (default: the owning grid's operating temperature,
300 K for the default gas grid and 356 K for the default water grid).

Three heat attribute names look alike and are deliberately distinct; the nodal
heat balance tells them apart by name, so never rename one into another:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Name
     - Carried by
   * - ``q_mw``
     - Heat-exchanger branches: the heat transferred across the branch.
   * - ``q_mw_heat``
     - :class:`~monee.model.child.HeatGenerator` and
       :class:`~monee.model.child.HeatLoad` children: nodal heat injection or
       withdrawal.
   * - ``heat_mw``
     - Coupler control nodes (P2H, G2H, CHP): the realized heat side of the
       coupling in generator sign, so negative means injection into the heat
       grid. A P2H built with a positive ``heat_energy_mw`` setpoint therefore
       reports ``heat_mw = -regulation * heat_energy_mw``.

Heat-loop wiring conventions (which junction is the hot one, supply and return
sides) are on :doc:`multi_energy`.

Removing components
===================

Every add operation on :class:`~monee.model.network.Network` has a removal
counterpart, addressed by the id the add returned:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Method
     - Removes
   * - ``remove_child(child_id)``
     - One child; also detaches it from its parent node.
   * - ``remove_node(node_id)``
     - The node plus every incident branch and attached child.
   * - ``remove_branch(branch_id)``
     - One branch by its ``(from, to, key)`` id tuple.
   * - ``remove_branch_between(node_one, node_two, key=0)``
     - The branch between two nodes; ``key`` picks among parallel branches.
   * - ``remove_compound(compound_id)``
     - The compound and, recursively, all of its subcomponents.

A typical replace-a-pipe workflow is ``remove_branch_between`` followed by a
new ``create_*`` call; the worked LinearValve example is on
:doc:`data_model`. To exclude a component from solves without deleting it,
use ``deactivate`` (or ``deactivate_by_id(cls, id)``) instead.

Writing attributes after construction
=====================================

A public attribute the model declares (or accepts as a constructor parameter,
such as ``cost``) is settable between solves and the equations read it. A
constructor argument that only sized a variable is not: it lives in that
variable's bounds from build time on, so a bigger battery is
``storage.e_mwh.max = 4.0`` and not ``storage.e_mwh_max = 4.0``. Where the
capacity is part of the model's contract it has a property that re-bounds the
variable, for example
:attr:`~monee.model.extension.islanding.el.GridFormingGenerator.p_mw_max`.

Assigning a plain value to any other public name stores an inert field that no
equation reads, and monee warns about it with
:class:`~monee.model.core.UnknownAttributeWarning`;
:func:`~monee.model.core.set_attribute_guard` (or
``MONEE_ATTRIBUTE_GUARD=error``) turns that into an ``AttributeError`` for CI.
Names starting with an underscore are never guarded, never solved and never
exported, which makes them the place for your own metadata such as
``load._priority_weight``. Details: :doc:`data_model`
(:ref:`settable-attributes`).

Setpoint versus served
======================

Every child carries a ``regulation`` attribute in 0.0 to 1.0 that scales its
setpoint; load-shedding problems promote it to a decision variable. The
result frames report the setpoint attribute unchanged next to the solved
``regulation``, so the served quantity is their product: a load with
``p_mw=2`` and ``regulation=0.5`` consumes 1 MW while its ``p_mw`` column
still reads 2. The same holds for compound setpoints such as
``CHPControlNode.gas_mass_flow_kgs`` and ``SubHE.q_mw_set``: ``*_set``
columns and control-node setpoints are inputs, while branch flows and ``q_mw``
are solved quantities. See :doc:`multi_energy` for the coupler result
columns and :doc:`../how-to/load_shedding` for shedding semantics.

Time
====

``dt_h`` is always in hours, for :func:`~monee.simulation.run_timeseries`,
:func:`~monee.simulation.run_multi_period` and
:meth:`~monee.simulation.stepper.Stepper.step` alike. A scalar means uniform
steps, a list gives per-step durations, and when a ``datetime_index`` is also
supplied the durations come from the index differences and ``dt_h`` is
ignored with a logged warning. Inter-step state integrates over ``dt_h``
implicitly, first order: a storage updates as
``e_mwh[t] == e_mwh[t-1] + dt_h * p_mw[t]``, so power in step ``t`` acts over
the duration of step ``t``. Details: :doc:`timeseries` and
:doc:`multi_period`.
