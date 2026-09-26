===================
Component reference
===================

Every grid component monee ships, grouped by energy carrier, with the two ways
to create it:

Object API
   The model class from :mod:`monee.model` (imported as ``mm``), added to a
   network with :meth:`~monee.model.network.Network.node`,
   :meth:`~monee.model.network.Network.branch`,
   :meth:`~monee.model.network.Network.child` or
   :meth:`~monee.model.network.Network.compound`. This is the complete
   interface and the one custom components plug into.

Express API
   The ``create_*`` function from :mod:`monee.express` (imported as ``mx``).
   It builds the model, attaches it to the network and returns the new
   component id. Single-carrier branch and child functions auto-create a
   missing endpoint; the coupling functions require existing nodes. ``mx.create_multi_energy_network()`` is
   an alias for ``mm.Network()``.

Both entry points produce the same component, so everything on these pages
applies to either. The API reference (:doc:`../api/monee.model`,
:doc:`../api/monee.express`) has the auto-generated signatures; these pages add
what a signature cannot say: the unit and meaning of every parameter, the
default each API applies, the columns a component contributes to the result
tables and where its physics is documented.

.. testcode::

   import monee.model as mm
   import monee.express as mx

   net = mx.create_multi_energy_network()

   # Object API: model class plus Network method.
   b0 = net.node(mm.Bus(base_kv=10), grid=mm.EL)
   b1 = net.node(mm.Bus(base_kv=10), grid=mm.EL)
   net.branch(mm.PowerLine(length_m=500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1), b0, b1)
   net.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0), attach_to_node_id=b0)

   # Express API: one call per component, same result.
   b2 = mx.create_bus(net, base_kv=10)
   mx.create_line(net, b1, b2, length_m=500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
   mx.create_power_load(net, b2, p_mw=0.2, q_mvar=0.05)

   print(len(net.nodes), len(net.branches), len(net.childs))

.. testoutput::

   3 2 2

The reference is split by sector:

.. toctree::
   :maxdepth: 1

   electricity
   gas
   water
   coupling
   grids
   extensions

Reading the tables
==================

Role
   Node (a connection point), branch (connects two nodes), child (attached to
   one node) or compound (a coupler that builds its own nodes and branches).
   :doc:`../concepts/data_model` explains the four types.

Parameters
   Every constructor argument with its unit, its default and what it does.
   Where the express function applies a different default than the class, the
   table lists both. The express functions additionally accept the container
   arguments ``constraints`` (a list of per-component constraints),
   ``overwrite_id`` (an explicit, still free component id), ``name`` (a label
   resolvable from :class:`~monee.simulation.TimeseriesData`), ``grid`` (the
   node's grid on node functions, the branch's own grid override on branch
   functions) and, on nodes, ``position`` (an ``(x, y)`` pair for plotting).

Results
   The table each component contributes to ``result.dataframes[Name]`` and
   ``result.get(Name)``, one row per component, with every column it
   carries and the value to expect. The kind column says where a value comes
   from: ``solved`` is written by the solver, ``input`` is a constructor
   argument or setpoint echoed back unchanged (with the internal sign where
   the constructor applied one), ``derived`` is computed from the inputs when
   the network is built, ``report`` is post-processed from the solution and
   ``internal`` is bookkeeping of a formulation that can be ignored. Every
   table also carries ``active`` (in service), ``id``, ``independent``
   (``False`` for the parts a compound built), ``ignored`` (dropped from the
   solve because its island has no reference: no external grid, or no
   grid-forming child once islanding is enabled) and, as soon as one row is
   named, ``name``. Child tables add ``node_id``. ``regulation``, the factor
   the nodal balance multiplies into the setpoint, appears on every child, on
   the coupler branches and control nodes and on heat exchangers; branch
   tables add ``on_off``. Columns the default formulation adds after a solve
   are listed below each table.

Signs
   Load convention throughout: positive is consumption, negative is
   generation. Generator-type constructors take positive magnitudes and apply
   the sign themselves. Hydraulic branches report flow from the from-node to
   the to-node as a negative ``mass_flow_kgs``; power branches report
   ``p_from_mw`` positive for the same direction. :doc:`../concepts/conventions`
   is the cheat sheet.

At a glance
===========

.. list-table::
   :header-rows: 1
   :widths: 24 10 12 27 27

   * - Component
     - Role
     - Carrier
     - Object API
     - Express API
   * - :ref:`Bus <components/electricity:Bus>`
     - node
     - electricity
     - ``mm.Bus``
     - ``mx.create_bus``
   * - :ref:`Power line <components/electricity:PowerLine>`
     - branch
     - electricity
     - ``mm.PowerLine``
     - ``mx.create_line``
   * - :ref:`Transformer <components/electricity:Trafo>`
     - branch
     - electricity
     - ``mm.Trafo``
     - ``mx.create_trafo``
   * - :ref:`Generic power branch <components/electricity:GenericPowerBranch>`
     - branch
     - electricity
     - ``mm.GenericPowerBranch``
     - ``mx.create_el_branch``
   * - :ref:`External power grid <components/electricity:ExtPowerGrid>`
     - child
     - electricity
     - ``mm.ExtPowerGrid``
     - ``mx.create_ext_power_grid``
   * - :ref:`Power load <components/electricity:PowerLoad>`
     - child
     - electricity
     - ``mm.PowerLoad``
     - ``mx.create_power_load``
   * - :ref:`Power generator <components/electricity:PowerGenerator>`
     - child
     - electricity
     - ``mm.PowerGenerator``
     - ``mx.create_power_generator``
   * - :ref:`Voltage-controlled generator <components/electricity:VoltageControlledGenerator>`
     - child
     - electricity
     - ``mm.VoltageControlledGenerator``
     - ``mx.create_el_child``
   * - :ref:`Shunt <components/electricity:PowerShunt>`
     - child
     - electricity
     - ``mm.PowerShunt``
     - ``mx.create_el_child``
   * - :ref:`Electric storage <components/electricity:ElectricStorage>`
     - child
     - electricity
     - ``mm.ElectricStorage``
     - ``mx.create_el_child``
   * - :ref:`Junction <components/gas:Junction>`
     - node
     - gas, water
     - ``mm.Junction``
     - ``mx.create_gas_junction``, ``mx.create_water_junction``
   * - :ref:`Gas pipe <components/gas:GasPipe>`
     - branch
     - gas
     - ``mm.GasPipe``
     - ``mx.create_gas_pipe``
   * - :ref:`Compressor <components/gas:GasCompressor>`
     - branch
     - gas
     - ``mm.GasCompressor``
     - ``mx.create_compressor``
   * - :ref:`External hydraulic grid <components/gas:ExtHydrGrid>`
     - child
     - gas, water
     - ``mm.ExtHydrGrid``
     - ``mx.create_gas_ext_grid``, ``mx.create_water_ext_grid``
   * - :ref:`Source <components/gas:Source>`
     - child
     - gas, water
     - ``mm.Source``
     - ``mx.create_gas_source``, ``mx.create_water_source``
   * - :ref:`Sink <components/gas:Sink>`
     - child
     - gas, water
     - ``mm.Sink``
     - ``mx.create_gas_sink``, ``mx.create_water_sink``
   * - :ref:`Gas storage <components/gas:GasStorage>`
     - child
     - gas
     - ``mm.GasStorage``
     - ``mx.create_gas_child``
   * - :ref:`Water pipe <components/water:WaterPipe>`
     - branch
     - water
     - ``mm.WaterPipe``
     - ``mx.create_water_pipe``
   * - :ref:`Heat exchanger <components/water:HeatExchanger>`
     - branch
     - water
     - ``mm.HeatExchangerLoad``, ``mm.HeatExchangerGenerator``
     - ``mx.create_heat_exchanger``
   * - :ref:`Passive heat exchanger <components/water:PassiveHeatExchanger>`
     - branch
     - water
     - ``mm.PassiveHeatExchangerLoad``, ``mm.PassiveHeatExchangerGenerator``
     - ``mx.create_passive_heat_exchanger``
   * - :ref:`Consumption point <components/water:ConsumeHydrGrid>`
     - child
     - water, gas
     - ``mm.ConsumeHydrGrid``
     - ``mx.create_consume_hydr_grid``
   * - :ref:`Heat generator and heat load <components/water:HeatGenerator and HeatLoad>`
     - child
     - water
     - ``mm.HeatGenerator``, ``mm.HeatLoad``
     - ``mx.create_heat_generator``, ``mx.create_heat_load``
   * - :ref:`Thermal storage <components/water:ThermalStorage>`
     - child
     - water
     - ``mm.ThermalStorage``
     - ``mx.create_water_child``
   * - :ref:`Power-to-heat <components/coupling:PowerToHeat>`
     - compound
     - electricity, water
     - ``mm.PowerToHeat``
     - ``mx.create_p2h``
   * - :ref:`Gas-to-heat <components/coupling:GasToHeat>`
     - compound
     - gas, water
     - ``mm.GasToHeat``
     - ``mx.create_g2h``
   * - :ref:`CHP <components/coupling:CHP>`
     - compound
     - gas, electricity, water
     - ``mm.CHP``
     - ``mx.create_chp``
   * - :ref:`Gas-to-power <components/coupling:GasToPower>`
     - branch
     - gas, electricity
     - ``mm.GasToPower``
     - ``mx.create_g2p``
   * - :ref:`Power-to-gas <components/coupling:PowerToGas>`
     - branch
     - electricity, gas
     - ``mm.PowerToGas``
     - ``mx.create_p2g``
   * - :ref:`Node-based couplers <components/coupling:Node-based couplers (HG variants)>`
     - branch, compound
     - mixed
     - ``mm.PowerToHeatHG``, ``mm.GasToHeatHG``, ``mm.CHPHG``
     - ``mx.create_p2h_hg``, ``mx.create_g2h_hg``, ``mx.create_chp_hg``
   * - :ref:`Grids <components/grids:Grids>`
     - grid
     - all
     - ``mm.PowerGrid``, ``mm.GasGrid``, ``mm.WaterGrid`` and the ``mm.create_*_grid`` factories
     - none, grids are passed to ``mm.Network`` or as ``grid=``
   * - :ref:`Grid-forming generator <components/extensions:GridFormingGenerator>`
     - child
     - electricity
     - ``mm.GridFormingGenerator``
     - ``mx.create_grid_forming_generator``
   * - :ref:`Grid-forming source <components/extensions:GridFormingSource>`
     - child
     - gas, water
     - ``mm.GridFormingSource``
     - ``mx.create_grid_forming_source``
   * - :ref:`Network aspects <components/extensions:Network aspects>`
     - aspect
     - gas, water
     - ``mm.GasLinepack``, ``mm.LumpedThermalCapacitance``
     - none, use ``net.add_extension``
