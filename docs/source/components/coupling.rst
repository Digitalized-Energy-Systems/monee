.. _components-coupling:

========
Coupling
========

:doc:`../concepts/multi_energy` explains each coupler's physics, the supply and
return wiring, the sign of every result column and how ``regulation``
dispatches them. Gas energy converts as ``MW = kg/s * 3.6 * hhv`` with the gas
grid's ``higher_heating_value_kwh_per_kg``.

The three compounds (P2H, G2H, CHP) build a control node, lossless
:class:`~monee.model.multi.GenericTransferBranch` links to the carrier grids
and a subordinate heat exchanger :class:`~monee.model.multi.SubHE` on the
water path from the cold inlet junction to the hot outlet junction. Their
results live in the control node table (``PowerToHeatControlNode``,
``GasToHeatControlNode``, ``CHPControlNode``), and the internal parts appear
in the ``SubHE`` and ``GenericTransferBranch`` tables with
``independent = False``. ``regulation`` is a column of every coupler branch,
control node and exchanger, not only of children.

Internal parts of the compounds
-------------------------------

The three exchanger-based compounds put a :class:`~monee.model.multi.SubHE`
branch on one leg of the water path and a lossless transfer link on the
other. P2H places the exchanger on the inlet side, from the cold junction to
the control node, and the link from the control node to the hot junction.
G2H and CHP place the link from the cold junction to the control node and
the exchanger from the control node to the hot junction. The exchanger is a
:ref:`heat exchanger <components/water:HeatExchanger>` whose duty the compound
sizes, so its row reads like an exchanger generator:

.. results: SubHE

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw``
     - solved
     - Heat delivered to the water in MW, negative: -0.1 for a 0.1 MW P2H.
   * - ``q_mw_set``
     - solved
     - The served duty (``regulation`` times the setpoint) as a solved variable, positive, so ``q_mw = -q_mw_set``.
   * - ``mass_flow_design_kgs``
     - solved
     - Flow sized from the served duty at the 30 K spread.
   * - ``mass_flow_kgs``
     - solved
     - Signed flow through the exchanger in kg/s, negative in the direction from the cold side to the hot side.
   * - ``mass_flow_pos_kgs``
     - solved
     - Reverse flow half, 0.
   * - ``mass_flow_neg_kgs``
     - solved
     - Forward flow half, the magnitude.
   * - ``t_from_pu``
     - solved
     - Inlet temperature in per unit.
   * - ``t_to_pu``
     - solved
     - Outlet temperature in per unit, above the inlet.
   * - ``direction``
     - internal
     - Direction indicator.

After a solve the ``SubHE`` table also carries ``q_mw_delivered``,
``t_in_pu``, ``t_out_pu`` and ``mass_flow_mag_kgs``.

The :class:`~monee.model.multi.GenericTransferBranch` rows are the lossless
links between the control node and the carrier grids. Before a solve they
carry only ``on_off``; after a solve the electrical links report
``p_from_mw``, ``p_to_mw``, ``q_from_mvar`` and ``q_to_mvar`` (equal at both
ends, since the link is lossless), the hydraulic links ``mass_flow_pos_kgs``,
``mass_flow_neg_kgs``, ``t_from_pu`` and ``t_to_pu``, and the gas links of
G2H, CHP and CHPHG additionally ``gas_mass_flow``, an alias of
``mass_flow_pos_kgs`` that reads 0 on compound-built links; the gas draw is
``mass_flow_neg_kgs``. Columns that do not apply to a link's carrier are
``NaN``. All of these rows have ``independent = False``.

PowerToHeat
-----------

Heat pump or electric boiler. Draws ``heat_energy_mw / efficiency`` at the bus
and heats the water passing from ``heat_node_id`` to ``heat_return_node_id``.

.. component: PowerToHeat | create_p2h

.. code-block:: python

   mm.PowerToHeat(heat_energy_mw, diameter_m, temperature_ext_k, efficiency, q_mvar_setpoint=0,
                  regulation=1)
   net.compound(mm.PowerToHeat(...), power_node_id=bus, heat_node_id=cold, heat_return_node_id=hot)
   mx.create_p2h(network, power_node_id, heat_node_id=None, heat_return_node_id=None,
                 heat_energy_mw=None, diameter_m=None, efficiency=None, temperature_ext_k=293,
                 q_mvar_setpoint=0, regulation=1, constraints=None, cold_node_id=None,
                 hot_node_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``heat_energy_mw``
     - required
     - Thermal output in MW, positive.
   * - ``diameter_m``
     - required
     - Required by the constructor and kept on the compound, but not read by
       the internal exchanger, which has no pipe geometry; it does not affect
       the solve.
   * - ``temperature_ext_k``
     - required, express 293
     - Kept on the compound, not read by the internal exchanger, which has no
       ambient loss; it does not affect the solve.
   * - ``efficiency``
     - required
     - Electric-to-heat conversion efficiency; a heat pump uses its COP.
   * - ``q_mvar_setpoint``
     - 0
     - Reactive power drawn at the bus in Mvar.
   * - ``regulation``
     - 1
     - Dispatch factor: 1 full output, 0 off.
   * - ``power_node_id``
     - required
     - Bus the unit draws from.
   * - ``heat_node_id``
     - required
     - Cold inlet junction of the internal exchanger (alias ``cold_node_id``).
   * - ``heat_return_node_id``
     - required
     - Hot outlet junction (alias ``hot_node_id``).

.. results: PowerToHeatControlNode

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``el_mw``
     - solved
     - Electric draw at the bus in MW, positive: 0.111 for a 0.1 MW unit at 0.9 efficiency.
   * - ``heat_mw``
     - solved
     - Heat delivered to the water in MW in generator sign, so -0.1 for that unit at full regulation.
   * - ``load_p_mw``
     - derived
     - Nameplate draw ``heat_energy_mw / efficiency``, echoed unchanged.
   * - ``load_q_mvar``
     - input
     - Reactive setpoint in Mvar.
   * - ``efficiency``
     - input
     - Conversion efficiency.
   * - ``t_pu``
     - solved
     - Water temperature at the control node in per unit of ``t_ref_k``, the hot side after heating.
   * - ``t_k``
     - solved
     - The same temperature in K.
   * - ``pressure_pu``
     - solved
     - Pressure at the control node in pu.
   * - ``pressure_squared_pu``
     - solved
     - Squared pressure.
   * - ``mass_flow_kgs``
     - internal
     - Child mass-flow sum of the control node; no child attaches there, so it stays at its seed.

The compound also adds a ``SubHE`` row (the internal exchanger, see
:ref:`Internal parts of the compounds <components/coupling:Internal parts of the compounds>`)
and ``GenericTransferBranch`` rows.

GasToHeat
---------

Gas boiler. Draws ``heat_energy_mw / (efficiency * 3.6 * hhv)`` kg/s of gas
and heats the water passing from ``heat_node_id`` to ``heat_return_node_id``.

.. component: GasToHeat | create_g2h

.. code-block:: python

   mm.GasToHeat(heat_energy_mw, diameter_m, temperature_ext_k, efficiency, regulation=1)
   net.compound(mm.GasToHeat(...), gas_node_id=junction, heat_node_id=cold, heat_return_node_id=hot)
   mx.create_g2h(network, gas_node_id, heat_node_id=None, heat_return_node_id=None,
                 heat_energy_mw=None, diameter_m=None, efficiency=None, temperature_ext_k=293,
                 regulation=1, constraints=None, cold_node_id=None, hot_node_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``heat_energy_mw``
     - required
     - Thermal output in MW, positive.
   * - ``diameter_m``
     - required
     - Required by the constructor and kept on the compound, but not read by
       the internal exchanger, which has no pipe geometry; it does not affect
       the solve.
   * - ``temperature_ext_k``
     - required, express 293
     - Kept on the compound, not read by the internal exchanger, which has no
       ambient loss; it does not affect the solve.
   * - ``efficiency``
     - required
     - Gas-to-heat conversion efficiency.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``gas_node_id``
     - required
     - Junction the unit draws gas from.
   * - ``heat_node_id``
     - required
     - Cold inlet junction (alias ``cold_node_id``).
   * - ``heat_return_node_id``
     - required
     - Hot outlet junction (alias ``hot_node_id``).

.. results: GasToHeatControlNode

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``heat_mw``
     - solved
     - Heat delivered to the water in MW in generator sign, -0.05 for a 0.05 MW boiler at full regulation.
   * - ``gas_mass_flow_kgs``
     - derived
     - Gas setpoint in kg/s, ``heat_energy_mw / (efficiency * 3.6 * hhv)``, about 0.0013 for 0.05 MW at 0.9 efficiency on the default gas grid; the served draw is ``regulation`` times it.
   * - ``efficiency_heat``
     - input
     - Conversion efficiency.
   * - ``pressure_pa``
     - report
     - Control node pressure in Pa.
   * - ``t_pu``
     - solved
     - Water temperature at the control node in per unit of ``t_ref_k``, the cold inlet before heating; only the P2H control node sits on the hot side.
   * - ``t_k``
     - solved
     - The same temperature in K.
   * - ``pressure_pu``
     - solved
     - Pressure at the control node in pu.
   * - ``pressure_squared_pu``
     - solved
     - Squared pressure.
   * - ``mass_flow_kgs``
     - internal
     - Child mass-flow sum of the control node; no child attaches there, so it stays at its seed.

The compound also adds a ``SubHE`` row and ``GenericTransferBranch`` rows.

CHP
---

Combined heat and power. Consumes a gas setpoint and produces
``efficiency_power * m * 3.6 * hhv`` MW of electricity at the bus and
``efficiency_heat * m * 3.6 * hhv`` MW of heat into the water passing from
``heat_node_id`` to ``heat_return_node_id``.

.. component: CHP | create_chp

.. code-block:: python

   mm.CHP(diameter_m, efficiency_power, efficiency_heat, mass_flow_setpoint_kgs, q_mvar_setpoint=0,
          temperature_ext_k=293, regulation=1)
   net.compound(mm.CHP(...), gas_node_id=junction, heat_node_id=cold, heat_return_node_id=hot,
                power_node_id=bus)
   mx.create_chp(network, power_node_id, heat_node_id=None, heat_return_node_id=None,
                 gas_node_id=None, diameter_m=None, efficiency_power=None, efficiency_heat=None,
                 mass_flow_setpoint_kgs=None, regulation=1, constraints=None,
                 remove_existing_branch=False, cold_node_id=None, hot_node_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``diameter_m``
     - required
     - Required by the constructor and kept on the compound, but not read by
       the internal exchanger, which has no pipe geometry; it does not affect
       the solve.
   * - ``efficiency_power``
     - required
     - Fraction of the gas energy converted to electricity.
   * - ``efficiency_heat``
     - required
     - Fraction of the gas energy converted to heat.
   * - ``mass_flow_setpoint_kgs``
     - required
     - Gas consumption in kg/s, positive.
   * - ``q_mvar_setpoint``
     - 0
     - Stored on the compound but not forwarded to its control node, so a CHP
       always injects 0 Mvar. CHPHG honours the value.
   * - ``temperature_ext_k``
     - 293
     - Kept on the compound, not read by the internal exchanger, which has no
       ambient loss; it does not affect the solve.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``power_node_id``
     - required
     - Bus receiving the electrical output.
   * - ``heat_node_id``
     - required
     - Cold inlet junction (alias ``cold_node_id``).
   * - ``heat_return_node_id``
     - required
     - Hot outlet junction (alias ``hot_node_id``).
   * - ``gas_node_id``
     - required
     - Junction the unit draws gas from.
   * - ``remove_existing_branch``
     - ``False``
     - Express only: drop a branch already connecting the two heat junctions
       before inserting the exchanger.

.. results: CHPControlNode

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``el_mw``
     - solved
     - Electric output at the bus in MW in generator sign: ``-efficiency_power * m * 3.6 * hhv``, about -0.025 for 0.002 kg/s at 0.3 efficiency on the default gas grid.
   * - ``heat_mw``
     - solved
     - Heat delivered to the water in MW in generator sign, about -0.042 for the same unit at 0.5 heat efficiency.
   * - ``gas_mass_flow_kgs``
     - input
     - Gas setpoint in kg/s, echoed; the served draw is ``regulation`` times it.
   * - ``gen_q_mvar``
     - input
     - Always 0; the compound does not forward ``q_mvar_setpoint``.
   * - ``efficiency_power``
     - input
     - Electric efficiency.
   * - ``efficiency_heat``
     - input
     - Heat efficiency.
   * - ``pressure_pa``
     - report
     - Control node pressure in Pa.
   * - ``t_pu``
     - solved
     - Water temperature at the control node in per unit of ``t_ref_k``, the cold inlet before heating; only the P2H control node sits on the hot side.
   * - ``t_k``
     - solved
     - The same temperature in K.
   * - ``pressure_pu``
     - solved
     - Pressure at the control node in pu.
   * - ``pressure_squared_pu``
     - solved
     - Squared pressure.
   * - ``mass_flow_kgs``
     - internal
     - Child mass-flow sum of the control node; no child attaches there, so it stays at its seed.

The compound also adds a ``SubHE`` row and ``GenericTransferBranch`` rows.

GasToPower
----------

Gas turbine or engine as a single branch from a gas junction to a bus.
Produces ``p_mw_setpoint`` and draws ``p_mw_setpoint / (efficiency * 3.6 *
hhv)`` kg/s of gas.

.. component: GasToPower | create_g2p

.. code-block:: python

   mm.GasToPower(efficiency, p_mw_setpoint, q_mvar_setpoint=0, regulation=1)
   net.branch(mm.GasToPower(...), gas_junction_id, bus_id)
   mx.create_g2p(network, from_node_id, to_node_id, efficiency, p_mw_setpoint, q_mvar_setpoint=0,
                 regulation=1, constraints=None, grid=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``efficiency``
     - required
     - Gas-to-electricity conversion efficiency.
   * - ``p_mw_setpoint``
     - required
     - Electrical output in MW, positive.
   * - ``q_mvar_setpoint``
     - 0
     - Reactive output in Mvar.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``from_node_id``
     - required
     - Express: the gas junction.
   * - ``to_node_id``
     - required
     - Express: the bus.

.. results: GasToPower

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_to_mw``
     - solved
     - Electric injection at the bus in MW in generator sign, -0.05 for a 0.05 MW unit at full regulation.
   * - ``from_mass_flow_kgs``
     - solved
     - Gas drawn at the junction in kg/s, positive: ``p / (efficiency * 3.6 * hhv)``, about 0.003 for 0.05 MW at 0.4 efficiency on the default gas grid.
   * - ``gas_mass_flow_kgs``
     - solved
     - The same gas draw.
   * - ``el_mw``
     - input
     - Electric setpoint with the generation sign, -0.05 for that unit.
   * - ``q_to_mvar``
     - input
     - Reactive setpoint, generation sign.
   * - ``efficiency``
     - input
     - Conversion efficiency.

PowerToGas
----------

Electrolyser as a single branch from a bus to a gas junction. Produces
``mass_flow_setpoint_kgs`` of gas and draws ``m * 3.6 * hhv / efficiency`` MW.

.. component: PowerToGas | create_p2g

.. code-block:: python

   mm.PowerToGas(efficiency, mass_flow_setpoint_kgs, consume_q_mvar_setpoint=0, regulation=1)
   net.branch(mm.PowerToGas(...), bus_id, gas_junction_id)
   mx.create_p2g(network, from_node_id, to_node_id, efficiency, mass_flow_setpoint_kgs,
                 consume_q_mvar_setpoint=0, regulation=1, constraints=None, grid=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``efficiency``
     - required
     - Electricity-to-gas conversion efficiency.
   * - ``mass_flow_setpoint_kgs``
     - required
     - Produced gas in kg/s, positive.
   * - ``consume_q_mvar_setpoint``
     - 0
     - Reactive power drawn at the bus in Mvar.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``from_node_id``
     - required
     - Express: the bus.
   * - ``to_node_id``
     - required
     - Express: the gas junction.

.. results: PowerToGas

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_from_mw``
     - solved
     - Electric draw at the bus in MW, positive: ``m * 3.6 * hhv / efficiency``, about 0.071 for 0.001 kg/s at 0.6 efficiency on the default gas grid.
   * - ``to_mass_flow_kgs``
     - solved
     - Gas injected at the junction in kg/s in generator sign, -0.001 for that unit.
   * - ``el_mw``
     - solved
     - The same electric draw.
   * - ``gas_mass_flow_kgs``
     - input
     - Gas setpoint with the generation sign, -0.001.
   * - ``q_from_mvar``
     - input
     - Reactive setpoint drawn at the bus.
   * - ``efficiency``
     - input
     - Conversion efficiency.

Node-based couplers (HG variants)
---------------------------------

Lighter variants that inject heat through the junction heat balance
(``q_mw_heat``) instead of an internal heat exchanger branch, so they need no
return junction and no diameter: :ref:`concepts/multi_energy:Node-based heat
couplers (HG variants)`. ``PowerToHeatHG`` and ``GasToHeatHG`` are single
branches ending at a water junction. ``CHPHG`` is a compound that attaches a
:class:`~monee.model.multi.SubHG` child at the heat junction and reports in the
``CHPHGControlNode`` table.

.. component: PowerToHeatHG | create_p2h_hg

.. code-block:: python

   mm.PowerToHeatHG(heat_energy_mw, efficiency, q_mvar_setpoint=0, regulation=1)
   net.branch(mm.PowerToHeatHG(...), bus_id, water_junction_id)
   mx.create_p2h_hg(network, power_node_id, heat_node_id, heat_energy_mw, efficiency,
                    q_mvar_setpoint=0, regulation=1, constraints=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``heat_energy_mw``
     - required
     - Thermal output in MW, positive; the bus draw is ``heat_energy_mw /
       efficiency``.
   * - ``efficiency``
     - required
     - Electric-to-heat conversion efficiency.
   * - ``q_mvar_setpoint``
     - 0
     - Reactive power drawn at the bus in Mvar.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``power_node_id``
     - required
     - Express: the bus.
   * - ``heat_node_id``
     - required
     - Express: the water junction receiving the heat.

.. results: PowerToHeatHG

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw_heat``
     - solved
     - Heat injected at the water junction in MW in generator sign, -0.02 for a 0.02 MW unit at full regulation.
   * - ``p_from_mw``
     - solved
     - Electric draw at the bus in MW, positive, ``heat_energy_mw / efficiency``.
   * - ``load_p_mw``
     - derived
     - Nameplate draw, echoed.
   * - ``heat_energy_mw``
     - input
     - Heat setpoint in MW, positive.
   * - ``load_q_mvar``
     - input
     - Reactive setpoint in Mvar.
   * - ``q_from_mvar``
     - input
     - Reactive draw at the bus.
   * - ``efficiency``
     - input
     - Conversion efficiency.

.. component: GasToHeatHG | create_g2h_hg

.. code-block:: python

   mm.GasToHeatHG(heat_energy_mw, efficiency, regulation=1)
   net.branch(mm.GasToHeatHG(...), gas_junction_id, water_junction_id)
   mx.create_g2h_hg(network, gas_node_id, heat_node_id, heat_energy_mw, efficiency, regulation=1,
                    constraints=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``heat_energy_mw``
     - required
     - Thermal output in MW, positive.
   * - ``efficiency``
     - required
     - Gas-to-heat conversion efficiency.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``gas_node_id``
     - required
     - Express: the gas junction.
   * - ``heat_node_id``
     - required
     - Express: the water junction receiving the heat.

.. results: GasToHeatHG

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw_heat``
     - solved
     - Heat injected at the water junction in MW in generator sign, -0.02 for a 0.02 MW unit at full regulation.
   * - ``from_mass_flow_kgs``
     - solved
     - Gas drawn at the junction in kg/s, positive, about 0.0005 for 0.02 MW at 0.9 efficiency on the default gas grid.
   * - ``gas_mass_flow_kgs``
     - derived
     - Gas setpoint in kg/s, the same number; 0 until a solve runs, since it needs the grid heating value.
   * - ``heat_energy_mw``
     - input
     - Heat setpoint stored with the generation sign, -0.02.
   * - ``efficiency``
     - input
     - Conversion efficiency.

.. component: CHPHG | create_chp_hg

.. code-block:: python

   mm.CHPHG(efficiency_power, efficiency_heat, mass_flow_setpoint_kgs, q_mvar_setpoint=0,
            regulation=1)
   net.compound(mm.CHPHG(...), gas_node_id=junction, heat_node_id=water_junction, power_node_id=bus)
   mx.create_chp_hg(network, power_node_id, heat_node_id, gas_node_id, efficiency_power,
                    efficiency_heat, mass_flow_setpoint_kgs, regulation=1, constraints=None,
                    name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``efficiency_power``
     - required
     - Fraction of the gas energy converted to electricity.
   * - ``efficiency_heat``
     - required
     - Fraction of the gas energy converted to heat.
   * - ``mass_flow_setpoint_kgs``
     - required
     - Gas consumption in kg/s, positive.
   * - ``q_mvar_setpoint``
     - 0
     - Reactive power injected at the bus in Mvar.
   * - ``regulation``
     - 1
     - Dispatch factor.
   * - ``power_node_id``
     - required
     - Bus receiving the electrical output.
   * - ``heat_node_id``
     - required
     - Water junction receiving the heat.
   * - ``gas_node_id``
     - required
     - Junction the unit draws gas from.

.. results: CHPHGControlNode

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``el_mw``
     - solved
     - Electric output at the bus in MW in generator sign, ``-efficiency_power * m * 3.6 * hhv``.
   * - ``heat_mw``
     - solved
     - Heat output in MW in generator sign, the same number the ``SubHG`` row carries as ``q_mw_heat``.
   * - ``gas_mass_flow_kgs``
     - input
     - Gas setpoint in kg/s, echoed.
   * - ``gen_q_mvar``
     - input
     - Reactive setpoint in Mvar.
   * - ``efficiency_power``
     - input
     - Electric efficiency.
   * - ``efficiency_heat``
     - input
     - Heat efficiency.
   * - ``pressure_pa``
     - internal
     - Unconstrained placeholder inherited from Junction; CHPHG has no water side, so ignore it.
   * - ``t_pu``
     - internal
     - Unconstrained placeholder inherited from Junction; CHPHG has no water side, so ignore it.
   * - ``t_k``
     - internal
     - Unconstrained placeholder inherited from Junction; CHPHG has no water side, so ignore it.
   * - ``pressure_pu``
     - internal
     - Unconstrained placeholder inherited from Junction; CHPHG has no water side, so ignore it.
   * - ``pressure_squared_pu``
     - internal
     - Unconstrained placeholder inherited from Junction; CHPHG has no water side, so ignore it.
   * - ``mass_flow_kgs``
     - internal
     - Child mass-flow sum of the control node; no child attaches there, so it stays at its seed.

The compound also adds a ``SubHG`` row at the heat junction:

.. results: SubHG

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw_heat``
     - solved
     - Heat injected at the junction in MW in generator sign, -0.021 for 0.001 kg/s at 0.5 heat efficiency on the default gas grid.
