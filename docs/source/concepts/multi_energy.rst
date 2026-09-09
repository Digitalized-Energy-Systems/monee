========================
Multi-energy coupling
========================

A multi-energy system (MES) combines two or more energy carriers (typically
electricity, gas, and heat) into a single network. monee models the coupling
points between these carriers as specialised compound components that sit
at the intersection of grids.

.. note::

   Compound components are created automatically by the express API functions
   below. You do not need to assemble them manually unless you are writing a
   custom coupling model.

----

Available coupling components
=============================

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Power-to-Heat (P2H)
      :shadow: sm

      **EL → Water**

      Heat pump or electric boiler. Draws electricity from a bus and injects
      thermal energy into a district heating supply pipe.

      :func:`monee.express.create_p2h` →
      :class:`~monee.model.multi.PowerToHeat`

   .. grid-item-card:: Power-to-Gas (P2G)
      :shadow: sm

      **EL → Gas**

      Electrolysis unit. Converts electricity to gas (e.g. green hydrogen).

      :func:`monee.express.create_p2g` →
      :class:`~monee.model.multi.PowerToGas`

   .. grid-item-card:: Gas-to-Power (G2P)
      :shadow: sm

      **Gas → EL**

      Gas turbine or engine. Converts gas to electricity.

      :func:`monee.express.create_g2p` →
      :class:`~monee.model.multi.GasToPower`

   .. grid-item-card:: Gas-to-Heat (G2H)
      :shadow: sm

      **Gas → Water**

      Gas boiler. Converts gas to district heat without generating electricity.

      :func:`monee.express.create_g2h` →
      :class:`~monee.model.multi.GasToHeat`

   .. grid-item-card:: Combined Heat and Power (CHP)
      :shadow: sm

      **Gas → EL + Water**

      Simultaneously produces electricity and heat from gas. Connects to
      three grids.

      :func:`monee.express.create_chp` →
      :class:`~monee.model.multi.CHP`

   .. grid-item-card:: Heat exchanger
      :shadow: sm

      **Water ↔ Water**

      Transfers thermal energy between two water circuits without mass
      transfer.

      :func:`monee.express.create_heat_exchanger` →
      :class:`~monee.model.branch.HeatExchanger`

----

Power-to-Heat (P2H)
===================

A P2H unit (e.g. a heat pump or electric boiler) draws electrical power from
a bus and heats the water passing through its internal heat exchanger. The
water enters that exchanger at ``heat_node_id`` and leaves it at
``heat_return_node_id``, so ``heat_return_node_id`` is the hot outlet and
``heat_node_id`` the cold inlet. The two parameters name the junctions of a
supply/return pair; they do not say which of the two is hot. Pass
``cold_node_id`` and ``hot_node_id`` instead if you prefer the wiring to state
it, as the example below does. The same holds for
:func:`~monee.express.create_g2h` and :func:`~monee.express.create_chp`, and
:ref:`District heating loops <dhs-loops>` has the full loop pattern.

The example builds a closed loop and solves it: a consumer takes 0.5 MW out of
the supply junction, the water returns to the plant reference at 330 K and the
P2H heats it back up.

.. testcode::

    import monee.express as mx
    import monee.model as mm
    from monee import run_energy_flow

    net = mx.create_multi_energy_network()

    # Electricity side
    bus_el = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus_el)

    # Heating side: return junction, supply junction, plant reference on the
    # return side and a consumer that carries the water back to it.
    junc_return = mx.create_water_junction(net)
    junc_supply = mx.create_water_junction(net)
    mx.create_water_ext_grid(net, junc_return, t_k=330)
    mx.create_passive_heat_exchanger(
        net, junc_supply, junc_return, q_mw=0.5, diameter_m=0.1
    )

    # Couple electricity to heat
    mx.create_p2h(
        net,
        power_node_id=bus_el,
        cold_node_id=junc_return,  # = heat_node_id, the exchanger inlet
        hot_node_id=junc_supply,   # = heat_return_node_id, the heated outlet
        heat_energy_mw=0.5,  # thermal output setpoint [MW]
        diameter_m=0.1,      # heat-exchange pipe diameter
        efficiency=0.95,     # electrical-to-thermal efficiency
    )

    result = run_energy_flow(net)
    junctions = result.get(mm.Junction)
    control = result.get(mm.PowerToHeatControlNode)

    print("slack import [MW]:", round(-result.get(mm.ExtPowerGrid)["p_mw"].iloc[0], 4))
    print("P2H el_mw    [MW]:", round(control["el_mw"].iloc[0], 4))
    print("P2H heat_mw  [MW]:", round(control["heat_mw"].iloc[0], 4))
    print("return, supply [K]:", round(junctions["t_k"].iloc[0], 1),
          round(junctions["t_k"].iloc[1], 1))

.. testoutput::

    slack import [MW]: 0.5263
    P2H el_mw    [MW]: 0.5263
    P2H heat_mw  [MW]: -0.5
    return, supply [K]: 330.0 360.0

The balance closes: the unit draws ``0.5 / 0.95 = 0.5263`` MW of electricity,
which the slack bus has to import, and delivers 0.5 MW of heat, which raises
the water from 330 K to 360 K. ``el_mw`` counts positive as consumption and
``heat_mw`` negative as injection; the sign table in
:ref:`District heating loops <dhs-loops>` lists the convention per coupler.

Key parameters:

- ``heat_energy_mw``: thermal output setpoint in megawatts. The solver
  derives the required electrical input as ``heat_energy_mw / efficiency``.
- ``efficiency``: ratio of thermal output to electrical input (≤ 1 for
  resistive heating, > 1 for heat pumps modelled with a fixed COP).
- ``diameter_m``: internal pipe diameter of the connecting heat-exchange
  branch.

Power-to-Gas (P2G)
==================

A P2G unit (electrolysis) converts electricity into gas (e.g. green hydrogen).

.. testcode::

    import monee.express as mx

    net_p2g  = mx.create_multi_energy_network()
    bus_el   = mx.create_bus(net_p2g)
    junc_gas = mx.create_gas_junction(net_p2g)

    mx.create_p2g(
        net_p2g,
        from_node_id=bus_el,
        to_node_id=junc_gas,
        efficiency=0.7,               # electrical-to-chemical efficiency
        mass_flow_setpoint_kgs=0.05,      # target gas production [kg/s]
    )

P2G is a plain two-endpoint branch coupler; its conversion loss is exposed
as :meth:`~monee.model.multi.PowerToGas.loss_percent`, which returns
``1 - efficiency``.

Gas-to-Power (G2P)
==================

A G2P unit (gas turbine, gas engine) converts gas to electricity.

.. testcode::

    import monee.express as mx

    net_g2p  = mx.create_multi_energy_network()
    junc_gas = mx.create_gas_junction(net_g2p)
    bus_el   = mx.create_bus(net_g2p)

    mx.create_g2p(
        net_g2p,
        from_node_id=junc_gas,
        to_node_id=bus_el,
        efficiency=0.40,         # gas-to-electrical efficiency
        p_mw_setpoint=2.0,       # target electrical output [MW]
        q_mvar_setpoint=0.0,
    )

Like P2G, G2P is a branch coupler and reports its conversion loss via
:meth:`~monee.model.multi.GasToPower.loss_percent` (= ``1 - efficiency``).

Combined Heat and Power (CHP)
==============================

A CHP unit simultaneously produces electricity and heat from gas. It connects
to three grids: gas (fuel input), electricity (power output), and district
heating (heat output).

.. testcode::

    import monee.express as mx

    net = mx.create_multi_energy_network()

    # Gas grid, with a pressure reference
    junc_gas = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, junc_gas)

    # Electricity grid
    bus_el = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus_el)

    # Heating loop: plant reference on the return side, one consumer sized to
    # the CHP's thermal output (0.1 kg/s of L-gas at 45 % is about 1.9 MW).
    junc_heat_return = mx.create_water_junction(net)
    junc_heat_supply = mx.create_water_junction(net)
    mx.create_water_ext_grid(net, junc_heat_return, t_k=330)
    mx.create_passive_heat_exchanger(
        net, junc_heat_supply, junc_heat_return, q_mw=1.9, diameter_m=0.15
    )

    # CHP coupling
    mx.create_chp(
        net,
        power_node_id=bus_el,
        cold_node_id=junc_heat_return,  # = heat_node_id, the exchanger inlet
        hot_node_id=junc_heat_supply,   # = heat_return_node_id, heated outlet
        gas_node_id=junc_gas,
        diameter_m=0.15,
        efficiency_power=0.35,   # gas → electricity
        efficiency_heat=0.45,    # gas → heat
        mass_flow_setpoint_kgs=0.1,  # gas consumption [kg/s]
    )

Solved, this network reports ``el_mw`` -1.4856 and ``heat_mw`` -1.9100 on the
CHP control node: both are generation, and the supply junction settles at
360 K against the 330 K return.

Key parameters:

- ``efficiency_power`` and ``efficiency_heat``: individual efficiencies for
  electrical and thermal output. Their sum must not exceed 1 (total fuel
  utilisation).
- ``mass_flow_setpoint_kgs``: gas consumption setpoint in kg/s.
- ``regulation``: a factor in [0, 1] that scales all outputs. Set as a
  :class:`~monee.model.core.Var` to let the optimiser dispatch the unit.

Compound couplers such as CHP also implement ``set_active(flag)``, which
:meth:`~monee.model.network.Network.deactivate` uses to switch the whole unit
off cleanly (zeroing gas flow and regulation) and to restore the previous
setpoints on reactivation.

Gas-to-Heat (G2H)
=================

A gas boiler converts gas to district heat without producing electricity.

.. testcode::

    import monee.express as mx

    net_g2h          = mx.create_multi_energy_network()
    junc_gas         = mx.create_gas_junction(net_g2h)
    junc_heat_return = mx.create_water_junction(net_g2h)
    junc_heat_supply = mx.create_water_junction(net_g2h)

    mx.create_g2h(
        net_g2h,
        gas_node_id=junc_gas,
        cold_node_id=junc_heat_return,  # = heat_node_id, the exchanger inlet
        hot_node_id=junc_heat_supply,   # = heat_return_node_id, heated outlet
        heat_energy_mw=0.5,  # thermal output [MW]
        diameter_m=0.1,
        efficiency=0.90,
    )

Node-based heat couplers (HG variants)
======================================

The heat-producing couplers above (CHP, G2H, P2H) deliver their thermal output
through an internal heat-exchanger branch that bridges the supply and return
sides of the heating network. That is why they require a ``heat_return_node_id``
and a ``diameter_m``. Each of them has a node-based "HG" variant (heat
generator) that injects the thermal power directly at a single heat junction.
The heat appears as a ``q_mw_heat`` term that the
:class:`~monee.model.node.Junction` heat balance picks up, exactly like a
node-level :class:`~monee.model.child.HeatGenerator` child.

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: CHP-HG
      :shadow: sm

      **Gas → EL + Water**

      CHP variant. Compound with an internal control node; injects heat via a
      hidden ``SubHG`` child at the heat node.

      :func:`monee.express.create_chp_hg` →
      :class:`~monee.model.multi.CHPHG`

   .. grid-item-card:: P2H-HG
      :shadow: sm

      **EL → Water**

      Power-to-Heat variant. A plain two-endpoint multi-grid branch from a bus
      to a heat junction.

      :func:`monee.express.create_p2h_hg` →
      :class:`~monee.model.multi.PowerToHeatHG`

   .. grid-item-card:: G2H-HG
      :shadow: sm

      **Gas → Water**

      Gas boiler variant. A plain two-endpoint multi-grid branch from a gas
      junction to a heat junction.

      :func:`monee.express.create_g2h_hg` →
      :class:`~monee.model.multi.GasToHeatHG`

The variants differ structurally from their classic counterparts:

- :class:`~monee.model.multi.CHPHG` is still a compound:
  ``create(gas_node, heat_node, power_node)`` wires gas and power through an
  internal control node, but the heat side is a subordinate ``SubHG``
  heat-generator child placed at ``heat_node`` instead of a heat-exchanger
  branch. It therefore needs no ``heat_return_node`` and no
  ``diameter_m``.
- :class:`~monee.model.multi.GasToHeatHG` (``heat_energy_mw, efficiency,
  regulation=1``) and :class:`~monee.model.multi.PowerToHeatHG`
  (``heat_energy_mw, efficiency, q_mvar_setpoint=0, regulation=1``) are plain
  two-endpoint branch couplers, just like G2P and P2G. They withdraw gas or
  electricity at the from-node and carry ``q_mw_heat`` directly on the branch,
  where the junction heat balance at the to-node picks it up. Both also expose
  ``loss_percent()`` (= ``1 - efficiency``).

Prefer the HG variants when

- you want simpler wiring: one heat junction instead of a supply/return
  pair, no heat-exchange pipe geometry to choose; or
- you use the McCormick-DHS formulation
  (:data:`~monee.model.formulation.HEAT_CONVEX_MILP_FORMULATION`) or a
  network built with node-based heat loads (e.g.
  ``node_based_heat_loads=True`` in the :mod:`monee.network` heat builders),
  where they are required: this formulation owns the nodal heat balance and
  only understands node-based heat injections, not heat-exchanger branches.

.. testcode::

    import monee.express as mx

    net_hg = mx.create_multi_energy_network()

    # Gas grid (fuel input)
    junc_gas = mx.create_gas_junction(net_hg)
    mx.create_gas_ext_grid(net_hg, junc_gas)

    # Electricity grid (power output)
    bus_el = mx.create_bus(net_hg)
    mx.create_ext_power_grid(net_hg, bus_el)

    # Heating grid - a single supply junction; no return node needed
    junc_heat = mx.create_water_junction(net_hg)
    mx.create_water_ext_grid(net_hg, junc_heat)

    # CHP coupling, node-based heat injection
    mx.create_chp_hg(
        net_hg,
        power_node_id=bus_el,
        heat_node_id=junc_heat,
        gas_node_id=junc_gas,
        efficiency_power=0.35,   # gas → electricity
        efficiency_heat=0.45,    # gas → heat
        mass_flow_setpoint_kgs=0.1,  # gas consumption [kg/s]
    )

The single-branch variants follow the same pattern as
:func:`~monee.express.create_g2p`:

.. testcode::

    mx.create_p2h_hg(
        net_hg,
        power_node_id=bus_el,
        heat_node_id=junc_heat,
        heat_energy_mw=0.5,  # thermal output setpoint [MW]
        efficiency=0.95,
    )
    mx.create_g2h_hg(
        net_hg,
        gas_node_id=junc_gas,
        heat_node_id=junc_heat,
        heat_energy_mw=0.5,
        efficiency=0.90,
    )

.. note::

   The internal control-node models (``CHPControlNode``, ``CHPHGControlNode``,
   ``GasToHeatControlNode``, ``PowerToHeatControlNode``) and the subordinate
   ``SubHE`` / ``SubHG`` components you may encounter in result dataframes are
   implementation details. They are wired up automatically by the compounds'
   ``create()``; you never instantiate them directly.

Heat exchanger
==============

A heat exchanger transfers thermal energy between two water circuits (for
example between a primary transmission network and a secondary distribution
loop) without mass transfer.

.. testcode::

    import monee.express as mx

    net_hx         = mx.create_multi_energy_network()
    junc_primary   = mx.create_water_junction(net_hx)
    junc_secondary = mx.create_water_junction(net_hx)

    mx.create_heat_exchanger(
        net_hx,
        from_node_id=junc_primary,
        to_node_id=junc_secondary,
        q_mw=0.2,  # heat transfer setpoint [MW]; positive draws heat at from-node
    )

----

.. _dhs-loops:

District heating loops
======================

CHP, G2H and P2H deliver their heat through an internal heat-exchanger branch
that spans two water junctions, so they only make sense inside a closed
hydraulic loop. Three things decide whether that loop solves: the orientation
of the coupler, where the references go, and which kind of exchanger the
consumers use.

Orientation
-----------

Water enters the internal exchanger at ``heat_node_id`` and leaves it at
``heat_return_node_id``, gaining the thermal output on the way. The junction
passed as ``heat_return_node_id`` is therefore the hot one, whatever the name
suggests, and the loop runs

.. code-block:: text

    plant reference -> heat_node_id -> [coupler] -> heat_return_node_id
                    -> consumer -> back to the plant reference

Use the ``cold_node_id`` and ``hot_node_id`` aliases if you want the wiring to
state which side is which; they map onto the same two parameters.

The plant reference
-------------------

Every loop needs a hydraulic reference.
:func:`~monee.express.create_water_ext_grid` (and the generic
:func:`~monee.express.create_ext_hydr_grid`) pins the junction's
``pressure_pu`` and, because ``pin_temperature`` defaults to ``True``, its
``t_k`` as well. Pinning the temperature makes that junction an unbounded heat
source and sink: whatever the couplers fail to deliver, or deliver in excess,
disappears into the reference without a violation. Put it on the cold side, as
the P2H example above does, so it fixes the return temperature and the coupler
still has to raise the water to the supply temperature. A reference on the
supply side silently covers the heat demand and the coupler contributes
nothing. Pass ``pin_temperature=False`` to keep the pressure reference and let
the temperature follow the energy balance.

The other end of the loop needs somewhere for the mass to go: a consumer
branch back to the reference junction, a :func:`~monee.express.create_sink`, or
a :func:`~monee.express.create_consume_hydr_grid` if you also want a fixed
pressure there. A coupler whose outlet is a dead end leaves the water without a
path back, and the solve then lands on a meaningless point or stops converging
altogether.

The gas-fed couplers need a reference on their gas side too. A gas junction fed
only by a :func:`~monee.express.create_source` has none, and a CHP hanging off
it comes back at ``regulation`` 0 with a zero electricity and heat balance
instead of an error, which is what makes that wiring mistake expensive. Use
:func:`~monee.express.create_gas_ext_grid` as in the CHP example above.

Fixed-flow and passive exchangers
---------------------------------

:func:`~monee.express.create_heat_exchanger` is a fixed-flow exchanger: a
numeric ``q_mw`` pins its mass flow to the design flow
``|q_mw| * 1e6 / (c_p * T_delta_design_K)``, a 30 K design spread by default,
scaled by ``regulation``. It prescribes the loop hydraulics instead of
following them, which is what you want for a plant-side exchanger, but two of
them on the same closed loop over-determine it and the solve fails to converge.
The internal exchanger of a coupler already is one of them, so the consumers on
the same loop should be :func:`~monee.express.create_passive_heat_exchanger`
instead: they inject or extract a fixed ``q_mw`` into a free-flowing stream and
let the temperature change follow from the actual mass flow.

Result columns and their signs
------------------------------

The compound couplers report their operating point on their internal control
node, the branch couplers on the branch itself. Consumption counts positive,
generation and heat injection negative:

.. list-table::
   :header-rows: 1
   :widths: 26 34 40

   * - Component
     - Column
     - Sign
   * - ``PowerToHeatControlNode``
     - ``el_mw``, ``heat_mw``
     - ``el_mw`` > 0: electricity consumed at ``power_node_id``.
       ``heat_mw`` < 0: heat injected at the hot outlet.
   * - ``GasToHeatControlNode``
     - ``gas_mass_flow_kgs``, ``heat_mw``
     - ``gas_mass_flow_kgs`` > 0: gas consumed. ``heat_mw`` < 0: heat injected.
   * - ``CHPControlNode``
     - ``el_mw``, ``heat_mw``, ``gas_mass_flow_kgs``
     - ``el_mw`` < 0 and ``heat_mw`` < 0: both are output.
       ``gas_mass_flow_kgs`` > 0: fuel consumed.
   * - ``PowerToHeatHG``
     - ``p_from_mw``, ``q_mw_heat``
     - ``p_from_mw`` > 0: electricity consumed at the from-bus.
       ``q_mw_heat`` < 0: heat injected at the to-junction.
   * - ``GasToHeatHG``
     - ``gas_mass_flow_kgs``, ``q_mw_heat``
     - ``gas_mass_flow_kgs`` > 0: gas consumed at the from-junction.
       ``q_mw_heat`` < 0: heat injected at the to-junction.
   * - ``PowerToGas``
     - ``p_from_mw``, ``gas_mass_flow_kgs``
     - ``p_from_mw`` > 0: electricity consumed.
       ``gas_mass_flow_kgs`` < 0: gas injected at the to-junction.

The slack is reported from the network's point of view instead, as described in
:ref:`Sign of the slack <data-model-slack-sign>`: an
:class:`~monee.model.child.ExtPowerGrid` reports ``p_mw`` negative while the
network imports, so the P2H example above shows ``-0.5263`` at the slack for
the 0.5263 MW the unit draws.

----

The heat slack is a backup plant
================================

A water loop usually carries an :class:`~monee.model.child.ExtHydrGrid` as its
hydraulic slack. That slack does more than close the mass balance: it supplies
or absorbs any mass flow at its pinned feed temperature, which makes it an
unlimited backup heat plant. When a coupling unit stops delivering heat, for
example a CHP whose gas supply is cut, the slack quietly takes over and every
heat consumer is still served in full. A contingency or resilience study that
leaves the slack uncapped will therefore report zero heat curtailment no
matter what happens on the gas or electricity side.

If backup supply should be limited or absent, cap the slack explicitly:

- pass ``max_import_kgs`` (and ``max_export_kgs``) to
  :func:`monee.express.create_water_ext_grid`; the caps bound the exchanged
  mass flow and can also be set later on the child's model, taking effect on
  the next solve, or
- pass ``bounds_ext_heat`` to the load-shedding problem
  (:doc:`../how-to/load_shedding`), which bounds the exchange inside the
  optimisation.

The same class serves as the gas slack: a child created via
:func:`monee.express.create_gas_ext_grid` is also an ``ExtHydrGrid``. Its
feed temperature defaults to the owning grid's operating temperature (300 K
for the default gas grid, 356 K for the default water grid).

----

Regulation and dispatch
=======================

All coupling components have a ``regulation`` attribute that scales their
output between 0 (off) and 1 (full setpoint). In a plain energy flow it is
fixed at 1. For optimisation, declare it as a solver variable:

.. testcode::

    import monee.problem as mp

    problem = mp.OptimizationProblem()
    problem.controllable_cps([
        (
            "regulation",
            mp.AttributeParameter(
                min=lambda a, v: 0,
                max=lambda a, v: 1,
                val=lambda a, v: 1,
            ),
        )
    ])

The optimiser then freely dispatches each coupling unit within its capacity.

.. tip::

   See :doc:`../tutorials/01_optimization_basics` and
   :doc:`../how-to/load_shedding` for end-to-end worked examples of coupling
   unit dispatch in an optimisation problem.
