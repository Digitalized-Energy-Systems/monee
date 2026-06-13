========================
Multi-energy coupling
========================

A multi-energy system (MES) combines two or more energy carriers - typically
electricity, gas, and heat - into a single network. monee models the coupling
points between these carriers as specialised **compound** components that sit
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
a bus and injects thermal power into a district heating supply pipe. The
return pipe closes the thermal loop.

.. testcode::

    import monee.express as mx

    net = mx.create_multi_energy_network()

    # Electricity side
    bus_el = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus_el)

    # Heating side (supply/return loop)
    junc_supply = mx.create_water_junction(net)
    junc_return = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, junc_supply)

    # Couple electricity to heat
    mx.create_p2h(
        net,
        power_node_id=bus_el,
        heat_node_id=junc_supply,
        heat_return_node_id=junc_return,
        heat_mw=0.5,  # thermal output setpoint [MW]
        diameter_m=0.1,      # heat-exchange pipe diameter
        efficiency=0.95,     # electrical-to-thermal efficiency
    )

**Key parameters:**

- ``heat_mw`` - thermal output setpoint in **megawatts**. The solver
  derives the required electrical input as ``heat_mw / efficiency``.
- ``efficiency`` - ratio of thermal output to electrical input (≤ 1 for
  resistive heating, > 1 for heat pumps modelled with a fixed COP).
- ``diameter_m`` - internal pipe diameter of the connecting heat-exchange
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

P2G is a plain two-endpoint **branch** coupler; its conversion loss is exposed
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

    # Gas grid
    junc_gas = mx.create_gas_junction(net)
    mx.create_source(net, junc_gas, mass_flow_kgs=0.5)

    # Electricity grid
    bus_el = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus_el)

    # Heating grid (supply + return)
    junc_heat_supply = mx.create_water_junction(net)
    junc_heat_return = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, junc_heat_supply)

    # CHP coupling
    mx.create_chp(
        net,
        power_node_id=bus_el,
        heat_node_id=junc_heat_supply,
        heat_return_node_id=junc_heat_return,
        gas_node_id=junc_gas,
        diameter_m=0.15,
        efficiency_power=0.35,   # gas → electricity
        efficiency_heat=0.45,    # gas → heat
        mass_flow_setpoint_kgs=0.1,  # gas consumption [kg/s]
    )

**Key parameters:**

- ``efficiency_power`` and ``efficiency_heat`` - individual efficiencies for
  electrical and thermal output. Their sum must not exceed 1 (total fuel
  utilisation).
- ``mass_flow_setpoint_kgs`` - gas consumption setpoint in kg/s.
- ``regulation`` - a factor in [0, 1] that scales all outputs. Set as a
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
    junc_heat_supply = mx.create_water_junction(net_g2h)
    junc_heat_return = mx.create_water_junction(net_g2h)

    mx.create_g2h(
        net_g2h,
        gas_node_id=junc_gas,
        heat_node_id=junc_heat_supply,
        heat_return_node_id=junc_heat_return,
        heat_mw=0.5,  # thermal output [MW]
        diameter_m=0.1,
        efficiency=0.90,
    )

Node-based heat couplers (HG variants)
======================================

The heat-producing couplers above (CHP, G2H, P2H) deliver their thermal output
through an internal heat-exchanger branch that bridges the supply and return
sides of the heating network - which is why they require a
``heat_return_node_id`` and a ``diameter_m``. Each of them has a **node-based
"HG" variant** ("heat generator") that instead injects the thermal power
directly at a single heat junction: the heat appears as a ``heat_mw`` term
that the :class:`~monee.model.node.Junction` heat balance picks up, exactly
like a node-level :class:`~monee.model.child.HeatGenerator` child.

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

- :class:`~monee.model.multi.CHPHG` is still a **compound** -
  ``create(gas_node, heat_node, power_node)`` wires gas and power through an
  internal control node, but the heat side is a subordinate ``SubHG``
  heat-generator child placed at ``heat_node`` instead of a heat-exchanger
  branch. It therefore needs **no** ``heat_return_node`` and **no**
  ``diameter_m``.
- :class:`~monee.model.multi.GasToHeatHG` (``heat_mw, efficiency,
  regulation=1``) and :class:`~monee.model.multi.PowerToHeatHG`
  (``heat_mw, efficiency, q_mvar_setpoint=0, regulation=1``) are plain
  two-endpoint **branch** couplers, just like G2P and P2G: they withdraw gas or
  electricity at the from-node and carry ``heat_mw`` directly on the branch,
  where the junction heat balance at the to-node picks it up. Both also expose
  ``loss_percent()`` (= ``1 - efficiency``).

Prefer the HG variants when

- you want **simpler wiring** - one heat junction instead of a supply/return
  pair, no heat-exchange pipe geometry to choose; or
- you use the **McCormick-DHS formulation**
  (:data:`~monee.model.formulation.HEAT_CONVEX_MILP_FORMULATION`) or a
  network built with node-based heat loads (e.g.
  ``node_based_heat_loads=True`` in the :mod:`monee.network` heat builders),
  where they are **required**: this formulation owns the nodal heat balance and
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
        heat_mw=0.5,  # thermal output setpoint [MW]
        efficiency=0.95,
    )
    mx.create_g2h_hg(
        net_hg,
        gas_node_id=junc_gas,
        heat_node_id=junc_heat,
        heat_mw=0.5,
        efficiency=0.90,
    )

.. note::

   The internal control-node models (``CHPControlNode``, ``CHPHGControlNode``,
   ``GasToHeatControlNode``, ``PowerToHeatControlNode``) and the subordinate
   ``SubHE`` / ``SubHG`` components you may encounter in result dataframes are
   implementation details. They are wired up automatically by the compounds'
   ``create()`` - you never instantiate them directly.

Heat exchanger
==============

A heat exchanger transfers thermal energy between two water circuits - for
example between a primary transmission network and a secondary distribution
loop - without mass transfer.

.. testcode::

    import monee.express as mx

    net_hx         = mx.create_multi_energy_network()
    junc_primary   = mx.create_water_junction(net_hx)
    junc_secondary = mx.create_water_junction(net_hx)

    mx.create_heat_exchanger(
        net_hx,
        from_node_id=junc_primary,
        to_node_id=junc_secondary,
        heat_mw=0.2,  # heat transfer setpoint [MW]; positive = from→to
    )

----

Regulation and dispatch
=======================

All coupling components have a ``regulation`` attribute that scales their
output between 0 (off) and 1 (full setpoint). In a plain energy flow it is
fixed at 1. For optimisation, declare it as a solver variable:

.. testcode::

    import monee.model as mm
    import monee.problem as mp

    problem = mp.OptimizationProblem()
    problem.controllable_cps((
        "regulation",
        mp.AttributeParameter(
            min=lambda a, v: 0,
            max=lambda a, v: 1,
            val=lambda a, v: 1,
        ),
    ))

The optimiser then freely dispatches each coupling unit within its capacity.

.. tip::

   See :doc:`../tutorials/01_optimization_basics` and
   :doc:`../how-to/load_shedding` for end-to-end worked examples of coupling
   unit dispatch in an optimisation problem.
