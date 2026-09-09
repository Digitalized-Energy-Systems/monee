District heating with couplers
==============================

This guide shows how to run a mass-flow-prescribed heat consumer next to a
CHP or P2H unit without over-determining the hydraulic loop, and how to drive
such a loop to part load from a demand profile. It builds on the loop
conventions in :ref:`District heating loops <dhs-loops>`; read that section
first if the supply/return orientation or the placement of the plant
reference is new to you.

The over-determined loop
------------------------

The internal heat exchanger of a coupler (:func:`~monee.express.create_chp`,
:func:`~monee.express.create_p2h`, :func:`~monee.express.create_g2h`) drives
its own mass flow through the loop. A consumer added with
:func:`~monee.express.create_heat_exchanger` is a fixed-flow exchanger too: a
numeric ``q_mw`` pins its branch flow to the design flow
``|q_mw| * 1e6 / (c_p * T_delta_design_K)`` (a 30 K design spread by
default), or to an explicit ``mass_flow_design_kgs``. Two junctions, one
coupler, one fixed-flow consumer: that closed loop now contains two branches
that each prescribe their own flow, and no branch that lets the hydraulics
give way. The mass balance at the two junctions can only hold if both
prescribed flows happen to be exactly equal.

The symptoms are solver specific. The default CasADi/IPOPT backend fails
with a ``CasADiSolveError`` (return status ``Restoration_Failed``); SCIP
returns without converging on the duty and reports a
``q_mw_delivered`` shortfall violation ("delivers 0 MW of its ... setpoint").
Both messages describe the symptom; the cause is the loop topology, and the
fix is one extra branch.

The bypass pattern
------------------

Add a plain water pipe between the supply and the return junction. The pipe
follows the pressure difference instead of prescribing a flow, so it absorbs
the mismatch between the coupler's driven flow and the consumer's design
flow, and the loop solves on every backend.

Build the electricity and gas sides first:

.. testcode::

    import monee.express as mx
    import monee.model as mm
    from monee import run_energy_flow

    net = mx.create_multi_energy_network()

    bus0 = mx.create_bus(net)
    bus1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, bus0)
    mx.create_line(net, bus0, bus1,
                   length_m=200, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, bus1, p_mw=0.3, q_mvar=0.0)

    junc_gas = mx.create_gas_junction(net)
    mx.create_gas_ext_grid(net, junc_gas)

Then the heat side: a supply/return pair with the plant reference on the
return side, the fixed-flow consumer, and the bypass pipe that closes the
hydraulics:

.. testcode::

    junc_return = mx.create_water_junction(net)
    junc_supply = mx.create_water_junction(net)
    mx.create_water_ext_grid(net, junc_return, t_k=330)

    mx.create_heat_exchanger(net, junc_supply, junc_return,
                             q_mw=0.5, name="substation")
    mx.create_water_pipe(net, junc_supply, junc_return,
                         diameter_m=0.15, length_m=50, name="bypass")

Finally the CHP. Its gas setpoint of 0.026 kg/s yields roughly 0.5 MW of
heat at ``efficiency_heat=0.45``, so plant and consumer are nearly matched:

.. testcode::

    mx.create_chp(
        net,
        power_node_id=bus1,
        cold_node_id=junc_return,
        hot_node_id=junc_supply,
        gas_node_id=junc_gas,
        diameter_m=0.15,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.026,
        name="chp",
    )

    result = run_energy_flow(net)
    chp = result.get(mm.CHPControlNode)
    substation = result.get(mm.HeatExchangerLoad)
    bypass = result.get(mm.WaterPipe)
    junctions = result.get(mm.Junction)

    print("CHP el, heat [MW]:", round(chp["el_mw"].iloc[0], 4),
          round(chp["heat_mw"].iloc[0], 4))
    print("substation duty [MW], flow [kg/s]:",
          round(substation["q_mw_delivered"].iloc[0], 4),
          round(substation["mass_flow_mag_kgs"].iloc[0], 3))
    print("bypass flow [kg/s]:", round(bypass["mass_flow_mag_kgs"].iloc[0], 3))
    print("return, supply [K]:", round(junctions["t_k"].iloc[1], 1),
          round(junctions["t_k"].iloc[2], 1))

.. testoutput::

    CHP el, heat [MW]: -0.3862 -0.4966
    substation duty [MW], flow [kg/s]: 0.5 3.983
    bypass flow [kg/s]: 0.027
    return, supply [K]: 330.0 359.8

The substation delivers its full 0.5 MW setpoint at its 3.98 kg/s design
flow; the CHP injects 0.4966 MW and drives its own, different flow. The
bypass carries the 0.027 kg/s difference, and the ``ExtHydrGrid`` reference
tops up the small remaining heat imbalance (the backup-plant section of
:doc:`../concepts/multi_energy` explains how to cap that slack). Remove the
``bypass`` pipe and the very same network raises ``CasADiSolveError`` on the
default solver.

Sizing barely matters: the bypass only needs enough capacity for the flow
mismatch, so a short pipe of roughly the loop diameter is fine. It also
keeps the loop solvable when a profile later drives the consumer and the
plant to different part loads (next section).

Alternatives to the bypass
--------------------------

Two other layouts avoid the over-determination without an extra pipe:

- Use :func:`~monee.express.create_passive_heat_exchanger` for the consumer.
  A passive exchanger extracts its ``q_mw`` from the free-flowing stream and
  lets the coupler drive the loop flow alone. Choose this when the consumer
  has no prescribed flow of its own; the concepts page discusses the
  distinction under :ref:`District heating loops <dhs-loops>`.
- Model the consumer as a nodal pair instead of a branch: a
  :func:`~monee.express.create_water_sink` with ``mass_flow_kgs`` on the
  supply junction (paired with the reference on the return side) prescribes
  the offtake flow without adding a second flow-driving branch to the loop.

Keep :func:`~monee.express.create_heat_exchanger` plus the bypass when the
consumer really is mass-flow prescribed, for example a substation with a
pump schedule from a co-simulation, or measured flows to replay.

.. _dh-part-load:

Part-load control
-----------------

A heat exchanger's ``q_mw`` result column is a solved output; registering a
timeseries on it does not drive the branch (the run warns once at start, see
:doc:`timeseries`). A demand profile drives one of the two setpoint
attributes instead, and the choice decides what the hydraulics do:

- ``regulation`` scales duty and mass flow together: the exchanger runs at
  its design temperature spread and the flow varies with the demand
  (variable-flow operation).
- ``q_mw_set`` changes the duty at the constant design mass flow: the flow
  stays at the value derived at construction and the temperature spread
  varies instead (constant-flow operation).

Variable flow, with the CHP dispatched down alongside the demand:

.. testcode::

    from monee.simulation import TimeseriesData, run_timeseries

    demand_mw = [0.5, 0.35, 0.2]

    td = TimeseriesData()
    td.add_branch_series_by_name("substation", "regulation",
                                 [d / 0.5 for d in demand_mw])
    td.add_compound_series_by_name("chp", "regulation", [1.0, 0.7, 0.4])

    res = run_timeseries(net, td)
    q = res.get_result_for(mm.HeatExchangerLoad, "q_mw")
    flow = res.get_result_for(mm.HeatExchangerLoad, "mass_flow_mag_kgs")
    print("duty [MW]: ", [round(v, 3) for v in q.iloc[:, 0]])
    print("flow [kg/s]:", [round(v, 3) for v in flow.iloc[:, 0]])

.. testoutput::

    duty [MW]:  [0.5, 0.35, 0.2]
    flow [kg/s]: [3.983, 2.788, 1.593]

Constant flow, driving ``q_mw_set`` with the demand in MW (positive is
consumption on a ``HeatExchangerLoad``):

.. testcode::

    td_cf = TimeseriesData()
    td_cf.add_branch_series_by_name("substation", "q_mw_set", demand_mw)

    res_cf = run_timeseries(net, td_cf)
    q_cf = res_cf.get_result_for(mm.HeatExchangerLoad, "q_mw")
    flow_cf = res_cf.get_result_for(mm.HeatExchangerLoad, "mass_flow_mag_kgs")
    print("duty [MW]: ", [round(v, 3) for v in q_cf.iloc[:, 0]])
    print("flow [kg/s]:", [round(v, 3) for v in flow_cf.iloc[:, 0]])

.. testoutput::

    duty [MW]:  [0.5, 0.35, 0.2]
    flow [kg/s]: [3.983, 3.983, 3.983]

The bypass absorbs the flow difference in both modes, which is why the loop
stays solvable while consumer and plant follow different profiles. The
couplers themselves are dispatched through their compound ``regulation``
series, as shown above; nodal :func:`~monee.express.create_heat_load`
children take their profile on ``q_mw_heat`` and have no mass flow of their
own, so the variable-flow vs constant-flow choice does not arise for them.

For dispatching ``regulation`` with an optimizer instead of a profile, see
the regulation section of :doc:`../concepts/multi_energy`; for infeasible
loops beyond the pattern covered here, see :doc:`diagnose_infeasibility`.
