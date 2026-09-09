===================================
03 · Coupled network: CHP dispatch
===================================

**Scenario.** An industrial site draws 0.8 MW of electricity from the public
grid and 1.2 MW of heat from a small district heating loop. On site sit two
coupling units: a gas-fired CHP that can cover most of the electric demand
while feeding the heating loop, and a power-to-heat unit (an electric boiler)
that turns surplus electricity into heat. A gas feeder supplies the CHP.

This tutorial builds the three-carrier network with the express API, solves a
plain energy flow, and then dispatches both couplers in an optimisation with a
capped grid connection. Along the way it spells out the conventions that trip
up first-time users: which way the coupler heat side is wired, what the slack
sign means, which formulation actually runs, and which result columns are
setpoints rather than served values.

The coupling components themselves are introduced in
:doc:`../concepts/multi_energy`; this page uses them.

----

Building the network
====================

One network object holds all three carriers. Nodes carry their grid, so a
coupler can simply point at a bus, a gas junction and two water junctions.

.. testcode::

    import monee.express as mx
    import monee.model as mm

    net = mx.create_multi_energy_network()

    # Electricity: grid connection, one line, the site load
    bus_grid = mx.create_bus(net)
    bus_site = mx.create_bus(net)
    mx.create_line(net, bus_grid, bus_site, 1000, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_grid)
    mx.create_power_load(net, bus_site, p_mw=0.8, q_mvar=0.0)

    # Gas: a feeder with a pressure reference for the CHP fuel
    junc_gas_feed = mx.create_gas_junction(net)
    junc_gas_site = mx.create_gas_junction(net)
    mx.create_gas_pipe(net, junc_gas_feed, junc_gas_site, diameter_m=0.15, length_m=500)
    mx.create_gas_ext_grid(net, junc_gas_feed)

    # Heat: a supply/return loop with the reference on the return side
    junc_return = mx.create_water_junction(net)
    junc_supply = mx.create_water_junction(net)
    mx.create_water_ext_grid(net, junc_return, t_k=330)
    mx.create_passive_heat_exchanger(
        net, junc_supply, junc_return, q_mw=1.2, diameter_m=0.15
    )

    # CHP: burns gas, generates electricity at bus_site, heats the loop
    mx.create_chp(
        net,
        power_node_id=bus_site,
        cold_node_id=junc_return,
        hot_node_id=junc_supply,
        gas_node_id=junc_gas_site,
        diameter_m=0.15,
        efficiency_power=0.35,
        efficiency_heat=0.45,
        mass_flow_setpoint_kgs=0.05,
    )

    # P2H: draws electricity at bus_site, heats the same loop
    mx.create_p2h(
        net,
        power_node_id=bus_site,
        cold_node_id=junc_return,
        hot_node_id=junc_supply,
        heat_energy_mw=0.3,
        diameter_m=0.1,
        efficiency=0.95,
    )

Three wiring conventions decide whether this loop solves, all covered in
detail in :ref:`District heating loops <dhs-loops>`:

- Couplers heat the water from cold to hot. Water enters the internal heat
  exchanger of a CHP or P2H at ``cold_node_id`` and leaves, heated, at
  ``hot_node_id``. These are aliases for ``heat_node_id`` and
  ``heat_return_node_id``; despite its name, ``heat_return_node_id`` is the
  hot outlet.
- The hydraulic reference sits on the return side.
  :func:`~monee.express.create_water_ext_grid` pins pressure and temperature,
  so on the cold side it fixes the 330 K return and the couplers still have to
  do the heating. On the supply side it would silently cover the heat demand
  itself.
- The consumer is a passive exchanger.
  :func:`~monee.express.create_passive_heat_exchanger` takes its 1.2 MW out of
  the free-flowing stream; the couplers' internal exchangers already prescribe
  the loop's mass flow, and a second fixed-flow exchanger in the loop would
  over-determine it.

The gas side needs a real reference too:
:func:`~monee.express.create_gas_ext_grid` pins the feeder pressure. A gas
junction fed only by a :func:`~monee.express.create_source` has no reference,
and a CHP hanging off it silently comes back with ``regulation`` 0 instead of
an error.

----

Choosing the formulation
========================

``run_energy_flow`` without arguments uses
:data:`~monee.model.formulation.bundles.DEFAULT_SIMULATION_FORMULATION`, a
hybrid of the polar AC power flow, a relaxed Weymouth gas model and the
bilinear Darcy-Weisbach heat model. The heat part of that default carries
binary flow-direction variables. The default CasADi/IPOPT back-end cannot
branch on binaries; it relaxes them and warns::

    UserWarning: IPOPT relaxes integer variables, but 3 component(s) carry a
    formulation that needs them enforced: PassiveHeatExchangerLoad((5, 4, 0)),
    ... Solve with a MIQCQP/MINLP back-end (e.g. solver='scip' or 'gurobi') or
    pass a binary-free formulation (e.g. formulation='smooth_nlp' / ...).

Take one of the two exits the warning names. For a pure electricity network
the default is fine as it is, and for gas the binary-free substitute is picked
automatically, but as soon as heat exchangers are involved, pass
``formulation="smooth_nlp"`` to solve the whole multi-energy network as one
smooth NLP, or keep the default formulation and solve with
``solver="scip"`` (open source) or ``solver="gurobi"`` (commercial). This
tutorial uses ``smooth_nlp`` throughout: on the default IPOPT back-end for the
energy flow below, and on SCIP for the dispatch optimisation further down,
where the coupler model is nonconvex and IPOPT converges to whichever local
solution its start point leads to. :doc:`../concepts/formulations` has the full
decision table.

----

Running the energy flow
=======================

.. testcode::

    from monee import run_energy_flow

    result = run_energy_flow(net, formulation="smooth_nlp")

    chp = result.get(mm.CHPControlNode)
    p2h = result.get(mm.PowerToHeatControlNode)
    ext = result.get(mm.ExtPowerGrid)

    print("grid import [MW]:", round(-ext["p_mw"].iloc[0], 4))
    print("CHP el_mw   [MW]:", round(chp["el_mw"].iloc[0], 4))
    print("CHP heat_mw [MW]:", round(chp["heat_mw"].iloc[0], 4))
    print("P2H el_mw   [MW]:", round(p2h["el_mw"].iloc[0], 4))
    print("P2H heat_mw [MW]:", round(p2h["heat_mw"].iloc[0], 4))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    grid import [MW]: 0.3833
    CHP el_mw   [MW]: -0.7428
    CHP heat_mw [MW]: -0.955
    P2H el_mw   [MW]: 0.3158
    P2H heat_mw [MW]: -0.3

.. note::

   Running this yourself also prints a few diagnostics on stderr: monee
   validates every returned result and summarises its findings, one line per
   category, in ``result.warnings``. On this network they are informational.
   :doc:`../how-to/diagnose_infeasibility` explains each category and when it
   matters.

Two conventions are at work in these numbers.

First the slack sign: :class:`~monee.model.child.ExtPowerGrid` reports
``p_mw`` from the network's point of view, negative while the network imports
(here ``p_mw`` is ``-0.3833``), so the import is read by negating the column.
:ref:`Sign of the slack <data-model-slack-sign>` explains why.

Second the coupler signs: on the control nodes, consumption is positive and
generation negative. The CHP burns its 0.05 kg/s setpoint and produces
0.7428 MW of electricity (``el_mw`` negative) plus 0.955 MW of heat
(``heat_mw`` negative). The P2H consumes 0.3158 MW (``el_mw`` positive, which
is ``0.3 / 0.95``) to inject its 0.3 MW of heat. The electric balance then
closes: 0.8 load plus 0.3158 P2H minus 0.7428 CHP leaves 0.373 MW, which the
grid covers together with the line losses, 0.3833 MW in total.

On the heat side the couplers inject 1.255 MW while the consumer takes 1.2 MW.
The pinned reference absorbs the 0.055 MW excess, and the loop temperatures
settle at the design spread:

.. testcode::

    junctions = result.get(mm.Junction)
    print("return [K]:", round(junctions["t_k"].iloc[2], 1))
    print("supply [K]:", round(junctions["t_k"].iloc[3], 1))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    return [K]: 330.0
    supply [K]: 360.0

----

Dispatching the couplers
========================

Now for the optimisation: the grid connection is capped at 0.2 MW (a
contracted limit), gas costs money, and the solver decides how hard to run
each coupler. Every coupler carries a ``regulation`` attribute that scales it
between 0 (off) and 1 (full setpoint); in the plain energy flow above it was
fixed at 1, and
:meth:`~monee.problem.core.OptimizationProblem.controllable_cps` promotes it
to a solver variable on every coupling component at once.

This step runs on SCIP. With a free ``regulation`` on both couplers the model
has several local solutions, and IPOPT starting from the energy flow above
settles in one of them instead of the cheapest one; SCIP searches globally and
returns the same answer whatever the network was solved for before.

.. testcode::

    import monee.problem as mp
    from monee import run_energy_flow_optimization

    problem = mp.OptimizationProblem()

    problem.controllable_cps([
        (
            "regulation",
            mp.AttributeParameter(
                min=lambda attr, val: 0,
                max=lambda attr, val: 1,
                val=lambda attr, val: 1,
            ),
        )
    ])

    # Import cap. ExtPowerGrid.p_mw is negative while importing, so a 0.2 MW
    # import limit is a lower bound on p_mw, not an upper one.
    constraints = mp.Constraints()
    constraints.select_types(mm.ExtPowerGrid).equation(
        lambda model: model.p_mw >= -0.2
    )
    problem.constraints = constraints

    # Fuel cost. On the CHP control node, gas_mass_flow_kgs is the capacity
    # setpoint and regulation the dispatch decision, so the burned fuel is
    # their product.
    gas_price = 100
    objectives = mp.Objectives()
    objectives.select(lambda model: isinstance(model, mm.CHPControlNode)).calculate(
        lambda models: sum(gas_price * m.gas_mass_flow_kgs * m.regulation for m in models)
    )
    problem.objectives = objectives

    opt = run_energy_flow_optimization(
        net, problem, formulation="smooth_nlp", solver="scip"
    )

    chp_opt = opt.get(mm.CHPControlNode)
    p2h_opt = opt.get(mm.PowerToHeatControlNode)
    ext_opt = opt.get(mm.ExtPowerGrid)

    chp_reg = max(0.0, chp_opt["regulation"].iloc[0])
    p2h_reg = max(0.0, p2h_opt["regulation"].iloc[0])
    print("CHP regulation:", round(chp_reg, 3))
    print("P2H regulation:", round(p2h_reg, 3))
    print("grid import [MW]:", round(-ext_opt["p_mw"].iloc[0], 4))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    CHP regulation: 0.812
    P2H regulation: 0.0
    grid import [MW]: 0.2

The dispatch is readable: with imports capped at 0.2 MW, the CHP must generate
the remaining 0.6 MW plus losses, which takes 81.2 percent of its capacity.
The P2H only adds electric demand that the CHP would have to cover with paid
fuel, so it is switched off. The ``max(0.0, ...)`` clip is the same habit as
in :doc:`01_optimization_basics`: an interior-point solver may land a hair
outside the declared [0, 1] bounds.

.. note::

   The heat setpoints of CHP and P2H are served through built-in objective
   terms rather than hard equations, so a custom objective competes with them
   and ``opt.objective`` contains both parts. When a coupler then delivers
   less heat than its setpoint, monee prints a diagnostic naming the
   component; the objective semantics section of :doc:`../how-to/load_shedding`
   explains how to read it. Prefer reading dispatch quantities from the result
   frames, as done here, over interpreting the raw objective value.

----

Setpoints and served values
===========================

As in :doc:`01_optimization_basics`, attributes you passed in stay put and
``regulation`` carries the decision. That extends to the coupler result
columns: those that echo an input parameter (``gas_mass_flow_kgs`` on the CHP,
``load_p_mw`` on the P2H, derived from ``heat_energy_mw / efficiency``) still
show the full setpoint after the optimisation, and the served value is
setpoint times regulation. Columns solved as variables (``el_mw`` and
``heat_mw`` on both the CHP and the P2H) already report the operating point.

.. testcode::

    fuel_kgs = chp_opt["gas_mass_flow_kgs"].iloc[0] * chp_reg
    print("CHP fuel [kg/s]:", round(fuel_kgs, 4))
    print("fuel cost:", round(gas_price * fuel_kgs, 2))
    print("CHP heat delivered [MW]:", round(-chp_opt["heat_mw"].iloc[0], 3))

.. testoutput::
   :options: +NORMALIZE_WHITESPACE

    CHP fuel [kg/s]: 0.0406
    fuel cost: 4.06
    CHP heat delivered [MW]: 0.775

One caveat deserves emphasis: the heating loop now receives only 0.775 MW
against its 1.2 MW demand, and the solve still succeeds. The temperature-pinned
reference on the return side absorbs any shortfall or excess, which is
what makes the loop solvable but also makes the heat demand non-binding in an
optimisation. If undersupplied heat must show up as a violation or as shed
load, keep the pressure reference but release the temperature with
``pin_temperature=False``, or model the demand with curtailable heat
components as in :doc:`../how-to/load_shedding`.

----

Next steps
==========

- :doc:`../concepts/multi_energy` documents every coupling component and the
  full sign table for their result columns.
- :doc:`../concepts/formulations` explains the formulation registry behind
  ``formulation="smooth_nlp"`` and when a MIQCQP back-end is the better exit.
- :doc:`../how-to/generate_mes` creates larger coupled benchmark networks in
  one call.
- :doc:`../how-to/load_shedding` prices unserved demand across all three
  carriers with a ready-made problem.
