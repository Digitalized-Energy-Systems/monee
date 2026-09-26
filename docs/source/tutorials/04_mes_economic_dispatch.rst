==============================
Multi-energy economic dispatch
==============================

**Scenario.** A small town is supplied by three carriers. Power comes from a
grid connection with an hourly price and from a 1.5 MW PV plant, gas from a
gas grid connection and a small biogas source, and heat from a district
heating line run at a supply temperature of 358 K, whose plant, a gas boiler
fired from outside the modelled gas network, sells heat at a fixed 40 per
MWh. Three coupling units link the carriers: a CHP that burns gas for power
and heat, an electric boiler that turns power into heat, and an electrolyser
that turns power into gas. Over a day condensed to eight hourly prices, the
economic dispatch decides which of them run, while the heating line keeps its
supply temperature.

The scenario is chosen so that the answer can be worked out by hand: every
unit has a breakeven power price at which it starts to pay off, so the
dispatch can be checked hour by hour and the saving recomputed from the
units' margins. Everything shown here comes from
``examples/mes_economic_dispatch.py`` in the repository, which prints the
same report and exits with an error when a check fails. The problem itself
is described in :doc:`../problems/economic_dispatch`, the coupling
components in :doc:`../concepts/multi_energy`.

----

The merit order to expect
=========================

Prices are in currency per MWh; each period is one hour, so they enter the
dispatch as currency per MW and period. Gas prices are per MWh of higher
heating value.

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :start-at: EL_PRICE = [
   :end-before: ALL_COUPLERS

The town draws 2 MW of power, 2 MW of heat through three consumers and
0.03 kg/s of gas, about 1.27 MW. ``SUPPLY_T_BAND_K`` is not a limit of the
dispatch, only the band the checks hold the supply temperature to.

Each unit displaces something with a known price:

- The electric boiler replaces heat from the boiler plant at 40. It needs
  :math:`1 / 0.95` MWh of power per MWh of heat, so it pays off while the
  power price is below :math:`40 \cdot 0.95 = 38`.
- The electrolyser replaces gas from the grid at 35. It needs
  :math:`1 / 0.65` MWh of power per MWh of gas, so it pays off below
  :math:`35 \cdot 0.65 = 22.75`.
- The CHP turns 1 MWh of gas at 35 into 0.35 MWh of power and 0.5 MWh of heat
  that would otherwise cost 40 at the boiler plant. It pays off when the power
  price :math:`p` satisfies :math:`0.35\, p + 0.5 \cdot 40 > 35`, that is
  above :math:`p = 42.86`.

The biogas source at 20 is always cheaper than the grid gas and runs in every
hour. The prices of the day cover every regime: the CHP should run in the four
expensive hours, the electric boiler in the three cheap ones, the
electrolyser in the two cheapest, and at 40 none of them.

----

Building the network
====================

All three carriers live in one network. Power and gas each run along a
string of three nodes joined by 1 km lines and pipes: the grid connection
at the first node, the town (its load and the PV plant, or its gas demand)
at the second, and the plant site at the third, where all three coupling
units connect next to the heat plant and the biogas source feeds in.

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :start-at: def build_network
   :end-before: dhs = mx.dhs_structure

Gas flows in monee are mass flows in kg/s; ``gas_mw_per_kgs`` returns the
higher heating value of the grid's gas, about 42.4 MW per kg/s, which
converts the MW sizes at the top of the script. ``Ids`` is a small dataclass
in the script that keeps the ids the checks read back later, so the ``ids.``
lines can be skipped on a first read. ``max_export_kgs=0`` keeps the town
from selling gas back; it does not bind on this day.

``dhs_structure`` lays out district heating pipes of one size.
``line(3, ...)`` creates three supply and three return junctions with a
consumer between each pair, and ``attach_heat_plant`` closes the loop with
the boiler plant, the line's water external grid: its ``cost`` prices the
heat it supplies, and by default it may supply heat but not absorb a surplus.

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :start-at: dhs = mx.dhs_structure
   :end-before: # Feed-in stations
   :dedent: 4

The CHP and the electric boiler sit at the heat plant as feed-in stations:
each one takes water from the return header (``cold``) and feeds it, heated,
into the supply header (``hot``). The electrolyser connects the plant site's
bus and gas junction. The ``couplers`` argument leaves units out, which
builds the two baselines the result is compared with further down: no
coupling units at all, and a CHP only.

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :start-at: # Feed-in stations
   :end-at: return net, ids
   :dedent: 4

Temperature control
-------------------

A district heating network is operated to a supply temperature setpoint: the
plants hold the supply at a target, here 358 K, and the consumers draw the
flow they need, cooling it by their design spread (``T_delta_design_K``,
30 K by default). A heat source must therefore feed in water at the
setpoint, whatever its output.

The feed-in stations (``create_chp`` and ``create_p2h`` with a
``cold_node_id`` and a ``hot_node_id``) do exactly that: their heat exchanger
heats return water by its 30 K design spread, back to the setpoint, at any
output. A running unit takes flow over from the plant instead of heating the
supply further.
The same units attached as nodal heat sources (``create_chp_hg``,
``create_p2h_hg``) downstream of the plant would not be temperature
controlled: their heat would go into water already at 358 K and push the
supply behind them 20 K or more above the setpoint.

The setpoint is not a separate constraint of the dispatch: the plant pins the
supply temperature and the feed-in stations always add their design spread,
so the physics already holds it, and a constraint on top would only restate
an equality and make the solve harder. The script checks it instead.

----

Dispatching the day
===================

Nothing links one hour to the next here: no unit stores energy and none is
ramp limited, so each hour's best dispatch depends on that hour's prices
alone. The day is therefore a timeseries of eight independent dispatches,
and :func:`~monee.simulation.timeseries.run_timeseries` solves one economic
dispatch per hour. A joint solve over the horizon with
:func:`~monee.simulation.multi_period.run_multi_period` becomes necessary once
the hours are coupled, by a storage, a ramp limit, gas linepack or thermal
inertia; on this network it returns the same dispatch and cost.

The electricity price and the PV availability vary per hour, and
``TimeseriesData`` carries both: the price as a series on the grid
connection's ``cost`` (``add_objective_data`` is ``add_child_series`` under a
name that marks a price), the availability as a series on the PV plant's
``regulation``, a factor on its 1.5 MW setpoint.

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :pyobject: solve_day

``ext_grid_cost_default`` puts the exchange with the power grid into the
objective; left out, imported power would be free. The hourly ``cost`` series
overrides its value, so hour 0's price there only switches the grid on.
``gas_cost_default`` and ``heat_cost_default`` do the same for gas and heat,
while the gas grid, the biogas source and the boiler plant keep their own
``cost``.

Once every carrier a coupling unit touches is dispatched, its operating
point, its ``regulation``, becomes a decision between 0 and 1
(``dispatch_coupling_points=True``, the default). With ``False`` the units
stay at their setpoint, which is full output; the must run CHP baseline
below is built that way.

``solver_options`` passes ``mu_strategy="adaptive"`` to IPOPT, the
third-party nonlinear solver monee hands the dispatch to by default, so the
two most expensive hours converge at the first attempt instead of after
monee's automatic retry. Setting it also switches that retry off (see
:ref:`concepts/solvers:Tuning solver options`), so if a change to the script
makes an hour fail, try the run without the option.

The script prints this report:

.. code-block:: text

   Hourly dispatch (1 = unit at full output)
   hour  price  import    pv   chp e-boil elyser boiler  expected
      0  120.0   0.951  0.00  1.00   0.00   0.00  0.507  CHP
      1   80.0   0.651  0.30  1.00   0.00   0.00  0.507  CHP
      2   55.0   0.200  0.75  1.00   0.00   0.00  0.507  CHP
      3   40.0   0.800  1.20  0.00   0.00   0.00  2.007  boiler plant only
      4   30.0   1.554  1.50  0.00   1.00   0.00  1.007  e-boiler
      5   18.0   2.557  1.50  0.00   1.00   1.00  1.007  e-boiler, electrolyser
      6   10.0   2.707  1.35  0.00   1.00   1.00  1.007  e-boiler, electrolyser
      7   60.0   0.501  0.45  1.00   0.00   0.00  0.507  CHP

   Breakevens: e-boiler below 38.00, electrolyser below 22.75, CHP above 42.86
   Supply temperature: 357.9 to 358.0 K, setpoint band 356 to 360 K

   day cost                total   saving      %  margins
   optimised dispatch    1331.39
   boiler only           1558.60   227.21   14.6   227.20
   must run CHP          1484.81   153.42   10.3   153.55

   Plausibility checks: 148 of 148 passed

``import``, ``pv`` and ``boiler`` (the boiler plant's heat) are in MW, the
unit columns are operating points, and ``expected`` names the units the
breakevens predict at that hour's price. The lower table compares the day's
cost with the two baselines, next to the saving the margins predict (see
`Checking the result`_).

The figure shows the same day. The top panel is the price against the three
breakevens; below it, one panel per carrier stacks the supply above zero and
the use below, so every unit shows up on each carrier it touches.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_dispatch.html" width="100%" height="1080" style="border:none;" loading="lazy" title="A coupled day: price and dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_dispatch.png
      :width: 100%

----

Checking the result
===================

A solver that reports success can still solve the wrong problem, so the
script checks the result against things that are known independently of it.
The figures on this page are rebuilt from the script at every docs build, and
a failing check stops them from being built.

Merit order. Each unit must run in exactly the hours its breakeven says. The
prices stay at least 2 per MWh away from every breakeven, so network losses
cannot flip a decision:

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :pyobject: check_merit_order

Physics and limits. Power, gas and heat must balance in every hour, with
small positive line and pipe losses; every consumer must receive its full
heat demand, and each coupling unit must convert at exactly its efficiency.
Voltages, line loadings and water temperatures must stay in their bands,
every supply junction in the setpoint band, the gas import must never turn
into an export, and the heat plant must never absorb heat.

Objective. The objective the solver reports for each hour must equal that
hour's cost recomputed from the result tables at the known prices. The
plant's heat there includes a small top-up of the feed-in water, which pipe
losses leave slightly below the setpoint.

Savings. The strongest check compares the optimum with two rule based
baselines built from the same network: the boiler plant alone, and a CHP that
runs every hour. Each unit's margin at full output follows from the prices
alone:

.. literalinclude:: ../../../examples/mes_economic_dispatch.py
   :language: python
   :pyobject: unit_margins

Against the boiler only baseline, the optimum should save the sum of all
positive margins; against the must run CHP baseline, the positive margins of
the other units plus the CHP losses it avoids in the cheap hours. The
solver's savings in the report match these predictions within the network
losses, which the margins ignore. They can ignore the network because no
line or pipe limit binds on this day; :doc:`mes_network_limits` tightens
both until they bind and the dispatch leaves the merit order.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_checks.html" width="100%" height="880" style="border:none;" loading="lazy" title="Checking the dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_checks.png
      :width: 100%

The left panel plots each unit's operating point against its margin in
every hour: positive margins sit at 1 and negative ones at 0. The right
panel compares the solver's savings with the margins' prediction; the bottom
panel shows the hottest and coolest supply junction, both at 358 K
throughout.

----

Running it yourself
===================

From the repository root::

    python examples/mes_economic_dispatch.py

The script prints the report above and exits with status 1 when any check
fails; ``tests/problem/test_mes_dispatch_example.py`` runs it with the test
suite. Change a price, an efficiency or a capacity at the top of the file
and the breakevens, the expected savings and the checks follow, as long as
the hand analysis still holds: prices clear of the breakevens, the CHP and
the electric boiler never paying off in the same hour, and each unit's full
output within the heat or gas it displaces. Otherwise units run at part load
and the merit order and saving checks fail although the dispatch is right.

Next steps
==========

- :doc:`mes_storage_and_ramps` adds a battery and ramp limits to this town,
  which couple the hours and call for the joint solve.
- :doc:`mes_network_limits` makes the cable and the gas pipe to the plant
  site bind, prices the congestion and values one more MW through each.
- :doc:`../problems/economic_dispatch` states the full formulation, including
  how the heat of the boiler plant is priced and which units become decisions.
- :doc:`03_coupled_network` wires the coupling units with supply and return
  junctions and explains their sign conventions.
- :doc:`../how-to/multi_period` covers horizons, storages and per-period
  prices in more depth.
