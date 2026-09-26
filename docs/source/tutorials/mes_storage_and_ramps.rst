=======================
Storage and ramp limits
=======================

**Scenario.** The town of the
:doc:`economic dispatch tutorial <04_mes_economic_dispatch>` gains a battery
and ramp limits. The battery sits at the town bus, stores 2 MWh and charges
or discharges at up to 1 MW. The CHP, the electric boiler and the
electrolyser may change their operating point by at most 0.5 from one hour
to the next. The prices, the PV and every unit are unchanged, so without the
two additions the day costs what the economic dispatch tutorial found,
1331.39.

Both additions couple the hours, so the day is no longer eight separate
dispatches but one problem over the whole horizon. This tutorial solves it
jointly, compares the result with the hour by hour solve, and measures what
the ramp limits cost and what foresight and the battery are worth, each
checked against a copper plate model that knows no network. Everything shown
here comes from ``examples/mes_storage_and_ramps.py``. The network, the
prices and the merit order are explained in the economic dispatch tutorial,
the joint form of the problem in
:ref:`problems/economic_dispatch:Multi-period dispatch`.

----

What couples the hours
======================

In the :doc:`economic dispatch tutorial <04_mes_economic_dispatch>` each
hour stood alone: nothing stored energy and no unit depended on where it had
run before, so :func:`~monee.simulation.timeseries.run_timeseries` solved
one independent dispatch per hour. Both additions end that independence.
What the battery sells in one hour it must have bought earlier or held since
the start of the day, and a unit can only move 0.5 per hour from where it
ran in the hour before, so the best decision for an hour depends on the
hours around it.

``run_timeseries`` still respects a ramp limit, against the hour it has just
solved, but it cannot look ahead, so it may run a unit into a position it
cannot leave in time. Nor can it dispatch a storage: the option
``include_storages=True`` of the dispatch problem, which makes a storage's
power a decision, takes effect only in
:func:`~monee.simulation.multi_period.run_multi_period` and
:func:`~monee.simulation.multi_period.run_mpc`; under ``run_timeseries``
monee warns and keeps every storage at its setpoint. ``run_multi_period``
solves the eight hours as one problem, with the operating points and the
battery's charge carried from each hour into the next.

The additions to the economic dispatch tutorial's constants; the battery
loses 10 % when it charges and 10 % when it discharges, so 81 % of the
energy it buys comes back:

.. literalinclude:: ../../../examples/mes_storage_and_ramps.py
   :language: python
   :start-at: RAMP_LIMIT = 0.5
   :end-before: UNITS = (

----

Adding the battery
==================

.. literalinclude:: ../../../examples/mes_storage_and_ramps.py
   :language: python
   :pyobject: build_network

The battery is an :class:`~monee.model.storage.ElectricStorage` at the town
bus, found as the PV's bus, holding 1 MWh before the first hour. In the load
convention its power ``p_mw`` is positive while it charges and negative
while it discharges. With efficiencies other than 1 the model splits that
power into a charging part ``p_charge_mw``, :math:`c_t`, and a discharging
part ``p_discharge_mw``, :math:`d_t`, both at least zero, and the energy
stored at the end of hour :math:`t`, ``e_mwh`` or :math:`e_t`, follows, with
one hour per period so that 1 MW over a period moves 1 MWh:

.. math::

   e_t = e_{t-1} + 0.9\, c_t - d_t / 0.9

The battery is the only store. The economic dispatch does not dispatch a
:class:`~monee.model.storage.ThermalStorage`, and a gas store would have
nothing to do, because the gas price is the same in every hour.

----

Ramp limits and the joint solve
===============================

.. literalinclude:: ../../../examples/mes_storage_and_ramps.py
   :language: python
   :pyobject: dispatch_problem

In monee a unit's operating point is its ``regulation``, the factor between
0 and 1 that scales its rated output. ``regulation_ramp`` limits the change
of every regulation that is a decision of the dispatch to 0.5 per period:
here the three coupling units, while the PV's regulation (its availability
series) and the battery's (fixed at 1) are data and stay untouched. The
``ramp_limit`` argument of ``create_economic_dispatch_problem`` would not
reach the coupling units: it limits generators, gas sources and heat units
in MW. ``include_storages=True`` makes the battery's power a decision.

.. literalinclude:: ../../../examples/mes_storage_and_ramps.py
   :language: python
   :pyobject: solve_jointly

The ``terminal_state`` pins the battery's charge at the end of the last hour
to the 1 MWh it started with. Without it the optimiser would sell the
starting charge and end the day empty, which makes the day look 26.49
cheaper without being better. Hour 0 is free of the ramp limit in both runs:
an ``initial_state`` could seed it with the previous day's operating points,
but ``run_timeseries`` has no such input.

The hour by hour solve is the same problem, one hour at a time and without
the battery, which ``run_timeseries`` would only keep idle:

.. literalinclude:: ../../../examples/mes_storage_and_ramps.py
   :language: python
   :pyobject: solve_hour_by_hour

----

The day
=======

The script solves four days: the economic dispatch tutorial's day without
limits as the reference, the day with ramp limits solved hour by hour, and
the joint day with ramp limits, without and with the battery. It prints
this report:

.. code-block:: text

   Operating points with ramp limits, joint and hour by hour
                chp           e-boiler      electrolyser
   hour  price  joint hourly  joint hourly  joint hourly
      0  120.0   1.00   1.00   0.00   0.00   0.00   0.00
      1   80.0   1.00   1.00   0.00   0.00   0.00   0.00
      2   55.0   1.00   1.00   0.00   0.00   0.00   0.00
      3   40.0   0.50   0.50   0.50   0.00   0.00   0.00
      4   30.0   0.00   0.00   1.00   0.50   0.00   0.00
      5   18.0   0.00   0.00   1.00   1.00   0.50   0.50
      6   10.0   0.00   0.00   1.00   1.00   0.50   1.00
      7   60.0   0.50   0.50   0.50   0.50   0.00   0.50

   Battery, joint (MW; MWh stored at the end of the hour)
   hour  price  charge   disch  stored  import
      0  120.0   0.000   0.900   0.000   0.050
      1   80.0   0.000   0.000   0.000   0.651
      2   55.0   0.000   0.000   0.000   0.200
      3   40.0   0.000   0.000   0.000   0.802
      4   30.0   0.222   0.000   0.200   1.777
      5   18.0   1.000   0.000   1.100   3.057
      6   10.0   1.000   0.000   2.000   3.208
      7   60.0   0.000   0.900   1.000   0.651

   Supply temperature in every run: 357.9 to 358.0 K, setpoint band 356 to 360 K

   day cost                        total
   no limits, hour by hour       1331.39
   ramp limits, hour by hour     1378.69
   ramp limits, joint            1363.25
   ramps and battery, joint      1235.89

   effect                         solver  copper plate
   cost of the ramp limits         31.86         31.88
   value of foresight              15.44         15.41
   value of the battery           127.36        127.33

   Plausibility checks: 675 of 675 passed

The first table sets the joint operating points, which are the same with and
without the battery, next to the hour by hour ones. The second follows the
battery through the joint day; ``import`` is the grid import, including the
battery's draw. Each effect is a difference of two day costs: the ramp
limits cost the joint day with them less the day without them, foresight is
worth the hour by hour day less the joint day, and the battery is worth the
joint day without it less the one with it. The copper plate column is the
same difference on the model without the network, described under
`Checking the result`_.

Where the battery trades
------------------------

The battery sells 0.9 MW at 120 in the first hour, which takes all of its
1 MWh, and so has nothing left for the hours at 80 and 55. It fills its
2 MWh at 30, 18 and 10 and sells 0.9 MW at 60 in the last hour to return to
the 1 MWh it must end with. Writing the 0.222 MW it buys at 30 as 0.2 / 0.9,
what it takes to store 0.2 MWh, it earns

.. math::

   120 \cdot 0.9 + 60 \cdot 0.9 - 30 \cdot 0.2 / 0.9 - 18 - 10 = 127.33

against the solver's 127.36. Losing 19 % on the round trip, it buys only at
prices up to 0.81 times the price at which it next sells. The coupling units
run exactly as without the battery: the grid connection trades any amount
at the hour's price, so the battery changes no unit's margin.

What foresight changes
----------------------

With ramp limits both solves move every unit by at most 0.5 per hour, but
only the joint solve sees the hours ahead. It moves a unit early, giving up
half a margin in one hour to be in position for the next:

- The electric boiler, which earns 40 minus :math:`p / 0.95` at full output,
  starts at 0.5 in hour 3, losing half of 2.11 at 40, so that it runs at
  full output in hour 4 and gains an extra half of 8.42 at 30. Hour by hour
  it only reaches 0.5 in hour 4.
- The electrolyser, which earns 22.75 minus the price, holds at 0.5 in hour
  6, giving up half of 12.75 at 10, so that it can be off in hour 7. Hour by
  hour it climbs to full output in hour 6 and is then stuck at 0.5 in hour 7,
  losing half of 37.25 at 60.

By hand, foresight is worth
:math:`(8.42 - 2.11) / 2 + (37.25 - 12.75) / 2 = 15.41`; the solver finds
15.44. The CHP runs the same in both: it still runs at 0.5 in hour 3, at 40
just below its breakeven of 42.86, because stepping down already in hour 2
would give up more at 55 than it saves at 40.

The figure shows the same day: the price with the breakevens on top, each
unit's operating point below it (solid joint, dashed hour by hour, grey
dotted without ramp limits) and the battery's power and stored energy at the
bottom.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_storage_dispatch.html" width="100%" height="1270" style="border:none;" loading="lazy" title="A coupled day with a battery and ramp limits"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_storage_dispatch.png
      :width: 100%

----

Checking the result
===================

The economic dispatch tutorial checked the merit order against the
breakevens. With ramp limits that check no longer holds: a unit may have to
run where its margin is negative, as the CHP does in hour 3, or stay below
full output where it is positive. A copper plate model takes its place: the
same dispatch as a linear program without the network, solved by SciPy's
HiGHS independently of monee and IPOPT. It is a fair twin because no network
limit binds in any of the four runs; :doc:`mes_network_limits` shows a day
in which two limits bind and the dispatch leaves the merit order.

Each unit :math:`u` earns its margin at full output from the economic
dispatch tutorial, :math:`M_u(p_t)`, times its operating point
:math:`x_{u,t}`; the battery pays the price for what it charges and earns it
for what it discharges; and the heat of the CHP and the electric boiler may
not exceed the 2 MW the consumers draw, because the boiler plant cannot
absorb heat:

.. math::

   \max \sum_{t=0}^{7} \Big[ \sum_u M_u(p_t)\, x_{u,t} + p_t\,(d_t - c_t) \Big]

   |x_{u,t} - x_{u,t-1}| \le 0.5 \quad (t = 1 \text{ to } 7), \qquad
   1.5\, x_{\mathrm{chp},t} + 1.0\, x_{\mathrm{eb},t} \le 2

   e_t = e_{t-1} + 0.9\, c_t - d_t / 0.9, \qquad e_{-1} = e_7 = 1

   0 \le x_{u,t} \le 1, \qquad 0 \le c_t, d_t \le 1, \qquad 0 \le e_t \le 2

Without the battery :math:`c_t = d_t = 0`. The hour by hour twin solves the
same program one hour at a time, each hour ramp limited against the one
before.

.. dropdown:: The copper plate in code

   .. literalinclude:: ../../../examples/mes_storage_and_ramps.py
      :language: python
      :pyobject: copper_plate

Every check recomputes what it tests from the result tables and the
script's constants, or compares it with the copper plate, so none relies on
the solver's own account of its result:

- every hour of every run passes the economic dispatch tutorial's balance,
  conversion, limit and supply temperature checks, with the battery's draw
  taken off the import before the power balance;
- in each ramp limited run no unit steps by more than 0.5, and every
  operating point and battery quantity matches the copper plate;
- each day's cost equals the sum of its hourly costs, and in every hour the
  battery saves exactly the price times what it trades, corrected for the
  change in line losses its flow causes;
- the ramp limits never make the day cheaper, foresight and the battery
  never make it dearer, and the three effects match the copper plate's
  within 1 %, the room the network losses need.

The battery check follows the bookkeeping above hour by hour. It also
requires that the battery never charges and discharges in the same hour,
which the model does not forbid, and that it buys only at prices up to 0.81
times the price at which it next sells:

.. literalinclude:: ../../../examples/mes_storage_and_ramps.py
   :language: python
   :pyobject: check_battery

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_storage_checks.html" width="100%" height="1200" style="border:none;" loading="lazy" title="Checking the joint dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_storage_checks.png
      :width: 100%

The figure plots these checks: the units' steps between hours inside the
band of 0.5 and the stored energy within 0 to 2 MWh, each in its copper
plate ring, the battery's cash flow on the cost it saves, the effects beside
the copper plate's, and the supply temperatures inside the setpoint band.

----

Running it yourself
===================

From the repository root::

    python examples/mes_storage_and_ramps.py

The script prints the report above and exits with status 1 when any check
fails; ``tests/problem/test_mes_storage_example.py`` runs it with the test
suite. It imports the network, the prices and the shared checks from
``examples/mes_economic_dispatch.py``, so keep the two files in the same
folder. Change a battery constant, or ``RAMP_LIMIT`` to another value below
1, at the top of the file, and the copper plate twin and the checks follow.
If an hour of the hour by hour run then fails to converge, drop
``solver_options`` from ``solve_hour_by_hour``: with ``ipopt.mu_strategy``
set, monee skips its own automatic retry.

Next steps
==========

- :doc:`04_mes_economic_dispatch` builds the town, derives the breakevens and
  explains the temperature control and the checks reused here.
- :doc:`mes_network_limits` keeps the hours independent and limits the
  network instead.
- :doc:`../problems/economic_dispatch` states the formulation, including the
  multi-period rows, ``regulation_ramp`` and ``include_storages``.
- :doc:`../how-to/multi_period` covers horizons, initial and terminal states
  and storages in more depth.
