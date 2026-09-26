=============================
Network limits and congestion
=============================

**Scenario.** The town of the
:doc:`economic dispatch tutorial <04_mes_economic_dispatch>` keeps its
prices, its PV and its three coupling units. The CHP, the electric boiler and
the electrolyser sit together at a plant site, and this time the site hangs
on two weaker links. Power comes over the grid feeder to the town bus, where
the town load and the PV sit, and on over a 1 km cable, now rated at
1.5 MVA, to the plant bus. Gas comes through a 150 mm main pipe to the town
junction, where the town draws its gas, and on through a 1 km pipe, now
30 mm wide instead of 150 mm, to the plant junction, where the biogas source
feeds in. In the economic dispatch tutorial no network limit binds, so its
dispatch is the merit order up to small losses. Here both limits bind, in
different hours, and the dispatch leaves the merit order.

This tutorial shows which limits the economic dispatch enforces, why these
two bind and when, what each of them costs, and what one more MW through
each bottleneck is worth. Everything shown here comes from
``examples/mes_network_limits.py`` in the repository, which checks every
answer against a hand calculation. The network, the prices and the merit
order are explained in the economic dispatch tutorial and not repeated here.

----

Where the network can bind
==========================

The economic dispatch solves the full network model, so every operating
limit of that model is a constraint of the dispatch, either as a row of the
formulation in :doc:`../problems/economic_dispatch` or as a bound of the
network model's own variables:

- the loading of every line, row (7) of the
  :ref:`problems/economic_dispatch:Mathematical formulation`, on the
  apparent power at both ends when the line has an MVA rating and on the
  current otherwise;
- the voltage band of every bus, row (6), 0.9 to 1.1 pu by default;
- the pressure band of every gas junction, 0.7 to 1.3 in squared per unit by
  default, which is 0.837 to 1.140 pu;
- the temperature band of the water junctions, row (15), and the velocity
  caps of the gas and water pipes;
- the caps of the external grids, such as the gas grid's
  ``max_export_kgs=0``.

In the economic dispatch tutorial none of them comes close. The plant site
is where the dispatch decides: all three coupling units sit there, so the
cable and the pipe to it carry exactly what the dispatch chooses, and
weakening them is the most direct way to make the network matter. The
heating line cannot bind, because its pipes carry what the consumers draw
whichever unit feeds the heat in, so a pipe limit would either make every
hour infeasible or never bind; the voltage band would need kilometres of
thin cable to bind at 20 kV.

The additions to the economic dispatch tutorial's constants, with two steps
that relax each limit a little for the marginal values further down:

.. literalinclude:: ../../../examples/mes_network_limits.py
   :language: python
   :start-at: CABLE_MVA = 1.5
   :end-before: # None keeps

----

Weakening the plant site's links
================================

.. literalinclude:: ../../../examples/mes_network_limits.py
   :language: python
   :pyobject: build_network

``build_network`` finds the cable as the line leaving the town bus and the
pipe as the one ending at the plant junction. ``max_s_mva`` gives the cable
an apparent power rating, and the dispatch then keeps
:math:`(P^2 + Q^2) / (\lambda \overline{S})^2 \le 1` at both of its ends,
with :math:`P` and :math:`Q` the active and reactive power at that end,
:math:`\overline{S}` the rating and :math:`\lambda` the upper value of
``bounds_lp``, 1 by default. ``None`` removes the rating, and the cable falls
back to its default current rating. The result's ``loading_from_pu`` and
``loading_to_pu`` still report the current against ``max_i_ka``, so the
script computes the apparent power from ``p`` and ``q`` at both ends.

In an optimisation every gas junction keeps its squared pressure within the
grid's ``pressure_squared_pu_min`` and ``pressure_squared_pu_max``. The floor
of 0.7 is the default; the script sets it explicitly because the relaxed day
lowers it. To bound single junctions instead, use ``problem.bounds`` on
``pressure_squared_pu`` as in :ref:`problems/economic_dispatch:Recipes`.
``pipe_diameter_m=None`` keeps the 150 mm pipe.

The script solves six days, each with its own settings:

.. literalinclude:: ../../../examples/mes_network_limits.py
   :language: python
   :start-at: # None keeps
   :end-before: LIMITS = {

``neither`` lifts both limits and is the economic dispatch tutorial's day.
The two days with one limit each price that limit, and the two relaxed days
measure what a little more capacity is worth.

----

Solving the day
===============

.. literalinclude:: ../../../examples/mes_network_limits.py
   :language: python
   :pyobject: solve_day

The limits cap each hour on its own and nothing else links the hours, so the
day is still eight independent dispatches and
:func:`~monee.simulation.timeseries.run_timeseries` is still the right tool.

Unlike the economic dispatch tutorial, ``solve_day`` passes no
``solver_options``. That tutorial's ``"ipopt.mu_strategy": "adaptive"``
stops hour 0 with ``Error_In_Step_Computation`` once the pipe is thin, and
monee retries a failed solve with the adaptive strategy only when
``ipopt.mu_strategy`` is not set. With IPOPT's default, the monotone
strategy, every limited day solves and the retry stays available. The day
without limits keeps that tutorial's own ``solve_day``.

----

The day
=======

The script prints this report:

.. code-block:: text

   Hourly dispatch with both limits (1 = unit at full output)
   hour  price    chp e-boil elyser  cable    psq  binding
      0  120.0  0.776  0.000  0.000  0.815  0.700  gas pipe
      1   80.0  0.776  0.000  0.000  0.815  0.700  gas pipe
      2   55.0  0.776  0.000  0.000  0.815  0.700  gas pipe
      3   40.0  0.000  0.000  0.000  0.000  1.009  none
      4   30.0  0.000  1.000  0.000  1.053  1.009  none
      5   18.0  0.000  1.000  0.446  1.500  1.030  cable
      6   10.0  0.000  1.000  0.446  1.500  1.030  cable
      7   60.0  0.776  0.000  0.000  0.815  0.700  gas pipe

   Where a limit binds: congestion cost and the value of one more MW
   hour  price  limit      congestion  lost margin  saving/MW  margin/MW
      0  120.0  gas pipe        18.15        18.15      27.01      27.00
      1   80.0  gas pipe         8.74         8.74      13.00      13.00
      2   55.0  gas pipe         2.85         2.86       4.24       4.25
      5   18.0  cable            2.60         2.63       4.70       4.75
      6   10.0  cable            7.04         7.06      12.72      12.75
      7   60.0  gas pipe         4.03         4.03       6.00       6.00

   Supply temperature in every run: 357.9 to 358.0 K, setpoint band 356 to 360 K

   day                    cost  congestion  lost margins
   both limits         1374.79       43.41         43.47
   cable only          1341.03        9.64          9.69
   gas pipe only       1365.16       33.77         33.78
   neither             1331.39
   cable at 1.6 MVA    1373.06
   floor at 0.68       1371.52

   Plausibility checks: 1144 of 1144 passed

``cable`` is the apparent power on the cable in MVA and ``psq`` the squared
pressure at the plant junction in per unit. `What the limits cost`_ and
`What one more MW is worth`_ explain the columns of the second table; the
last table gives each day's cost and its congestion summed over the day.

The merit order still decides which units run; the limits decide how far.
The pipe holds the CHP at part load wherever it runs, and the cable holds
the electrolyser at part load in the two cheapest hours. In hours 3 and 4 no
limit binds and the dispatch is the economic dispatch tutorial's.

The figure shows the same day: the price, the flows through the cable and
the pipe against their limits, and each unit's operating point, with the day
without limits in grey. That day's flows rise above the limits in exactly
the hours in which the limits bind.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_limits_dispatch.html" width="100%" height="1330" style="border:none;" loading="lazy" title="Where the network binds"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_limits_dispatch.png
      :width: 100%

Why the cable cuts the electrolyser
-----------------------------------

In hours 5 and 6 the electric boiler and the electrolyser would draw
:math:`1 / 0.95 + 1 = 2.05` MW at the plant bus. The cable, with a
resistance of :math:`r = 0.16\,\Omega` at :math:`U = 20` kV and next to no
reactive power, loses :math:`r S^2 / U^2 = 0.0009` MW at its rating and
delivers 1.4991 MW. The dispatch gives that power to the unit that earns
more per MW of it. The electric boiler saves heat at 40 with 0.95 MWh of
heat per MWh of power, so it earns :math:`38 - p` per MW; the electrolyser
earns :math:`35 \cdot 0.65 - p = 22.75 - p`. The electric boiler therefore
runs at full output and the electrolyser gets the rest,
:math:`1.4991 - 1 / 0.95 = 0.4465` MW, the same in both hours because the
town's load and PV sit on the other side of the cable.

Why the gas pipe cuts the CHP
-----------------------------

The default gas formulation drops the squared pressure along a pipe by the
Weymouth equation with the constant friction of a fully rough pipe:

.. math::

   p_i^2 - p_j^2 = K\, m\, |m|, \qquad
   K = \frac{f}{C^2 p_\text{ref}^2}, \qquad
   f = \frac{0.25}{\log_{10}^2\left(\varepsilon / (3.7 D)\right)}, \qquad
   C^2 = \frac{\pi^2 D^5}{16 L R_s T Z}

with the pressures :math:`p` in per unit of :math:`p_\text{ref} = 10` bar,
the flow :math:`m` in kg/s, the pipe's diameter :math:`D`, length :math:`L`
and roughness :math:`\varepsilon`, and the specific gas constant
:math:`R_s`, temperature :math:`T` and compressibility :math:`Z` of the gas.
For the 30 mm pipe, 1 km long with a roughness of 0.01 mm, and the default
L gas (:math:`R_s = 459` J/(kg K), :math:`T = 300` K, :math:`Z = 0.974`),
``weymouth_k`` in the script gives :math:`K = 136.63`, 4300 times the 150 mm
pipe's 0.0318.

The grid connection holds its junction at a squared pressure of 1 and the
main pipe loses only 0.0002 of it, so the plant junction reaches the floor
of 0.7 at

.. math::

   m = \sqrt{(0.99981 - 0.7) / 136.63} = 0.04684 \text{ kg/s} = 1.988 \text{ MW}

at 42.44 MW per kg/s. With the biogas source's 0.340 MW, which enters at the
plant junction itself, the CHP can burn 2.328 MW of its 3 MW, an operating
point of 0.776. ``pipe_capacity_kgs`` solves the same balance exactly. The
pipe's velocity cap allows more than twice this flow and does not bind. In
hours 3 to 6 gas flows back to the town, and the plant junction sits above
the grid connection's pressure.

What the limits cost
--------------------

The congestion cost of a limit is what the day costs more with it than
without it. Each hour, it equals the margins the units lose against the day
without limits plus the line losses that change with the dispatch, priced at
the hour's power price:

.. math::

   C_t = \sum_u M_u(t)\, (x^0_{u,t} - x_{u,t}) + p_t (L_t - L^0_t)

where :math:`M_u(t)` is a unit's hourly margin at full output from the
economic dispatch tutorial, :math:`x` its operating point and :math:`L` the
line losses, with the superscript 0 for the day without limits. The loss
term is why ``congestion`` and ``lost margin`` differ slightly: the cable
costs a little less than the margin it takes, because the smaller flow saves
line losses. The two limits bind in different hours, so their costs add up,
and the day with both costs 3.3 % more than the day without either.

What one more MW is worth
-------------------------

The two relaxed days measure what a little more capacity would save. Each
hour's saving, divided by the extra MW that passes the bottleneck, is the
report's ``saving/MW``. It equals the margin per MW of the unit that runs at
part load, the ``margin/MW`` column: :math:`22.75 - p` for the electrolyser
per MW of power and :math:`0.35\, p + 0.5 \cdot 40 - 35` for the CHP per MW
of fuel. The small gaps on the cable are the marginal line losses: a MW
delivered to the plant bus costs a little more than the grid price.

The ``saving/MW`` is the shadow price of the limit: it separates the price
at the plant site from the price outside, as in nodal pricing. While the
cable binds, one more MW of demand at the plant bus costs the grid price
delivered there, with its marginal losses, plus the shadow price:
:math:`18.05 + 4.70 = 22.75` in hour 5, the electrolyser's breakeven and
well above the grid's 18. While the pipe binds, gas at the plant junction
is worth :math:`35 + 27.01 = 62.01` in hour 0, which is what the CHP makes
of it, :math:`0.35 \cdot 120 + 20 = 62`. A limit that binds splits one price
into two, and on the scarce side the unit at the margin sets the price.

----

Checking the result
===================

The economic dispatch tutorial's check of the merit order fails here by
design, because the limits hold units at part load. A hand dispatch takes
its place: the merit order, cut back where a limit binds, with the cable
going first to the electric boiler and the CHP burning what the pipe brings
plus the biogas.

.. literalinclude:: ../../../examples/mes_network_limits.py
   :language: python
   :pyobject: expected_dispatch

As in the economic dispatch tutorial, the checks do not take the solver's
word: they recompute each quantity from the result tables and compare it
with a hand calculation or an independent identity. They cover

- each unit's operating point against this hand dispatch, in every hour of
  all six days;
- that tutorial's balances, conversions, bands and hourly cost;
- the cable against its rating, the gas junctions against the floor, and
  the pipe's pressure drop and capacity against the Weymouth equation;
- where the limits bind: in exactly the hours in which the day without
  limits would overload them, so none binds by accident and none is missed;
- the congestion identity above, and each ``saving/MW`` against its
  ``margin/MW`` up to the marginal line losses;
- that relaxing or lifting a limit never makes an hour dearer.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_limits_checks.html" width="100%" height="1200" style="border:none;" loading="lazy" title="Checking the constrained dispatch"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_limits_checks.png
      :width: 100%

The figure sets the marginal values, the congestion costs and every
operating point against their hand values, and draws the pipe's Weymouth
curve with the two floors and every hour on it; the grey ring is the flow of
the day without limits, far below the floor.

----

Local optima and the scope of the checks
========================================

IPOPT is a local solver: it stops at a point from which no small step lowers
the cost, and the model has such points away from the optimum, corners such
as a feed-in exchanger whose flow has reached zero or the smoothed split of
the thin pipe's flow into its two directions. Close to this tutorial's
settings, some combinations stop at such a point while still reporting
``Solve_Succeeded``, and which ones depends on the CasADi build; the
settings above reach the optimum on every build we tried, some only through
monee's automatic retry. If you change a limit, rerun the script: the
operating point check catches a point that is not the optimum.
:doc:`mes_local_optima` takes such stops apart and shows how SCIP, a global
solver, finds the optimum and proves it.

The checks themselves assume this tutorial's settings, for example that the
electrolyser is the unit at part load on the cable and that the two limits
bind in different hours. A cable too weak for the electric boiler or the CHP
breaks these assumptions, and the checks then fail although the solver found
the optimum.

----

Running it yourself
===================

From the repository root::

    python examples/mes_network_limits.py

The script prints the report above and exits with status 1 when any check
fails; ``tests/problem/test_mes_network_limits_example.py`` runs it with the
test suite. It imports the network, the prices and the shared checks from
``examples/mes_economic_dispatch.py``.

Next steps
==========

- :doc:`04_mes_economic_dispatch` builds the town, derives the breakevens and
  explains the temperature control and the checks reused here.
- :doc:`mes_storage_and_ramps` couples the hours with a battery and ramp
  limits instead of limiting the network.
- :doc:`mes_local_optima` solves this day with APOPT, which stops short of
  the optimum, traces the stop to the feed-in exchangers, and finds and
  proves the optimum with SCIP on the exact quadratic formulation.
- :doc:`../problems/economic_dispatch` states the formulation, including the
  line loading limit and the gas pressure band.
