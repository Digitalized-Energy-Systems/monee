===============================
Local optima and global solvers
===============================

**Scenario.** The town of the
:doc:`network limits tutorial <mes_network_limits>` keeps its prices, its
three coupling units and its two weak links to the plant site, the 1.5 MVA
cable and the 30 mm gas pipe. That tutorial solved the day with IPOPT, the
third-party solver monee calls by default, and closed with a warning: close
to its settings, IPOPT can stop at a point that is not the optimum and still
report success. Here the same day goes to APOPT, a mixed integer solver that
comes with the GEKKO package, which monee can call instead. APOPT reports
every hour as optimal, yet it leaves the CHP off in the four hours in which
it should run, and the day costs 117.15 more than the optimum of 1374.79.

The point of this tutorial is that neither of the two local solvers, APOPT
and IPOPT, is guaranteed to find the global optimum. It compares each of
them one to one with SCIP, an open-source global solver developed at the
Zuse Institute Berlin, on the same hours: APOPT on the network limits day,
and IPOPT on nearby settings where it stops early on some builds. It works
out why a local solver stops where it does, and it shows that the way to the
optimum lies in the formulation: on the exact quadratic formulation SCIP
finds and proves the optimum of every hour, IPOPT confirms both solvers'
dispatches on its own formulation, and a relaxation of that formulation gives a lower
bound that no dispatch can undercut. Everything
shown here comes from ``examples/mes_local_optima.py`` in the repository,
which prints the same report and exits with an error when a check fails. The
network, the prices, the two limits and the hand dispatch are explained in
the network limits tutorial and not repeated here; read it first.

----

Local and global solvers
========================

A solver is local when all it can establish is that no small step from its
answer lowers the cost. The economic dispatch of this town is nonconvex: the
AC power flow, the Weymouth equation of the gas pipes and the products of
flow and temperature in the heating line all bend the feasible set. On such a
problem a point with that property need not be the optimum. None of the
three solvers below is part of monee. They are third-party packages,
developed and maintained elsewhere; monee builds the model from the network
and its formulation and hands it to them. The packages that bring them,
CasADi, GEKKO, Pyomo and pyscipopt, are dependencies that pip installs along
with monee. The three differ in exactly this property, and each needs a
formulation of the network that suits it:

- IPOPT, an interior point solver from the COIN-OR project, which monee
  calls by default through the CasADi package, whose builds bring IPOPT and
  its linear solver along. It solves smooth problems; on this network it
  solves the same problem as ``formulation="smooth_nlp"``, polar AC power
  flow with sines and cosines, and pipe flows whose direction is smoothed
  with square roots instead of decided by a binary. It is local.
- APOPT, a solver that the GEKKO package ships as a binary, selected with
  ``solver="apopt"``. It solves the default formulation and branches on its
  integer variables, here the flow direction binaries of the gas and water
  pipes, solving a smooth problem with a local method at each node of its
  search tree. Its ``optimal`` says that no branch found a better point from
  where that local method stopped, so it is local as well.
- SCIP, from the Zuse Institute Berlin, selected with ``solver="scip"``;
  monee passes the model to it through the Pyomo and pyscipopt packages. A
  spatial branch and bound solver: it bounds every nonconvex term from below
  by a convex relaxation, and it splits the ranges of continuous variables as
  well as integer ones until the best point it has found lies within a set
  gap of the lower bound it has proved. Its ``optimal`` is a proof, up to
  that gap and its tolerances.

SCIP needs a formulation whose nonconvex terms it can bound well.
``nonconvex_miqcqp`` writes every carrier with quadratic terms only: the
power lines as a branch flow model in squared voltages and currents, with
the cone :math:`p^2 + q^2 = w\,\ell` as an equality, the gas pipes with the
Weymouth equation in squared pressures and a direction binary, and the
heating line with its products of flow and temperature. SCIP bounds such
quadratic terms tightly and cheaply, and the sines, cosines and square roots
of the smooth formulation more loosely. In our measurements on this day,
SCIP proved the optimum of ``nonconvex_miqcqp`` at the root node of its
search tree, in about 0.07 s per hour. Given ``smooth_nlp``, the problem
IPOPT solves, it searched 200 to 1400 nodes per hour, and with the tighter
options of `The global optimum`_ it ran into its time limit in five of the
eight hours, returned a dispatch 63 above the optimum in hour 1 and reported
hour 7 infeasible. The formulation decides how much searching a global
solve takes, and with it how tight a tolerance the solve can afford.

The script names the solver and formulation of each approach once:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :start-at: APOPT = {"solver": "apopt"}
   :end-before: DAY_LIMITS = {

``APOPT`` needs no formulation: monee writes the default formulation as a
GEKKO model, and APOPT solves that. ``EXACT`` is SCIP on
``nonconvex_miqcqp`` with the options of `The global optimum`_,
and ``RELAXATION`` loosens it into a lower bound
(`A lower bound from a relaxation`_). IPOPT is the solver monee calls when
none is named, so it needs no entry.

----

Solving the day four ways
=========================

Every solve goes through one function. It builds the network limits
tutorial's network with the given cable, pipe and floor, pins some units'
operating points if asked to, and solves the chosen hours one by one:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :pyobject: solve_hours

``on_step_error="skip"`` records a failed hour instead of raising, so that
an IPOPT failure becomes data rather than the end of the run. ``StepLog``
collects the time of each step and monee's warnings, the only place where
monee reports its automatic retry of a failed IPOPT solve.
``describe_hour`` recomputes each hour's cost from the result tables,
because APOPT's result carries no objective of its own.

A unit's ``regulation`` is its operating point, and the economic dispatch
keeps a variable it finds there, so ``pin_unit`` fixes a unit by setting
both bounds of that variable to the value:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :pyobject: pin_unit

``solve_day`` calls ``solve_hours`` with the network limits tutorial's
cable, pipe and floor once each for APOPT, SCIP, the relaxation and IPOPT,
and once more for the CHP hours by SCIP with the CHP pinned off.

----

The day
=======

The script prints this report:

.. code-block:: text

   The network limits day by APOPT and by SCIP (1 = unit at full output)
                  SCIP dispatch       APOPT      cost of the hour
   hour  price    chp e-boil elyser     chp     APOPT     SCIP    bound
      0  120.0  0.776  0.000  0.000   0.000    359.94   296.99   296.99
      1   80.0  0.776  0.000  0.000   0.000    255.84   225.53   225.53
      2   55.0  0.776  0.000  0.000   0.000    188.53   178.62   178.62
      3   40.0  0.000  0.000  0.000   0.000    151.75   151.75   151.75
      4   30.0  0.000  1.000  0.000   0.000    126.37   126.37   126.37
      5   18.0  0.000  1.000  0.446   0.000    105.62   105.62   105.62
      6   10.0  0.000  1.000  0.446   0.000     91.11    91.11    91.11
      7   60.0  0.776  0.000  0.000   0.000    212.80   198.81   198.81
   day                                        1491.95  1374.79  1374.79

   APOPT costs 117.15 more: 116.97 of CHP margins in hours 0, 1, 2 and 7
   and 0.18 of added line losses.
   SCIP with the CHP pinned off in these hours: 359.94, 255.84, 188.53 and 212.80,
   exactly APOPT's costs.

   Hour 0 by SCIP with the CHP pinned: no second valley on the way to the optimum
   chp         0.000    0.250    0.500    0.750    0.776
   cost       359.94   339.64   319.36   299.10   296.99

   IPOPT on the NLP formulation at each solver's dispatch, cost within 0.001
   of the solver's own: SCIP in 8 of 8 hours, APOPT in 8 of 8.

   Hours 4 and 5 with the 34 mm pipe at a floor of 0.72 (1 = unit at full output)
                        SCIP dispatch         cost of the hour
    cable  hour  price  e-boil  elyser     SCIP    bound  e-boiler off
     1.50     4   30.0   1.000   0.000   126.37   126.37        134.75
     1.50     5   18.0   1.000   0.446   105.62   105.62        124.02
     1.60     4   30.0   1.000   0.000   126.37   126.37        134.75
     1.60     5   18.0   1.000   0.546   105.15   105.15        124.02
     1.70     4   30.0   1.000   0.000   126.37   126.37        134.75
     1.70     5   18.0   1.000   0.646   104.68   104.68        124.02
     1.80     4   30.0   1.000   0.000   126.37   126.37        134.75
     1.80     5   18.0   1.000   0.746   104.21   104.21        124.02
     1.90     4   30.0   1.000   0.000   126.37   126.37        134.75
     1.90     5   18.0   1.000   0.846   103.74   103.74        124.02
     2.00     4   30.0   1.000   0.000   126.37   126.37        134.75
     2.00     5   18.0   1.000   0.946   103.27   103.27        124.02
     2.10     4   30.0   1.000   0.000   126.37   126.37        134.75
     2.10     5   18.0   1.000   1.000   103.02   103.02        124.02

   IPOPT on the NLP formulation at SCIP's dispatch, cost within 0.001 of
   SCIP's own: 14 of 14 hours of the sweep.

   Plausibility checks: 502 of 502 passed

The first table is the network limits day: SCIP's dispatch, APOPT's CHP
(its other two units agree with SCIP's in every hour), and each hour's cost
by APOPT, by SCIP and as the relaxation's lower bound. The lines below it
belong to `Where APOPT stops`_, the last table to
`IPOPT near these settings`_, and the two lines on IPOPT at the solvers'
dispatches to `Checking both dispatches on IPOPT's formulation`_. IPOPT's
own results depend on the build, so the script prints them after the
report.

SCIP's dispatch is the network limits tutorial's hand dispatch, hour for
hour. APOPT reports ``optimal`` in every hour but runs the CHP at 0 in hours
0, 1, 2 and 7, where the hand dispatch has it at 0.776, the most the gas pipe
allows. Its point passes every balance, conversion and limit check of the
economic dispatch tutorial: a valid dispatch, just not the best one. It
forgoes the CHP's margins and imports the CHP's power over the feeder
instead of sending it over the cable from the plant bus, which adds line
losses.

----

Where APOPT stops
=================

Two results in the report locate the stop. With the CHP pinned off, SCIP's
best dispatch costs exactly what APOPT's does in each of the four CHP hours:
APOPT solved the town without its CHP, and solved it well. And with the CHP
pinned along its operating point, hour 0's cost falls almost linearly all the
way from APOPT's stop to the optimum. The dispatch has a single valley; the
second one, which holds APOPT, lies in the model's flows and temperatures.

The zero-flow corner
--------------------

The CHP and the electric boiler feed their heat in through the exchanger of
a feed-in station, which heats water from the return by its design spread of
30 K. Its model ties the heat :math:`q` it feeds in, proportional to the
unit's operating point, to its flow :math:`\dot m` twice, once through the
design flow and once through the energy balance:

.. math::

   \dot m = \frac{q}{c_p \cdot 30\ \text{K}}, \qquad
   \dot m\, c_p\, (T_\text{out} - T_\text{in}) = q

with :math:`c_p` the heat capacity of water and :math:`T_\text{in}` and
:math:`T_\text{out}` the temperatures at the exchanger's inlet and outlet.
Substituting the first into the second leaves

.. math::

   q \left( \frac{T_\text{out} - T_\text{in}}{30\ \text{K}} - 1 \right) = 0

whose solutions are two lines crossing at :math:`q = 0` and a spread of
30 K: the unit off at any spread, or the unit at any output with a spread of
exactly 30 K. A local solver sees only the linearisation. At a point on the
first line with spread :math:`s_0`, it reads

.. math::

   \left( \frac{s_0}{30\ \text{K}} - 1 \right) \mathrm{d}q = 0

so unless :math:`s_0` is exactly 30 K, :math:`\mathrm{d}q = 0` and no first
order step turns the unit on. Every point nearby that satisfies the balance
has :math:`q = 0` as well, and so the same cost: the point is a true local
minimum of the model, and a local solver is right by its own standard to
stop there. APOPT's four stops sit on that line, with practically no flow
through the CHP's exchanger and a spread of about 32 K.

A second lock sits at the unit's control node, which the station's water
passes through. Its heat balance comes down to
:math:`\dot m\, (T_\text{inflow} - T_\text{node}) = 0`: at zero flow the
node's temperature is free, and the linearisation again forbids any flow
while the node differs from the water flowing in. At APOPT's stops the
CHP's node, and with it the exchanger's inlet, sits 4.5 to 4.7 K above the
return. Moving the node to the return temperature and the spread to 30 K
would cost nothing, but nothing in the cost points there either, so no local
method takes that step. Other starts did not help APOPT when we tried, and
neither does its branching, which acts on the pipes' flow directions, not
on a unit's operating point.

The corner is a property of how these balances are written, not of the
dispatch. Writing them linearly, :math:`T_\text{out} = T_\text{in} + 30` K
for the exchanger and :math:`T_\text{node} = T_\text{inflow}` for the
control node, keeps every dispatch and its cost, since water that does not
flow carries no heat, and removes both locks. monee does not write them that
way today; in a test with the two balances patched in, APOPT found the
optimum of this day.

In the figure, the top panel compares the hourly costs, APOPT's hatched grey
and SCIP's blue with the relaxation's bound as diamonds, and the second the
CHP's operating point. The third row shows how much of the cable's rating
and of the gas pipe's capacity at the pressure floor each dispatch uses: in
the four CHP hours SCIP runs the pipe at its limit and sends the CHP's power
over the cable, while APOPT leaves the cable idle and the pipe carrying only
the biogas back to the town. In the other hours the two load both links
alike. The bottom left panel is the solution set of the CHP's exchanger in
the plane of heat output and spread, with APOPT's four stops on the line of
zero output just above 30 K and SCIP's optimum on the 30 K line; the bottom
right panel is hour 0's cost with the CHP pinned.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_local_optima_day.html" width="100%" height="1570" style="border:none;" loading="lazy" title="Where APOPT stops"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_local_optima_day.png
      :width: 100%

----

IPOPT near these settings
=========================

IPOPT reaches the optimum on the network limits tutorial's own day on every
build we tried; the IPOPT output the script prints after the report opens
with that day, so you can check yours. Its early stops show up close by. The
second part of the script solves hours 4 and 5, at prices of 30 and 18, with
the plant pipe widened to 34 mm, its floor raised to 0.72 and the cable
rated at 1.50 to 2.10 MVA in steps of 0.1 MVA:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :pyobject: solve_sweep

We chose this pipe and floor because a scan of 1539 settings found IPOPT's
early stops clustered there with CasADi 3.7.2; nothing in the physics
singles them out. The hand dispatch still follows the network limits
tutorial, which derives the pipe's and the cable's capacities: the CHP is
off in these hours, the cable serves the electric boiler first, and in hour
5 the electrolyser takes what the cable has left, so the optimum falls with
the rating until the electrolyser reaches full output. The
``e-boiler off`` column is the best dispatch with the electric boiler pinned
off; its excess over the optimum is the margins the units give up plus the
change in line losses, the network limits tutorial's congestion identity
with a pin in place of a limit.

Two behaviours
--------------

Near these settings IPOPT does not reliably reach the optimum, and how it
misses depends on the build of CasADi, and so of IPOPT and its linear
solver. We saw two behaviours:

- With CasADi 3.7.2, IPOPT stopped in most hour 4 solves, and at one or
  more ratings in hour 5, with the electric boiler off, and reported
  ``Solve_Succeeded`` at the first attempt. These stops cost exactly the
  ``e-boiler off`` column: IPOPT found the best dispatch without the
  electric boiler, with the
  boiler's exchanger at zero flow and its spread off 30 K, the same corner
  that holds APOPT's CHP.
- With CasADi 3.8.1, the same settings never stopped silently. Most hours
  reached the optimum only through monee's automatic retry, and a few
  retries ended at ``Solved_To_Acceptable_Level`` far from the optimum.

The first is the worse behaviour, because nothing in the result says that
anything went wrong. The table the script prints after the report shows
IPOPT's status, dispatch, excess and retry for every sweep hour, so you can
see which behaviour your build shows.

In hour 4 the cable never binds, and yet IPOPT's answer changes with the
rating. IPOPT keeps every inequality strictly satisfied and adds a barrier
term :math:`-\mu \log s` for each slack :math:`s`, lowering :math:`\mu` as it
converges. The barrier vanishes at the solution, but it shapes the path that
leads there, and a limit shapes it whether or not it binds at the end. In
the cases we traced, IPOPT's first iterations pull all units down from their
start at full output, and a unit whose flow reaches zero with its spread off
30 K on the way stays there.

monee's safety nets cannot catch such a stop. Its automatic retry reruns the
IPOPT solve with IPOPT's adaptive barrier strategy only when the first
attempt ends with a status other than ``Solve_Succeeded``, and
``solver="auto"`` falls back to SCIP only after a numerical failure (see
:ref:`concepts/solvers:Automatic back-end fallback`). A silent stop is
``Solve_Succeeded``, so both return it unchanged.

Starting IPOPT from several points mitigates this. Solving each hour also
with all units off and with all at half output, set like the pin in
``pin_unit`` but with the bounds 0 and 1, and keeping the cheapest result
that ended ``Solve_Succeeded`` reached the optimum in every sweep hour on
every build we tried. It certifies nothing, though, and it did not help
APOPT.

The figure is computed when these pages are built, so its IPOPT markers show
the behaviour of the machine that built them, not necessarily of yours. The
top two panels compare IPOPT one to one with SCIP in hours 4 and 5, with a
pair of bars for every cable rating: SCIP's global optimum in dark blue, and
IPOPT's cost in sky blue where it reached the optimum, hatched if only after
the retry, and in red, labelled with its excess, where it did not. The
bottom panels count these outcomes and give the seconds per hourly solve of
each approach.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_local_optima_sweep.html" width="100%" height="1170" style="border:none;" loading="lazy" title="IPOPT near the network limits"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_local_optima_sweep.png
      :width: 100%

----

The global optimum
==================

SCIP on ``nonconvex_miqcqp``, ``EXACT`` above, reaches the hand dispatch in
every hour of both parts and ends ``optimal`` in each. It needs three
options:

- ``numerics/feastol`` at 1e-9. The thin pipe's squared flows are of the
  order of :math:`10^{-4}\ (\text{kg/s})^2`, so SCIP's default feasibility
  tolerance of :math:`10^{-6}` is a noticeable fraction of them. With it,
  the pipe's pressure drop missed the Weymouth equation by up to 0.4 %, more
  than the script's check allows, although the dispatch was right.
- ``limits/gap`` at 1e-6. With the gap of :math:`10^{-4}` that monee passes
  to SCIP unless told otherwise, the relaxation below could stop slightly
  above the exact optimum with a loose cone.
- ``limits/time`` at 60 s, so that a hard hour cannot stall the script. A
  SCIP solve that runs into the limit still returns its best point so far,
  which ``run_timeseries`` accepts, so the script checks that every SCIP
  solve ended ``optimal``.

The two tighter tolerances cost no measurable time. ``optimal`` means that
SCIP proved no dispatch cheaper by more than the gap, within its
tolerances. The result carries the cost of the dispatch SCIP found, not the
lower bound it proved, so the status is the certificate; the relaxation
below gives a second one that can be read off a result.

On this day a SCIP hour took about as long as an IPOPT hour. Gurobi, a
commercial solver that monee can also call, reached SCIP's optimum in every
sweep hour with its tolerances tightened the same way, but in our
measurements it was not exact on the day: the thin pipe missed the Weymouth
equation by more than the script's check allows.

----

Checking both dispatches on IPOPT's formulation
===============================================

The comparison of APOPT's day with SCIP's rests on two different models, and
neither is the one IPOPT solves. SCIP solves ``nonconvex_miqcqp``: the power
lines as a branch flow model in squared voltages and currents instead of
polar AC, the gas pipes with the exact Weymouth equation and a direction
binary instead of a flow direction smoothed with a square root, and the
heating line with a direction binary instead of smoothed upwinding. APOPT
solves the default formulation, polar AC like IPOPT's, but with the Weymouth
equation relaxed into an inequality as in `A lower bound from a relaxation`_.
A dispatch that is cheap only in one of these models would make the
comparison unfair.

The script therefore takes both solvers' dispatches of the day, and SCIP's
of the sweep, and solves every hour once more with IPOPT on its own
formulation, with every unit pinned at the solver's operating point:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :start-at: def pinned_dispatch
   :end-before: def solve_day

A unit that is off is pinned at 1e-6 rather than at zero. At zero flow its
feed-in exchanger sits in the corner of `Where APOPT stops`_, and pinned
there, IPOPT failed in four of the five hours in which APOPT runs no unit; 1e-6
of a unit's output moves an hour's cost by far less than the check's
tolerance of 0.001.

In every hour both dispatches cost on IPOPT's formulation what they cost on
their own, within 0.0002 in our runs, as the report's two lines on it say.
On IPOPT's formulation SCIP's day costs 1374.80 and APOPT's 1491.95, the same
gap of 117.15, so the comparison holds in IPOPT's own model. For SCIP's
dispatch the physical state agrees as well: voltages within
:math:`10^{-8}` pu, squared pressures within :math:`3 \cdot 10^{-7}` and
water temperatures within :math:`10^{-5}` K, and the limits bind where they
bind for SCIP, the plant junction at the floor of 0.7 in the CHP hours and
the cable at its rating in hours 5 and 6. APOPT's relaxed gas model is
looser: in hour 0 it puts the town and plant junctions at squared pressures
of 0.85 and 0.86, where the Weymouth equation gives 1.0 and 1.009 for the
same flows. APOPT's own pressures are not physical there, although its dispatch
is valid on IPOPT's formulation. This held with CasADi 3.7.2 and 3.8.1, on
Windows and on Linux.

In the figure, the top panel is the cost of each hour on IPOPT's formulation
for APOPT's and SCIP's dispatch, with each solver's cost on its own
formulation as a black cap on its bar. The bottom panels show how far the
cost and the squared pressure move between IPOPT's formulation and the
solver's own: the cost far inside the check's tolerance, the pressure only
in APOPT's hour 0.

.. only:: html

   .. raw:: html

      <iframe src="../_static/interactive/tutorial_mes_local_optima_nlp.html" width="100%" height="920" style="border:none;" loading="lazy" title="Both dispatches on IPOPT's formulation"></iframe>

.. only:: latex

   .. image:: /_static/interactive/tutorial_mes_local_optima_nlp.png
      :width: 100%

----

A lower bound from a relaxation
===============================

A relaxation loosens constraints so that every point of the exact model
stays feasible in it. Its optimum can then only be lower, and it is usually
easier to solve. ``RELAXATION`` loosens two carriers. ``el_misocp`` turns
the cone equality into the inequality :math:`p^2 + q^2 \le w\,\ell`, and
``gas_convex_miqcqp`` turns the Weymouth equation into an inequality that
allows a pipe more pressure drop than its flow needs. The heating line keeps
its exact bilinear model, so this is a relaxation of power and gas only, and
SCIP still has to solve it globally. The fully convex bundle
``convex_miqcqp`` would relax the heating line as well, but the economic
dispatch rejects it: pricing heat needs the nodal heat balance that its
McCormick envelopes replace.

A lower bound turns any dispatch into a verdict. Its cost minus the bound is
the most that a better solver could still save: 117.15 on APOPT's day, all
of which SCIP's exact solve finds, and 8.38 on a silent IPOPT stop in
hour 4. On SCIP's dispatch the report's ``bound`` columns leave nothing to
save, to two decimals, in every hour, so the optimum is confirmed twice, by
SCIP's status and by a bound computed separately, up to SCIP's gap and the
few thousandths per hour of small terms that the formulations add to the
objective to keep their relaxations tight.

The relaxed point itself is only partly physical. The power flow cone is
tight, :math:`w\,\ell` and :math:`p^2 + q^2` agree on every line, and on a
radial network like this one a tight cone means that the relaxed power flow
is an AC power flow. The gas inequality is loose where the floor does not
bind: in hours 3 to 6 the relaxed plant pipe drops up to 1.29 times the
Weymouth drop of its flow, so the relaxed gas pressures are not physical,
although the flows are.

----

Checking the result
===================

The script runs 502 checks, a count that depends neither on the machine nor
on what IPOPT does. Most compare a result with something computed without a
solver: the hand dispatch, the economic dispatch tutorial's balance,
conversion and limit checks, the cost recomputed from the result tables and
the Weymouth equation of the thin pipe. They cover

- SCIP's exact solve: ``optimal``, at the hand dispatch, passing those
  checks and on the Weymouth curve;
- the relaxation: ``optimal``, at or just below the optimum, with a tight
  power flow cone;
- APOPT: passing those checks, never cheaper than the optimum, and its stop
  and hour 0's cost curve as the report shows them;
- the electric boiler pinned off: the congestion identity;
- IPOPT at SCIP's and APOPT's dispatch: at that dispatch and its cost,
  passing those checks, and within the pipe's floor and the cable's rating.

The check of APOPT's stop, for example:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :pyobject: check_apopt_stop

The checks at the two dispatches rest on IPOPT and held on every build we
tried. IPOPT's own result can only be checked for what holds on every
build: it must not undercut the global optimum and, if it is off the hand
dispatch, must cost more than it, while a failed solve passes:

.. literalinclude:: ../../../examples/mes_local_optima.py
   :language: python
   :pyobject: check_ipopt

The checks of APOPT's stop assume that it stops. Should monee one day write
the two balances linearly, those checks fail and flag that this page needs
rewriting.

----

What did not work
=================

Besides ``solver="auto"``, ``convex_miqcqp``, SCIP with its default
options or on the smooth formulation and other starts for APOPT, all covered
above, we tried the following while this tutorial was written:

- ``ipopt.mu_strategy="adaptive"`` failed outright in many sweep hours, and
  setting it switches off monee's retry.
- ``formulation="smooth_nlp"`` with IPOPT builds the same problem as the
  default with the variables in another order, and left more sweep hours
  off the optimum than the default did.
- ``ipopt.s_max=1e10`` avoided the silent stops, but hours failed outright
  or needed the retry, and the sweep ran much slower.
- Warm starting each hour from the previous one's solution, as the
  :class:`~monee.simulation.stepper.Stepper` can, starts a unit that was off
  the hour before right in its corner; near this day's settings it left
  hours off the optimum that a cold start solved.
- Narrower pipes did not make IPOPT stop in the CHP hours.

----

Running it yourself
===================

From the repository root::

    python examples/mes_local_optima.py

The script takes about 30 s. It prints the report above, then IPOPT's
results, and exits with status 1 when any check fails;
``tests/problem/test_mes_local_optima_example.py`` runs it and compares the
report with the one on this page. It builds on
``examples/mes_network_limits.py`` and ``examples/mes_economic_dispatch.py``,
so the hand dispatch and the checks inherit the assumptions of the network
limits tutorial; outside them the checks fail although SCIP found the
optimum.

Next steps
==========

- :doc:`mes_network_limits` builds the two limits, prices the congestion and
  values one more MW through each bottleneck.
- :doc:`04_mes_economic_dispatch` builds the town, derives the breakevens and
  explains the temperature control and the checks reused here.
- :doc:`../concepts/formulations` describes every formulation named here and
  how to combine them per carrier.
- :doc:`../concepts/solvers` covers the back-ends, their options, the
  automatic retry and the fallback to SCIP.
