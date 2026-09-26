=====================
Optimization problems
=====================

An energy flow answers "what state does the network settle in for these
setpoints?". An optimization problem turns some of those setpoints into
decisions and asks for the best choice under an objective and a set of
operational limits. monee ships two ready-made problems and the building
blocks to write your own:

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Problem
     - Decides
     - Minimises
   * - :doc:`economic_dispatch`
     - Active power of every generator; with gas and heat dispatched also
       the gas sources, the heat units and the operating point of the
       coupling units. The external grids are always variables that balance
       the network, priced or not.
     - Generation cost, plus the exchange at its price when one is given
       (imports charged, exports credited at the same price), plus the gas
       and heat supply cost of the dispatched carriers.
   * - :doc:`load_shedding`
     - A service level between 0 and 1 for every demand, generator and
       coupling unit, the state of reserve lines and optionally storage
       dispatch.
     - Weighted unserved power (MW) across electricity, gas and heat.
   * - Your own, see :ref:`problems/index:Building your own problem`
     - Any attribute you promote to a decision variable.
     - Any expression you write over the model attributes.

.. toctree::
   :maxdepth: 1
   :hidden:

   economic_dispatch
   load_shedding

The general problem
===================

Every problem monee solves, built-in or custom, has the same shape. For a
network with state :math:`x` (voltages, angles, flows, pressures,
temperatures) and decision variables :math:`u` over the periods
:math:`h \in \mathcal{T}`:

.. math::

   \begin{aligned}
   \min_{x,\,u} \quad & \sum_{h \in \mathcal{T}} \Big( \sum_{j \in \mathcal{J}_h} f_j(x_h, u_h) + A(x_h) \Big) \\
   \text{s.t.} \quad & F_h(x_h, u_h) = 0 && \forall h \in \mathcal{T} && \text{network equations} \\
   & c_h(x_h, u_h) \le 0 && \forall h \in \mathcal{T} && \text{problem constraints} \\
   & \underline{x}_h \le x_h \le \overline{x}_h, \quad \underline{u}_h \le u_h \le \overline{u}_h && \forall h \in \mathcal{T} && \text{variable bounds} \\
   & d(x_{h-1}, u_{h-1}, x_h, u_h) \le 0 && \forall h \in \mathcal{T} && \text{inter-temporal coupling}
   \end{aligned}

The subscript :math:`h` marks what can change between periods. The timeseries
sets the setpoints inside :math:`F_h` and the bounds promoted from them, and
:math:`\mathcal{J}_h` and :math:`c_h` hold only the objective terms and
constraints whose ``when_period`` filter admits the period (called with the
0-based index :math:`h-1`). The filter applies only in
:func:`~monee.simulation.multi_period.run_multi_period`; a timeseries run
ignores it and uses every term in every step. For the first period,
:math:`x_0` and :math:`u_0` are the values carried in from before the run: an
entry of ``initial_state`` passed to
:func:`~monee.simulation.multi_period.run_multi_period` where it names one,
otherwise (and always in a timeseries run) a storage's own initial state of
charge such as ``e_mwh_initial``. An
inter-temporal row with no such value, such as a regulation ramp, is omitted
at :math:`h = 1`. The problem and inter-temporal rows may be equalities as
well as inequalities; the storage state of charge, for example, is an
equality.

A single optimization run is the special case :math:`\mathcal{T} = \{1\}`.
Only :func:`~monee.simulation.multi_period.run_multi_period` solves the summed
problem over a horizon jointly. A timeseries run solves one period at a time:
it fixes :math:`(x_{h-1}, u_{h-1})` at the previous step's solution and
minimises that period's objective
:math:`\sum_j f_j(x_h, u_h) + A(x_h)` alone. Each piece maps onto one part of
the API:

.. list-table::
   :header-rows: 1
   :widths: 22 36 42

   * - Symbol
     - Meaning
     - Where it comes from
   * - :math:`x`
     - Network state.
     - The attributes the models declare as :class:`~monee.model.core.Var`,
       see :doc:`../concepts/data_model`.
   * - :math:`u`
     - Decision variables.
     - Attributes promoted by :meth:`~monee.problem.OptimizationProblem.controllable`
       or one of the typed ``controllable_*`` helpers. Promotion turns a fixed
       setpoint into a bounded :class:`~monee.model.core.Var`.
   * - :math:`F_h`
     - Network equations: nodal balances, branch physics, coupling
       equations. Written :math:`F` rather than the customary :math:`g`
       because the problem pages use :math:`g` for generators.
     - The formulation in force for each carrier, see
       :doc:`../concepts/domains` and :doc:`../concepts/formulations`. They
       are the same equations an energy flow solves.
   * - :math:`\underline{x}, \overline{x}, \underline{u}, \overline{u}`
     - Bounds on the state and the decisions.
     - For :math:`x`, the models' own ``Var`` bounds, the grid bands and any
       :meth:`~monee.problem.OptimizationProblem.bounds` override. For
       :math:`u`, the ``AttributeParameter`` passed at promotion, or the
       interval between zero and the setpoint when none is given.
       :meth:`~monee.problem.OptimizationProblem.bounds` is applied after
       promotion and replaces either kind.
   * - :math:`c_h`
     - Problem constraints.
     - :class:`~monee.problem.Constraints`, one ``.equation(...)`` per row.
   * - :math:`f_j`
     - Objective terms.
     - :class:`~monee.problem.Objectives`, one ``.calculate(...)`` per term;
       all terms are summed.
   * - :math:`A`
     - Auxiliary terms the formulations add to keep their relaxations tight:
       the branch losses :math:`\sum_\ell r_\ell\, \ell_\ell` of the MISOCP
       electricity model, the :math:`\varepsilon`-weighted epigraph terms of
       the relaxed gas and water pipe models, the
       :math:`\varepsilon`-weighted temperature pull
       :math:`10^{-6}(1 - T_\text{pu})` of the McCormick heat model, and the
       duty pull of active heat exchangers.
     - Added automatically, not scaled against the user objective and in mixed
       units: per-unit losses on the grid base and the delivered
       heat-exchanger duty in MW at weight 1 (with a sign that rewards
       delivery), squared mass flows in :math:`(\text{kg/s})^2` at
       :math:`10^{-5}` and a per-unit temperature deficit at
       :math:`10^{-6}`. With the default formulations, the default CasADi
       backend keeps only the heat-exchanger duty pull, so :math:`A` is zero
       on networks without active heat exchangers; a formulation you select
       explicitly keeps its terms on CasADi too. A mixed-integer backend also
       keeps the small epigraph terms of the default gas pipe, water pipe and
       passive heat-exchanger models.
   * - :math:`d`
     - Coupling between periods: storage state of charge, ramp limits,
       linepack, thermal inertia.
     - ``temporal_equation`` on a constraint, the storage models and the
       network aspects; only active in
       :func:`~monee.simulation.multi_period.run_multi_period` and timeseries
       runs.

The sum over periods carries no time-step weight: every period, :math:`A`
included, enters with weight 1 whatever its ``dt_h``. A price per MWh applied
to a power in MW therefore gives a cost rate, not a cost. With equal steps the
reported ``user_objective`` is the energy cost divided by ``dt_h``, and the
optimal dispatch changes only through the relative weight of :math:`A`. With
unequal steps the short periods are over-weighted and the dispatch itself
changes. To minimise energy cost, multiply each period's price by its
``dt_h``.

Building your own problem
=========================

The built-in problems are assembled from the same public pieces you can use
directly. The example below minimises generation cost with the import capped
at 0.3 MW, a stripped-down :doc:`economic_dispatch` without voltage or loading
limits and with an unpriced import. As on that page, :math:`P_g` is the
generation (the generator's ``p_mw`` is :math:`-P_g`), :math:`c_g = 10` its
cost, and :math:`p_e` is the external grid's ``p_mw``, negative for import:

.. math::

   \min_{x,\,P_g} \; c_g P_g \quad \text{s.t.} \quad F(x, P_g) = 0, \quad
   0 \le P_g \le 0.6\,\text{MW}, \quad p_e \ge -0.3\,\text{MW}

.. testcode::

   import monee
   import monee.express as mx
   import monee.model as mm
   import monee.problem as mp

   net = mx.create_multi_energy_network()
   b0, b1 = mx.create_bus(net), mx.create_bus(net)
   mx.create_line(net, b0, b1, 300, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
   mx.create_ext_power_grid(net, b0)
   mx.create_power_generator(net, b1, p_mw=0.6, q_mvar=0.0, cost=10)
   mx.create_power_load(net, b1, p_mw=0.8, q_mvar=0.0)

   problem = mp.OptimizationProblem()
   problem.controllable_generators(["p_mw"])  # u: 0 <= P_g <= 0.6

   objectives = mp.Objectives()  # f_j: c_g * P_g, with P_g = -p_mw
   objectives.select(lambda m, _c: isinstance(m, mm.PowerGenerator)).calculate(
       lambda models: sum(m.cost * -m.p_mw for m in models)
   )
   constraints = mp.Constraints()  # c: import at most 0.3 MW
   constraints.select_types(mm.ExtPowerGrid).equation(lambda m: m.p_mw >= -0.3)
   problem.objectives = objectives
   problem.constraints = constraints

   result = monee.run_energy_flow_optimization(net, problem)
   gen = result.get(mm.PowerGenerator)["p_mw"].iloc[0]
   ext = result.get(mm.ExtPowerGrid)["p_mw"].iloc[0]
   print(f"generation: {-gen:.3f} MW")
   print(f"import: {-ext:.3f} MW")
   print(f"objective: {result.user_objective:.2f}")

.. testoutput::

   generation: 0.503 MW
   import: 0.300 MW
   objective: 5.03

The import cap binds, so the generator covers the rest of the 0.8 MW demand
plus the line losses. Promoting ``p_mw`` without an explicit
:class:`~monee.problem.AttributeParameter` bounds it between zero and the
setpoint at the time the problem is applied, which is why the generator's
capacity is 0.6 MW. A setpoint of zero would lock the variable at zero; pass an
:class:`~monee.problem.AttributeParameter` or call
:meth:`~monee.problem.OptimizationProblem.bounds` to set real bounds. Signs
follow the load convention throughout: a generator's ``p_mw`` is negative and
an import shows as a negative ``p_mw`` on the external grid, see
:doc:`../concepts/conventions`.

:doc:`../tutorials/01_optimization_basics` builds a larger custom problem step
by step, and :doc:`../api/monee.problem` documents every selector, the
selector conventions and the ``AttributeParameter`` options.

Solving a problem
=================

Pass the problem to :func:`~monee.run_energy_flow_optimization` for a single
period, or to :func:`~monee.simulation.multi_period.run_multi_period` together
with a :class:`~monee.simulation.timeseries.TimeseriesData` for a horizon. The
same problem object works for both. :doc:`../concepts/solvers` explains which
backend fits which problem class, and :doc:`../how-to/multi_period` walks
through a horizon.

The result reports three objective values. ``result.objective`` is the whole
minimised value, summed over all periods and including :math:`A`.
``result.user_objective`` is :math:`\sum_h \sum_j f_j(x_h, u_h)` alone and
``result.aux_objective`` is :math:`\sum_h A(x_h)`. These two are ``None`` on
the GEKKO backend, when the problem declares no objective term, and when a
Pyomo or Gurobi solve fails. Compare ``user_objective`` against hand
calculations and external references.

Integer decision variables are treated as continuous between their own bounds
by the default CasADi and IPOPT backend, without any message, so a binary such
as the switch of a reserve line can take any value in :math:`[0, 1]`; the
relaxation warning the backend does emit covers only the direction binaries of
the discrete network formulations. A reserve line can therefore come back half
closed. Solve with a mixed-integer backend when integrality matters, see
:doc:`../how-to/use_pyomo_solver`.
