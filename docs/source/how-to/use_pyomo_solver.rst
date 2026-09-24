====================
Use the Pyomo solver
====================

The Pyomo solver interface lets you back monee with any solver that Pyomo
supports, including Gurobi, SCIP, GLPK, CBC, and CPLEX. Use it when
you need a MILP or MIQCP back-end, for example to solve an AC optimal power
flow with the :data:`~monee.model.formulation.EL_MISOCP_FORMULATION`.

----

Prerequisites
=============

Install Pyomo and at least one solver back-end:

.. tab-set::

   .. tab-item:: SCIP (recommended)

      Open-source MINLP / MILP solver. It handles the MIQCQP and MISOCP
      models monee generates for optimal power flow, so it is the safe
      default for the Pyomo back-end.

      .. code-block:: bash

          conda install -c conda-forge pyscipopt

   .. tab-item:: Gurobi

      Commercial solver with a free academic licence available at
      `gurobi.com <https://www.gurobi.com/academia/academic-program-and-licenses/>`_.
      It also solves the MIQCQP and MISOCP models and often outperforms SCIP
      on them when a licence is available. Install the Python bindings after
      obtaining a licence:

      .. code-block:: bash

          pip install gurobipy

   .. tab-item:: GLPK

      Open-source LP / MILP solver. It cannot solve the MIQCQP / MISOCP
      models, so use it only for linear formulations.

      .. code-block:: bash

          conda install -c conda-forge glpk

----

Pick a solver by name
=====================

Pass a solver name to the top-level entry points. Any name other than a GEKKO
solver (``"apopt"``, ``"bpopt"``, ``"ipopt"``) routes to Pyomo automatically:

.. code-block:: python

    import monee

    # Plain energy flow via Pyomo + SCIP
    result = monee.run_energy_flow(net, solver="scip")

    # Optimisation via Pyomo + Gurobi
    result = monee.solve(net, optimization_problem=problem, solver="gurobi")

``"ipopt"`` is available in several back-ends and defaults to CasADi (the
in-process IPOPT backend), falling back to GEKKO's bundled IPOPT when CasADi is
not installed. Force the Pyomo back-end for such ambiguous names with
``backend="pyomo"``:

.. code-block:: python

    result = monee.solve(net, solver="ipopt", backend="pyomo")

To reuse a solver object across solves, or to override the solver per solve,
instantiate :class:`~monee.solver.PyomoSolver` directly:

.. code-block:: python

    solver = monee.PyomoSolver(solver_name="scip")
    result = solver.solve(net)

    # Per-solve override: same instance, different back-end solver
    result = solver.solve(net, solver_name="gurobi")

----

Minimal example
===============

Build a small electricity grid and solve an AC optimal power flow with the
MISOCP relaxation and SCIP:

.. code-block:: python

    import monee
    import monee.express as mx

    # Build the network
    net = mx.create_multi_energy_network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    # Switch to the MISOCP formulation
    net.apply_formulation(monee.EL_MISOCP_FORMULATION)

    # Solve with Pyomo + SCIP
    result = monee.solve(net, solver="scip")
    print(result.objective)

----

Passing an optimisation problem
================================

Pass an :class:`~monee.problem.core.OptimizationProblem` the same way as with
the GEKKO solver:

.. code-block:: python

    import monee
    import monee.problem as mp

    problem = monee.OptimizationProblem()
    problem.controllable_demands([
        (
            "regulation",
            mp.AttributeParameter(
                min=lambda a, v: 0,
                max=lambda a, v: 1,
                val=lambda a, v: 1,
            ),
        )
    ])

    result = monee.PyomoSolver().solve(
        net,
        optimization_problem=problem,
        solver_name="gurobi",
    )

----

Warm starts, options, and robustness
====================================

The Pyomo back-end applies a few conveniences automatically:

.. note::

   **Warm starts:** after every solve, the solution is persisted back into the
   *input* network, and ``warmstart=True`` is passed to the solver whenever it
   reports warm-start capability. Repeated solves on the same network, for
   example in a timeseries loop, therefore start from the previous solution
   without any extra code.

.. note::

   **Per-solver option presets:** monee ships tuned default options for some
   solvers. For Gurobi these are ``ScaleFlag=2``, ``MIPFocus=2``,
   ``MIPGap=1e-3``, and ``TimeLimit=300``, chosen for the MISOCP and
   McCormick formulations. See ``PER_SOLVER_OPTIONS`` in
   ``monee.solver.pyo`` if you need to inspect or adjust them.

.. note::

   **Trivial equations are skipped:** equations that reduce to a plain
   ``True`` or ``False`` during model construction, typically produced by
   deactivated or islanded components, are dropped instead of crashing the
   build. Always-``True`` (tautology) equations are dropped silently, while
   always-``False`` (structurally infeasible) equations are dropped but
   logged at warning level so a genuine modelling error, such as an
   over-deactivated node or branch, is surfaced. This lets problems such as
   load shedding drive the solution even when parts of the network are
   switched off.

Lexicographic objectives (Pyomo only)
-------------------------------------

Some formulations (e.g. MISOCP, McCormick) add auxiliary
relaxation-tightening terms to the objective, which normally have to be
weighted against your own objectives. Constructing the problem with
``OptimizationProblem(lex_objectives=True)`` removes this weight tuning. The
Pyomo back-end then runs a two-phase solve: it first minimises your objectives
alone, then minimises the formulation's auxiliary terms under a cap that holds
the user objective at its phase-one optimum (up to solver tolerance). The GEKKO
back-end ignores the flag and falls back to a single summed objective. See
:doc:`../concepts/solvers` for how this fits into the solver architecture.

----

Solver name reference
=====================

The ``solver_name`` argument is forwarded to ``pyomo.environ.SolverFactory``:

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Value
     - Solver
     - Notes
   * - ``"scip"``
     - SCIP
     - Open-source MINLP / MILP; solves the MIQCQP / MISOCP models.
       Recommended default. ``conda install -c conda-forge pyscipopt``.
   * - ``"gurobi"``
     - Gurobi
     - Commercial, requires a valid licence. Solves the MIQCQP / MISOCP
       models and often outperforms SCIP on them.
   * - ``"glpk"``
     - GLPK
     - Open-source, LP / MILP only.
   * - ``"cbc"``
     - CBC
     - Open-source, LP / MILP.
   * - ``"ipopt"``
     - IPOPT
     - Open-source NLP. Requires a separate IPOPT binary; the string
       ``"ipopt"`` routes to CasADi by default, so pass ``backend="pyomo"``.

Unknown or uninstalled names raise a :exc:`ValueError` that lists the Pyomo
solvers actually installed on your system.

----

Formulation compatibility
=========================

Not every formulation is compatible with every solver back-end. The
GEKKO-specific helpers ``if2``, ``max2``, and ``sign2`` are not available
in the Pyomo translation layer and will raise :exc:`NotImplementedError` if a
formulation tries to use them.

.. note::

   Piecewise-linear constraints are translated differently per back-end:
   Pyomo uses an SOS2 ``Piecewise`` construct, which requires a MIP-capable
   solver (SCIP, Gurobi, CBC, ...), whereas GEKKO approximates the
   same data with cubic splines. A PWL formulation that solves fine under
   GEKKO + IPOPT therefore needs a MIP solver under Pyomo.

.. tip::

   For the Pyomo back-end, prefer:

   - :data:`~monee.model.formulation.EL_MISOCP_FORMULATION` for
     electricity optimal power flow.
   - Custom formulations written with standard Pyomo / monee expressions.

   See :doc:`../concepts/formulations` for the full list of built-in
   formulations and a guide to writing custom ones.

----

Troubleshooting
===============

If the solver reports infeasibility, the result carries a structured
diagnosis: ``result.infeasibility_report`` is attached automatically on
infeasible termination, with constraint residuals, bound violations, and an
optional minimal infeasible subsystem. See
:doc:`diagnose_infeasibility` for how to read and act on it.
