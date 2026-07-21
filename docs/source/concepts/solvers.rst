===================
Solvers & Backends
===================

monee turns the network model and its formulation into a mathematical programme
and hands it to a numerical solver back-end. Four back-ends ship with monee:

- CasADi: in-process IPOPT, the default when ``casadi`` is installed. It builds
  the problem once as an in-memory expression graph and calls IPOPT directly,
  with no subprocess. Best for smooth nonlinear energy flow and NLP optimisation.
- GEKKO: bundled, the fallback default when ``casadi`` is absent. Ships its own
  APOPT, BPOPT, and IPOPT binaries.
- Pyomo: routes to any Pyomo-registered solver (Gurobi, SCIP, GLPK, CBC,
  CPLEX, …). Best for MILP and MIQCP problems.
- gurobipy: native in-memory Gurobi, single-period only, an alternative to
  driving Gurobi through Pyomo's file round-trip.

In everyday use you pick a back-end by naming a solver.

Selecting a solver by name
==========================

:func:`~monee.solve`, :func:`~monee.run_energy_flow`, and the other top-level
entry points accept a ``solver=`` string and route it to the right back-end
automatically:

.. testcode::

    import monee.express as mx
    from monee import run_energy_flow, solve

    net = mx.create_multi_energy_network()
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    result = run_energy_flow(net)                # default: CasADi + IPOPT
    result = solve(net, solver="ipopt")          # explicit, same back-end

.. code-block:: python

    # Names other than ipopt/apopt/bpopt route to Pyomo (solver must be installed):
    result = solve(net, optimization_problem=problem, solver="gurobi")
    result = run_energy_flow(net, solver="scip")

The routing rules (implemented in :func:`~monee.solver.resolve_solver` in
``monee.solver.dispatch``) are:

- ``"ipopt"`` routes to the in-process CasADi back-end when ``casadi`` is
  installed, and otherwise to GEKKO's bundled IPOPT binary. This is the default
  solver, so ``solver=None`` resolves the same way.
- ``"apopt"`` and ``"bpopt"`` select the GEKKO back-end (its discrete-capable
  solvers).
- Any other name (``"gurobi"``, ``"scip"``, ``"glpk"``,
  ``"cbc"``, …) is forwarded to Pyomo's ``SolverFactory``.
- ``backend="casadi"`` / ``"gekko"`` / ``"pyomo"`` / ``"gurobipy"`` overrides
  the automatic routing. The CasADi and gurobipy back-ends each provide one
  solver only (``"ipopt"`` and ``"gurobi"`` respectively).
- Misspelt or uninstalled Pyomo solver names raise a :exc:`ValueError` that
  lists the solvers actually installed on your system, so a typo never
  silently falls back to a different solver.

For multi-period problems, :func:`~monee.run_multi_period` follows the same
convention via :func:`~monee.solver.resolve_multi_period_solver`.

Direct instantiation
--------------------

You can also construct a solver object yourself, which is handy when you want to
reuse one instance across many solves or pass backend-specific arguments:

.. code-block:: python

    from monee.solver import GEKKOSolver, PyomoSolver

    result = GEKKOSolver(solver=3).solve(net)            # 1=APOPT, 2=BPOPT, 3=IPOPT
    result = PyomoSolver(solver_name="scip").solve(net)  # name overridable per solve

A concrete instance also works as the ``solver=`` argument of
:func:`~monee.solve` (in that case ``backend=`` must stay unset).

----

.. tab-set::

   .. tab-item:: CasADi (default)

      The CasADi back-end (``monee.solver.casadi``) is the default when the
      optional ``casadi`` package is installed. It builds the NLP once as an
      in-memory `CasADi <https://web.casadi.org>`_ expression graph and calls
      IPOPT in-process, with no subprocess and no text round-trip, so it is
      typically much faster than GEKKO on repeated solves.

      Suitable for:

      - Smooth nonlinear energy flow: AC power flow, Weymouth gas flow,
        Darcy-Weisbach water/heat flow, with gas/heat tables realised as smooth
        cubic B-splines.
      - General NLP optimisation, storage and temporal coupling, and
        multi-period problems.

      Limitations:

      - IPOPT only, so integer variables are relaxed. Models that must enforce
        integrality (MINLP) need APOPT (GEKKO) or a MIP solver (Pyomo).
      - A non-converged solve raises :exc:`~monee.solver.casadi.CasADiSolveError`.

      Usage:

      .. code-block:: python

          result = solve(net, solver="ipopt")   # or backend="casadi"

   .. tab-item:: GEKKO

      The :class:`~monee.solver.GEKKOSolver` (``monee.solver.gekko``) wraps the
      `GEKKO <https://gekko.readthedocs.io>`_ optimisation suite, which ships its
      own solver binaries (APOPT, BPOPT, IPOPT) and needs no extra installation
      beyond ``pip install monee``. It is the fallback default when ``casadi``
      is not installed.

      Suitable for:

      - Nonlinear energy-flow simulation: AC power flow, Weymouth gas flow,
        Darcy-Weisbach water/heat flow.
      - General NLP and MINLP optimisation problems.
      - Fast square steady-state simulation via IMODE=1 (see
        :ref:`concepts/solvers:Simulation vs. optimisation mode`).

      Limitations:

      - MILP / MIQCP problems are handled by APOPT's built-in MINLP solver,
        which can be slow for large mixed-integer models. For those cases use
        the Pyomo back-end with a dedicated MILP solver.
      - Lexicographic objectives fall back to a single summed objective.

      Usage:

      .. code-block:: python

          result = solve(net, solver="apopt")   # or "bpopt"; "ipopt" prefers CasADi

   .. tab-item:: Pyomo

      The :class:`~monee.solver.PyomoSolver` (``monee.solver.pyo``) translates
      the monee model into a `Pyomo <https://www.pyomo.org>`_
      ``ConcreteModel`` and delegates to any solver supported by Pyomo,
      including Gurobi, GLPK, CBC, SCIP, and CPLEX.

      Suitable for:

      - MILP and MIQCP problems, e.g. optimal power flow with binary switching
        decisions using
        :data:`~monee.model.formulation.EL_MISOCP_FORMULATION`.
      - Lexicographic objective ordering
        (``OptimizationProblem(lex_objectives=True)``).
      - Situations where a specific commercial or open-source solver is
        required.

      Limitations:

      - GEKKO-specific operators (``if2``, ``max2``, ``sign2``) are not
        available in the Pyomo back-end. Formulations that rely on them raise
        :exc:`NotImplementedError`. Use Pyomo-compatible formulations or write
        your own with standard Pyomo expressions.
      - Requires the chosen solver binary / Python API installed separately.

      Usage:

      .. code-block:: python

          result = solve(net, solver="scip")   # or "gurobi"

      See :doc:`../how-to/use_pyomo_solver` for a complete worked example
      including MISOCP optimal power flow.

   .. tab-item:: gurobipy (native Gurobi)

      The :class:`~monee.solver.gurobipy.GurobipySolver`
      (``monee.solver.gurobipy``) builds the model in-memory through the
      `gurobipy <https://docs.gurobi.com/projects/optimizer/en/current/reference/python.html>`_
      API and calls Gurobi directly, instead of routing through Pyomo's
      ``SolverFactory`` (which serialises the model to an LP / NL file and
      shells out). It is a direct alternative to ``backend="pyomo"`` with
      ``solver="gurobi"`` for users who have Gurobi installed.

      Suitable for:

      - MIP, MIQCQP, and MISOCP problems, e.g. optimal power flow with
        :data:`~monee.model.formulation.EL_MISOCP_FORMULATION`. Linear,
        quadratic, and second-order-cone terms map straight onto Gurobi's
        ``LinExpr`` / ``QuadExpr`` cones.
      - Smooth NLP energy flow (Weymouth gas, Darcy-Weisbach water / heat),
        built with ``gurobipy.nlfunc`` general nonlinear constraints. This
        path requires Gurobi 12 or newer.
      - Lexicographic objectives
        (``OptimizationProblem(lex_objectives=True)``), via Gurobi's native
        hierarchical multi-objective for linear / quadratic tiers and a
        two-phase fallback for nonlinear ones.

      Advantages over the Pyomo / Gurobi path:

      - No LP / NL file round-trip and no subprocess, so repeated solves and
        timeseries loops are faster.
      - IIS-based infeasibility diagnosis (``computeIIS``), surfaced as a
        :class:`~monee.solver.gurobipy.GurobiIISReport` on
        ``result.infeasibility_report``.
      - Direct warm starts (``Var.Start``), including a build-once timeseries
        driver that re-bounds inputs in place rather than rebuilding the model.

      Limitations:

      - Single-period only. For multi-period problems use ``backend="pyomo"``
        or ``backend="gekko"``.
      - GEKKO-specific operators (``if2``, ``max2``, ``sign2``) are not
        available; formulations that rely on them raise
        :exc:`NotImplementedError`.
      - Requires the ``gurobipy`` package and a valid Gurobi licence
        installed separately (Gurobi 12 or newer for the NLP formulations).

      Usage (note the explicit ``backend=``: a bare ``solver="gurobi"`` routes
      to Pyomo, not to this backend):

      .. code-block:: python

          result = solve(net, solver="gurobi", backend="gurobipy")

          # Or construct it directly to pass Gurobi parameters:
          from monee.solver.gurobipy import GurobipySolver

          result = GurobipySolver(params={"TimeLimit": 60}).solve(net)

----

Simulation vs. optimisation mode
================================

Every solver accepts a ``simulation=`` flag (``solve(net, simulation=True)``;
:func:`~monee.run_energy_flow` defaults to ``simulation=True``). In simulation
mode the formulations re-declare their variables via
``ensure_var(model, simulation=True)``, which pins phantom degrees of freedom
to constants and drops operational flow limits, so the model becomes
square: exactly as many equations as unknowns, with a unique steady
state.

Only GEKKO has a dedicated fast path for this:

- GEKKO attempts a square IMODE=1 solve, which is substantially faster than the
  optimisation mode. If the model turns out not to be square, or carries an
  objective of any kind (IMODE=1 would silently ignore it), the solver logs a
  warning and falls back to IMODE=3 on the same model.
- CasADi and Pyomo have no IMODE concept. A square system solves as an
  objective-free feasibility problem, yielding the same steady state.

Check :attr:`~monee.solver.core.SolverResult.mode_used` to see which path
actually ran on GEKKO: ``"simulation"`` (the fast square path) or
``"optimization"`` (IMODE=3, the signal that a requested simulation silently
fell back). The CasADi and Pyomo back-ends do not distinguish the two: CasADi
always reports ``"optimization"`` and Pyomo reports ``None``.

.. code-block:: python

    result = solve(net, solver="apopt", simulation=True)
    assert result.mode_used == "simulation"   # square IMODE=1 path ran (GEKKO)

----

Working with results
====================

Every solve returns a :class:`~monee.solver.core.SolverResult`:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Field
     - Description
   * - ``network``
     - The solved network copy (the input network is updated separately for
       warm starting, see below).
   * - ``dataframes``
     - ``dict`` mapping model-type name → result :class:`~pandas.DataFrame`.
   * - ``objective``
     - Objective value; ``0.0`` for a plain energy flow, ``None`` when not
       meaningful.
   * - ``success``
     - Whether the solver converged.
   * - ``violations``
     - Bound violations beyond a tolerance of ``1e-6``, keyed
       ``"<ModelType>.<id>.<attr>"`` with the violation magnitude as value.
   * - ``mode_used``
     - ``"simulation"`` / ``"optimization"`` / ``None``, see above.

Convenience accessors:

.. code-block:: python

    import monee.model as mm

    result.summary()      # compact per-type statistics (same as repr(result))
    print(result)         # full per-type table dump
    result.full()         # raw result-DataFrame dict, stringified ids
    result.get(mm.Bus)    # result DataFrame by model class (typo-safe)
    result[bus_1]         # single result row by component id (pandas Series)
    result.plot()         # interactive Plotly network graph

In Jupyter, a ``SolverResult`` renders as collapsible HTML tables. Bound
violations are reported on every result: if ``violations`` is non-empty,
the summary prints a ``VIOLATIONS`` section, making solver tolerance issues
visible without extra checks.

Warm starting
-------------

After every solve (including, on a best-effort basis, *failed* GEKKO solves)
the solved variable values are copied back into the input network
(``persist_solution``). A subsequent solve of the same network therefore
warm-starts from the previous solution, which speeds up timeseries loops and
repeated what-if studies considerably. The Pyomo back-end additionally passes
``warmstart=True`` to solvers that support it (e.g. Gurobi), and the native
gurobipy back-end seeds ``Var.Start`` from the persisted values directly.

The :class:`~monee.simulation.Stepper` carries each successful step's
solution into its working network the same way (``warm_start``, on by
default for every backend except gurobipy, whose ``Var.Start`` seeding
measured neutral), so externally paced loops warm-start step over step. For IPOPT a
near-optimal starting point under the flat-start defaults can even hurt:
the default initial barrier pushes the point back into the interior and
occasionally fails the step computation. The Stepper therefore hints the
CasADi back-end (``CasADiSolver.set_warm_start_hint``, a no-op on the other
back-ends) whenever the injected values are a previous solution, and that
solve runs with IPOPT warm start options (``mu_init=1e-6``, small bound
pushes) instead. On the bundled multi-energy benchmark network this roughly
halves the wall time per step (see ``benchmarks/stepper_warm_start.py``).

----

Lexicographic objectives
========================

By default, all objective terms (your own plus the small formulation-internal
tightening terms, e.g. MISOCP loss minimisation) are summed into a single
objective, which requires careful weighting so that the auxiliary terms never
dominate. With

.. code-block:: python

    from monee.problem import OptimizationProblem

    problem = OptimizationProblem(lex_objectives=True)
    result = solve(net, optimization_problem=problem, solver="gurobi")

the Pyomo back-end instead performs a two-phase solve: phase 1 minimises
only the user objectives; phase 2 minimises the formulation-tightening
auxiliary terms under the constraint that the user objective stays within a
small tolerance of its phase-1 optimum. This removes weight tuning between
user and formulation objectives entirely. The native gurobipy back-end
supports the same ordering (via Gurobi's hierarchical multi-objective for
linear / quadratic tiers, two-phase otherwise); the CasADi and GEKKO
back-ends fall back to the single summed objective.

----

Tuning solver options
=====================

The back-ends ship sensible defaults per solver:

- GEKKO: APOPT (``solver=1``) receives MINLP options (1000 branch iterations,
  gap tolerance ``1e-3``, …); IPOPT (``solver=3``) receives NLP-only options
  (``max_iter 3000``, ``tol 1e-6``, ``constr_viol_tol 1e-6``), since IPOPT
  rejects the ``minlp_*`` keys. See ``DEFAULT_SOLVER_OPTIONS`` and
  ``IPOPT_SOLVER_OPTIONS`` in ``monee.solver.gekko``.
- CasADi: IPOPT runs with ``tol 1e-8`` and ``max_iter 3000`` (``_IPOPT_OPTS``
  in ``monee.solver.casadi``).
- Pyomo: per-solver defaults live in ``PER_SOLVER_OPTIONS`` in
  ``monee.solver.pyo``; for example, Gurobi runs with ``MIPGap=1e-3``
  (roughly kW precision at MW scale) and ``TimeLimit=300``. Extend or
  override these dictionaries to tune a specific solver.
- gurobipy: the native back-end runs with ``MIPGap=1e-3`` and
  ``TimeLimit=300`` (``DEFAULT_GUROBI_PARAMS`` in ``monee.solver.gurobipy``).
  Pass ``GurobipySolver(params={...})`` to merge in any other Gurobi
  parameter.

----

Diagnosing infeasible models
============================

When a solve fails, the modelling back-ends produce a structured diagnosis. The
Pyomo back-end attaches an :class:`~monee.solver.InfeasibilityReport`
(constraint residuals, bound violations, minimal infeasible subsystem) to the
result as ``result.infeasibility_report``. The GEKKO back-end raises a
:class:`~monee.solver.GekkoSolveError` carrying a parsed APMonitor report in its
``.report`` attribute. The native gurobipy back-end attaches a
:class:`~monee.solver.gurobipy.GurobiIISReport` (the irreducible inconsistent
subsystem from ``computeIIS``) to ``result.infeasibility_report``. The CasADi
back-end raises a :class:`~monee.solver.casadi.CasADiSolveError` with the IPOPT
return status. See
:doc:`../how-to/diagnose_infeasibility` for how to read and act on these
reports.

----

Which back-end is fastest?
==========================

The same network and formulation can go to several back-ends, and they reach
the same solution (agreement to ``1e-4`` or better across every benchmarked
case, most to ``1e-8``), so the only thing that separates them is speed. The
in-process back-ends win, because they skip a subprocess or a file round-trip:

- For the smooth NLPs (``EL_NLP``, ``GAS_NLP``, ``HEAT_NLP``, ``SMOOTH_NLP``),
  the in-process CasADi/IPOPT back-end solves the same NLP as GEKKO 3.5 to 18
  times faster, with the largest lead on timeseries loops where GEKKO pays its
  per-step subprocess cost repeatedly. This is the default IPOPT path, so it
  needs no configuration.
- For the conic and quadratic formulations (``EL_MISOCP``, the
  ``*_CONVEX_MIQCQP`` and ``*_NONCONVEX_MIQCQP`` variants), the native gurobipy
  back-end is 2 to 7 times faster than driving the same Gurobi through Pyomo,
  since it skips Pyomo's model-translation layer. The gap grows with model size,
  reaching about 4 times on the coupled multi-sector OPF.

GEKKO and Pyomo stay valuable for what they add rather than their speed: GEKKO
ships its own binaries and an MINLP solver with no extra install, and Pyomo
drives any registered solver (SCIP for global non-convex models, GLPK,
CBC) plus lexicographic objectives.

See :doc:`../benchmarks/backend_selection` for the head-to-head timings and the
per-formulation recommendation behind this summary.

----

Choosing a solver
=================

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Scenario
     - Recommended solver
     - Formulation
   * - AC power flow (simulation)
     - default (``"ipopt"``)
     - ``EL_NLP_FORMULATION``
   * - Gas flow (simulation)
     - default (``"ipopt"``)
     - ``GAS_CONVEX_MIQCQP_FORMULATION``
   * - Water / heat flow (simulation)
     - default (``"ipopt"``)
     - ``HEAT_NONCONVEX_MIQCQP_FORMULATION``
   * - AC optimal power flow (convex relaxation)
     - Pyomo (``"scip"`` / ``"gurobi"``)
     - ``EL_MISOCP_FORMULATION``
   * - Custom MILP problem
     - Pyomo + MILP solver
     - custom formulation

.. tip::

   When in doubt, start with the default (``"ipopt"``): it handles all smooth
   nonlinear problems and runs on CasADi when installed, GEKKO otherwise.
   Switch to Pyomo only when you need a MILP / MIQCP solver, lexicographic
   objectives, or a specific commercial solver back-end.

----

See also
========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Formulations
      :link: formulations
      :link-type: doc
      :shadow: sm

      The equation sets the solvers assemble, and which back-end each
      formulation targets.

   .. grid-item-card:: Use a Pyomo solver
      :link: ../how-to/use_pyomo_solver
      :link-type: doc
      :shadow: sm

      Install SCIP or Gurobi and run a MISOCP optimal power flow.

   .. grid-item-card:: Diagnose infeasibility
      :link: ../how-to/diagnose_infeasibility
      :link-type: doc
      :shadow: sm

      Read ``InfeasibilityReport`` / ``GekkoSolveError`` diagnostics when a
      solve fails.

   .. grid-item-card:: Choosing a backend
      :link: ../benchmarks/backend_selection
      :link-type: doc
      :shadow: sm

      Head-to-head timings: which back-end solves each formulation class
      fastest.
