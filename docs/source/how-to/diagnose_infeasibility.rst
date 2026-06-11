=====================================
Diagnose infeasible and failed solves
=====================================

A solve can fail for many reasons: an over-constrained optimisation problem,
a disconnected island without a slack, bounds that exclude the only physical
solution, or a solver that simply runs out of iterations. monee ships
structured **infeasibility diagnostics** for both back-ends that turn a bare
"Solution Not Found" into a ranked list of violated constraints and variables
with their values and bounds - translated back to monee component names.

This page shows what to check first, and how to read the diagnostic reports
produced by the Pyomo and GEKKO back-ends.

----

First checks
============

Every solve returns a :class:`~monee.solver.core.SolverResult`. Before
reaching for the back-end-specific reports, inspect three fields:

.. code-block:: python

    result = monee.run_energy_flow_optimization(net, problem)

    result.success      # True if the solver reported convergence
    result.violations   # {"<ModelType>.<id>.<attr>": magnitude}
    result.mode_used    # "simulation" | "optimization" | None (Pyomo)

``result.violations`` lists every variable whose solved value lies outside
its declared bounds by more than ``1e-6`` - for example
``{"Bus.4.vm_pu": 0.0123}`` means the voltage magnitude at bus 4 exceeds its
bound by 0.0123 p.u. The dict is also printed at the bottom of
``repr(result)``, so violations are visible in any summary printout. A
"successful" solve with non-empty violations usually means the solver
converged to a point that only *barely* satisfies the constraints, or that a
relaxation left bound noise behind - treat it as a warning sign.

``result.mode_used`` tells you which solve path actually ran. When you
request a simulation (square steady-state solve) but the model is not square,
the GEKKO back-end silently falls back to optimisation mode -
``mode_used == "optimization"`` is the signal that this happened.

Two modelling mistakes cause the vast majority of infeasible solves:

.. note::

   **Disconnected nodes.** Buses or junctions that are topologically
   disconnected (e.g. because a line or pipe is deactivated) have no path to
   any slack and make the whole system unsolvable. Pass
   ``exclude_unconnected_nodes=True`` to the solve call so disconnected
   components are excluded automatically - see
   :doc:`load_shedding` for details on how excluded components are reported.

.. note::

   **Missing slack.** Every electrical island needs an ``ExtPowerGrid`` (or a
   grid-forming generator when islanding is enabled), and every gas/water
   island needs an ``ExtHydrGrid`` or grid-forming source. Without one, the
   balance equations have no degree of freedom to close. See
   :doc:`islanding` for the grid-forming components per carrier.

----

Pyomo back-end
==============

When a Pyomo solve terminates as *infeasible*, the
:class:`~monee.solver.PyomoSolver` **automatically** runs the diagnostics and
attaches the report to the result:

.. code-block:: python

    import monee

    result = monee.PyomoSolver().solve(
        net, optimization_problem=problem, solver_name="scip"
    )

    if not result.success:
        print(result.infeasibility_report.summary(max_items=10))

The report is also logged at ``ERROR`` level at solve time, so it appears in
your console even if you never touch the result object.

``result.infeasibility_report`` is an
:class:`~monee.solver.InfeasibilityReport` dataclass:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Contents
   * - ``constraint_residuals``
     - Constraints whose residual at the final point exceeds the tolerance,
       sorted by magnitude (largest first), each with body value, bounds, and
       the constraint expression.
   * - ``bound_violations``
     - Variables whose value violates their declared bounds, sorted by
       violation magnitude.
   * - ``variables_at_bounds``
     - Variables sitting *exactly on* a bound (within tolerance) - these are
       the active bounds, often the ones to relax.
   * - ``mis_constraints``
     - The constraints and variable bounds forming a **Minimal Infeasible
       Subsystem** (MIS): a smallest set that is infeasible on its own.
   * - ``summary(max_items=10)``
     - Human-readable report of all of the above (``str(report)`` gives the
       same output).

Manual diagnosis
----------------

If you work with the underlying Pyomo ``ConcreteModel`` directly (for
example in a custom solver workflow), build the same report yourself:

.. code-block:: python

    from monee.solver import diagnose_infeasibility

    report = diagnose_infeasibility(
        pm,                     # the Pyomo ConcreteModel after a failed solve
        solver_name="scip",     # used for the MIS computation
        tol=1e-4,               # residual / bound tolerance
        compute_mis_flag=True,  # MIS analysis is slow - disable for speed
    )
    print(report.summary())

The individual primitives are also exported from :mod:`monee.solver`:

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Function
     - Returns
   * - ``collect_constraint_residuals(pm, tol=1e-4)``
     - Violated constraints, sorted by residual.
   * - ``collect_variable_bound_violations(pm, tol=1e-4)``
     - Variables outside their bounds, sorted by violation.
   * - ``collect_variables_at_bounds(pm, tol=1e-4)``
     - Variables within ``tol`` of a bound (active bounds).
   * - ``compute_mis(pm, solver_name="scip")``
     - Constraint / bound names of a Minimal Infeasible Subsystem.

.. note::

   The MIS computation requires **SCIP, Gurobi, or CPLEX** and runs a series
   of auxiliary solves; each internal solve is capped at **10 seconds** so a
   hard MIS computation cannot stall your script. On large models, pass
   ``compute_mis_flag=False`` and start from the constraint residuals
   instead.

Reading variable names
----------------------

Internally, Pyomo variables are named ``{cat}_{id}__{attr}`` (with an
additional ``_t{period}`` segment in multi-period solves). The report
translates them back to readable component descriptions:

.. code-block:: text

    child_3__p_mw        →  child[3].p_mw
    node_7_t2__vm_pu     →  node[7].vm_pu (t=2)

Time limits and other non-OK statuses
-------------------------------------

Only *infeasible* terminations trigger a report. Other non-OK outcomes -
most commonly a time limit reached with a feasible incumbent - are treated
as **success with a warning** ("using witness solution"): the result carries
the best solution found so far, ``result.success`` is ``True``, and no
report is attached. Check the log for the warning if your objective value
looks suboptimal.

----

GEKKO back-end
==============

APMonitor (the engine behind GEKKO) normally fails with nothing more than
``@error: Solution Not Found``. monee instead parses the artifacts APMonitor
leaves in its run directory (``infeasibilities.txt``, ``results.json``) and
raises a :class:`~monee.solver.GekkoSolveError` - a ``RuntimeError`` subclass
whose ``.report`` attribute carries the full diagnosis:

.. code-block:: python

    from monee import run_energy_flow_optimization
    from monee.solver import GekkoSolveError

    try:
        result = run_energy_flow_optimization(net, problem)
    except GekkoSolveError as e:
        print(e.report.summary(max_items=10))

The exception message already embeds the summary, so an unhandled
``GekkoSolveError`` prints the diagnosis in the traceback; the report is
also logged at ``ERROR`` level.

``e.report`` is a :class:`~monee.solver.GekkoInfeasibilityReport` dataclass:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Contents
   * - ``solver_message``
     - The raw error message from GEKKO/APMonitor.
   * - ``infeasible_equations``
     - The equations APMonitor flagged as possibly infeasible at the final
       iterate, ranked by infeasibility magnitude - each with the involved
       variables (value, bounds, marginal).
   * - ``bound_violations``
     - Variables whose failed-iterate value violates their declared bounds.
   * - ``variables_at_bounds``
     - Variables sitting on a bound at the failed iterate.
   * - ``summary(max_items=10)``
     - Human-readable report (``str(report)`` gives the same output).

APMonitor sanitises variable names irreversibly
(``powergenerator-5.p_mw`` becomes ``powergenerator_5_p_mw``); monee records
the mapping while building the model and translates the names in the report
back to the original monee form, both in the equation strings and in the
per-variable listings.

.. note::

   GEKKO diagnostics are **best-effort**: they require the run-directory
   artifacts of a *local* solve. :class:`~monee.solver.GEKKOSolver` always
   solves locally, but if you call the lower-level
   :func:`~monee.solver.diagnose_gekko_infeasibility` on a remote GEKKO
   instance, no artifacts exist and it returns ``None`` instead of a report;
   the diagnosis itself never raises. When no report can be built, the
   original GEKKO exception propagates unchanged.

Even on failure, the partial final iterate is written back to the input
network as a warm start - so after fixing the modelling issue (relaxing a
bound, adding a slack), the next solve resumes from where the failed one
stopped instead of starting cold.

----

See also
========

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: Solvers
      :link: ../concepts/solvers
      :link-type: doc
      :shadow: sm

      GEKKO vs Pyomo back-ends, solver selection, and when to use which.

   .. grid-item-card:: Use the Pyomo solver
      :link: use_pyomo_solver
      :link-type: doc
      :shadow: sm

      Installing Pyomo solver back-ends and solving MISOCP optimal power
      flow.
