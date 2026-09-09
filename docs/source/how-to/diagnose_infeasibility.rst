=====================================
Diagnose infeasible and failed solves
=====================================

A solve can fail for many reasons: an over-constrained optimisation problem,
a disconnected island without a slack, bounds that exclude the only physical
solution, or a solver that runs out of iterations. monee turns a bare
"Solution Not Found" into a ranked list of violated constraints and variables,
each with its value and bounds, translated back to monee component names.

This page shows what to check first, then how to read the diagnostic reports
from the Pyomo and GEKKO back-ends.

----

First checks
============

Every solve returns a :class:`~monee.solver.core.SolverResult`. Before
reaching for the back-end-specific reports, inspect three fields:

.. code-block:: python

    result = monee.run_energy_flow_optimization(net, problem)

    result.success      # True if the solver reported convergence
    result.violations   # {"<ModelType>.<id>.<attr>": magnitude}
    result.warnings     # typed validation findings, see the next section
    result.mode_used    # "simulation" | "optimization" | None (Pyomo)

``result.violations`` lists every variable whose solved value lies outside
its declared bounds by more than ``1e-6``. For example
``{"Bus.4.vm_pu": 0.0123}`` means the voltage magnitude at bus 4 exceeds its
bound by 0.0123 p.u. The dict also prints at the bottom of ``repr(result)``,
so violations show up in any summary printout. A successful solve with
non-empty violations is a warning sign: the solver converged to a point that
only barely satisfies the constraints, or a relaxation left bound noise
behind.

``result.mode_used`` tells you which solve path actually ran. When you request
a simulation (a square steady-state solve) but the model is not square, the
GEKKO back-end silently falls back to optimisation mode. A
``mode_used == "optimization"`` is the signal that this happened.

Both of the modelling mistakes below, and a few more, are caught before any
solve by the pre-flight lint (``net.check()``, next section). Two modelling
mistakes cause the vast majority of infeasible solves:

.. note::

   **Disconnected nodes.** Buses or junctions that are topologically
   disconnected (e.g. because a line or pipe is deactivated) have no path to
   any slack and make the whole system unsolvable. Plain energy-flow solves
   exclude such components automatically. For optimisation solves, attach an
   islanding config (see :doc:`islanding`), the preferred mechanism, or pass
   the legacy ``exclude_unconnected_nodes=True`` flag to the solve call. See
   :doc:`load_shedding` for how excluded components are reported.

.. note::

   **Missing slack.** Every electrical island needs an ``ExtPowerGrid`` (or a
   grid-forming generator when islanding is enabled), and every gas/water
   island needs an ``ExtHydrGrid`` or grid-forming source. Without one, the
   balance equations have no degree of freedom to close. See
   :doc:`islanding` for the grid-forming components per carrier.

----

Pre-flight lint
===============

Many failed or silently wrong solves are visible in the network structure
before any solve. :meth:`~monee.model.network.Network.check` runs cheap
structural checks, prints each finding and returns them as a list of typed
:class:`~monee.model.network.CheckFinding` entries; an empty list means the
network passed:

.. code-block:: python

    findings = net.check()
    if findings:
        raise SystemExit("fix the network before solving")

Each finding has a ``category``, a ``message`` and, where the finding is
local, a ``component``:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Category
     - What was found
   * - ``missing_slack``
     - A carrier has nodes but no active slack or grid forming child
       (``ExtPowerGrid``, ``ExtHydrGrid``, or a grid forming component with
       islanding enabled). The solver excludes every island without one, so
       the whole carrier would be pruned from the solve.
   * - ``heat_only_dead_end``
     - A junction of degree 1 or less whose only children are heat only
       (``HeatLoad`` / ``HeatGenerator``). With no mass flow to carry the
       heat, the solver prunes the junction and its children; connect a
       return pipe or add a mass flow child (``Sink`` / ``Source``).
   * - ``mixed_base_kv``
     - A branch that is not a transformer connects buses with different
       ``base_kv``. Per unit branch parameters are computed from one side's
       base, so voltages and flows come out silently skewed; insert a
       :class:`~monee.model.branch.Trafo` or align the bases.
   * - ``dof``
     - The simulation mode model is not square: the equation system was
       assembled (without solving) and the variable count does not match the
       equation count. Variables that appear in no equation are listed by
       name. A simulation then returns one feasible point, not a unique
       setpoint.
   * - ``check_error``
     - The squareness preview could not assemble the model at all; a solve
       would most likely fail the same way, and the message carries the
       assembly error.

The ``dof`` preview is the only check that builds the equation system (it
still performs no solve); pass ``net.check(dof_preview=False)`` to skip it on
very large networks.

Checking a network headed into an optimisation
----------------------------------------------

``net.check()`` lints for a simulation by default, so it reports every
variable no equation pins. In an optimisation those free variables (storage
energies, coupler setpoints, load regulation, ...) are exactly what the
problem is expected to bind, and the ``dof`` finding is pure noise. Declare
the intent instead:

.. code-block:: python

    findings = net.check(mode="optimization")

``mode="optimization"`` keeps the graph checks (``missing_slack``,
``heat_only_dead_end``, ``mixed_base_kv``) and skips the squareness preview;
``mode="simulation"`` is the default and keeps it. Any other value raises
``ValueError``. A solve that runs in optimisation mode without any objective
still reports its undetermined variables afterwards, as a ``dof`` entry on
``result.warnings`` (see the table below).

Pruned components at solve time
-------------------------------

When a solve excludes components you created (an island without a slack, a
dead end stub, a heat only dead end junction), it emits one warning naming
the removed components and the rule that removed them, and attaches the same
text as a ``pruned`` entry on ``result.warnings``. ``net.check()`` reports
the same situations before solving, so a clean pre-flight means no surprise
pruning later.

----

Result validation and strict mode
=================================

Failure is not the only bad outcome: a solver can report success on a result
that quietly differs from what you asked for. After every solve, monee runs a
set of cheap invariant checks on the solved values (no extra solves) and
attaches the findings to ``result.warnings`` as typed
:class:`~monee.solver.core.ResultWarning` entries. When any are found, one
summary line is emitted through Python's ``warnings`` module (category
:class:`~monee.solver.core.ResultValidationSummary`, so
``warnings.filterwarnings("ignore", category=...)`` silences it) and the full
entries stay on the result. Later solves in the same process with the exact
same finding profile, for example every step of a timeseries run, are counted
instead of re-reported;
:func:`~monee.solver.core.validation_summary_counts` returns the counts and
:func:`~monee.solver.core.reset_validation_summary_counts` clears them
between independent runs. The checks apply noise floors so solver tolerance
does not surface as findings: served or delivered deltas below 1e-4 MW (or
kg/s), balance residuals below tolerance and relaxation gaps at negligible
flow are not reported, and a child whose ``regulation`` the solved problem
itself made controllable (intentional shedding) produces no ``served_delta``
entry. Anything beyond a floor still fires:

.. code-block:: python

    result = monee.solve_load_shedding_problem(net, bounds_ext_el=(-2, 0))
    for w in result.warnings:
        print(w.category, w.component, w.message)

Each entry has a ``category`` (a stable key), a ``message``, and where the
finding is local, a ``component`` and a ``value`` with its magnitude:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Category
     - What was found
   * - ``energy_balance``
     - The per carrier balance does not close: summed child injections and
       branch flows leave a net imbalance beyond tolerance. The check runs on
       carriers not touched by multi energy couplers (there the closure
       identity spans carriers and is skipped).
   * - ``served_delta``
     - A demand was served below its setpoint: a load or sink whose
       regulation ended below 1, or a fixed duty heat exchanger whose
       delivered power fell short of its setpoint.
   * - ``relaxation``
     - A relaxed formulation ran and its relaxation is not tight (epigraph
       gap beyond the formulation's tolerance). The reported physics is then
       not exact; solve with SCIP or Gurobi, or use a smooth formulation.
   * - ``integrality``
     - Components carry a formulation with integer variables that the
       back-end relaxes (for example a direction binary on CasADi/IPOPT).
       Solve with SCIP or Gurobi to enforce them.
   * - ``pruned``
     - Components the solve excluded through pruning or ignore logic. The
       entry names each removed component and the rule that removed it (no
       slack or grid forming child in its island, dead end stub, heat only
       dead end junction); the same text is emitted as a warning at solve
       time. They show up as NaN or ignored in the result frames, and
       ``net.check()`` reports the same situations before solving.
   * - ``bound_active``
     - A user given bound binds at the optimum (within tolerance), for
       example an external grid exchange cap sitting exactly at its limit.
       The bound is shaping the answer; check that it is intentional. A
       variable sitting at its own setpoint bound (an islanding gated
       injection, or a load at the served end of a zero to setpoint gate) is
       the normal fully served state and is not reported.
   * - ``dof``
     - A simulation solve was requested but the model is not square, so the
       solved values are one feasible point rather than a unique setpoint.
       In optimisation mode the same category fires when there is no
       objective at all and more variables than equations: nothing selects
       among the feasible points, so the solver returns an arbitrary one.
       Pin the free variable with an equation, give it a plain float or
       ``Const`` value, or minimise an objective over it. An optimisation
       that does carry an objective is never reported here: picking a point
       is exactly its job.

A clean solve has an empty ``result.warnings`` list. A failed solve is not
validated at all: its values are meaningless and the failure is already
reported through the mechanisms below.

Strict mode
-----------

Interactive exploration tolerates warnings; CI runs and paper results should
not. Passing ``strict=True`` to any solve entry point
(:func:`monee.run_energy_flow`, :func:`monee.run_energy_flow_optimization`,
:func:`monee.solve_load_shedding_problem`, or a backend ``solve``) turns the
warnings into a raised :class:`~monee.solver.core.ValidationError` that lists
every entry:

.. code-block:: python

    try:
        result = monee.run_energy_flow(net, strict=True)
    except monee.ValidationError as e:
        for w in e.warnings:
            print(w)

The default stays permissive: warnings are recorded and summarised, the
result is returned. Note that on a back-end that relaxes integer variables,
any network whose formulation needs them (an ``integrality`` entry) will
always raise under strict; that is the intended nudge towards SCIP or Gurobi
or a binary free formulation.

----

Pyomo back-end
==============

When a Pyomo solve terminates as infeasible, the
:class:`~monee.solver.PyomoSolver` runs the diagnostics and attaches the
report to the result:

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
     - Variables sitting on a bound (within tolerance); these are the active
       bounds, often the ones to relax.
   * - ``mis_constraints``
     - The constraints and variable bounds forming a Minimal Infeasible
       Subsystem (MIS): a smallest set that is infeasible on its own.
   * - ``mis_message``
     - Why no MIS is available, when ``mis_constraints`` is empty. Never
       empty in that case: it names the failure and, when the MIS routine's
       internal solves did not reach optimality, how many of them ended in
       what condition.
   * - ``violation_hotspots(limit=10)``
     - The violated constraints collapsed onto the components that own them,
       ranked by how many of a component's equations are violated and then by
       the largest residual.
   * - ``summary(max_items=10)``
     - Human-readable report of all of the above (``str(report)`` gives the
       same output). Every list is capped at ``max_items`` entries, the
       residual list per carrier group.

Reading a report with hundreds of violations
--------------------------------------------

An infeasible multi-energy solve routinely produces several hundred violated
equations, because a solve that never found a feasible point reports the
residuals of wherever it stopped, and one broken sector shows up as balance
residuals everywhere. The report is ordered so the raw list is the last thing
you read:

1. The Minimal Infeasible Subsystem, when one could be isolated, or
   ``mis_message`` saying why not.
2. The infeasibility explanation (the elastic phase's "feasible after
   relaxing" set), which often survives even when the MIS refinement fails.
3. The violation hotspots: a few hundred equations collapsed onto a handful
   of named components, most-violated first. This is the closest thing to a
   localisation the report has when no MIS was isolated.
4. Variable bound violations and the active bounds.
5. The raw residuals, grouped by carrier and capped per group.

Start at the top and stop as soon as a section names a component. A component
with several violated equations is a better lead than a single large residual
somewhere else, because residual magnitudes are in different units per carrier
and are not comparable across them.

.. code-block:: python

    report = result.infeasibility_report
    for spot in report.violation_hotspots(limit=5):
        print(spot["component"], spot["count"], spot["max_residual"])

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
        compute_mis_flag=True,  # MIS analysis is slow; disable for speed
        mis_time_limit=10.0,    # seconds per internal MIS solve
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
   * - ``compute_mis(pm, solver_name="scip", mis_time_limit=None)``
     - Constraint / bound names of a Minimal Infeasible Subsystem.

.. note::

   The MIS computation requires SCIP, Gurobi, or CPLEX and runs a series
   of auxiliary solves; each internal solve is capped at 10 seconds so a
   hard MIS computation cannot stall your script. On large models, pass
   ``compute_mis_flag=False`` and start from the constraint residuals
   instead.

When no MIS comes back
----------------------

MIS isolation goes through the Pyomo deletion filter, which relaxes the model
elastically and then re-solves once per candidate constraint. On a non-convex
multi-energy MIQCQP those internal solves are themselves hard, so the routine
frequently returns nothing. It never returns nothing silently: ``mis_message``
carries the reason, and the summary prints it in place of the MIS section.

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - What ``mis_message`` says
     - What to do
   * - The solver is not available to Pyomo
     - Install SCIP, Gurobi or CPLEX, or pass another ``solver_name``.
   * - The MIS solver found the model feasible
     - No MIS exists. The original failure was numerical or solver specific,
       so re-run the failing solve with a different back end.
   * - The elastic phase could not relax any constraint into feasibility,
       and the internal solves came back ``infeasible``
     - The branch-and-bound solver declared the elastically relaxed model
       infeasible, which it should not be. That is the usual outcome on a
       non-convex multi-energy MIQCQP. No cap change helps; read the hotspots
       and isolate by carrier instead.
   * - The elastic phase could not relax any constraint into feasibility,
       and the internal solves hit a limit
     - The message names how many of the internal solves ended in what
       condition. Retry with a larger ``mis_time_limit``, or shrink the model
       (solve one carrier at a time) before asking for a MIS again.
   * - The elastic phase could not relax any constraint into feasibility,
       and every internal solve was optimal
     - The model is likely numerically unstable rather than time limited.
       Check the scaling of the reported hotspots first.
   * - The MIS refinement did not isolate a minimal subsystem
     - The infeasibility explanation section printed above it is the best
       available localisation; read that and the hotspots.

On a full multi-energy network expect the isolation to fail more often than it
succeeds. The hotspot ranking, one carrier at a time, is the reliable path.

Reading variable names
----------------------

Internally, Pyomo variables are named ``{cat}_{id}__{attr}`` (with an
additional ``_t{period}`` segment in multi-period solves). The report
translates them back to readable component descriptions:

.. code-block:: text

    child_3__p_mw        →  child[3].p_mw
    node_7_t2__vm_pu     →  node[7].vm_pu (t=2)

Constraints follow the same scheme, ``{cat}_{id}_eq_{i}`` with the same
``_t{period}`` segment in multi-period solves, so ``branch_1_2_0_t2_eq_0`` is
the first equation of that branch in period 2.

Time limits and other non-OK statuses
-------------------------------------

Only infeasible terminations trigger a report. Other non-OK outcomes, most
commonly a time limit reached with a feasible incumbent, count as success with
a warning ("using witness solution"): the result carries the best solution
found so far, ``result.success`` is ``True``, and no report is attached. Check
the log for the warning if your objective value looks suboptimal.

----

GEKKO back-end
==============

APMonitor (the engine behind GEKKO) normally fails with nothing more than
``@error: Solution Not Found``. monee instead parses the artifacts APMonitor
leaves in its run directory (``infeasibilities.txt``, ``results.json``) and
raises a :class:`~monee.solver.GekkoSolveError` (a ``RuntimeError`` subclass)
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
       iterate, ranked by infeasibility magnitude, each with the involved
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

   GEKKO diagnostics are best-effort: they need the run-directory artifacts of
   a local solve. :class:`~monee.solver.GEKKOSolver` always solves locally, but
   if you call the lower-level
   :func:`~monee.solver.diagnose_gekko_infeasibility` on a remote GEKKO
   instance, no artifacts exist and it returns ``None`` instead of a report.
   The diagnosis itself never raises. When no report can be built, the original
   GEKKO exception propagates unchanged.

Even on failure, the partial final iterate is written back to the input
network as a warm start. After you fix the modelling issue (relax a bound, add
a slack), the next solve resumes from where the failed one stopped instead of
starting cold.

----

CasADi back-end
===============

Failure semantics differ by back-end: CasADi and GEKKO raise an exception on
a failed solve, while the Pyomo back-end (``solver="scip"``) returns the
result with ``success=False`` and the structured report attached as
``result.infeasibility_report``. The CasADi error message points this out, so
a raised solve can always be re-run on SCIP to obtain the report.

The default back-end raises a :class:`~monee.solver.CasADiSolveError` (a
``RuntimeError`` subclass) when IPOPT stops without converging. The message
carries the IPOPT return status, which says what kind of failure it was:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Return status
     - What it means
   * - ``Infeasible_Problem_Detected``
     - IPOPT proved the constraints inconsistent. This is a modelling problem:
       work through the first checks above, and repeat the solve with
       ``solver="scip"`` to get a Minimal Infeasible Subsystem.
   * - ``Restoration_Failed`` / ``Error_In_Step_Computation``
     - IPOPT could not find a feasible point from the starting point it was
       given. The model may still be perfectly feasible; this is a convergence
       failure, not a proof of infeasibility.
   * - ``Maximum_Iterations_Exceeded``
     - The iteration limit was hit. First read the rest of the error message:
       when the final iterate still violates constraints, the message says so
       and names the largest residual (and, on a multi-period solve, the first
       infeasible period). In that case the problem is likely infeasible and a
       higher limit will not help; treat it like
       ``Infeasible_Problem_Detected``. Only when no such line is present,
       raise the limit by passing a higher value to the solve call, for
       example
       ``monee.run_energy_flow(net, solver_options={"ipopt.max_iter": 5000})``,
       or improve the start point.

CasADi reports Restoration_Failed
---------------------------------

``Restoration_Failed`` means IPOPT entered its feasibility restoration phase
and gave up there. The most common trigger in monee is a multi-energy compound
(:class:`~monee.model.multi.CHP`, :class:`~monee.model.multi.PowerToHeat`,
:class:`~monee.model.multi.GasToHeat`) next to an AC branch: the compound's
transfer branches carry one signed value at both ends, which the flat start of
the AC neighbourhood does not match, and the interior-point iterates never
recover.

The raised :class:`~monee.solver.CasADiSolveError` appends a hint to retry on
SCIP only when both conditions hold: the network actually contains a
multi-energy compound, and the return status is ``Restoration_Failed`` or
``Error_In_Step_Computation``. A status of ``Infeasible_Problem_Detected``
never carries the hint; IPOPT then proved the constraints inconsistent, so no
retry from a different start point can help and the checks at the top of this
page are the way forward.

Before assuming the network is wrong, retry it on SCIP, which uses a spatial
branch and bound instead of a local interior-point method:

.. code-block:: python

    import monee

    try:
        result = monee.run_energy_flow(net)
    except monee.solver.CasADiSolveError:
        result = monee.run_energy_flow(net, solver="scip")

If SCIP returns a solution, the model was feasible all along and you hit the
convergence failure; report the case, and keep the SCIP fallback in the
meantime. If SCIP also fails, treat it as a real modelling problem and read
its infeasibility report as described above.

Two further things help when the retry is not an option:

* Size the compound so its setpoint is reachable. A CHP burning 0.1 kg/s of
  gas releases about 1.9 MW of heat, which a 0.5 kg/s water flow cannot absorb
  (it would need a temperature rise of about 900 K, past the 800 K bound of the
  control node). Match the heat-side mass flow to the fuel setpoint, or lower
  the setpoint.
* Warm start from a solved neighbouring operating point. A timeseries run
  reuses the previous step's solution, so the first step is the one to get
  through; solve it with SCIP and let the remaining steps run on CasADi.

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
