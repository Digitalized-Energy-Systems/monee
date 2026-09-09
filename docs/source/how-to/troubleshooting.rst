===============
Troubleshooting
===============

The traps on this page share one shape: the solve succeeds, or fails with a
message that points somewhere else, and the number in front of you is not the
one you think it is. Each entry gives the symptom, the cause, the fix and
where to read on. The conventions behind most of them are collected on
:doc:`../concepts/conventions`.

Before anything else, check ``result.success``. A solve that does not converge
returns a result rather than raising, so a failure that is never checked reads
like a solution.

My import cap does nothing
==========================

Symptom
    You passed ``bounds_ext_el=(0, cap)``, or wrote the constraint
    ``model.p_mw <= cap``, and the solve succeeds while importing freely, or
    it sheds load in a network that has plenty of headroom.

Cause
    The load convention. An import into the network is a negative ``p_mw``
    on the external grid, so an import cap is a lower bound. A tuple whose
    lower bound is zero or positive does not cap the import, it forbids
    importing at all, and the shedding problem then serves the load by
    curtailing it.

Fix
    Cap the import at ``X`` MW with ``bounds_ext_el=(-X, export_cap)`` or the
    constraint ``model.p_mw >= -X``; ``max_import_mw=X`` on the factory does
    the same. :func:`~monee.problem.create_min_load_shedding_problem` warns
    when a supplied ``bounds_ext_el`` lower bound is not negative.

Read on
    :ref:`data-model-slack-sign` and :doc:`load_shedding`. The same sign rule
    applies to pricing objectives, see :doc:`multi_period`.

My PV runs at nameplate at midnight
===================================

Symptom
    A generator ignores its registered profile and produces its constructor
    setpoint at every step; the external grid shows a phantom export.

Cause
    The series bound to nothing. :class:`~monee.simulation.TimeseriesData`
    resolves ``add_child_series_by_name`` keys against component names, and a
    key that matches no component is reported as a warning by the run
    drivers, then the constructor setpoint is used unchanged for the whole
    run.

Fix
    Give the component a ``name=`` at creation and register the series under
    exactly that name (or use the id-based ``add_child_series``). Check the
    log for the unmatched-keys warning, and verify the binding by printing
    the solved value at a step where the profile is zero.

Read on
    :doc:`timeseries` and :doc:`../concepts/timeseries`.

The solver says infeasible
==========================

Work through these in order; each step catches a distinct class of failures.

1. Does every carrier have a slack? Each connected component needs an active
   :class:`~monee.model.child.ExtPowerGrid` or
   :class:`~monee.model.child.ExtHydrGrid` (or a grid-forming child under
   islanding). A deactivated branch can split the network into an island
   without one.
2. Are your bounds the problem? Retry with the bounds relaxed
   (``bounds_vm=(0.5, 1.5)``, ``check_pressure=False``, ``check_t=False``)
   to see whether an operational band excludes the only physical solution.
   Remember that the pressure band is stated in squared per unit.
3. Is it the back-end rather than the model? Coupled multi-energy networks
   can defeat the default IPOPT start while the model is feasible; retry
   with ``formulation="smooth_nlp"`` and with ``solver="scip"``, or let
   ``solver="auto"`` do the retry for you. Read the failure message: it
   names the true return status first, and only adds the retry hint when
   compounds are present. The next entry tells the two cases apart.
4. Still infeasible? Read the bound-violation and back-end reports on
   :doc:`diagnose_infeasibility`, which walks through the Pyomo
   infeasibility report, the GEKKO diagnostics and the CasADi return-status
   table.

A failed solve is not always an infeasible model
================================================

Symptom
    A solve comes back with ``result.success == False`` and a status such as
    ``Restoration_Failed``, ``Error_In_Step_Computation`` or
    ``Maximum_Iterations_Exceeded``, and it is not clear whether the model is
    wrong or the solver simply gave up.

Cause
    Two different outcomes arrive through the same channel. A numerics failure
    means the interior-point search broke down: the model may well be feasible,
    and a different back-end or a different starting point still solves it. A
    proven infeasibility (IPOPT ``Infeasible_Problem_Detected``, Pyomo
    ``infeasible``) is a verdict about the model itself, and every back-end
    reproduces it. Only the second one should end a study; treating the first
    one as an answer throws away cases that are perfectly solvable, which is
    how a contingency screening loses one case in ten.

Fix
    Read the classification off the result rather than guessing from the text:

    .. code-block:: python

        result = run_energy_flow(net)
        if not result.success:
            report = result.infeasibility_report
            if getattr(report, "numerics_failure", True):
                result = run_energy_flow(net, solver="scip")   # worth retrying
            else:
                ...                                            # the model has to change

    The same rule is available as
    :func:`monee.solver.casadi.is_numerics_failure`, and
    ``solver="auto"`` applies it for you: it runs the default back-end and
    retries once on SCIP on a numerics failure only, recording
    ``result.fallback_from`` and a ``backend_fallback`` warning when the retry
    produced the result. Use it for sweeps, where one unsolvable case should
    not stop the run, and name the back-end explicitly for a single headline
    solve.

    If SCIP also fails, the model is the problem: work through the checklist
    above, and read the structured report SCIP attaches to
    ``result.infeasibility_report``.

    Watch for two statuses that look numerical but are not:
    ``Not_Enough_Degrees_Of_Freedom`` and ``Invalid_Problem_Definition`` say
    the model is badly posed, so no retry helps. ``Network.check()`` usually
    names the cause before the solve.

Read on
    :doc:`diagnose_infeasibility` and :doc:`../concepts/solvers`.

My contingency loop needs a try/except around every solve
=========================================================

Symptom
    Code that loops over outages or samples has to wrap each
    :func:`~monee.run_energy_flow` call in ``try`` / ``except`` and rebuild the
    network afterwards, because one case in ten raised.

Cause
    A non-converged CasADi solve used to raise. It now returns a result with
    ``success=False`` carrying the status, the message and the diagnostics, so
    the failure is data instead of control flow. The network you passed in is
    left untouched by a failed solve, so there is nothing to rebuild.

Fix
    Branch on ``result.success`` and keep the failed results:

    .. code-block:: python

        outcomes = []
        for case in cases:
            result = run_energy_flow(build(case), solver="auto")
            outcomes.append((case, result))

        solved = [c for c, r in outcomes if r.success]
        failed = [(c, r.termination_condition) for c, r in outcomes if not r.success]

    ``strict=True`` restores the raise where you want a hard stop, and existing
    ``try`` / ``except`` blocks stay correct either way. Inside
    :func:`~monee.run_timeseries` and the :class:`~monee.simulation.Stepper`,
    a failed step is still a failed step: ``on_step_error='skip'`` records it
    and the run continues.

Read on
    :doc:`../concepts/solvers`.

Temperatures look wrong on IPOPT
================================

Symptom
    District-heating temperatures come out several Kelvin below the analytic
    value on the default CasADi/IPOPT path, or pressures and flows look
    blended, and a warning about relaxed integer variables appeared.

Cause
    The default optimisation formulation for gas and heat carries integer
    flow-direction variables. A continuous back-end relaxes them, so the
    solve converges on a fractional blend of both flow directions, and the
    terms that would tighten the relaxation can stay slack.

Fix
    On IPOPT pass the binary-free ``formulation="smooth_nlp"`` (or
    ``"heat_nlp"`` / ``"gas_nlp"`` per carrier) for exact physics, or keep
    the default formulation and solve with a back-end that enforces
    integrality, ``solver="scip"`` or ``solver="gurobi"``. Treat the
    relaxed-integrality warning and the relaxation-gap warning as answers,
    not noise.

Read on
    :doc:`../concepts/formulations` and :doc:`../concepts/solvers`.

A component is missing from the results
=======================================

Symptom
    A bus or junction you created does not behave like part of the solve:
    its row carries ``ignored=True`` with a stale or arbitrary value, or a
    coupler delivers nothing after an outage elsewhere.

Cause
    Pruning. Before solving, monee severs the multi-energy coupling points
    and then ignores every connected component that is left without an
    active external grid child, because such an island can never be
    energised. Components reachable only through a coupler are therefore
    ignored together with their island, and an electrical outage can
    amputate a compound's port so its heat or gas side idles at zero.

Fix
    Filter result frames on the ``ignored`` column before reading values.
    If the island is meant to run, give it a slack
    (:func:`~monee.express.create_ext_power_grid`,
    :func:`~monee.express.create_ext_hydr_grid`) or a grid-forming child
    with islanding enabled. Deactivating a compound deactivates all of its
    subcomponents.

Read on
    :doc:`diagnose_infeasibility`, :doc:`../concepts/islanding` and the
    compound handling notes in :doc:`load_shedding`.

result.objective is not my objective
====================================

Symptom
    ``result.objective`` is slightly negative at zero curtailment, differs
    between back-ends at the same optimum, or drops when a contingency gets
    worse.

Cause
    ``result.objective`` is the value the solver minimised: your objective
    terms plus the formulation's own small tightening terms, with ignored
    and inactive components dropped entirely. It is a solver diagnostic, not
    a report of your cost function.

Fix
    Read ``result.user_objective`` for your terms alone
    (``result.aux_objective`` holds the formulation part), or compute the
    quantity you care about directly from the result frames, for example
    served load as ``sum(p_mw * regulation)``.

Read on
    :ref:`objective-semantics` on the load-shedding page.
