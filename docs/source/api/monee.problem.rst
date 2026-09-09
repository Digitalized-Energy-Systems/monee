monee optimization problem
==========================

Problem definitions
-------------------

.. note::

   A custom objective is summed into the same objective as the terms the
   formulations add to tighten their own relaxations, and those terms are
   unscaled physical quantities (the heat-exchanger duty pull is in MW). An
   objective with large coefficients can therefore outbid them, most visibly by
   letting a fixed-duty heat exchanger under-deliver. ``result.objective`` is
   the combined value; ``result.user_objective`` reports your part alone and
   ``result.aux_objective`` the formulation part. See
   :doc:`../how-to/load_shedding` for the duty shortfall this can cause and how
   it is reported.

.. note::

   Selector convention. Every selection callable accepts a one argument or a
   two argument predicate; the arity is detected automatically. The
   recommended form is the two argument predicate
   ``lambda model, component: ...``, which behaves identically on
   ``objectives.select``, ``constraints.select`` and
   ``prob.controllable(attrs, component_condition=...)``: the first argument
   is the component model, the second the ``Component`` container (use it for
   ``component.grid``, ``component.active`` or ``component.id``). Ignore the
   second argument when you do not need it, for example
   ``lambda m, _c: isinstance(m, PowerGenerator)``.

   A one argument predicate receives each selector's historical argument: the
   model on ``objectives.select``, the ``Component`` container on
   ``constraints.select`` and ``controllable``. A one argument predicate
   written for the other convention is no longer a silent no-op: when it
   matches nothing under its selector's native argument but does match under
   the other one, the other interpretation is applied and a warning explains
   the mix-up. A predicate that matches nothing at all is reported as a
   warning as well.

   ``prob.bounds(minmax, component_condition, attributes)`` differs for
   historical reasons: its two argument condition receives ``(model, grid)``
   rather than ``(model, component)``, and a one argument condition receives
   the model alone.

   ``constraints.select_types(PowerGenerator)`` and
   ``constraints.select_grids(PowerGrid)`` cover the common type tests without
   a lambda.

.. automodule:: monee.problem
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:

Utilities
---------

Shared constraint helpers used by the problem factories.

.. automodule:: monee.problem.utils
   :members:
   :undoc-members:
   :show-inheritance:

Metrics
-------

Post-solve performance and resilience metrics.

.. automodule:: monee.problem.metric
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:
