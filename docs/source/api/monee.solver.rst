monee solver
===================

The public names below are re-exported from :mod:`monee.solver`
(e.g. ``from monee.solver import GEKKOSolver, PyomoSolver``). Most code
does not instantiate a back-end directly - pass a solver name such as
``"ipopt"`` or ``"gurobi"`` to the top-level entry points and let
:func:`~monee.solver.dispatch.resolve_solver` pick the back-end.

.. automodule:: monee.solver
   :members:
   :undoc-members:

Core
----

Back-end-agnostic machinery: the abstract :class:`~monee.solver.core.SolverInterface`,
the :class:`~monee.solver.core.SolverResult` returned by every solve, and the
shared solve-pipeline helpers (variable injection/withdrawal, islanding-aware
component pruning, warm starting, post-processing).

.. automodule:: monee.solver.core
   :members:
   :undoc-members:
   :show-inheritance:

GEKKO back-end
--------------

.. automodule:: monee.solver.gekko
   :members:
   :undoc-members:
   :show-inheritance:

Pyomo back-end
--------------

.. automodule:: monee.solver.pyo
   :members:
   :undoc-members:
   :show-inheritance:

Dispatch
--------

.. automodule:: monee.solver.dispatch
   :members:
   :undoc-members:
   :show-inheritance:

.. py:data:: monee.solver.dispatch.GEKKO_SOLVERS
   :type: dict[str, int]
   :value: {"apopt": 1, "bpopt": 2, "ipopt": 3}

   Mapping of GEKKO solver name to the GEKKO ``SOLVER`` option integer.
   These are the names :func:`~monee.solver.dispatch.resolve_solver` routes
   to the GEKKO back-end.

Infeasibility diagnostics
-------------------------

.. automodule:: monee.solver.infeasibility
   :no-index:

Pyomo
~~~~~

.. automodule:: monee.solver.infeasibility.pyo
   :members:
   :undoc-members:
   :show-inheritance:

GEKKO / APMonitor
~~~~~~~~~~~~~~~~~

.. automodule:: monee.solver.infeasibility.apm
   :members:
   :undoc-members:
   :show-inheritance:
