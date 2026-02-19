====================
Use the Pyomo solver
====================

The Pyomo solver interface lets you back monee with any solver that Pyomo
supports — including Gurobi, HiGHS, GLPK, CBC, and CPLEX. Use it when you
need a MILP or MIQCP back-end, for example to solve an AC optimal power flow
with the :data:`~monee.model.formulation.MISOCP_NETWORK_FORMULATION`.

----

Prerequisites
=============

Install Pyomo and at least one solver back-end:

.. tab-set::

   .. tab-item:: HiGHS (recommended)

      Open-source LP / MILP / MIQCP solver. Works out of the box on most
      platforms.

      .. code-block:: bash

          pip install highspy

      or via conda:

      .. code-block:: bash

          conda install -c conda-forge highs

   .. tab-item:: GLPK

      Open-source LP / MILP solver.

      .. code-block:: bash

          conda install -c conda-forge glpk

   .. tab-item:: Gurobi

      Commercial solver with a free academic licence available at
      `gurobi.com <https://www.gurobi.com/academia/academic-program-and-licenses/>`_.
      Install the Python bindings after obtaining a licence:

      .. code-block:: bash

          pip install gurobipy

----

Minimal example
===============

The following snippet builds a small electricity grid and solves an AC
optimal power flow using the MISOCP relaxation and HiGHS:

.. code-block:: python

    import monee.express as mx
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION
    from monee.solver.pyo import PyomoSolver

    # Build the network
    net = mx.create_multi_energy_network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    # Switch to the MISOCP formulation
    net.set_formulation(MISOCP_NETWORK_FORMULATION)

    # Solve with Pyomo + HiGHS
    result = PyomoSolver().solve(net, solver_name="highs")
    print(result.objective_value)

----

Passing an optimisation problem
================================

Pass an :class:`~monee.problem.core.OptimizationProblem` the same way as with
the GEKKO solver:

.. code-block:: python

    import monee.problem as mp
    from monee.solver.pyo import PyomoSolver

    problem = mp.OptimizationProblem()
    problem.controllable_demands((
        "regulation",
        mp.AttributeParameter(
            min=lambda a, v: 0,
            max=lambda a, v: 1,
            val=lambda a, v: 1,
        ),
    ))

    result = PyomoSolver().solve(
        net,
        optimization_problem=problem,
        solver_name="gurobi",
    )

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
   * - ``"highs"``
     - HiGHS
     - Open-source LP / MILP / MIQCP. ``pip install highspy``.
   * - ``"gurobi"``
     - Gurobi
     - Commercial, requires a valid licence.
   * - ``"glpk"``
     - GLPK
     - Open-source, LP / MILP only.
   * - ``"cbc"``
     - CBC
     - Open-source, LP / MILP.
   * - ``"ipopt"``
     - IPOPT
     - Open-source NLP. Requires a separate IPOPT binary.

----

Formulation compatibility
=========================

Not every formulation is compatible with every solver back-end. The
GEKKO-specific helpers ``if2``, ``max2``, and ``sign2`` are **not** available
in the Pyomo translation layer and will raise :exc:`NotImplementedError` if a
formulation tries to use them.

.. tip::

   For the Pyomo back-end, prefer:

   - :data:`~monee.model.formulation.MISOCP_NETWORK_FORMULATION` for
     electricity optimal power flow.
   - Custom formulations written with standard Pyomo / monee expressions.

   See :doc:`../concepts/formulations` for the full list of built-in
   formulations and a guide to writing custom ones.
