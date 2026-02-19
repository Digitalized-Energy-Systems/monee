=======
Solvers
=======

monee solves energy-flow problems by translating the network model and its
formulation into a mathematical programme and delegating to a numerical solver
back-end. Two solver interfaces are provided.

----

.. tab-set::

   .. tab-item:: GEKKO (default)

      The ``GekkoSolver`` (``monee.solver.gekko``) is the default back-end. It
      wraps the `GEKKO <https://gekko.readthedocs.io>`_ optimisation suite,
      which ships its own solver binaries (APOPT, IPOPT, BPOPT) and **requires
      no extra installation** beyond ``pip install monee``.

      **Suitable for:**

      - Nonlinear energy-flow simulation — AC power flow, Weymouth gas flow,
        Darcy–Weisbach water/heat flow.
      - General NLP and MINLP optimisation problems.

      **Limitations:**

      - MILP / MIQCP problems are handled by APOPT's built-in MINLP solver,
        which can be slow for large mixed-integer models. For those cases use
        the Pyomo back-end with a dedicated MILP solver.

      **Usage**

      :func:`monee.run_energy_flow` uses ``GekkoSolver`` by default:

      .. testcode::

          import monee.express as mx
          from monee import run_energy_flow

          net = mx.create_multi_energy_network()
          bus_0 = mx.create_bus(net)
          bus_1 = mx.create_bus(net)
          mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
          mx.create_ext_power_grid(net, bus_0)
          mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

          result = run_energy_flow(net)

      You can also instantiate the solver directly to pass solver options:

      .. code-block:: python

          from monee.solver.gekko import GekkoSolver

          result = GekkoSolver().solve(net, solver=3)  # solver=3 → IPOPT

   .. tab-item:: Pyomo

      The ``PyomoSolver`` (``monee.solver.pyo``) translates the monee model
      into a `Pyomo <https://www.pyomo.org>`_ ``ConcreteModel`` and delegates
      to any solver supported by Pyomo — including Gurobi, GLPK, HiGHS, CBC,
      and CPLEX.

      **Suitable for:**

      - MILP and MIQCP problems, e.g. optimal power flow with binary switching
        decisions using
        :data:`~monee.model.formulation.MISOCP_NETWORK_FORMULATION`.
      - Situations where a specific commercial or open-source solver is
        required.

      **Limitations:**

      - GEKKO-specific mathematical operators (``if2``, ``max2``, ``sign2``)
        are **not** available in the Pyomo back-end. Formulations that rely on
        these operators will raise :exc:`NotImplementedError`. Use
        Pyomo-compatible formulations or write your own using standard Pyomo
        expressions.
      - Requires Pyomo and a compatible solver installed separately.

      **Usage**

      .. code-block:: python

          from monee.solver.pyo import PyomoSolver

          result = PyomoSolver().solve(
              net,
              solver_name="highs",   # or "gurobi", "glpk", "cbc", ...
          )

      See :doc:`../how-to/use_pyomo_solver` for a complete worked example
      including MISOCP optimal power flow.

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
     - GEKKO
     - ``AC_NETWORK_FORMULATION``
   * - Gas flow (simulation)
     - GEKKO
     - ``NL_WEYMOUTH_NETWORK_FORMULATION``
   * - Water / heat flow (simulation)
     - GEKKO
     - ``NL_DARCY_WEISBACH_NETWORK_FORMULATION``
   * - AC optimal power flow (convex relaxation)
     - Pyomo + Gurobi / HiGHS
     - ``MISOCP_NETWORK_FORMULATION``
   * - Custom MILP problem
     - Pyomo + MILP solver
     - custom formulation

.. tip::

   When in doubt, start with **GEKKO** — it works out of the box and handles
   all nonlinear problems. Switch to **Pyomo** only when you need a MILP /
   MIQCP solver or a specific commercial solver back-end.
