============
Formulations
============

A **formulation** defines the mathematical equations that describe how energy
flows through a grid component. Separating formulations from the data model
means you can swap equation sets without redefining the network topology — for
example, replacing a nonlinear AC power flow with a convex MISOCP relaxation,
or experimenting with a new linearised hydraulic model, without touching nodes
and branches.

Architecture
============

The formulation layer consists of four base classes in
:mod:`monee.model.formulation.core`:

:class:`~monee.model.formulation.core.NodeFormulation`
    Provides variables and balance equations for a node (bus, junction).

:class:`~monee.model.formulation.core.BranchFormulation`
    Provides variables and physics equations for a branch (line, pipe).

:class:`~monee.model.formulation.core.ChildFormulation`
    Provides variables and equations for a child component (load, generator).

:class:`~monee.model.formulation.core.CompoundFormulation`
    Provides equations for multi-grid compound components (e.g. heat pumps).

A :class:`~monee.model.formulation.core.NetworkFormulation` collects these
specialised formulations and maps them to concrete model types:

.. code-block:: python

    from monee.model.formulation import NetworkFormulation

    my_formulation = NetworkFormulation(
        branch_type_to_formulations={MyPipeModel: MyPipeBranchFormulation()},
        node_type_to_formulations={(MyJunctionModel, MyGrid): MyJunctionNodeFormulation()},
    )

The solver receives the ``NetworkFormulation`` and dispatches the right
formulation object for every element it encounters during model assembly.

----

Built-in formulations
=====================

monee ships four ready-to-use formulations, importable from
:mod:`monee.model.formulation`:

.. testcode::

    from monee.model.formulation import (
        AC_NETWORK_FORMULATION,
        MISOCP_NETWORK_FORMULATION,
        NL_WEYMOUTH_NETWORK_FORMULATION,
        NL_DARCY_WEISBACH_NETWORK_FORMULATION,
    )

.. tab-set::

   .. tab-item:: AC power flow

      ``AC_NETWORK_FORMULATION``

      Full nonlinear AC power flow using **voltage magnitude** and **angle** as
      decision variables. Handles any network topology. Compatible with NLP
      solvers (GEKKO / IPOPT).

      *Best for:* standard electricity simulation.

   .. tab-item:: MISOCP (convex OPF)

      ``MISOCP_NETWORK_FORMULATION``

      Second-order cone relaxation of the AC optimal power flow. The
      formulation is **convex** and compatible with MILP / MIQCP solvers such
      as Gurobi or HiGHS. Minimises resistive losses in the objective.

      *Best for:* large-scale optimal power flow where global optimality or
      mixed-integer decisions are needed.

   .. tab-item:: Weymouth (gas)

      ``NL_WEYMOUTH_NETWORK_FORMULATION``

      Nonlinear Weymouth pipe-flow equations. Pressure is represented as
      *pressure-squared* (p²) with a first-order Taylor linearisation of
      √p around the nominal operating point to keep the formulation smooth.

      *Best for:* gas network simulation and optimisation.

   .. tab-item:: Darcy–Weisbach (water/heat)

      ``NL_DARCY_WEISBACH_NETWORK_FORMULATION``

      Nonlinear Darcy–Weisbach hydraulic equations with laminar-flow friction
      approximation and full heat-transfer modelling — temperature in/out per
      pipe, insulation losses, and external temperature influence.

      *Best for:* district heating and water network simulation.

----

Choosing a formulation
======================

.. list-table::
   :header-rows: 1
   :widths: 38 28 34

   * - Scenario
     - Solver back-end
     - Formulation constant
   * - AC power flow simulation
     - GEKKO (IPOPT)
     - ``AC_NETWORK_FORMULATION``
   * - Gas flow simulation
     - GEKKO (IPOPT)
     - ``NL_WEYMOUTH_NETWORK_FORMULATION``
   * - Water / heat flow simulation
     - GEKKO (IPOPT)
     - ``NL_DARCY_WEISBACH_NETWORK_FORMULATION``
   * - AC optimal power flow (convex)
     - Pyomo + Gurobi / HiGHS
     - ``MISOCP_NETWORK_FORMULATION``
   * - Custom MILP model
     - Pyomo + MILP solver
     - custom formulation

See :doc:`solvers` for guidance on picking the right solver back-end.

----

Writing a custom formulation
============================

Subclass the appropriate base class and implement ``ensure_var`` (declare
model variables) and ``equations`` (return a list of equality/inequality
constraints):

.. code-block:: python

    from monee.model.formulation.core import (
        BranchFormulation,
        NodeFormulation,
        NetworkFormulation,
    )
    from monee.model.core import Var


    class MyNodeFormulation(NodeFormulation):
        def ensure_var(self, model):
            # Declare a decision variable on the node model
            model.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")

        def equations(self, node, grid, from_branch_models, to_branch_models,
                      connected_child_models, **kwargs):
            # Return a list of solver constraints
            return []


    class MyBranchFormulation(BranchFormulation):
        def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
            return [
                from_node_model.vars["pressure_pu"]
                - to_node_model.vars["pressure_pu"]
                == branch.resistance * branch.mass_flow
            ]


    MY_NETWORK_FORMULATION = NetworkFormulation(
        branch_type_to_formulations={MyPipe: MyBranchFormulation()},
        node_type_to_formulations={(MyJunction, MyGrid): MyNodeFormulation()},
    )

Pass ``MY_NETWORK_FORMULATION`` to your solver call in place of one of the
built-in constants. See :doc:`solvers` for the available solver interfaces.
