============
Formulations
============

A **formulation** defines the mathematical equations that describe how energy
flows through a grid component. Separating formulations from the data model
means you can swap equation sets without redefining the network topology - for
example, replacing a nonlinear AC power flow with a convex MISOCP relaxation,
trading the mixed-integer gas model for a smooth, binary-free NLP variant, or
experimenting with a new linearised hydraulic model, without touching nodes
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

Lifecycle
---------

Each formulation object participates in three phases:

``ensure_var(model, simulation=False)``
    Declares (or replaces) solver variables on the component model - ``Var``,
    ``Const``, ``Intermediate`` or ``PostProcess`` attributes. It is called
    once when the formulation is attached to a component, and the solvers
    **re-call it with** ``simulation=True`` at solve time to *square* the
    model: phantom degrees of freedom are pinned to constants or demoted to
    post-solve reports so that a steady-state simulation can run as a square
    equation system. See :doc:`solvers` for how the solver's ``simulation``
    flag drives this.

``equations(...)``
    Returns the list of constraint expressions for one component. The solver
    back-end injects keyword arguments that the formulation may use:
    ``sqrt_impl``, ``cos_impl``, ``sin_impl``, ``log_impl``, ``exp_impl``
    (backend-specific math implementations), ``pwl_impl`` (a piecewise-linear
    builder exposing ``piecewise_eq(y, x, xs, ys)``) and ``simulation`` (a
    bool mirroring the solver mode, used e.g. to drop operational flow-limit
    inequalities that would break a square solve).

``minimize(...)``
    Returns auxiliary objective terms appended to the solver objective - for
    example the ε-tightening terms that keep a convex epigraph relaxation
    tight at the optimum, or the loss term that tightens the MISOCP cone.

Attaching formulations
----------------------

Every :class:`~monee.model.network.Network` starts with
:data:`~monee.model.formulation.bundles.DEFAULT_SIMULATION_FORMULATION` - a
deliberate hybrid of the polar-AC NLP (electricity), the epigraph-relaxed
Weymouth (gas) and the bilinear Darcy-Weisbach (water/heat). To switch, call
:meth:`~monee.model.network.Network.apply_formulation`: it retro-applies the
mapping to all existing components *and* stores it as the network default, so
components added afterwards pick it up too.

Registration keys may be a plain model type (``GasPipe``) or a
``(model_type, grid_type)`` tuple - the latter is needed when a model is
shared between carriers, e.g. ``Junction`` is used by both gas and water
grids, so the node formulations are keyed ``(Junction, GasGrid)`` and
``(Junction, WaterGrid)``.

For a single component, pass the ``formulation=`` keyword to any of the
``Network`` builder methods (:meth:`~monee.model.network.Network.node`,
:meth:`~monee.model.network.Network.branch`,
:meth:`~monee.model.network.Network.child`,
:meth:`~monee.model.network.Network.compound`) to override the network-wide
choice.

.. note::

   Native IO (:func:`~monee.io.native.write_omef_network` /
   :func:`~monee.io.native.load_to_network`) does **not** persist non-default
   formulations. After loading a network from disk, re-apply your formulation
   with ``net.apply_formulation(...)``.

----

Built-in formulations
=====================

All built-in formulations are importable from :mod:`monee.model.formulation`
(and re-exported at the top level). Names follow the **optimization class**:
sector constants are ``<SECTOR>_<CLASS>_FORMULATION``, and three
sector-complete bundles cover the whole network with one consistent class:

.. testcode::

    # Per-sector constants
    from monee import (
        EL_NLP_FORMULATION,              # polar AC (smooth non-convex NLP)
        EL_MISOCP_FORMULATION,           # branch-flow SOC relaxation
        EL_NONCONVEX_MIQCQP_FORMULATION, # exact branch flow (SOC equality)
        GAS_NLP_FORMULATION,             # smooth Weymouth
        GAS_CONVEX_MIQCQP_FORMULATION,   # epigraph-relaxed Weymouth
        GAS_NONCONVEX_MIQCQP_FORMULATION,# exact Weymouth
        HEAT_NLP_FORMULATION,            # smooth Darcy-Weisbach
        HEAT_CONVEX_MILP_FORMULATION,    # McCormick district heating
        HEAT_NONCONVEX_MIQCQP_FORMULATION,  # bilinear Darcy-Weisbach
    )

    # Sector-complete bundles
    from monee.model.formulation import (
        SMOOTH_NLP_FORMULATION,       # pure NLP across all carriers
        CONVEX_MIQCQP_FORMULATION,    # certifiable relaxations everywhere
        NONCONVEX_MIQCQP_FORMULATION, # exact quadratic models everywhere
    )

The pre-restructure names (``AC_NETWORK_FORMULATION``,
``NL_WEYMOUTH_NETWORK_FORMULATION``, ``SMOOTH_NETWORK_FORMULATION``, …)
remain importable but are deprecated.

.. tab-set::

   .. tab-item:: Electricity

      ``EL_NLP_FORMULATION`` *(default)*

      Full nonlinear polar AC power flow using **voltage magnitude** and
      **angle** as decision variables - a smooth non-convex NLP. Handles any
      network topology. Compatible with NLP solvers (GEKKO / IPOPT). A single
      formulation serves both modes: under a simulation solve,
      ``ensure_var(simulation=True)`` demotes the unused ``vm_pu_squared``
      variable to a post-solve report so the model stays square.

      *Best for:* standard electricity simulation and NLP optimisation.

      ``EL_MISOCP_FORMULATION``

      Second-order cone relaxation of the AC optimal power flow (branch-flow
      model with squared voltages and currents). The formulation is **convex**
      and compatible with MILP / MIQCP solvers such as Gurobi or SCIP; its
      ``minimize()`` adds a resistive-loss term that keeps the cone tight.
      Supports branch currents (``i_from_ka``/``i_to_ka``) and line-loading
      limits.

      *Best for:* large-scale optimal power flow where global optimality or
      mixed-integer decisions are needed.

      ``EL_NONCONVEX_MIQCQP_FORMULATION``

      The branch-flow model with the SOC pinned to **equality**
      ``P² + Q² = (W/tap²)·ell`` - exact AC power flow in the lifted
      variables, a non-convex MIQCQP for global solvers (SCIP, Gurobi).

      *Best for:* exact mixed-integer AC studies without trigonometric terms.

   .. tab-item:: Gas

      ``GAS_CONVEX_MIQCQP_FORMULATION`` *(default)*

      Epigraph-relaxed (MISOCP-shaped) Weymouth pipe flow - a **convex
      MIQCQP**. Pressure is represented as *pressure-squared* (p²), flow
      direction by a binary plus a convex epigraph relaxation
      ``m² ≤ m_squared`` that ``minimize()`` keeps tight with a small
      ε-tightening term. Friction is pinned to the turbulent Swamee-Jain
      asymptote per pipe - accurate for typical gas distribution (Re ≳ 10⁵),
      but it under-estimates the pressure drop in the laminar regime.

      *Best for:* gas optimisation with branch-and-bound solvers
      (Gurobi/SCIP) and gas simulation.

      ``GAS_NONCONVEX_MIQCQP_FORMULATION``

      The same model with the epigraph pinned to **equality**
      ``m² = m_squared`` - exact Weymouth, a non-convex MIQCQP for global
      solvers. No ε term in the objective.

      :func:`~monee.model.formulation.make_gas_milp_pwl_formulation` ``(n_breakpoints=12)``

      Opt-in **variable-friction MILP** variant: replaces the constant
      friction with a per-pipe piecewise linearisation of
      ``φ(m) = friction(Re(m))·m²`` over log-spaced breakpoints, so laminar
      and turbulent regimes both resolve. Use it when lightly-loaded pipes
      (Re < 2300) matter.

      :func:`~monee.model.formulation.make_gas_nlp_formulation` ``(friction_model="constant", smoothing_eps=1e-3, n_breakpoints=12)``
      / ``GAS_NLP_FORMULATION``

      **Pure-NLP, binary-free** Weymouth: one signed mass-flow variable drives
      a smooth pressure drop ``friction·m·|m|`` with ``|m| ≈ √(m² + ε²)``
      (``smoothing_eps``, in kg/s). No direction binary, no epigraph - the
      pressure equation is also row-normalised for IPOPT conditioning.
      ``friction_model`` selects ``"constant"`` (turbulent asymptote),
      ``"pwl"`` (one odd spline replacing the whole drop term) or
      ``"nonlinear"`` (smooth laminar-turbulent friction blend with a
      Reynolds closure).

      *Best for:* GEKKO IPOPT/APOPT solves, especially full multi-energy
      systems where the MISOCP-shaped default stalls IPOPT.

   .. tab-item:: Water / heat

      ``HEAT_NONCONVEX_MIQCQP_FORMULATION`` *(default)*

      Darcy-Weisbach hydraulics (epigraph-relaxed, like the gas side) with
      full heat-transfer modelling: per-pipe inlet/outlet temperatures,
      insulation losses and external-temperature influence, with a direction
      binary for temperature upwinding. The temperature side carries true
      continuous bilinears (``alpha·mag``, ``direction·t``), making the model
      a **non-convex MIQCQP**. Active heat exchangers
      (:class:`~monee.model.branch.HeatExchanger`) use the LP
      ``FixedFlowHeatExchangerFormulation`` (fixed design mass flow, energy
      balance in MW); passive heat exchangers use the bilinear free-flow
      formulation.

      *Best for:* district heating and water network simulation and
      optimisation with global solvers.

      :func:`~monee.model.formulation.make_heat_nonconvex_pwl_formulation` ``(n_breakpoints=12)``

      Opt-in **variable-friction** PWL variant, analogous to the gas one:
      restores laminar-regime accuracy for lightly-loaded pipes (Re < 2300).
      Heat-exchanger formulations are unchanged; the temperature bilinears
      keep the model a non-convex MIQCQP.

      :func:`~monee.model.formulation.make_heat_nlp_formulation` ``(friction_model="constant", smoothing_eps=1e-3, n_breakpoints=12)``
      / ``HEAT_NLP_FORMULATION``

      **Pure-NLP, binary-free** water/heat: smooth signed flow plus smooth
      temperature upwinding written in multiplied form (no direction binary,
      no division by the flow magnitude), and active/passive heat-exchanger
      formulations with their binaries pinned to constants. Same
      ``friction_model`` options as the smooth gas formulation.

      *Best for:* GEKKO IPOPT/APOPT solves of heat networks and full MES.

   .. tab-item:: District heating (McCormick)

      ``HEAT_CONVEX_MILP_FORMULATION`` /
      :func:`~monee.model.formulation.make_heat_convex_milp_formulation` ``(num_partitions=1, include_heat_exchangers=True)``

      Convex **relaxation** of the district-heating thermal side after Deng et
      al. (2021): the bilinear enthalpy ``H = c·m·τ`` is replaced by McCormick
      envelopes, the heat loss is Taylor-linearised, and hydraulics are
      omitted (pipes are unidirectional). With ``num_partitions=1`` the result
      is a plain McCormick **LP**; ``num_partitions > 1`` upgrades to a
      piecewise **MILP** with tighter envelopes. By default the active and
      passive heat exchangers map to their convex variants
      (``McCormickHeatExchangerFormulation`` /
      ``McCormickPassiveHeatExchangerFormulation``) so the sector has no
      non-convex fallback; pass ``include_heat_exchangers=False`` for the
      legacy pipes-only apply.

      Because it is a relaxation, check the worst-case gap before raising
      ``num_partitions``:

      .. code-block:: python

          from monee.model.formulation.milp.heat import (
              mccormick_dhs_gap_bound_mw,  # worst-case H_out gap [MW]
              mccormick_dhs_gap_bound_k,   # same gap as a sender-temperature error [K]
          )

      Tighten the envelopes via ``WaterGrid.t_pu_min_env`` /
      ``WaterGrid.t_pu_max_env`` or a per-pipe ``m_U_design`` cap.

      *Best for:* fast LP/MILP heat-dispatch studies where guaranteed bounds
      matter more than exact hydraulics.

   .. tab-item:: Full-network bundles

      :func:`~monee.model.formulation.make_smooth_nlp_formulation` ``(friction_model="constant")``
      / ``SMOOTH_NLP_FORMULATION``

      Polar AC + smooth gas + smooth water/heat in a single
      ``NetworkFormulation`` - one ``apply_formulation`` call turns a
      multi-energy network into a pure NLP across all three carriers.

      *Best for:* full-MES simulation and optimisation with IPOPT/APOPT where
      the MISOCP-shaped defaults stall the interior-point solver.

      :func:`~monee.model.formulation.make_convex_miqcqp_formulation` ``(num_partitions=1)``
      / ``CONVEX_MIQCQP_FORMULATION``

      Branch-flow MISOCP electricity + epigraph-relaxed Weymouth gas +
      McCormick district heating (incl. the convex heat-exchanger variants).
      Every constraint is certifiable by a convex MIQCQP/MISOCP solver;
      solutions are relaxation optima - check the documented gap bounds where
      exactness matters.

      ``NONCONVEX_MIQCQP_FORMULATION``

      Exact branch flow + exact Weymouth + bilinear Darcy-Weisbach: the exact
      quadratic models across all carriers, with no trigonometric terms - for
      global MIQCQP solvers (SCIP, Gurobi).

----

Choosing a formulation
======================

For each carrier monee offers formulations in several **optimization
classes**: the MIQCQP family (direction binary plus epigraph - ideal for
branch-and-bound solvers such as Gurobi or SCIP, in a convex relaxed and an
exact non-convex flavour), the LP/MILP family (PWL variable-friction
variants and the McCormick heating relaxation), and the smooth pure-NLP
family (when you want IPOPT/APOPT to solve the whole system without integer
variables).

.. list-table::
   :header-rows: 1
   :widths: 38 28 34

   * - Scenario
     - Solver back-end
     - Formulation
   * - AC power flow / NLP OPF
     - GEKKO (IPOPT)
     - ``EL_NLP_FORMULATION``
   * - Convex / mixed-integer OPF
     - Pyomo + Gurobi / SCIP
     - ``EL_MISOCP_FORMULATION``
   * - Gas, water/heat optimisation (MIQCQP)
     - Pyomo + Gurobi / SCIP
     - ``GAS_CONVEX_MIQCQP_FORMULATION``, ``HEAT_NONCONVEX_MIQCQP_FORMULATION``
   * - Laminar-heavy hydraulics (Re < 2300)
     - Pyomo + Gurobi / SCIP
     - ``make_gas_milp_pwl_formulation()``, ``make_heat_nonconvex_pwl_formulation()``
   * - Single-carrier pure-NLP solve
     - GEKKO (IPOPT / APOPT)
     - ``GAS_NLP_FORMULATION``, ``HEAT_NLP_FORMULATION``
   * - Full multi-energy pure-NLP solve
     - GEKKO (IPOPT / APOPT)
     - ``SMOOTH_NLP_FORMULATION``
   * - Full multi-energy convex solve
     - Pyomo + Gurobi / SCIP
     - ``CONVEX_MIQCQP_FORMULATION``
   * - Full multi-energy exact MIQCQP solve
     - Pyomo + SCIP / Gurobi
     - ``NONCONVEX_MIQCQP_FORMULATION``
   * - Heat dispatch with optimality bounds
     - Pyomo + LP / MILP solver
     - ``HEAT_CONVEX_MILP_FORMULATION``
   * - Custom MILP model
     - Pyomo + MILP solver
     - custom formulation

See :doc:`solvers` for guidance on picking the right solver back-end.

A minimal smooth-NLP example - build a gas network, swap the formulation,
solve:

.. testcode::

    from monee import mx, run_energy_flow
    from monee.model.formulation import GAS_NLP_FORMULATION

    net = mx.create_multi_energy_network()

    j0 = mx.create_gas_junction(net)
    j1 = mx.create_gas_junction(net)
    mx.create_gas_pipe(net, j0, j1, diameter_m=0.3, length_m=1000)
    mx.create_ext_hydr_grid(net, j0)
    mx.create_sink(net, j1, mass_flow=0.5)

    net.apply_formulation(GAS_NLP_FORMULATION)
    result = run_energy_flow(net)

Simulation vs optimisation
--------------------------

There are **no separate simulation formulations**. Whether a model is solved
as a square steady-state simulation or as an optimisation problem is decided
entirely by the solver's ``simulation`` flag
(:func:`~monee.run_energy_flow` passes ``simulation=True`` by default): the
solver re-calls ``ensure_var(model, simulation=True)`` on every component,
which pins phantom degrees of freedom, demotes report-only quantities to
``PostProcess`` values and (in the smooth formulations) drops operational
flow-limit inequalities so the equation system becomes square. GEKKO then
attempts a fast IMODE=1 solve, falling back to IMODE=3 when the model is not
square - check ``result.mode_used``. See :doc:`solvers` for details.

.. tip::

   **Warm starting.** In simulation mode the smooth formulations seed the
   flow magnitude ``mass_flow_mag`` from ``branch.mass_flow_nominal`` (the
   design flow stored on pipes by the MES network generators). A good
   magnitude seed alone cuts the smooth-NLP iteration count by roughly 20× -
   the sign of the flow is recovered cheaply, the magnitude is what
   conditions the pressure-drop and temperature-transport equations.

----

Writing a custom formulation
============================

Subclass the appropriate base class and implement ``ensure_var`` (declare
model variables), ``equations`` (return a list of equality/inequality
constraints) and optionally ``minimize`` (return auxiliary objective terms):

.. code-block:: python

    from monee.model.formulation.core import (
        BranchFormulation,
        NodeFormulation,
        NetworkFormulation,
    )
    from monee.model.core import Const, Var


    class MyNodeFormulation(NodeFormulation):
        def ensure_var(self, model, simulation=False):
            # Declare a decision variable on the node model. Solvers call
            # this again with simulation=True at solve time - pin variables
            # your equations never constrain so the model stays square.
            model.pressure_pu = Var(1, min=0, max=2, name="pressure_pu")
            if simulation and isinstance(model.t_pu, Var):
                model.t_pu = Const(model.t_pu.value)

        def equations(self, node, grid, from_branch_models, to_branch_models,
                      connected_child_models, **kwargs):
            # Return a list of solver constraints
            return []


    class MyBranchFormulation(BranchFormulation):
        def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
            sqrt = kwargs["sqrt_impl"]  # solver-injected math implementation
            eqs = [
                from_node_model.vars["pressure_pu"]
                - to_node_model.vars["pressure_pu"]
                == branch.resistance * branch.mass_flow
            ]
            if not kwargs.get("simulation", False):
                # Operational limits would unbalance a square simulation
                # solve - emit them only in optimisation mode.
                eqs.append(branch.mass_flow <= branch.flow_max)
            return eqs

        def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
            # Optional: auxiliary objective-tightening terms.
            return []


    MY_NETWORK_FORMULATION = NetworkFormulation(
        branch_type_to_formulations={MyPipe: MyBranchFormulation()},
        node_type_to_formulations={(MyJunction, MyGrid): MyNodeFormulation()},
    )

The keyword arguments received by ``equations`` and ``minimize`` are injected
by the solver back-end:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword
     - Meaning
   * - ``sqrt_impl``, ``cos_impl``, ``sin_impl``, ``log_impl``, ``exp_impl``
     - Backend-specific math implementations (GEKKO operators or Pyomo
       intrinsics) - use these instead of ``math.*`` so the same formulation
       runs on every back-end.
   * - ``pwl_impl``
     - Piecewise-linear builder; call ``pwl_impl.piecewise_eq(y, x, xs, ys)``
       to constrain ``y = f(x)`` along breakpoints ``xs``/``ys`` (GEKKO cubic
       spline, Pyomo SOS2 piecewise).
   * - ``simulation``
     - ``True`` when the solver runs in simulation mode - drop inequalities
       that would unbalance a square solve and verify them post-hoc instead.

Activate the formulation with ``net.apply_formulation(MY_NETWORK_FORMULATION)``
(network-wide) or pass it via the ``formulation=`` keyword to a single
``Network`` builder call. See :doc:`solvers` for the available solver
interfaces.
