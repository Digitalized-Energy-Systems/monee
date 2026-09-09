===========
Quickstart
===========

monee represents multi-energy grids as directed graphs and solves steady-state
energy flows or optimisation problems over them. This page walks through the
core workflow in five minutes.

.. tip::

   You will need monee installed before running any of the examples below.
   See the :doc:`install` page if you have not set it up yet.

----

Solving an electricity network
================================

Build a radial electricity network, attach an external grid as the slack bus,
add a load, and run the energy-flow calculation:

.. testcode::

    from monee import mx, run_energy_flow

    net = mx.create_multi_energy_network()

    # Two buses connected by a 100 m line
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, length_m=100,
                   r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    # Slack bus and demand
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    result = run_energy_flow(net)

By default :func:`~monee.run_energy_flow` runs in simulation mode
(``simulation=True``): the model is squared and solved as a steady-state flow.
With the default in-process CasADi back-end this runs as a single NLP solve and
``result.mode_used`` reports ``"optimization"``; on the GEKKO back-end the same
square solve runs as ``IMODE=1`` and reports ``"simulation"`` (falling back to
``IMODE=3``/``"optimization"`` if the model is not square). Check
``result.mode_used`` to see which path actually ran. To pick a solver by
name, pass ``run_energy_flow(net, solver="ipopt")`` or ``solver="gurobi"``. See
:doc:`concepts/solvers` for the available back-ends.

Check your network before solving: ``net.check()`` runs a pre-flight lint
(missing slack, dead end junctions, mixed ``base_kv``, unpinned variables)
and prints what it finds; an empty return means clean. When the network is
headed into an optimisation problem, call ``net.check(mode="optimization")``:
free variables such as storage energies are exactly what the problem will
bind, so the unpinned-variable preview is skipped there. Then check your result
before reading numbers: ``result.termination_condition`` says whether the
solver actually converged, ``result.violations`` and ``result.warnings`` are
non-empty when a solved value sits outside its declared bounds or a
post-solve validation check flagged something, and ``result.mode_used`` says
which solve path ran. When any of them surprises you, start at
:doc:`how-to/troubleshooting` and :doc:`how-to/diagnose_infeasibility`. The
signs, units and defaults behind every solved value are collected on
:doc:`concepts/conventions`.

Inspecting the result
---------------------

:func:`~monee.run_energy_flow` returns a
:class:`~monee.solver.core.SolverResult` with two main attributes:

``result.network``
    The solved network object with all variables updated in-place.

``result.dataframes``
    A ``dict[str, pandas.DataFrame]`` keyed by model class name.  Each row is
    one component; each column is a model variable or parameter.

.. testcode::

    bus_df = result.dataframes["Bus"]
    print(len(bus_df))                    # one row per bus

.. testoutput::

    2

Access individual columns by name:

.. code-block:: python

    print(bus_df[["id", "vm_pu", "va_degree"]])

Sign cheat for hydraulic results: a negative ``mass_flow_kgs`` on a pipe means
flow running from the from node to the to node, and a negative
``mass_flow_kgs`` on an ``ExtHydrGrid`` means the network imports from it. Take ``abs()`` for
throughput and read direction off the sign only. All signs and units are
collected on :doc:`concepts/conventions`.

----

Multi-energy networks
=====================

Couple several energy carriers in a single simulation. The example below
connects an electricity grid to a district heating grid through a power-to-heat
(P2H) unit:

.. testcode::

    from monee import mx, run_energy_flow

    net = mx.create_multi_energy_network()

    # Electricity grid
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, length_m=100,
                   r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    # District heating grid (hot-water pipes)
    j_cold = mx.create_water_junction(net)  # cold header, plant reference
    j_mid  = mx.create_water_junction(net)  # junction after pipe
    j_hot  = mx.create_water_junction(net)  # heated outlet of the P2H

    mx.create_ext_hydr_grid(net, j_cold)
    mx.create_water_pipe(net, j_cold, j_mid, diameter_m=0.12, length_m=100)
    mx.create_sink(net, j_hot, mass_flow_kgs=1)

    # Couple the two grids. The water is heated on its way from j_mid to
    # j_hot, so the second junction is the hot one.
    mx.create_p2h(net, bus_1, cold_node_id=j_mid, hot_node_id=j_hot,
                  heat_energy_mw=0.1, diameter_m=0.1, efficiency=0.9)

    result = run_energy_flow(net)

    # Both carriers appear in the result
    print("Bus" in result.dataframes)
    print("Junction" in result.dataframes)

.. testoutput::

    True
    True

.. tip::

   Other coupling units, :func:`~monee.express.create_g2p` (gas-to-power),
   :func:`~monee.express.create_p2g` (power-to-gas),
   :func:`~monee.express.create_g2h` (gas-to-heat), and
   :func:`~monee.express.create_chp` (combined heat and power), follow the
   same pattern.  See :doc:`concepts/multi_energy` for details on all
   supported coupling types.

----

Optimisation
============

To solve an optimisation problem over the network, replace
:func:`~monee.run_energy_flow` with
:func:`~monee.run_energy_flow_optimization`. monee ships a ready-made
load-shedding formulation and also supports custom objectives and constraints:

.. testcode::

    from monee import mx, run_energy_flow_optimization
    from monee.problem import create_min_load_shedding_problem

    opt_net   = mx.create_multi_energy_network()
    opt_bus_0 = mx.create_bus(opt_net)
    opt_bus_1 = mx.create_bus(opt_net)
    mx.create_line(opt_net, opt_bus_0, opt_bus_1, length_m=100,
                   r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(opt_net, opt_bus_0)
    mx.create_power_load(opt_net, opt_bus_1, p_mw=0.5, q_mvar=0.0)

    problem = create_min_load_shedding_problem(
        bounds_vm=(0.9, 1.1),
        check_pressure=False,      # no gas grid in this network
        check_t=False,   # no heat grid in this network
    )

    result = run_energy_flow_optimization(opt_net, problem)
    print(f"Objective: {max(0.0, result.objective):.4f}")

Nothing needs shedding here, so the objective is zero. The ``max(0.0, ...)``
clip removes interior-point noise of about 1e-9 that would otherwise flip the
sign of the printed zero.

.. testoutput::

    Objective: 0.0000

See :doc:`tutorials/01_optimization_basics` and :doc:`how-to/load_shedding`
for end-to-end worked examples.

----

Extending with custom models
==============================

The express API covers the built-in component library. For custom physics,
subclass the appropriate model base class and implement ``equations``:

.. testcode::

    import monee.model as mm
    from monee import run_energy_flow

    @mm.model
    class FlexibleLoad(mm.PowerLoad):
        """A load whose active power is a free variable constrained by *c*."""

        def __init__(self, c, **kwargs):
            # Declare p_mw and q_mvar as solver decision variables
            super().__init__(mm.Var(c, min=0, max=c), mm.Var(0), **kwargs)
            self._c = c

        def equations(self, grid, node_model, **kwargs):
            return [
                self.p_mw <= self._c,
                self.q_mvar == 0,
            ]

    # Build a single-bus network using the custom load
    pn = mm.Network(mm.PowerGrid(name="power", sn_mva=1))

    child_id = pn.child(FlexibleLoad(c=0.5))
    node_id = pn.node(mm.Bus(base_kv=1), child_ids=[pn.child(mm.ExtPowerGrid(p_mw=1, q_mvar=1))], grid=mm.EL)                       # slack
    node_load = pn.node(mm.Bus(base_kv=1), child_ids=[child_id], grid=mm.EL) # load bus
    pn.branch(mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
           from_node_id=node_id, to_node_id=node_load, grid=mm.EL)
    print(run_energy_flow(pn))

.. skipped: p_mw is a free degree of freedom that no equation pins, so the
   printed operating point is back-end dependent (see the paragraph below).

.. testoutput::
   :options: +SKIP

    SolverResult

      Bus  (2 instances)
      ────────────────────────────────────────────────────────────────────
       id  ignored  base_kv  vm_pu  vm_pu_squared  va_radians  va_degree    p_mw     q_mvar
        0    False        1      1              1           0          0 -0.3059 -0.0009353
        1    False        1 0.9969         0.9939   -0.003059    -0.1753   0.305          0

      FlexibleLoad  (1 instance)
      ────────────────────────────────────────────────────────────────────
       id  ignored  regulation  p_mw  q_mvar  node_id
        0    False           1 0.305       0        1

      ExtPowerGrid  (1 instance)
      ────────────────────────────────────────────────────────────────────
       id  ignored  regulation    p_mw     q_mvar  vm_pu  va_degree  regulate_vm  node_id
        1    False           1 -0.3059 -0.0009353      1          0         True        0

      PowerLine  (1 instance)
      ────────────────────────────────────────────────────────────────────
             id  ignored  on_off  tap  shift  br_r_pu  br_x_pu  g_fr_pu  b_fr_pu  g_to_pu  b_to_pu  max_i_ka max_s_mva  backup  p_from_mw  q_from_mvar  i_from_ka  loading_from_pu  p_to_mw  q_to_mvar  i_to_ka  loading_to_pu  length_m  r_ohm_per_m  x_ohm_per_m  parallel
      (0, 1, 0)    False       1    1      0     0.01     0.01        0        0        0        0      3.19      None   False     0.3059    0.0009353     0.1766          0.05537   -0.305          0   0.1766        0.05537       100       0.0001       0.0001         1

``p_mw`` here is a decision variable that no equation determines: the model
only bounds it from above, so in simulation mode it is a free degree of freedom
and the printed value is whichever feasible point the solver landed on, not a
setpoint. The run says so, with a warning that the model is not square, and a
different back-end will print a different number. Give a custom model a fixed
setpoint as a plain float or a :class:`~monee.model.Const`, and keep
:class:`~monee.model.Var` for quantities an equation pins or an optimisation
problem drives. See :doc:`concepts/solvers` for the squaring rules in
simulation mode.

The :func:`~monee.model.core.model` decorator (``@mm.model``) registers the
class in monee's component registry. Registration lets the model round-trip
through the native OMEF serialisation (:mod:`monee.io.native`). For the full
model contract, including custom branches and nodes, read the
:doc:`concepts/data_model` concept page.

----

Next steps
==========

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Tutorials
      :link: tutorials/index
      :link-type: doc
      :shadow: sm

      End-to-end worked examples: optimisation and time-series simulation.

   .. grid-item-card:: Physical models
      :link: concepts/domains
      :link-type: doc
      :shadow: sm

      The equations for electricity, gas, and heat networks.

   .. grid-item-card:: Multi-energy coupling
      :link: concepts/multi_energy
      :link-type: doc
      :shadow: sm

      All coupling components (P2H, G2P, CHP, G2H) in detail.

   .. grid-item-card:: Formulations
      :link: concepts/formulations
      :link-type: doc
      :shadow: sm

      Choose between AC, MISOCP, and nonlinear formulations.

   .. grid-item-card:: Solvers
      :link: concepts/solvers
      :link-type: doc
      :shadow: sm

      GEKKO versus Pyomo: when to use each.

   .. grid-item-card:: API Reference
      :link: api/index
      :link-type: doc
      :shadow: sm

      Complete reference for every public function and class.
