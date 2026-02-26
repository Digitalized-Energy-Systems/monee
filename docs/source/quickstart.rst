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

----

Multi-energy networks
=====================

The key feature of monee is coupling several energy carriers in a single
simulation. The example below connects an electricity grid to a district
heating grid via a **power-to-heat (P2H)** unit:

.. testcode::

    from monee import mx, run_energy_flow

    net = mx.create_multi_energy_network()

    # ── Electricity grid ──────────────────────────────────────────────────
    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)

    mx.create_line(net, bus_0, bus_1, length_m=100,
                   r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(net, bus_0)
    mx.create_power_load(net, bus_1, p_mw=0.1, q_mvar=0.0)

    # ── District heating grid (hot-water pipes) ───────────────────────────
    j_supply = mx.create_water_junction(net)  # supply header
    j_mid    = mx.create_water_junction(net)  # junction after pipe
    j_return = mx.create_water_junction(net)  # return header

    mx.create_ext_hydr_grid(net, j_supply)
    mx.create_water_pipe(net, j_supply, j_mid, diameter_m=0.3, length_m=100)
    mx.create_sink(net, j_return, mass_flow=1)

    # ── Couple the two grids ──────────────────────────────────────────────
    mx.create_p2h(net, bus_1, j_mid, j_return,
                  heat_energy_w=10_000, diameter_m=0.1, efficiency=0.9)

    result = run_energy_flow(net)

    # Both carriers appear in the result
    print("Bus" in result.dataframes)
    print("Junction" in result.dataframes)

.. testoutput::

    True
    True

.. tip::

   Other coupling units — :func:`~monee.express.create_g2p` (gas-to-power),
   :func:`~monee.express.create_p2g` (power-to-gas),
   :func:`~monee.express.create_g2h` (gas-to-heat), and
   :func:`~monee.express.create_chp` (combined heat and power) — follow the
   same pattern.  See :doc:`concepts/multi_energy` for details on all
   supported coupling types.

----

Optimisation
============

Replace :func:`~monee.run_energy_flow` with
:func:`~monee.run_energy_flow_optimization` to solve an optimisation problem
over the network. monee ships a ready-made **load-shedding** formulation and
supports fully custom objectives and constraints:

.. testcode::

    from monee import mx, run_energy_flow_optimization
    from monee.problem import create_load_shedding_optimization_problem

    opt_net   = mx.create_multi_energy_network()
    opt_bus_0 = mx.create_bus(opt_net)
    opt_bus_1 = mx.create_bus(opt_net)
    mx.create_line(opt_net, opt_bus_0, opt_bus_1, length_m=100,
                   r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    mx.create_ext_power_grid(opt_net, opt_bus_0)
    mx.create_power_load(opt_net, opt_bus_1, p_mw=0.5, q_mvar=0.0)

    problem = create_load_shedding_optimization_problem(
        bounds_el=(0.9, 1.1),
        check_pressure=False,   # no gas grid in this network
        check_t=False,          # no heat grid in this network
    )

    result = run_energy_flow_optimization(opt_net, problem)
    print(f"Objective: {result.objective:.4f}")

.. testoutput::
   :options: +SKIP

    Objective: ...

See :doc:`tutorials/01_optimization_basics` and :doc:`how-to/load_shedding`
for end-to-end worked examples.

----

Extending with custom models
==============================

The express API covers the built-in component library. For custom physics,
subclass the appropriate model base class and implement ``equations``:

.. testcode::

    import monee.model as mm

    @mm.model
    class FlexibleLoad(mm.PowerLoad):
        """A load whose active power is a free variable constrained by *c*."""

        def __init__(self, c, **kwargs):
            # Declare p_mw and q_mvar as solver decision variables
            super().__init__(mm.Var(c, min=0, max=c), mm.Var(0), **kwargs)
            self._c = c

        def equations(self, grid, node, **kwargs):
            return [
                self.p_mw <= self._c,
                self.q_mvar == 0,
            ]

    # Build a single-bus network using the custom load
    pn = mm.Network(mm.PowerGrid(name="power", sn_mva=1))

    child_id = pn.child(FlexibleLoad(c=0.5))
    pn.node(mm.Bus(base_kv=1), grid=mm.EL)                       # slack
    pn.node(mm.Bus(base_kv=1), child_ids=[child_id], grid=mm.EL) # load bus

Read the :doc:`concepts/data_model` concept page for the full model contract
and how to implement custom branches and nodes.

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

      All coupling components — P2H, G2P, CHP, G2H — in detail.

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
