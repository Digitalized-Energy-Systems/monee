========================
Bulk topology builders
========================

:mod:`monee.express` provides structure builders that create whole line,
ring, and star topologies (and paired district-heating supply and return
networks) in a few calls, instead of one ``mx.create_*`` call per node,
branch, and load.

Each ``*_structure`` factory returns a builder that remembers carrier-level
defaults (pipe diameter, segment length, line impedance, and so on) so the
shape methods stay short. Shape methods return lightweight segment handles
that you pass back via ``start_from=`` to compose larger structures.

----

Quick start
===========

Build a five-bus radial feeder with a 0.1 MW load on every bus, attach the
slack bus, and run the energy flow:

.. testcode::

    import monee.model as mm
    import monee.express as mx
    from monee import run_energy_flow

    net = mm.Network()

    # One builder call fixes the line parameters for every branch ...
    s = mx.el_structure(net, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    # ... one shape call creates 5 buses, 4 lines, and 5 loads.
    seg = s.line(5, load_p_mw=0.1)
    s.attach_ext_grid(seg.first)

    result = run_energy_flow(net)
    print(result.dataframes["Bus"][["vm_pu"]].shape)
    print(len(result.dataframes["PowerLine"]))

.. testoutput::

    (5, 1)
    4

``mm.Network()`` and :func:`~monee.express.create_multi_energy_network` are
interchangeable: the second is a plain alias returning the same empty network.
This page writes ``mm.Network()`` because it imports :mod:`monee.model`
anyway; pages that stay inside the express API, such as :doc:`../quickstart`
and :doc:`storage`, prefer ``mx.create_multi_energy_network()``.

----

Builders and their defaults
===========================

Every builder takes the network first, then keyword-only carrier
parameters. These are stored at construction and applied to every branch
the builder creates. Per-call overrides are not supported, so create a
second builder on the same network if you need different parameters.

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Factory
     - Required parameters
     - Optional parameters (defaults)
   * - :func:`~monee.express.el_structure`
     - ``length_m``, ``r_ohm_per_m``, ``x_ohm_per_m``
     - ``parallel=1``, ``base_kv=1``, ``grid=None``
   * - :func:`~monee.express.gas_structure`
     - ``diameter_m``, ``length_m``
     - ``temperature_ext_k=296.15``, ``roughness_m=1e-5``, ``grid=None``
   * - :func:`~monee.express.water_structure`
     - ``diameter_m``, ``length_m``
     - ``temperature_ext_k=296.15``, ``roughness_m=0.001``,
       ``lambda_insulation_w_per_m_k=0.025``, ``insulation_thickness_m=0.2``,
       ``unidirectional=False``, ``grid=None``
   * - :func:`~monee.express.dhs_structure`
     - ``diameter_m``, ``length_m``
     - same as ``water_structure``, but ``unidirectional=True``
       (see `District heating structures`_ below)

``grid=None`` means the network's default grid for that carrier:
:class:`~monee.model.PowerGrid` ``sn_mva=1`` for ``el_structure``,
:class:`~monee.model.WaterGrid` at ``t_ref_k=356`` K for ``water_structure``
and ``dhs_structure``, and ``create_gas_grid("gas")`` for ``gas_structure``,
which is L-gas at 300 K with ``pressure_ref_pa=1e6``, that is 10 bar absolute.
Node pressures are reported in per unit of that reference, so a gas structure
left at the default runs a 10 bar network. To model a medium-pressure network
instead, pass the grid explicitly:

.. testcode::

   import monee.model as mm
   import monee.express as mx

   net_mp = mm.Network()
   gas_mp = mx.gas_structure(
       net_mp,
       diameter_m=0.15,
       length_m=400,
       grid=mm.create_gas_grid(
           "mp",
           pressure_ref_pa=1.5e5,
           pressure_ambient_pa=mm.STANDARD_ATMOSPHERE_PA,
       ),
   )
   seg = gas_mp.line(3)
   print(net_mp.grids[-1].pressure_ref_pa)

.. testoutput::

   150000.0

Build the grid with its operating pressure rather than assigning
``pressure_ref_pa`` afterwards: the compressibility is derived in
``__post_init__``, so a later assignment leaves it stale.
``pressure_ambient_pa`` switches node pressures from absolute (the default) to
gauge. See :doc:`../concepts/data_model` for the full list of per carrier
constants.

The builders delegate to the regular express creators
(:func:`~monee.express.create_bus` and
:func:`~monee.express.create_line`,
:func:`~monee.express.create_gas_junction` and
:func:`~monee.express.create_gas_pipe`,
:func:`~monee.express.create_water_junction` and
:func:`~monee.express.create_water_pipe`), so all units and sign
conventions match the rest of the express API.

----

Shapes: line, ring, star
========================

All single-carrier builders (``ElStructure``, ``GasStructure``,
``WaterStructure``) share three shape methods:

``.line(n, *, start_from=None, **load_kwargs)``
    A chain of ``n`` nodes (``n >= 1``) connected by ``n - 1`` branches.

``.ring(n, *, start_from=None, **load_kwargs)``
    A line of ``n`` nodes (``n >= 3``) plus one closing branch from the
    last node back to the first.

``.star(arms, *, start_from=None, hub_loads=False, **load_kwargs)``
    A hub node with one radial arm per entry of ``arms``, where each entry
    is the arm length in nodes (excluding the hub), e.g. ``[2, 3, 2]``.
    The hub is treated as a pure branching or feed-in point: by default it
    gets no load children, only the arm nodes do. Pass ``hub_loads=True``
    to attach the load setpoints to the hub as well.

Passing ``start_from=<node id>`` composes a new shape onto an existing
node. That node becomes the first node of the line or ring (or the hub of
the star), no new node is created in its place, and the per-node loads are
not re-attached to it.

.. warning::

   ``el_structure`` never changes an existing bus: its ``base_kv`` applies
   only to buses the builder creates. Growing a shape from a ``start_from``
   bus whose ``base_kv`` differs from the builder's therefore produces a
   mixed-base network, which skews the per-unit line parameters and can
   depress voltages while the solve still reports success. The builder
   emits a ``UserWarning`` naming both ``base_kv`` values when this
   happens; fix it by creating the bus with the matching ``base_kv`` or by
   constructing the builder with the bus's ``base_kv``.

The ``load_kwargs`` attach a child to every newly created node (star hubs
excluded unless ``hub_loads=True``):

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Carrier
     - Consumption kwargs
     - Injection kwargs
   * - Electrical
     - ``load_p_mw`` (with optional ``load_q_mvar``, default 0)
     - ``gen_p_mw`` (with optional ``gen_q_mvar``, default 0)
   * - Gas
     - ``sink_mass_flow`` (kg/s)
     - ``source_mass_flow`` (kg/s)
   * - Water
     - ``sink_mass_flow``, ``heat_load_q_mw``
     - ``source_mass_flow``, ``heat_generator_q_mw``

A keyword outside the carrier's row is a typo, and the shape methods raise
``TypeError`` naming the accepted keys rather than quietly attaching nothing:
``gas_structure(...).line(3, sink_mass_flow_kgs=0.02)`` fails instead of
building three junctions with no sink on them. The accepted set per builder is
also readable at runtime as ``GasStructure.LOAD_KWARGS`` and its siblings.

Pass generator and source magnitudes as positive numbers. The underlying
express creators negate them internally, exactly as
:func:`~monee.express.create_power_generator` and
:func:`~monee.express.create_gas_source` do.

Finally, each builder offers ``.attach_ext_grid(node_id, **kwargs)``, which
forwards to the carrier's external-grid creator
(:func:`~monee.express.create_ext_power_grid`,
:func:`~monee.express.create_gas_ext_grid`, or
:func:`~monee.express.create_water_ext_grid`).

Composing shapes
----------------

A gas star fed at the hub, with a three-junction extension grafted onto
the end of the middle arm:

.. testcode::

    import monee.model as mm
    import monee.express as mx
    from monee import run_energy_flow

    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=200)

    star = g.star([2, 3, 2], sink_mass_flow=0.05)
    g.attach_ext_grid(star.hub)

    # Extend the middle arm by a 3-junction line (2 new junctions).
    feeder = g.line(3, start_from=star.arms[1].last, sink_mass_flow=0.02)
    print(feeder.first == star.arms[1].last)

    result = run_energy_flow(net)
    print(len(result.dataframes["GasPipe"]))

.. testoutput::

    True
    9

Feeder with a load-free head bus
--------------------------------

``load_kwargs`` attach a child to every node the shape creates, including the
one you then hand to ``attach_ext_grid``, so ``s.line(5, load_p_mw=0.25)``
puts a load on the slack bus as well. The standard feeder layout, an external
grid at the head bus and loads on the rest, is a two-shape composition: create
the head bus on its own, then grow the loaded feeder from it. ``start_from``
does not re-attach the load kwargs to the node it starts from:

.. testcode::

    import monee.model as mm
    import monee.express as mx
    from monee import run_energy_flow

    net = mm.Network()
    s = mx.el_structure(net, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)

    head = s.line(1)               # one bus, no load kwargs
    s.attach_ext_grid(head.first)

    # 4 new buses with a load each, chained onto the head bus.
    feeder = s.line(5, start_from=head.first, load_p_mw=0.25)

    result = run_energy_flow(net)
    print(len(feeder), len(feeder.children))
    print(len(result.dataframes["Bus"]), len(result.dataframes["PowerLoad"]))

.. testoutput::

    5 4
    5 4

``len(feeder)`` counts the head bus, which the segment lists in ``.nodes``
and reports as ``.first``, but ``feeder.children`` holds only the four loads
the call created. The same idiom works for a gas or water structure fed at a
junction that carries no sink.

----

Segment handles
===============

Shape methods return handles that record everything they created, so you
can keep wiring without tracking node ids by hand:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Handle
     - Contents
   * - :class:`~monee.express.structures.Segment`
     - Returned by ``line``/``ring``.  ``.nodes``, ``.branches``, and
       ``.children`` list the created ids in order; ``.first`` and
       ``.last`` are the end nodes, and ``len(seg)`` is the node count.
       On a ring, ``.branches`` includes the closing branch.
   * - :class:`~monee.express.structures.StarSegment`
     - Returned by ``star``.  ``.hub`` is the hub node id and ``.arms``
       is a list of ``Segment`` objects whose ``.first`` is the shared
       hub.  ``.nodes``, ``.branches``, and ``.children`` are derived
       from the arms (each node listed once).
   * - :class:`~monee.express.structures.DhsSegment`
     - Returned by the DHS builder.  ``.supply`` and ``.return_`` are the
       two water ``Segment`` objects, ``.heat_exchangers`` lists the
       bridging heat-exchanger branches, and the derived
       ``.nodes``/``.branches``/``.children`` cover both sides.

.. tip::

   The handles store plain ids, so anything you build with a structure can
   be refined afterwards with the regular express API. For example,
   attach a CHP to ``seg.last`` or convert a load on ``star.hub`` into a
   controllable demand.

Parameter studies: editing what a structure built
-------------------------------------------------

A reinforcement or sensitivity study reuses one built topology and varies a
few components. Resolve the id a handle stores to its component with the
network lookups, then edit the ``.model``:

* ``net.node_by_id(seg.nodes[i])``
* ``net.branch_by_id(seg.branches[i])``
* ``net.child_by_id(seg.children[i])``

Each returns the container object; the physical parameters live on its
``.model``. Children are removed with ``net.remove_child(child_id)`` (see
:doc:`../concepts/conventions` for the full set of removal methods):

.. testcode::

    import monee.model as mm
    import monee.express as mx
    from monee import run_energy_flow

    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=200)
    seg = g.ring(6, sink_mass_flow=0.1)
    g.attach_ext_grid(seg.nodes[0])

    # Reinforce one pipe ...
    net.branch_by_id(seg.branches[0]).model.diameter_m = 0.25

    # ... raise one consumer ...
    net.child_by_id(seg.children[1]).model.mass_flow_kgs = 0.35

    # ... and drop the sink at the feed-in junction.
    net.remove_child(seg.children[0])

    result = run_energy_flow(net)
    print(len(result.dataframes["Sink"]))
    print(round(result.dataframes["Sink"]["mass_flow_kgs"].max(), 2))

.. testoutput::

    5
    0.35

What is safe to change after construction:

* Constructor arguments of node, branch and child models are stored as plain
  attributes and nothing is derived from them at construction, so assigning a
  new value before the next solve is enough: ``diameter_m``, ``length_m``,
  ``roughness_m`` and ``on_off`` on pipes, ``r_ohm_per_m``, ``x_ohm_per_m``
  and ``parallel`` on lines, and the setpoints and ``regulation`` on children.
* Attributes that hold solver state are not inputs. A pipe's
  ``mass_flow_kgs``, ``velocity_mps`` and ``friction``, and a bus's
  ``vm_pu``, carry start values that every solve overwrites, so writing them
  changes the starting point at most. Setpoints of the same name on children
  (a ``Sink``'s ``mass_flow_kgs``, a ``PowerLoad``'s ``p_mw``) are ordinary
  numbers and are the ones to edit.
* Grid objects are the exception, because they derive constants in
  ``__post_init__``: a ``GasGrid``'s compressibility follows from
  ``pressure_ref_pa`` and ``t_k``, a ``WaterGrid``'s density and viscosity
  from ``t_ref_k``. Build a new grid with the operating point you want, as
  under `Builders and their defaults`_, rather than assigning those fields on
  an existing one.

To sweep a parameter without carrying solved values between runs, build the
variants from ``net.copy()`` and solve each copy.

----

District heating structures
===========================

:func:`~monee.express.dhs_structure` returns a
:class:`~monee.express.structures.DhsStructure` that mirrors every shape on
two water networks at once: a supply side and a return side. Both are
regular ``WaterStructure`` builders, reachable as ``.supply`` and
``.return_``. Its shape methods accept three extra keyword arguments:

* ``start_from_supply=`` and ``start_from_return=``: compose onto existing
  supply or return nodes, analogous to ``start_from=`` above.
* ``heat_exchanger_q_mw=``: bridge each supply node to its return
  counterpart with a heat exchanger
  (:func:`~monee.express.create_heat_exchanger`, oriented supply to return,
  hot side into cold side). Pass a scalar to put the same consumer on every
  node pair, or a list with one entry per node. Entries of ``None`` or
  ``0`` skip that node, and negative values create heat-exchanger
  generators.

A ring of four consumers supplied by one heat plant:

.. testcode::

    import monee.model as mm
    import monee.express as mx
    from monee import run_energy_flow

    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    # 4-node supply ring, 4-node return ring, and 4 heat exchangers.
    seg = dhs.ring(4, heat_exchanger_q_mw=0.01)

    # Heat plant: pressure and temperature reference on the supply side,
    # consumption point on the return side.
    dhs.attach_heat_plant(seg.supply.first, seg.return_.first, name="plant")

    result = run_energy_flow(net)
    print(len(seg.supply), len(seg.return_), len(seg.heat_exchangers))
    print(result.dataframes["Junction"][["t_k"]].shape)

.. testoutput::

    4 4 4
    (8, 1)

``.attach_heat_plant(supply_node, return_node, *, t_k=358.0, name=None)``
closes the hydraulic loop. It attaches an ``ExtHydrGrid`` at
``supply_node`` (via :func:`~monee.express.create_water_ext_grid`), holding
the feed temperature ``t_k``, and a ``ConsumeHydrGrid`` at ``return_node``
(via :func:`~monee.express.create_consume_hydr_grid`). It returns the tuple
``(supply_child_id, return_child_id)``. If ``name`` is given, the children
are named ``<name>_supply`` and ``<name>_return``.

.. note::

   ``dhs_structure`` creates unidirectional pipes by default, oriented from
   the first node towards the last. A *ring* is the exception: its pipes are
   always created bidirectional, whatever ``unidirectional`` the builder was
   given, because a closed loop splits the flow at the plant and at least one
   pipe then carries water against the ring orientation. On a *line*, water
   leaving the heat exchangers must be able to flow towards the plant's return
   connection:
   attach it at ``seg.return_.last`` (not ``seg.return_.first``), or pass
   ``unidirectional=False`` to the builder so the solver picks flow
   directions freely.

----

See also
========

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Multi-energy coupling
      :link: ../concepts/multi_energy
      :link-type: doc
      :shadow: sm

      Connect the carriers you just built with P2H, G2P, P2G, and CHP
      coupling components.

   .. grid-item-card:: Reference networks
      :link: reference_networks
      :link-type: doc
      :shadow: sm

      Ready-made multi-energy benchmark networks if you would rather not
      build a topology at all.

   .. grid-item-card:: Quickstart
      :link: ../quickstart
      :link-type: doc
      :shadow: sm

      The node-by-node express workflow that the structure builders
      condense.
