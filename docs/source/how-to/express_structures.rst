========================
Bulk topology builders
========================

:mod:`monee.express` includes a set of **structure builders** that create
whole line, ring, and star topologies - and even paired district-heating
supply/return networks - in a few calls instead of one
``mx.create_*`` call per node, branch, and load.

Each ``*_structure`` factory returns a builder that remembers carrier-level
defaults (pipe diameter, segment length, line impedance, ...) so the shape
methods stay short.  Shape methods return lightweight segment handles that
you can pass back via ``start_from=`` to compose larger structures.

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

----

Builders and their defaults
===========================

Every builder takes the network first and keyword-only carrier parameters.
The parameters are stored at construction time and applied to **every**
branch the builder creates; per-call overrides are not supported - create a
second builder on the same network instead.

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

Internally the builders delegate to the familiar express creators:
:func:`~monee.express.create_bus` / :func:`~monee.express.create_line`,
:func:`~monee.express.create_gas_junction` /
:func:`~monee.express.create_gas_pipe`, and
:func:`~monee.express.create_water_junction` /
:func:`~monee.express.create_water_pipe` - so all units and sign
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

``.star(arms, *, start_from=None, **load_kwargs)``
    A hub node with one radial arm per entry of ``arms``, where each entry
    is the arm length in nodes (excluding the hub), e.g. ``[2, 3, 2]``.

Passing ``start_from=<node id>`` composes a new shape onto an existing
node: it becomes the first node of the line/ring (or the hub of the star),
no new node is created in its place, and the per-node loads are **not**
re-attached to it.

The ``load_kwargs`` attach a child to every newly created node:

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

Generator and source magnitudes are passed **positive** - the underlying
express creators negate them internally, exactly as with
:func:`~monee.express.create_power_generator` and
:func:`~monee.express.create_gas_source`.

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

----

Segment handles
===============

Shape methods return handles that record everything they created, so you
can keep wiring without bookkeeping node ids by hand:

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
   be refined afterwards with the regular express API - for example,
   attach a CHP to ``seg.last`` or convert a load on ``star.hub`` into a
   controllable demand.

----

District heating structures
===========================

:func:`~monee.express.dhs_structure` returns a
:class:`~monee.express.structures.DhsStructure` that mirrors every shape
on **two** water networks at once - a supply side and a return side
(accessible as ``.supply`` and ``.return_``, each a regular
``WaterStructure``).  Its shape methods accept three extra keyword
arguments:

* ``start_from_supply=`` / ``start_from_return=`` - compose onto existing
  supply/return nodes, analogous to ``start_from=`` above.
* ``heat_exchanger_q_mw=`` - bridge each supply node to its return
  counterpart with a heat exchanger
  (:func:`~monee.express.create_heat_exchanger`, oriented supply ->
  return, i.e. hot side into cold side).  Pass a scalar to put the same
  consumer on every node pair, or a list with one entry per node -
  entries of ``None`` or ``0`` skip that node, and negative values create
  heat-exchanger *generators*.

A ring of four consumers supplied by one heat plant:

.. testcode::

    import monee.model as mm
    import monee.express as mx
    from monee import run_energy_flow

    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    # ── 4-node supply ring + 4-node return ring + 4 heat exchangers ──────
    seg = dhs.ring(4, heat_exchanger_q_mw=0.01)

    # Heat plant: pressure/temperature reference on the supply side,
    # consumption point on the return side.
    dhs.attach_heat_plant(seg.supply.first, seg.return_.first, name="plant")

    result = run_energy_flow(net)
    print(len(seg.supply), len(seg.return_), len(seg.heat_exchangers))
    print(result.dataframes["Junction"][["t_k"]].shape)

.. testoutput::

    4 4 4
    (8, 1)

``.attach_heat_plant(supply_node, return_node, *, t_k=358.0, name=None)``
closes the hydraulic loop: it attaches an ``ExtHydrGrid`` (via
:func:`~monee.express.create_water_ext_grid`) at ``supply_node``, holding
the feed temperature ``t_k``, and a ``ConsumeHydrGrid`` (via
:func:`~monee.express.create_consume_hydr_grid`) at ``return_node``.  It
returns the tuple ``(supply_child_id, return_child_id)``; if ``name`` is
given, the children are named ``<name>_supply`` and ``<name>_return``.

.. note::

   ``dhs_structure`` creates **unidirectional** pipes by default, oriented
   from the first node towards the last.  On a *ring* the loop closes
   itself, so the plant can sit on any node pair.  On a *line*, water
   leaving the heat exchangers must be able to flow towards the plant's
   return connection - attach it at ``seg.return_.last`` (not
   ``seg.return_.first``), or pass ``unidirectional=False`` to the builder
   if you want the solver to pick flow directions freely.

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
