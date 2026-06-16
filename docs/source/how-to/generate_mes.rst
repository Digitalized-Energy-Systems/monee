==========================================
Generate synthetic multi-energy networks
==========================================

Electrical test grids are abundant, but realistic multi-energy benchmark
cases are scarce. The generator suite in :mod:`monee.network.mes` turns any
electrical :class:`~monee.model.network.Network` into a full multi-energy
system (MES). It overlays a gas grid and a district heating system (DHS) on
the spanning tree of the power topology, sizes the new loads and sources from
the local electrical demand and generation, and places coupling points (CHP,
P2G, P2H) across the shared buses.

Choosing a generator
====================

.. list-table::
   :header-rows: 1
   :widths: 42 18 40

   * - Function
     - Randomness
     - Use when
   * - :func:`~monee.network.mes.generate_supply_return_mes_based_on_power_net`
     - Seedable
     - Recommended. Deterministic gas and heat overlays plus seedable
       coupling placement, with a supply/return DHS suited to the McCormick
       and smooth NLP formulations.
   * - :func:`~monee.network.mes.generate_mes_based_on_power_net`
     - Seedable
     - Legacy. Randomised single-pipe heat grid and gas grid with deployment
       rates instead of deterministic sizing. Takes a ``seed`` argument.
   * - :func:`~monee.network.bench.mes_simbench.generate_mes_based_on_simbench_id`
     - Global ``random``
     - Legacy generator applied directly to a `simbench
       <https://simbench.de>`_ grid (requires the ``simbench`` extra). Has no
       seed parameter.

All generators work on a copy of the input power network. The original is
never modified.

----

One-call wrapper (recommended)
==============================

:func:`~monee.network.mes.generate_supply_return_mes_based_on_power_net`
chains the three deterministic builders described below (gas tree,
supply/return DHS, coupling points) into a single call:

.. code-block:: python

    from monee.network import generate_supply_return_mes_based_on_power_net

    mes = generate_supply_return_mes_based_on_power_net(
        net_power,
        coupling_density=0.2,
        coupling_kwargs={"seed": 42},
    )

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword argument
     - Description
   * - ``coupling_density``
     - Probability or share of buses that receive a coupling point
       (default ``0.2``, see :ref:`density semantics <how-to/generate_mes:Density semantics>`).
   * - ``centralized`` / ``central_node_id``
     - Concentrate all coupling points on one hub bus instead of spreading
       them across the network.
   * - ``couplings``
     - Tuple of unit types to draw from; default ``("chp", "p2g", "p2h")``.
   * - ``heat_kwargs`` / ``gas_kwargs`` / ``coupling_kwargs``
     - Dicts forwarded verbatim to
       :func:`~monee.network.mes.create_heat_supply_return_net_for_power`,
       :func:`~monee.network.mes.create_gas_tree_net_for_power`, and
       :func:`~monee.network.mes.create_coupling_points_for_mes`.

.. tip::

   The gas and heat overlays are fully deterministic. The only random choices
   are which buses receive coupling points and, optionally, where extra gas
   mesh pipes land. Both take explicit seeds:
   ``coupling_kwargs={"seed": 42}`` and ``gas_kwargs={"mesh_seed": 7}``.

Legacy randomised generator
---------------------------

:func:`~monee.network.mes.generate_mes_based_on_power_net` is the older
interface. Instead of deterministic sizing it deploys heat exchangers and gas
sinks at random with the given rates, and places coupling points with per-type
densities:

.. code-block:: python

    from monee.network import generate_mes_based_on_power_net

    mes = generate_mes_based_on_power_net(
        net_power,
        heat_deployment_rate=0.5,
        gas_deployment_rate=0.5,
        chp_density=0.1,
        p2g_density=0.02,
        p2h_density=0.1,
        seed=42,
    )

.. warning::

   :func:`~monee.network.mes.generate_mes_based_on_power_net` takes a ``seed``
   argument; pass it for reproducible networks. Leaving ``seed=None`` falls
   back to the global :mod:`random` state. The simbench wrapper below has no
   seed parameter at all, so seed the global :mod:`random` module before
   calling it.

:func:`~monee.network.bench.mes_simbench.generate_mes_based_on_simbench_id`
applies the same overlay directly to a simbench code:

.. code-block:: python

    from monee.network.bench.mes_simbench import generate_mes_based_on_simbench_id

    mes = generate_mes_based_on_simbench_id(
        "1-LV-rural3--0-no_sw",
        heat_deployment_rate=0.5,
        gas_deployment_rate=0.5,
    )

----

Gas overlay
===========

:func:`~monee.network.mes.create_gas_tree_net_for_power` mirrors every bus of
the power spanning tree with a gas junction, connects them with pipes, and
sizes per-bus sinks and sources from the electrical load and generation
magnitudes. The energy-to-mass conversion uses the grid's higher heating
value (lgas, about ``42.44`` MJ/kg), so a bus with 1 MW of electrical load and
``gas_load_share=3.0`` receives a gas sink of
:math:`3 / 42.44 \approx 0.071` kg/s.

.. code-block:: python

    from monee.network import create_gas_tree_net_for_power

    mes = net_power.copy()
    bus_to_gas_junc = create_gas_tree_net_for_power(
        net_power,
        mes,
        gas_load_share=3.0,
        extra_mesh_pipes=2,
        mesh_seed=7,
    )

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword argument
     - Description
   * - ``gas_load_share`` / ``gas_gen_share``
     - Multipliers on the electrical load/generation MW before HHV
       conversion (defaults ``3.0`` / ``1.0``).
   * - ``default_diameter_m``
     - Pipe diameter, default ``0.3`` m.
   * - ``min_load_kgs`` / ``min_source_kgs``
     - Per-bus floors (default ``1.8e-4`` kg/s ≈ 7.6 kW thermal).
   * - ``slack_node_id``
     - Bus that receives the gas external grid; defaults to the first node
       of the spanning tree.
   * - ``extra_mesh_pipes``
     - Number of random tie-line pipes added between non-adjacent
       junctions, giving every junction more than one path to the slack
       (N-1 resilience on the otherwise radial tree).  Default ``0``.
   * - ``mesh_diameter_factor`` / ``mesh_length_factor``
     - Tie-line sizing relative to ``default_diameter_m`` /
       ``default_length`` (defaults ``0.5`` / ``2.0``).
   * - ``mesh_seed``
     - Seed for tie-line placement; ``None`` falls back to the global
       :mod:`random` state.

The function returns a ``{power_node_id: gas_junction_id}`` mapping, which
the coupling-point builder consumes.

----

Heat overlay
============

:func:`~monee.network.mes.create_heat_supply_return_net_for_power` builds a
supply/return DHS: supply junctions mirror the power buses along the
spanning tree (pipes oriented outward from the slack by BFS), one shared
return junction collects all consumer returns, and heat-exchanger branches
bridge supply and return. Loads are sized as
``max(min_load_mw, p_load_mw * heat_load_share)``. Generators are sized the
same way and rescaled to total demand when ``balance_gen_to_load=True``.

It returns ``({power_node_id: supply_junction_id}, return_junction_id)``.

Heat-plant modes
----------------

``heat_plant_mode`` selects how the slack heat plant closes the loop:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Mode
     - Behaviour
   * - ``"closing_pipe"``
     - Default and most robust. A single supply-side slack plus a
       return-to-supply closing pipe. Works under the default
       linear-heat-exchanger formulation, but the slack temperature pin
       collapses some of the supply/return ΔT.
   * - ``"two_port"``
     - A second hydraulic slack at the return junction: cleaner physics,
       but requires the McCormick-DHS path, that is
       ``node_based_heat_loads=True``.
   * - ``"screening"``
     - Closing pipe plus an oversized return sink. Fast for screening
       studies, but the resulting mass flow is not physical.

Node-based heat loads (McCormick-DHS)
-------------------------------------

Setting ``node_based_heat_loads=True`` switches consumers and producers
from two-port heat-exchanger branches to node-level
:class:`~monee.model.child.HeatLoad` / :class:`~monee.model.child.HeatGenerator`
children, as required by
:data:`~monee.model.formulation.HEAT_CONVEX_MILP_FORMULATION`.
In this mode the builder additionally:

- prunes the heat grid to the Steiner subtree of consumer buses
  (dead-end junctions would otherwise be forced to zero flow and drag the
  temperature to ambient),
- tightens the McCormick envelopes on the heat grid
  (``t_pu`` ∈ [0.75, 1.15], ``v_max_mps = 2``) and sets a per-pipe
  ``m_U_design`` flow bound from the cumulative downstream demand,
- distributes :class:`~monee.model.child.HeatGenerator` children at
  generator buses according to ``node_heat_gen_share`` (set ``0.0`` for a
  slack-only plant).

Automatic pipe sizing
---------------------

By default every pipe gets the flat ``default_diameter_m`` (0.12 m). On large
radial networks this is physically infeasible: the trunk pipes near the slack
must carry the whole network's flow, exceeding the velocity cap while the
Darcy pressure drop (proportional to :math:`1/D^5`) blows up. Enable
``auto_diameter=True`` to size each supply pipe (and the closing pipe) from
the cumulative downstream consumer demand instead:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword argument
     - Description
   * - ``auto_diameter``
     - Enable capacity-based sizing (default ``False``).
   * - ``auto_diameter_v_mps``
     - Design velocity; defaults to the heat grid's ``v_max_mps``.
   * - ``auto_diameter_headroom``
     - Safety margin applied to the design flow before sizing
       (default ``1.5``).
   * - ``auto_min_diameter_m``
     - Floor for the sized diameter; defaults to ``default_diameter_m`` so
       leaf pipes are never thinned below the flat default.

.. note::

   In node-based mode each supply pipe also receives a
   ``mass_flow_nominal_kgs`` attribute equal to its design flow. The smooth NLP
   formulations read this as a warm start for the pipe's flow variable, which
   improves conditioning: on a simbench MV network it cuts IPOPT from roughly
   600 to 30 iterations.

----

Coupling points
===============

:func:`~monee.network.mes.create_coupling_points_for_mes` adds CHP, P2G,
and P2H units to a network that already carries the gas and heat overlays,
consuming the junction mappings returned by the two builders above:

.. code-block:: python

    from monee.network import create_coupling_points_for_mes

    created = create_coupling_points_for_mes(
        mes,
        bus_to_gas_junc=bus_to_gas_junc,
        bus_to_heat_supply_junc=bus_to_heat_supply,
        heat_return_junc=heat_return,
        density=0.3,
        seed=42,
    )
    # → [{"type": "chp", "node": 3, "id": ...}, ...]

Density semantics
-----------------

``density`` ∈ [0, 1] is interpreted per placement mode:

- **Decentralised** (``centralized=False``, default): every candidate bus
  (one that has both a gas and a heat junction) receives a coupling point
  with probability ``density`` (a per-node Bernoulli draw).
- **Centralised** (``centralized=True``): ``round(density * N)`` units are
  all placed on one hub bus (``central_node_id``, or the lowest candidate
  id by default), an energy-hub layout.

Each placed unit draws its type uniformly from ``couplings`` and is sized
from the bus's local reference power (generation magnitude if present,
otherwise load; buses with neither are skipped) via the per-type
``*_p_share`` and the global ``cp_size_multiplier``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword argument
     - Description
   * - ``couplings``
     - Unit types to draw from, default ``("chp", "p2g", "p2h")``.
   * - ``chp_p_share`` / ``p2g_p_share`` / ``p2h_p_share``
     - Per-type share of the bus reference power (defaults ``0.5`` /
       ``1.0`` / ``1.0``).
   * - ``chp_efficiency_power`` / ``chp_efficiency_heat``
     - CHP electrical/thermal efficiencies (``0.4`` / ``0.45``).
   * - ``p2g_efficiency`` / ``p2h_efficiency``
     - Conversion efficiencies (``0.7`` / ``0.95``).
   * - ``cp_size_multiplier``
     - Global scaling on all unit capacities (default ``1.0``).
   * - ``regulation``
     - Initial dispatch level in [0, 1] for every created unit (default
       ``1.0``).
   * - ``use_hg_variants``
     - Use the node-based CHP-HG / P2H-HG compounds
       (:func:`~monee.express.create_chp_hg` /
       :func:`~monee.express.create_p2h_hg`) instead of return-branch
       variants. Required with ``node_based_heat_loads=True``
       (McCormick-DHS only sees node-level heat injection).
   * - ``replace_primary_generation``
     - Drain the matching primary fleet (PowerGenerator, gas Source,
       HeatGenerator/HX-Gen) by the rated output of the new units so the
       total rated production per carrier stays invariant: the coupling
       points substitute capacity instead of adding it. Raises ``ValueError``
       if the new units exceed the primary pool and some capacity cannot be
       absorbed. Default ``False`` (additive).
   * - ``seed``
     - Seed for placement and type choice; ``None`` falls back to the
       global :mod:`random` state.

----

Complete example
================

Build a small feeder, overlay gas and heat, add coupling points, and run a
multi-energy power flow:

.. testcode::

    import monee.express as mx
    from monee import run_energy_flow
    from monee.network import generate_supply_return_mes_based_on_power_net

    # Electricity grid
    pn = mx.create_multi_energy_network()
    buses = [mx.create_bus(pn, base_kv=20) for _ in range(5)]
    mx.create_ext_power_grid(pn, buses[0])
    for a, b in zip(buses, buses[1:]):
        mx.create_line(pn, a, b, length_m=100, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    for bus in buses[1:]:
        mx.create_power_load(pn, bus, p_mw=0.1, q_mvar=0.02)

    # Overlay gas + district heating + coupling points
    mes = generate_supply_return_mes_based_on_power_net(
        pn,
        coupling_density=0.4,
        coupling_kwargs={"seed": 42},
    )

    result = run_energy_flow(mes)
    print(result.dataframes["Bus"][["vm_pu"]])

.. testoutput::
   :options: +SKIP

          vm_pu
    0  1.000000
    1  0.999997
    2  0.999994
    3  0.999992
    4  0.999989

For a larger starting point, import a standard electrical test case first,
for example the CIGRE MV grid via pandapower, or a simbench grid:

.. code-block:: python

    import pandapower.networks as ppn
    from monee.io.from_pandapower import from_pandapower_net
    from monee.io.from_simbench import obtain_simbench_net

    pn = from_pandapower_net(ppn.create_cigre_network_mv(with_der="pv_wind"))
    # or: pn = obtain_simbench_net("1-MV-urban--0-no_sw")

    mes = generate_supply_return_mes_based_on_power_net(
        pn,
        coupling_density=0.2,
        heat_kwargs={"node_based_heat_loads": True, "auto_diameter": True},
        coupling_kwargs={"use_hg_variants": True, "seed": 42},
    )

monee also ships two pre-generated MES cases built with this machinery:
:func:`~monee.network.mes.create_monee_benchmark_net` (a fixed 7-bus,
120 kV multi-energy grid with backup lines) and
:func:`~monee.network.mes.create_mv_multi_cigre` (CIGRE MV with gas, heat,
P2G, and CHP).

----

See also
========

.. grid:: 1 2 2 3
   :gutter: 3

   .. grid-item-card:: Reference networks
      :link: reference_networks
      :link-type: doc
      :shadow: sm

      Hand-built, fixed multi-energy reference networks in
      :mod:`monee.network.res` for benchmarking without randomness.

   .. grid-item-card:: Formulations
      :link: ../concepts/formulations
      :link-type: doc
      :shadow: sm

      The formulation layer, including the McCormick-DHS and smooth NLP
      relaxations the heat overlay is tuned for.

   .. grid-item-card:: Multi-energy coupling
      :link: ../concepts/multi_energy
      :link-type: doc
      :shadow: sm

      CHP, P2G, P2H and their node-based HG variants: the building blocks
      the coupling-point factory places.
