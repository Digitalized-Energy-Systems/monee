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
       coupling placement, with a supply/return DHS. Its ``node_based_heat_loads``
       variant is shaped for the McCormick DHS relaxation, which is the only
       formulation that reads the envelope hints it writes; the branch-based
       default is the one to pair with the smooth NLP formulation.

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

``auto_diameter`` is on by default, so every supply pipe (and the closing
pipe) is sized for the flow it has to carry rather than getting the flat
``default_diameter_m`` of 0.12 m. Two criteria are applied per pipe and the
wider of the two wins:

- velocity: the pipe must carry its cumulative downstream design flow without
  exceeding the design velocity;
- pressure: the Darcy drop, proportional to :math:`1/D^5`, must keep the whole
  worst slack-to-leaf path inside a share of the grid's reference pressure.

The design flow of a pipe is the larger of the cumulative downstream consumer
demand and the cumulative downstream HX-Gen injection, because a generator
pushes its own flow back up the tree.

The velocity criterion on its own is not enough on grids with long branches.
A CIGRE MV overlay sized at 5 m/s stays under the velocity cap everywhere and
still drops far below the Junction ``pressure_pu`` floor of 0.5, which is why
that example used to come back ``Infeasible_Problem_Detected``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Keyword argument
     - Description
   * - ``auto_diameter``
     - Enable demand-based sizing (default ``True``). Set ``False`` for the
       flat ``default_diameter_m`` everywhere.
   * - ``auto_diameter_v_mps``
     - Design velocity; defaults to the heat grid's ``v_max_mps``.
   * - ``auto_diameter_headroom``
     - Safety margin applied to the design flow before sizing
       (default ``1.5``).
   * - ``auto_min_diameter_m``
     - Floor for the sized diameter; defaults to ``default_diameter_m`` so
       leaf pipes are never thinned below the flat default.
   * - ``auto_diameter_pressure_budget_pu``
     - Share of the grid reference pressure the worst supply path may spend
       (default ``0.25``). ``None`` sizes on velocity alone.

Sizing also raises the heat grid's ``max_mass_flow_kgs`` to admit the trunk
design flow. That grid-level cap is a per-branch bound of its own, so leaving
it at the 200 kg/s default would make the sized diameters moot on any overlay
whose trunk carries more than that.

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

          vm_pu
    0  1.000000
    1  0.999988
    2  0.999978
    3  0.999972
    4  0.999970

For a larger starting point, start from a standard electrical test case.
:func:`~monee.network.mes.create_mv_multi_cigre` imports the CIGRE MV grid
through :func:`~monee.io.from_pandapower.from_pandapower_net` and overlays it
with hand-tuned pipe geometry and power scaling. It does not call the wrapper
above: it uses the older single-pipe builders
:func:`~monee.network.mes.create_gas_net_for_power` and
:func:`~monee.network.mes.create_heat_net_for_power` (one heat pipe per
branch, no separate return network) and then places its generators, P2G and
CHP by hand. Treat it as a fixed benchmark case rather than as an example of
the recommended generator. It solves in a few seconds:

.. testcode::

    from monee import run_energy_flow
    from monee.network import create_mv_multi_cigre

    mes = create_mv_multi_cigre()
    result = run_energy_flow(mes)
    print(result.success)

.. testoutput::

    True

monee also ships a second fixed MES case,
:func:`~monee.network.mes.create_monee_benchmark_net`, a 7-bus, 120 kV
multi-energy grid with backup lines. It is built from the same older
single-pipe builders as ``create_mv_multi_cigre``.

To generate a supply and return MES on the same CIGRE MV import yourself, hand
the converted network to the wrapper. The demand-based sizing described above
is on by default, so no ``heat_kwargs`` are needed to make it solve:

.. testcode::

    import pandapower.networks as ppn

    from monee import run_energy_flow
    from monee.io.from_pandapower import from_pandapower_net
    from monee.network import generate_supply_return_mes_based_on_power_net

    power_net = from_pandapower_net(ppn.create_cigre_network_mv(with_der="pv_wind"))
    mes = generate_supply_return_mes_based_on_power_net(power_net, coupling_density=0.2)
    print(run_energy_flow(mes, formulation="smooth_nlp").success)

.. testoutput::

    True

.. note::

   The pandapower converter does not carry line lengths across, so the layer
   builders fall back on the geodesic distance between the bus positions. On
   grids whose bus coordinates are a schematic drawing rather than geographic
   ones, CIGRE MV among them, that fallback overstates every segment by an
   order of magnitude or two. The overlay still solves, because the pressure
   criterion widens the pipes until it does, but the resulting diameters are
   not a design you would build. Pass ``heat_kwargs={"length_scale": ...}`` or
   an explicit ``default_length`` when the derived lengths look wrong, the way
   :func:`~monee.network.mes.create_mv_multi_cigre` does with
   ``length_scale=0.001``.

.. warning::

   With ``auto_diameter=False`` every pipe keeps the flat
   ``default_diameter_m`` of 0.12 m, which carries only a few kg/s at the
   design velocity. An MV feeder overlaid that way asks for pressure drops far
   outside the junction bounds and the solve comes back infeasible; the
   generator warns when the design flow outgrows the flat diameter. Either
   leave ``auto_diameter`` on, or lower ``heat_load_share`` so the heat demand
   matches the pipes you asked for. The gas tree takes the same
   ``auto_diameter`` switch.

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
