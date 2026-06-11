====================
Reference networks
====================

monee ships a family of **fixed multi-energy reference networks** that are
ready to use for testing, benchmarking, and tutorial examples.  Every factory
is deterministic - hand-built or internally seeded - so repeated calls always
return the same grid.  All of them couple electricity, gas, and/or heat via
standard conversion units (CHP, P2H, P2G, G2P) and are importable directly
from :mod:`monee.network`.

If you want to *generate* a multi-energy overlay on top of an arbitrary
power network instead - gas and heat grids derived from the electrical
topology, with configurable coupling-point density - see
:doc:`generate_mes`.

----

Available networks
==================

.. list-table::
   :header-rows: 1
   :widths: 32 14 54

   * - Function
     - Scale
     - Description
   * - :func:`~monee.network.res.create_urban_district_net`
     - Small
     - 20 kV residential district.  5 buses · 5 gas junctions · 6 heat
       junctions.  CHP-centred topology with high coupling-point density.
       Suitable for load-shedding and demand-side flexibility studies.
   * - :func:`~monee.network.res.create_urban_district_net_with_ties`
     - Small
     - Urban district plus **normally-open tie branches** (``on_off=0``,
       named ``tie_*``) in all three carriers and a second heat consumer.
       Built for restoration and reconfiguration studies.
   * - :func:`~monee.network.res.create_balanced_urban_mes_net`
     - Small
     - Urban district with **balanced per-carrier demands** (~1.7 MW power ·
       ~1.4 MW gas · 0.85 MW heat) so single-carrier failures shed comparable
       load.  Comes with a synthetic timeseries generator.
   * - :func:`~monee.network.res.create_resilient_urban_mes_net`
     - Small
     - Redundancy-rich urban variant: two gas sources, two CHPs, wind and
       solar generation, three heat consumers, G2P backup.
   * - :func:`~monee.network.res.create_industrial_hub_net`
     - Medium
     - Industrial ring grid (meshed).  8 buses · 7 gas junctions.
       Three G2P units and two P2G units with redundant supply paths.
       Good for topology studies and N-1 contingency analysis.
   * - :func:`~monee.network.res.create_regional_mes_net`
     - Medium
     - Regional grid spanning all coupling types.  8 buses ·
       8 gas junctions · 6 heat junctions.  CHP, G2P, P2G, P2H all present,
       making it the broadest of the hand-built grids.
   * - :func:`~monee.network.res.create_large_urban_mes_net`
     - Scalable
     - ``n_districts`` replicated urban districts under a shared HV slack
       and a shared gas feeder.  ~96 nodes at the default 6 districts,
       ~320 at 20.
   * - :func:`~monee.network.mes.create_monee_benchmark_net`
     - Small
     - Seeded 7-bus 120 kV MES with gas and heat overlay, CHP, P2G, two G2P
       units, a grid-forming generator, and two normally-open **backup
       lines** (``backup=True``, ``on_off=0``).
   * - :func:`~monee.network.mes.create_mv_multi_cigre`
     - Medium
     - Seeded MES built on the pandapower CIGRE MV grid (with PV and wind
       DER), overlaid with gas and heat plus P2G and CHP, and one open
       backup line.  Requires the optional ``pandapower`` dependency.
   * - :func:`~monee.network.bench.restoration.create_restoration_benchmark`
     - Large
     - Curated multi-energy restoration benchmark - see
       :ref:`the dedicated section below <how-to/reference_networks:The restoration benchmark>`.

----

Quick start
===========

Load and run an energy-flow calculation on the urban district network:

.. testcode::

    import monee.model as mm
    from monee import run_energy_flow
    from monee.network import create_urban_district_net

    net = create_urban_district_net()
    result = run_energy_flow(net)

    print(result.get(mm.Bus)[["vm_pu"]].shape)
    print(result.get(mm.GasPipe)[["mass_flow"]].shape)

.. testoutput::
   :options: +SKIP

   (5, 1)
   (5, 1)

----

Restoration and reconfiguration variants
=========================================

Normally-open ties
------------------

:func:`~monee.network.res.create_urban_district_net_with_ties` keeps the
primary topology of the urban district but adds **normally-open tie
branches** - switchable alternative paths created with ``on_off=0`` and
named ``tie_*``:

* a power tie ``b3↔b4`` between the two large loads,
* gas ties ``g0↔g2`` (bypass around ``g1``) and ``g3↔g4`` (meshing the two
  sink junctions),
* a heat supply tie ``s1↔s3`` bypassing ``s2``,
* and a second heat consumer, so a single heat-exchanger failure does not
  collapse all heat demand.

A restoration or reconfiguration algorithm can discover the ties through the
``on_off`` attribute of the branch models and close one by setting it to
``1`` before re-solving:

.. testcode::

    from monee.network import create_urban_district_net_with_ties

    net = create_urban_district_net_with_ties()
    ties = [b for b in net.branches if getattr(b.model, "on_off", 1) == 0]
    print(sorted(tie.name for tie in ties))

    ties[0].model.on_off = 1  # close the first tie

.. testoutput::

   ['tie_b3_b4', 'tie_g0_g2', 'tie_g3_g4', 'tie_s1_s3']

Scaling up: replicated districts
--------------------------------

:func:`~monee.network.res.create_large_urban_mes_net` replicates the
urban-district pattern (including its internal ties) ``n_districts`` times
under a **shared HV slack bus and a shared external gas feeder**.  Adjacent
districts are linked by normally-open MV power ties and live gas trunks, so
electricity and gas each remain a single connected component.  The heat
sector is intentionally **local per district** - one supply-return consumer
pair each - and therefore decomposes into one connected component per
district.

.. testcode::

    from monee.network import create_large_urban_mes_net

    net = create_large_urban_mes_net(n_districts=6)   # ~96 nodes
    big = create_large_urban_mes_net(n_districts=20)  # ~320 nodes

Each district contributes 5 power buses, 4 gas junctions, 4 water junctions,
2 heat consumers, 3 coupling points (CHP, P2G, G2P), and 4 internal ties.

Seeded benchmark nets with backup lines
---------------------------------------

Two further fixed MES grids live in :mod:`monee.network.mes`.  Both are
seeded internally, so they are reproducible despite using the randomized
overlay generators under the hood:

* :func:`~monee.network.mes.create_monee_benchmark_net` - a 7-bus 120 kV
  power grid with full gas and heat overlay, a CHP, a P2G, two G2P units,
  and a :class:`~monee.model.GridFormingGenerator` (useful together with
  :doc:`islanding <islanding>`).  Two extra power lines are created as
  **backup assets** with ``backup=True`` and ``on_off=0`` - open in normal
  operation, available to a restoration algorithm.
* :func:`~monee.network.mes.create_mv_multi_cigre` - the pandapower CIGRE MV
  benchmark (with PV and wind DER) converted to monee and overlaid with gas
  and heat grids, a P2G, a CHP, and one open backup line.  Requires
  ``pandapower``.

----

Balanced demands and synthetic timeseries
=========================================

In the other reference networks the carrier demands differ by an order of
magnitude, so a gas outage and a power outage are hard to compare.
:func:`~monee.network.res.create_balanced_urban_mes_net` scales the
shed-able demand per carrier to comparable energy levels:

* **Power**: 0.5 + 0.7 + 0.5 MW of direct loads = 1.7 MW
* **Gas**: 0.015 + 0.010 kg/s of direct sinks ≈ 1.4 MW
* **Heat**: 0.55 + 0.30 MW of heat-exchanger consumers = 0.85 MW

The companion function
:func:`~monee.network.res.create_balanced_urban_mes_timeseries` builds a
ready-to-run :class:`~monee.simulation.TimeseriesData` with synthetic
winter-weekday demand profiles for every direct consumer in the network:

* ``PowerLoad`` children - residential/commercial pattern with morning ramp
  and evening peak; both ``p_mw`` and ``q_mvar`` are scaled together so the
  power factor stays constant,
* gas ``Sink`` children (only those on the gas grid - heat-return sinks are
  skipped) - twin breakfast/dinner spikes on ``mass_flow``,
* ``HeatExchangerLoad`` branches - space-heating pattern on ``q_mw_set``,
  anti-correlated with outdoor temperature.

All profiles are expressed as fractions of the rated setpoints already
stored in the network, and the 24-point reference patterns are resampled by
piecewise-linear interpolation whenever ``n_steps != 24``:

.. testcode::

    from monee.network import (
        create_balanced_urban_mes_net,
        create_balanced_urban_mes_timeseries,
    )

    net = create_balanced_urban_mes_net()
    td = create_balanced_urban_mes_timeseries(net, n_steps=24)
    print(type(td).__name__)

.. testoutput::

   TimeseriesData

Pass the result straight to :func:`~monee.simulation.run_timeseries`:

.. code-block:: python

    from monee.simulation import run_timeseries

    ts_result = run_timeseries(net, td, steps=24)

See :doc:`timeseries` for everything you can do with the result.

Redundancy-rich variant
-----------------------

:func:`~monee.network.res.create_resilient_urban_mes_net` extends the
balanced network with deliberate redundancy, making carrier dependence and
backup paths directly observable: **two gas sources** (an external grid plus
a secondary source, inter-connected so either can serve both CHPs), **two
CHPs** in independent gas sub-trees, **wind and solar** generation that is
independent of the gas system, three heat-exchanger consumers on separate
supply-return pairs, and a **G2P** unit providing gas-backed power.

----

The restoration benchmark
=========================

:func:`~monee.network.bench.restoration.create_restoration_benchmark` (also
re-exported from :mod:`monee.network`) is a curated multi-energy grid built
for restoration-sequence planning, resilience studies, and multi-period
optimisation:

.. code-block:: python

    def create_restoration_benchmark(*, linepack=False, ltc=False, misocp=True)

* **Electricity** - two 110 kV feeders (each with its own external grid)
  joined by an HV tie, four 20 kV substations behind 63 MVA transformers,
  and three five-bus load chains (industrial, commercial, residential) with
  distributed solar and wind generation.
* **Gas** - two high-pressure feeders with a trunk pipe, three compressors
  stepping down into three eight-junction medium-pressure chains plus two
  spur junctions, and fourteen gas sinks.
* **Heat** - a long district-heating supply chain (120 junctions) fed by a
  358 K heat plant, with a common return junction and consumer heat
  exchangers along the chain.
* **Coupling points** - two CHP-HG units (gas → power + heat, node-based
  heat injection), one P2G electrolyser, and one G2P peaker.

The keyword-only flags select optional features:

* ``linepack=True`` attaches the :class:`~monee.model.GasLinepack`
  extension (inherent gas storage in pipes),
* ``ltc=True`` attaches :class:`~monee.model.LumpedThermalCapacitance`
  (thermal inertia at heat junctions),
* ``misocp=True`` (the default) applies
  :data:`~monee.model.formulation.MISOCP_NETWORK_FORMULATION` to the power
  grid, replacing the nonlinear AC formulation.

Both extensions only take effect in multi-period runs - see
:doc:`../concepts/temporal_extensions`.

.. tip::

   With the default ``misocp=True``, pair the benchmark with a
   MIQCP-capable solver such as **Gurobi**.  For the nonlinear variant
   (``misocp=False``), use the :class:`~monee.solver.PyomoSolver` with
   **ipopt** and relaxed tolerances.

.. code-block:: python

    import monee
    from monee.network import create_restoration_benchmark

    # Default: MISOCP formulation already applied - solve with Gurobi.
    net = create_restoration_benchmark()
    result = monee.run_energy_flow(net, solver="gurobi")

    # Nonlinear variant for interior-point solvers.
    from monee.solver.pyo import DEFAULT_SOLVER_OPTIONS

    DEFAULT_SOLVER_OPTIONS["max_iter"] = 5000
    DEFAULT_SOLVER_OPTIONS["tol"] = 1e-4

    net_nl = create_restoration_benchmark(misocp=False)
    result = monee.run_energy_flow(net_nl, solver="ipopt", backend="pyomo")

----

Load shedding on a reference network
=====================================

The regional MES network is a convenient starting point for load-shedding
experiments because it exercises all coupling types simultaneously:

.. code-block:: python

    import monee
    from monee.network import create_regional_mes_net

    net = create_regional_mes_net()
    net.apply_formulation(monee.MISOCP_NETWORK_FORMULATION)

    problem = monee.create_min_load_shedding_problem(
        bounds_el=(0.9, 1.1),
        bounds_heat=(0.9, 1.1),
        bounds_gas=(0.9, 1.1),
        include_ext_grids=True,
    )

    result = monee.run_energy_flow_optimization(
        net,
        problem,
        solver="gurobi",
        exclude_unconnected_nodes=True,
    )

    print(result.dataframes["PowerLoad"][["regulation"]])

----

Import / export capacity limits
================================

Each external grid connection accepts optional capacity bounds that cap
the power or mass-flow that can be exchanged with the upstream network.
These are set at network construction time:

.. testcode::

    import monee.express as mx

    net_cap = mx.create_multi_energy_network()
    bus_cap = mx.create_bus(net_cap)

    # Limit the electrical connection to 4 MW import, 1 MW export.
    mx.create_ext_power_grid(net_cap, bus_cap, max_import_mw=4.0, max_export_mw=1.0)

    # Gas connection: unlimited import, no export allowed.
    junc_cap = mx.create_gas_junction(net_cap)
    mx.create_ext_hydr_grid(net_cap, junc_cap, max_import_kgs=0.1)

The reference networks use these limits to reflect realistic import
capacities for each grid type.

----

See also
========

.. grid:: 1 2 3 3
   :gutter: 3

   .. grid-item-card:: Generate MES overlays
      :link: generate_mes
      :link-type: doc
      :shadow: sm

      Derive gas and heat grids from any power network with the parametric
      generators in :mod:`monee.network.mes`.

   .. grid-item-card:: Load shedding
      :link: load_shedding
      :link-type: doc
      :shadow: sm

      Set up and interpret minimum-load-shedding problems.

   .. grid-item-card:: Pyomo solvers
      :link: use_pyomo_solver
      :link-type: doc
      :shadow: sm

      Install solver binaries and run MISOCP formulations with Gurobi or
      HiGHS.
