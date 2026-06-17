====================
Reference networks
====================

monee ships fixed multi-energy reference networks for testing,
benchmarking, and tutorials.  Import them from :mod:`monee.network`.
Every factory is deterministic (hand-built or internally seeded), so
repeated calls return the same grid.  They couple electricity, gas, and
(in most cases) heat through conversion units (CHP, P2H, P2G, G2P).

To generate a multi-energy overlay on top of an arbitrary power network
instead (gas and heat grids derived from the electrical topology, with
configurable coupling-point density), see :doc:`generate_mes`.

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
     - 20 kV residential district.  5 buses · 5 gas junctions · 4 heat
       junctions.  CHP-centred topology with high coupling-point density.
       Best for load-shedding and demand-side flexibility studies.
   * - :func:`~monee.network.res.create_large_urban_mes_net`
     - Scalable
     - ``n_districts`` replicated urban districts under a shared HV slack
       and a shared gas feeder.  ~96 nodes at the default 6 districts,
       ~320 at 20.
   * - :func:`~monee.network.mes.create_monee_benchmark_net`
     - Small
     - Seeded 7-bus 120 kV MES with gas and heat overlay, CHP, P2G, two G2P
       units, a grid-forming generator, and two normally-open backup
       lines (``backup=True``, ``on_off=0``).
   * - :func:`~monee.network.mes.create_mv_multi_cigre`
     - Medium
     - Seeded MES built on the pandapower CIGRE MV grid (with PV and wind
       DER), overlaid with gas and heat plus P2G and CHP, and one open
       backup line.  Requires the optional ``pandapower`` dependency.
   * - :func:`~monee.network.bench.restoration.create_restoration_benchmark`
     - Large
     - Curated multi-energy restoration benchmark, see
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
    print(result.get(mm.GasPipe)[["mass_flow_kgs"]].shape)

.. testoutput::
   :options: +SKIP

   (5, 1)
   (4, 1)

----

Scaling up: replicated districts
================================

:func:`~monee.network.res.create_large_urban_mes_net` replicates the
urban-district pattern ``n_districts`` times under a shared HV slack bus and a
shared external gas feeder.  Each replicated district adds its own
normally-open internal ties (``on_off=0``, named ``tie_*``) across all three
carriers plus a second heat consumer.  Adjacent districts are linked by
normally-open MV power ties and live gas trunks, so electricity and gas each
remain a single connected component.  The heat sector is intentionally local
per district (one supply-return consumer pair each) and therefore decomposes
into one connected component per district.

.. testcode::

    from monee.network import create_large_urban_mes_net

    net = create_large_urban_mes_net(n_districts=6)   # ~96 nodes
    big = create_large_urban_mes_net(n_districts=20)  # ~320 nodes

Each district contributes 5 power buses, 4 gas junctions, 5 water junctions,
2 heat consumers, 3 coupling points (CHP, P2G, G2P), and 3 internal ties.

Seeded benchmark nets with backup lines
---------------------------------------

Two further fixed MES grids live in :mod:`monee.network.mes`.  Both are
seeded internally, so they are reproducible despite using the randomized
overlay generators under the hood:

* :func:`~monee.network.mes.create_monee_benchmark_net`: a 7-bus 120 kV
  power grid with full gas and heat overlay, a CHP, a P2G, two G2P units,
  and a :class:`~monee.model.GridFormingGenerator` (useful together with
  :doc:`islanding <islanding>`).  Two extra power lines are created as
  backup assets with ``backup=True`` and ``on_off=0``: open in normal
  operation, available to a restoration algorithm.
* :func:`~monee.network.mes.create_mv_multi_cigre`: the pandapower CIGRE MV
  benchmark (with PV and wind DER) converted to monee and overlaid with gas
  and heat grids, a P2G, a CHP, and one open backup line.  Requires
  ``pandapower``.

----

The restoration benchmark
=========================

:func:`~monee.network.bench.restoration.create_restoration_benchmark` (also
re-exported from :mod:`monee.network`) is a curated multi-energy grid built
for restoration-sequence planning, resilience studies, and multi-period
optimisation:

.. code-block:: python

    def create_restoration_benchmark(*, linepack=False, ltc=False, misocp=True)

* **Electricity:** two 110 kV feeders (each with its own external grid)
  joined by an HV tie, four 20 kV substations behind 63 MVA transformers,
  and three five-bus load chains (industrial, commercial, residential) with
  distributed solar and wind generation.
* **Gas:** two high-pressure feeders with a trunk pipe, three compressors
  feeding three eight-junction medium-pressure chains plus two spur
  junctions, and fourteen gas sinks.
* **Heat:** a long district-heating supply chain (120 junctions) fed by a
  358 K heat plant, with a common return junction and consumer heat
  exchangers along the chain.
* **Coupling points:** two CHP-HG units (gas → power + heat, node-based
  heat injection), one P2G electrolyser, and one G2P peaker.

The keyword-only flags select optional features:

* ``linepack=True`` attaches the :class:`~monee.model.GasLinepack`
  extension (inherent gas storage in pipes),
* ``ltc=True`` attaches :class:`~monee.model.LumpedThermalCapacitance`
  (thermal inertia at heat junctions),
* ``misocp=True`` (the default) applies
  :data:`~monee.model.formulation.EL_MISOCP_FORMULATION` to the power
  grid, replacing the nonlinear AC formulation.

Both extensions only take effect in multi-period runs, see
:doc:`../concepts/temporal_extensions`.

.. tip::

   With the default ``misocp=True``, pair the benchmark with a
   MIQCP-capable solver such as Gurobi.  For the nonlinear variant
   (``misocp=False``), use the :class:`~monee.solver.PyomoSolver` with
   ipopt and relaxed tolerances.

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

The urban district network is a good starting point for load-shedding
experiments: it couples power, gas, and heat through a CHP, a P2G, and a G2P.

.. code-block:: python

    import monee
    from monee.network import create_urban_district_net

    net = create_urban_district_net()
    net.apply_formulation(monee.EL_MISOCP_FORMULATION)

    problem = monee.create_min_load_shedding_problem(
        bounds_vm=(0.9, 1.1),
        bounds_t=(0.9, 1.1),
        bounds_pressure=(0.9, 1.1),
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

    # Gas connection: import capped at 0.1 kg/s, export unbounded.
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
