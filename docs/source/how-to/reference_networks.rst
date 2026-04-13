====================
Reference networks
====================

monee ships three parametric multi-energy reference networks (RES) in
:mod:`monee.network.res` that are ready to use for testing, benchmarking,
and tutorial examples.  All three include electricity, gas, and heat grids
coupled via standard conversion units (CHP, P2H, P2G, G2P).

----

Available networks
==================

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Function
     - Scale
     - Description
   * - :func:`~monee.network.res.create_urban_district_net`
     - Small
     - 20 kV residential district.  5 buses · 5 gas junctions · 6 heat
       junctions.  CHP-centred topology with high coupling-point density.
       Suitable for load-shedding and demand-side flexibility studies.
   * - :func:`~monee.network.res.create_industrial_hub_net`
     - Medium
     - Industrial ring grid (meshed).  8 buses · 7 gas junctions.
       Three G2P units and two P2G units with redundant supply paths.
       Good for topology studies and N-1 contingency analysis.
   * - :func:`~monee.network.res.create_regional_mes_net`
     - Medium
     - Regional grid spanning all coupling types.  8 buses ·
       8 gas junctions · 5 heat junctions.  CHP, G2P, P2G, P2H all present,
       making it the most comprehensive of the three.

----

Quick start
===========

Load and run an energy-flow calculation on the urban district network:

.. testcode::

    from monee import run_energy_flow
    from monee.network import create_urban_district_net

    net = create_urban_district_net()
    result = run_energy_flow(net)

    print(result.get("Bus")[["vm_pu"]].shape)
    print(result.get("GasPipe")[["mass_flow"]].shape)

.. testoutput::
   :options: +SKIP

   (5, 1)
   (5, 1)

----

Load shedding on a reference network
=====================================

The regional MES network is a convenient starting point for load-shedding
experiments because it exercises all coupling types simultaneously:

.. code-block:: python

    from monee import run_energy_flow_optimization, PyomoSolver
    from monee.network import create_regional_mes_net
    from monee.model.formulation import MISOCP_NETWORK_FORMULATION
    from monee.problem import create_load_shedding_optimization_problem

    net = create_regional_mes_net()
    net.apply_formulation(MISOCP_NETWORK_FORMULATION)

    problem = create_load_shedding_optimization_problem(
        bounds_el=(0.9, 1.1),
        bounds_heat=(0.9, 1.1),
        bounds_gas=(0.9, 1.1),
        use_ext_grid_bounds=False,
    )

    result = run_energy_flow_optimization(
        net,
        problem,
        solver=PyomoSolver(),
        solver_name="gurobi",
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
