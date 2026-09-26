.. _components-grids:

=====
Grids
=====

A grid carries the per-carrier constants and bounds that every component of
that carrier reads: bases, reference pressure and temperature, fluid
properties, operational bands. A network creates a default grid per carrier;
pass your own to :class:`~monee.model.network.Network` when the defaults do
not fit, and pass a grid instance as ``grid=`` to ``net.node`` or the express
node functions for a second grid of the same carrier (a second voltage level,
a second pressure stage). Details: :ref:`concepts/data_model:Grids`, defaults
and conventions: :doc:`../concepts/conventions`.

.. testcode::

   import monee.model as mm

   net = mm.Network(
       el_model=mm.create_power_grid("el", sn_mva=100),
       gas_model=mm.create_gas_grid("gas", type="methane", pressure_ref_pa=5e5),
       water_model=mm.create_water_grid("water", t_ref_k=343),
   )
   hv = net.node(mm.Bus(base_kv=110), grid=mm.EL)
   mv = net.node(mm.Bus(base_kv=20), grid=mm.create_power_grid("mv", sn_mva=10))
   print(net.node_by_id(hv).grid.sn_mva, net.node_by_id(mv).grid.sn_mva)

.. testoutput::

   100 10

.. component: PowerGrid | create_power_grid

.. code-block:: python

   mm.PowerGrid(name, sn_mva=1, vm_pu_max=1.5, vm_pu_min=0.5)
   mm.create_power_grid(name, sn_mva=1)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``name``
     - required
     - Label of the grid.
   * - ``sn_mva``
     - 1
     - Apparent-power base in MVA that scales the flow equations. Imported
       grids usually carry 100.
   * - ``vm_pu_max``, ``vm_pu_min``
     - 1.5, 0.5
     - Voltage band read by the MISOCP branch-flow formulation, the islanding
       extension and the load-shedding problem. Under the default AC
       formulation a bus carries its own 0.5 to 1.5 bounds; tighten them with
       the problem's ``bounds_vm``.

.. component: GasGrid | create_gas_grid

.. code-block:: python

   mm.create_gas_grid(name, type="lgas", t_ref_k=None, pressure_ref_pa=None,
                      pressure_ambient_pa=None)
   mm.GasGrid(name, molar_mass, dynamic_visc_pas, higher_heating_value_kwh_per_kg,
              universal_gas_constant, t_k, t_ref_k, pressure_ref_pa, nominal_pressure_pu,
              max_mass_flow_kgs, pressure_squared_pu_max, pressure_squared_pu_min,
              compressibility=None, pressure_ambient_pa=0.0)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``name``
     - required
     - Label of the grid.
   * - ``type``
     - ``"lgas"``
     - Fluid preset of ``create_gas_grid``: ``"lgas"`` (11.79 kWh/kg) or
       ``"methane"`` (15.3 kWh/kg).
   * - ``t_ref_k``
     - preset 356
     - Reference temperature in K, the base of ``t_pu``.
   * - ``pressure_ref_pa``
     - preset 1e6
     - Reference pressure in Pa, the base of ``pressure_pu``.
   * - ``pressure_ambient_pa``
     - 0.0
     - 0 keeps node pressures absolute; ``mm.STANDARD_ATMOSPHERE_PA`` makes
       them gauge.
   * - ``t_k``
     - preset 300
     - Operating temperature in K, also the default feed temperature of a gas
       slack.
   * - ``molar_mass``
     - preset
     - Molar mass in kg/mol.
   * - ``dynamic_visc_pas``
     - preset
     - Dynamic viscosity in Pa s.
   * - ``higher_heating_value_kwh_per_kg``
     - preset
     - Heating value in kWh/kg used by every gas energy conversion.
   * - ``universal_gas_constant``
     - preset 8.314
     - In J/(mol K).
   * - ``nominal_pressure_pu``
     - preset 1
     - Nominal pressure in pu: linearisation point of the linear and conic gas
       formulations, the pressure pinned on a floating island without a
       grid-forming source, and the linepack reference.
   * - ``max_mass_flow_kgs``
     - preset 20
     - Per-pipe mass-flow cap (the smaller of this and the velocity cap) and
       the bound of a ConsumeHydrGrid's free flow; an ExtHydrGrid is capped by
       its own ``max_import_kgs`` and ``max_export_kgs``.
   * - ``pressure_squared_pu_max``, ``pressure_squared_pu_min``
     - preset 1.3, 0.7
     - Operational band in squared per unit, applied in optimisation runs.
   * - ``compressibility``
     - ``None``
     - Real-gas factor Z; ``None`` derives it from pressure, temperature and
       molar mass.

.. component: WaterGrid | create_water_grid

.. code-block:: python

   mm.WaterGrid(name, t_ref_k=356, pressure_ref_pa=1000000, max_mass_flow_kgs=200, v_max_mps=5.0,
                fluid_density_kg_per_m3=None, dynamic_visc_pas=None, node_heat_reg_kgs=0.0)
   mm.create_water_grid(name, t_ref_k=356, pressure_ref_pa=1000000, node_heat_reg_kgs=0.0)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``name``
     - required
     - Label of the grid.
   * - ``t_ref_k``
     - 356
     - Reference temperature in K, the base of ``t_pu`` and the default feed
       temperature of a water slack.
   * - ``pressure_ref_pa``
     - 1e6
     - Reference pressure in Pa, the base of ``pressure_pu``.
   * - ``max_mass_flow_kgs``
     - 200
     - Per-pipe mass-flow cap (the smaller of this and the velocity cap) and
       the bound of a ConsumeHydrGrid's free flow; an ExtHydrGrid is capped by
       its own ``max_import_kgs`` and ``max_export_kgs``.
   * - ``v_max_mps``
     - 5.0
     - Velocity cap that bounds each pipe's mass flow via its cross section.
   * - ``fluid_density_kg_per_m3``
     - ``None``
     - Density; ``None`` derives it from ``t_ref_k``.
   * - ``dynamic_visc_pas``
     - ``None``
     - Viscosity; ``None`` derives it from ``t_ref_k``.
   * - ``node_heat_reg_kgs``
     - 0.0
     - Conduction-style regulariser of the nodal heat balance.

:class:`~monee.model.grid.NoGrid` (``mm.NO_GRID``) marks components bound to
no carrier.
