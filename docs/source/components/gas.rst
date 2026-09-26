.. _components-gas:

===
Gas
===

Physics: the Weymouth pipe flow and friction model sections of
:doc:`../concepts/domains`. Grid defaults and the pressure convention:
:ref:`concepts/conventions:Gas`.

Junction
--------

Hydraulic node for gas and for water. Holds pressure and temperature in per
unit of the grid references and sums the child mass flows. The gas
formulations solve ``pressure_squared_pu``; the water formulations solve
``pressure_pu``. Temperature is solved by the nodal heat balance on water
only. Gas pipes carry no temperature, so a gas junction keeps its initial
1 pu (356 K on the default grid) unless a slack pins its own ``t_k``, and the
gas physics uses the grid ``t_k`` instead.

.. component: Junction | create_gas_junction, create_water_junction, create_junction

.. code-block:: python

   mm.Junction()
   mx.create_gas_junction(network, *, grid=mm.GAS, constraints=None, overwrite_id=None,
                          name=None, position=None)
   mx.create_water_junction(network, *, grid=mm.WATER, constraints=None, overwrite_id=None,
                            name=None, position=None)
   mx.create_junction(network, grid, *, constraints=None, overwrite_id=None, name=None,
                      position=None)

The class takes no parameters. The carrier is the grid the node is added to:
``mm.GAS`` or ``mm.WATER`` as the ``grid`` argument of ``net.node`` selects the
network's default grid of that carrier, a grid instance from
:ref:`Grids <components/grids:Grids>` selects that one.

.. results: Junction

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``pressure_pu``
     - solved
     - Pressure in per unit of the grid ``pressure_ref_pa``; 1.0 at a slack with default settings, lower downstream of it.
   * - ``pressure_squared_pu``
     - solved
     - Squared pressure, the state the gas formulations solve; equals ``pressure_pu ** 2`` on gas junctions. On water junctions it is an unused placeholder that stays at 1.0; read ``pressure_pu``.
   * - ``t_pu``
     - solved
     - Temperature in per unit of the grid ``t_ref_k``. Solved on water; on gas it is not modelled and stays at its initial value except where a slack pins it, and an optimisation leaves it a free, meaningless variable.
   * - ``t_k``
     - report
     - Temperature in K, ``t_pu * t_ref_k``.
   * - ``mass_flow_kgs``
     - solved
     - Net child withdrawal at the junction in kg/s, load convention: sinks positive, sources and the slack negative. Branch flows are not part of it.

After a solve with the default formulation the table also carries
``pressure_pa``, the absolute pressure in Pa.

GasPipe
-------

Gas pipeline between two junctions. Flow is split into two non-negative
halves and reported as ``mass_flow_kgs = mass_flow_pos_kgs -
mass_flow_neg_kgs``, negative for flow from the from-node to the to-node.

.. component: GasPipe | create_gas_pipe

.. code-block:: python

   mm.GasPipe(diameter_m, length_m, temperature_ext_k=296.15, roughness_m=0.0001, on_off=1,
              friction=None)
   mx.create_gas_pipe(network, from_node_id, to_node_id, diameter_m, length_m,
                      temperature_ext_k=296.15, roughness_m=1e-05, on_off=1,
                      constraints=None, grid=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``diameter_m``
     - required
     - Inner diameter in m.
   * - ``length_m``
     - required
     - Pipe length in m.
   * - ``temperature_ext_k``
     - 296.15
     - Stored and echoed only; no built-in gas formulation reads it. The gas
       physics uses the grid's ``t_k``, 300 K on the default grid.
   * - ``roughness_m``
     - 1e-4, express 1e-5
     - Absolute wall roughness in m for the friction correlation. Note the
       two APIs apply different defaults.
   * - ``on_off``
     - 1
     - In service flag.
   * - ``friction``
     - ``None``
     - Class only. Accepted but not read by any built-in formulation; the
       friction factor comes from the formulation's ``friction_model``, by
       default a constant fully rough value from ``diameter_m`` and
       ``roughness_m``.

.. results: GasPipe

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - solved
     - Signed flow in kg/s: negative for flow from the from-node to the to-node. A pipe carrying 0.2 kg/s from the slack to a sink reports -0.2.
   * - ``mass_flow_pos_kgs``
     - solved
     - Flow from the to-node to the from-node in kg/s, never negative.
   * - ``mass_flow_neg_kgs``
     - solved
     - Flow from the from-node to the to-node in kg/s, never negative.
   * - ``velocity_mps``
     - report
     - Mean velocity in m/s with the sign of the flow, ``mass_flow_kgs / (rho * A)`` with the grid's reference density. Optimisation solves cap the mass flow at the grid ``v_max_mps``; a simulation does not.
   * - ``gas_density_kg_per_m3``
     - solved
     - Gas density along the pipe.
   * - ``reynolds_scaled``
     - internal
     - Reynolds number divided by 1e6 when the formulation uses ``friction_model="nonlinear"``; 0 under the default constant friction model.
   * - ``friction``
     - derived
     - Darcy friction factor set by the formulation: the fully rough asymptote from diameter and roughness under the default constant model, a solved variable only with ``friction_model="nonlinear"``.
   * - ``direction``
     - internal
     - Direction binary of the MIQCQP formulations; pinned at 1 and meaningless under the default smooth formulation, read the sign of ``mass_flow_kgs`` instead.
   * - ``mass_flow_pos_kgs_squared``
     - internal
     - Squared flow half used by the Weymouth equation.
   * - ``mass_flow_neg_kgs_squared``
     - internal
     - Squared flow half used by the Weymouth equation.
   * - ``q_mw``
     - internal
     - Unused on a gas pipe, always 0.
   * - ``diameter_m``
     - input
     - Inner diameter in m.
   * - ``length_m``
     - input
     - Length in m.
   * - ``temperature_ext_k``
     - input
     - Stored only; the gas physics uses the grid ``t_k``.
   * - ``roughness_m``
     - input
     - Wall roughness in m.

After a solve the table also carries ``mass_flow_mag_kgs``, the flow
magnitude, and with :ref:`GasLinepack <components/extensions:Network aspects>`
registered ``linepack_kg`` and ``net_pack_kgs``.

GasCompressor
-------------

Ideal compressor with a fixed pressure ratio, ``p_to = compression_ratio *
p_from``, passing flow only from the from-node to the to-node.

.. component: GasCompressor | create_compressor

.. code-block:: python

   mm.GasCompressor(compression_ratio=1.5, max_flow_kgs=10.0)
   mx.create_compressor(network, from_node_id, to_node_id, compression_ratio=1.5,
                        max_flow_kgs=10.0, constraints=None, grid=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``compression_ratio``
     - 1.5
     - Discharge over suction pressure.
   * - ``max_flow_kgs``
     - 10.0
     - Upper bound of the throughput in kg/s.

.. results: GasCompressor

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - solved
     - Throughput in kg/s, negative because flow always runs from the from-node to the to-node.
   * - ``mass_flow_neg_kgs``
     - solved
     - The throughput magnitude, bounded by ``max_flow_kgs``.
   * - ``compression_ratio``
     - input
     - Discharge over suction pressure; the to-node pressure is this times the from-node pressure.
   * - ``max_flow_kgs``
     - input
     - Throughput bound in kg/s.

ExtHydrGrid
-----------

The hydraulic slack, shared by gas and water: pins the junction pressure and,
by default, its temperature, and exchanges whatever mass flow the balance
needs. Negative ``mass_flow_kgs`` is injection into the network. On a heat
network it acts as an unlimited backup plant unless capped
(:ref:`concepts/multi_energy:The heat slack is a backup plant`). The result
table name is ``ExtHydrGrid`` for both carriers.

.. component: ExtHydrGrid | create_gas_ext_grid, create_water_ext_grid, create_ext_hydr_grid

.. code-block:: python

   mm.ExtHydrGrid(mass_flow_kgs=-1, pressure_pu=1, t_k=None, max_import_kgs=None,
                  max_export_kgs=None, pin_temperature=True, free_pressure_bounds=None,
                  cost=None)
   mx.create_gas_ext_grid(network, node_id, mass_flow_kgs=1, pressure_pu=1, t_k=None,
                          max_import_kgs=None, max_export_kgs=None, pin_temperature=True,
                          free_pressure_bounds=None)
   mx.create_water_ext_grid(network, node_id, mass_flow_kgs=1, pressure_pu=1, t_k=None,
                            max_import_kgs=None, max_export_kgs=None, pin_temperature=True,
                            free_pressure_bounds=None)
   mx.create_ext_hydr_grid(network, node_id, mass_flow_kgs=1, pressure_pu=1, t_k=None,
                           grid_key=mm.GAS_KEY, max_import_kgs=None, max_export_kgs=None,
                           pin_temperature=True, free_pressure_bounds=None,
                           constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``mass_flow_kgs``
     - -1, express 1
     - Initial guess for the free exchange variable in kg/s, not a setpoint.
   * - ``pressure_pu``
     - 1
     - Pinned pressure in per unit of the grid ``pressure_ref_pa`` (10 bar
       for the default gas and water grids).
   * - ``t_k``
     - ``None``
     - Feed temperature in K. ``None`` resolves at solve time to the grid's
       operating temperature: 300 K on the default gas grid, 356 K on the
       default water grid.
   * - ``max_import_kgs``
     - ``None``
     - Import cap as a positive magnitude; bounds ``mass_flow_kgs`` from below
       at ``-max_import_kgs``. Re-read at every solve.
   * - ``max_export_kgs``
     - ``None``
     - Export cap as a positive magnitude; bounds ``mass_flow_kgs`` from above.
   * - ``pin_temperature``
     - ``True``
     - Pin the junction temperature to ``t_k``. ``False`` leaves it to the
       network.
   * - ``free_pressure_bounds``
     - ``None``
     - ``(lo, hi)`` in pu replaces the pressure pin with a bounded variable.
   * - ``cost``
     - ``None``
     - Price for :doc:`../problems/economic_dispatch`: per MW of higher
       heating value on a gas grid, per MW of heat supplied to its island on
       a water grid. The express functions forward it. The result table gains
       a ``cost`` column only when it is set.
   * - ``grid_key``
     - ``mm.GAS_KEY``
     - Generic express function only: carrier of the auto-created junction.

.. results: ExtHydrGrid

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - solved
     - Exchange in kg/s, load convention: a network drawing 0.2 kg/s reports -0.2, and a positive value means the network pushes gas or water out.
   * - ``pressure_pu``
     - input
     - Pinned pressure in pu.
   * - ``t_k``
     - input
     - Feed temperature in K; a ``None`` argument shows up resolved when the temperature is pinned, 300 on the default gas grid and 356 on the default water grid, and stays ``None`` with ``pin_temperature=False``.
   * - ``max_import_kgs``
     - input
     - Import cap, ``None`` when unbounded.
   * - ``max_export_kgs``
     - input
     - Export cap, ``None`` when unbounded.
   * - ``pin_temperature``
     - input
     - Whether the junction temperature is pinned.
   * - ``free_pressure_bounds``
     - input
     - The pressure band when the pin is replaced, otherwise ``None``.

Source
------

Fixed mass-flow injection at a junction, for gas and for water. Takes a
positive magnitude and stores the generation sign.

.. component: Source | create_gas_source, create_water_source, create_source

.. code-block:: python

   mm.Source(mass_flow_kgs, t_k=None, cost=None)
   mx.create_gas_source(network, node_id, mass_flow_kgs=1, t_k=None)
   mx.create_water_source(network, node_id, mass_flow_kgs=1, t_k=None)
   mx.create_source(network, node_id, mass_flow_kgs=1, grid_key=mm.GAS_KEY, t_k=None,
                    constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``mass_flow_kgs``
     - required, express 1
     - Injected mass flow in kg/s as a positive magnitude.
   * - ``t_k``
     - ``None``
     - Temperature of the injected stream in K. ``None`` credits it at the
       junction's own mixed temperature.
   * - ``cost``
     - ``None``
     - Price per MW of higher heating value for
       :doc:`../problems/economic_dispatch`; only a gas source is priced. The
       express functions forward it, and the result table gains a ``cost``
       column only when it is set.
   * - ``grid_key``
     - ``mm.GAS_KEY``
     - Generic express function only: carrier of the auto-created junction.

.. results: Source

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - input
     - Injected setpoint in kg/s with the generation sign: a 0.2 kg/s source reports -0.2. The junction sees ``regulation`` times it.
   * - ``injection_t_k``
     - input
     - Temperature of the injected stream in K, ``None`` when it takes the junction temperature.

Sink
----

Fixed mass-flow withdrawal at a junction, for gas and for water.

.. component: Sink | create_gas_sink, create_water_sink, create_sink

.. code-block:: python

   mm.Sink(mass_flow_kgs)
   mx.create_gas_sink(network, node_id, mass_flow_kgs=1)
   mx.create_water_sink(network, node_id, mass_flow_kgs=1)
   mx.create_sink(network, node_id, mass_flow_kgs=1, grid_key=mm.GAS_KEY, constraints=None,
                  overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``mass_flow_kgs``
     - required, express 1
     - Withdrawn mass flow in kg/s, positive.
   * - ``grid_key``
     - ``mm.GAS_KEY``
     - Generic express function only: carrier of the auto-created junction.

.. results: Sink

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - input
     - Withdrawn setpoint in kg/s, positive. The junction sees ``regulation`` times it, which load shedding lowers.

GasStorage
----------

Pressurised gas store at a junction with a tracked mass. The state update
integrates ``mass_flow_kgs`` over ``dt_h * 3600`` seconds; positive dispatch
is charging. Attach it with ``mx.create_gas_child`` or ``net.child``.

.. component: GasStorage | create_gas_child

.. code-block:: python

   mm.GasStorage(m_stored_kg_initial, m_stored_kg_max, flow_max_kgs, mass_flow_initial_kgs=0.0,
                 efficiency_charge=1.0, efficiency_discharge=1.0, regulation=1)
   mx.create_gas_child(network, model, node_id, constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``m_stored_kg_initial``
     - required
     - Stored mass at the first step in kg.
   * - ``m_stored_kg_max``
     - required
     - Capacity in kg; upper bound of ``m_stored_kg``.
   * - ``flow_max_kgs``
     - required
     - Symmetric charge and discharge limit in kg/s, enforced once the
       dispatch is controllable.
   * - ``mass_flow_initial_kgs``
     - 0.0
     - Dispatch in kg/s while the storage is not controllable.
   * - ``efficiency_charge``
     - 1.0
     - Charging efficiency.
   * - ``efficiency_discharge``
     - 1.0
     - Discharging efficiency.
   * - ``regulation``
     - 1
     - Scales the dispatch seen by the junction and by the state update.
   * - ``model``
     - required
     - Express only: the child model instance to attach.

.. results: GasStorage

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - input
     - Dispatch in kg/s, positive charging. Solved once the storage is controllable.
   * - ``m_stored_kg``
     - solved
     - Stored mass in kg at the end of the step. Meaningful in a timeseries or multi-period run; a single solve leaves it a free variable, so ignore the value there.
   * - ``efficiency_charge``
     - input
     - Charging efficiency.
   * - ``efficiency_discharge``
     - input
     - Discharging efficiency.
