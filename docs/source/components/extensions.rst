.. _components-islanding:

=============================
Islanding and network aspects
=============================

Grid-forming units and the islanding modes are the components of
:doc:`../concepts/islanding`; :doc:`../how-to/islanding` walks through a solve.
Network aspects (:doc:`../concepts/network_aspects`) are registered on the
network with ``net.add_extension(...)`` and have no express function.

GridFormingGenerator
--------------------

Island slack for electricity: free ``p_mw`` and ``q_mvar`` within symmetric
capacities and a pinned voltage while it leads an island. Requires
``monee.enable_islanding(net, electricity=True)``. While an external grid
leads its component the unit holds ``nominal_p_mw`` and ``nominal_q_mvar``.
Both capacities stay writable on the model.

.. component: GridFormingGenerator | create_grid_forming_generator

.. code-block:: python

   mm.GridFormingGenerator(p_mw_max, q_mvar_max, vm_pu=1.0, nominal_p_mw=0.0, nominal_q_mvar=0.0)
   mx.create_grid_forming_generator(network, node_id, p_mw_max, q_mvar_max, vm_pu=1.0,
                                    nominal_p_mw=0.0, nominal_q_mvar=0.0, constraints=None,
                                    overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``p_mw_max``
     - required
     - Active capacity in MW, applied as ``-p_mw_max <= p_mw <= p_mw_max``.
   * - ``q_mvar_max``
     - required
     - Reactive capacity in Mvar, applied symmetrically.
   * - ``vm_pu``
     - 1.0
     - Voltage magnitude pinned while leading an island.
   * - ``nominal_p_mw``
     - 0.0
     - Injection held while not leading; ``None`` leaves it free.
   * - ``nominal_q_mvar``
     - 0.0
     - Reactive injection held while not leading; ``None`` leaves it free.

.. results: GridFormingGenerator

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - solved
     - Injection in MW in load convention, so a unit supplying its island 2 MW reports -2.0. Held at ``nominal_p_mw`` while not leading.
   * - ``q_mvar``
     - solved
     - Reactive injection in Mvar, same convention.

GridFormingSource
-----------------

Island reference for gas and water: pins pressure and temperature while it
leads an island and exchanges a free mass flow within ``mass_flow_max_kgs``.
Requires ``enable_islanding`` for the carrier.

.. component: GridFormingSource | create_grid_forming_source, create_gas_grid_forming_source, create_water_grid_forming_source

.. code-block:: python

   mm.GridFormingSource(pressure_pu=1.0, t_k=356.0, mass_flow_max_kgs=1000000.0,
                        nominal_mass_flow_kgs=None)
   mx.create_grid_forming_source(network, node_id, pressure_pu=1.0, t_k=356.0,
                                 mass_flow_max_kgs=1000000.0, grid_key=mm.GAS_KEY,
                                 constraints=None, overwrite_id=None, name=None)
   mx.create_gas_grid_forming_source(network, node_id, pressure_pu=1.0, t_k=356.0,
                                     mass_flow_max_kgs=1000000.0, **kwargs)
   mx.create_water_grid_forming_source(network, node_id, pressure_pu=1.0, t_k=356.0,
                                       mass_flow_max_kgs=1000000.0, **kwargs)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``pressure_pu``
     - 1.0
     - Pressure pinned while leading, in pu.
   * - ``t_k``
     - 356.0
     - Temperature pinned while leading, in K.
   * - ``mass_flow_max_kgs``
     - 1e6
     - Symmetric bound of the exchange in kg/s.
   * - ``nominal_mass_flow_kgs``
     - ``None``
     - Class only: flow held while not leading; ``None`` leaves it free.
   * - ``grid_key``
     - ``mm.GAS_KEY``
     - Generic express function only: carrier of the auto-created junction.

.. results: GridFormingSource

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - solved
     - Exchange in kg/s in load convention, negative for injection into the island. Held at ``nominal_mass_flow_kgs`` while not leading, when that was given.

Islanding modes
---------------

:class:`~monee.model.extension.islanding.el.ElectricityIslandingMode`,
:class:`~monee.model.extension.islanding.gas.GasIslandingMode` and
:class:`~monee.model.extension.islanding.water.WaterIslandingMode` are the
per-carrier aspects that ``enable_islanding`` registers; pass an instance
instead of ``True`` to tune ``big_m_conn`` on any carrier and, on the
electricity mode, ``angle_bound`` and ``branch_gating``. They add the
energisation binary ``e_el``, ``e_gas`` or ``e_water`` to every node table of
their carrier, ``c_src_*`` to grid-forming nodes and the connectivity flows
``c_*_fwd`` and ``c_*_rev`` to every branch table of the carrier; gated
electricity branches also get ``on_off`` promoted to a decision.

Network aspects
---------------

:class:`~monee.model.extension.linepack.GasLinepack` gives every gas pipe a
stored mass (``linepack_kg``, ``net_pack_kgs``) that couples consecutive
steps of a timeseries or multi-period solve;
:class:`~monee.model.extension.ltc.LumpedThermalCapacitance` gives water
junctions thermal inertia. In a single-step solve GasLinepack still reports
``linepack_kg`` (the stored mass at the solved pressure) with
``net_pack_kgs`` pinned to 0; the temporal coupling activates only in a
timeseries. LumpedThermalCapacitance adds no equations in a single step.
:doc:`../concepts/temporal_extensions` has the physics and worked examples.

.. code-block:: python

   net.add_extension(mm.GasLinepack(overrides=None))
   net.add_extension(mm.LumpedThermalCapacitance(default_t_init=None, t_init_overrides=None,
                                                 first_step_steady_state=False))
