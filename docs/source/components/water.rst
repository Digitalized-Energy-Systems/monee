.. _components-water:

==============
Water and heat
==============

Physics: the Darcy-Weisbach hydraulic flow and temperature propagation
sections of :doc:`../concepts/domains`. The three heat attribute
names ``q_mw``, ``q_mw_heat`` and ``heat_mw`` are distinct on purpose:
:ref:`concepts/conventions:Water and heat`. Supply and return wiring of a
district heating loop: :ref:`concepts/multi_energy:District heating loops`.
Junctions, the slack, sources and sinks are the shared hydraulic components
described under :ref:`Gas <components-gas>`; add them to the water grid with
the ``create_water_*`` functions or ``grid=mm.WATER``.

WaterPipe
---------

Insulated pipe of a district heating network. Carries mass flow, a pressure
drop and the temperature at both ends; heat is lost to the surroundings
through the insulation.

.. component: WaterPipe | create_water_pipe

.. code-block:: python

   mm.WaterPipe(diameter_m, length_m, temperature_ext_k=283.15, roughness_m=4.5e-05,
                lambda_insulation_w_per_m_k=0.025, insulation_thickness_m=0.12, on_off=1,
                friction=None, unidirectional=False)
   mx.create_water_pipe(network, from_node_id, to_node_id, diameter_m, length_m,
                        temperature_ext_k=296.15, roughness_m=0.001,
                        lambda_insulation_w_per_m_k=0.025, insulation_thickness_m=0.2,
                        on_off=1, unidirectional=False, constraints=None, grid=None,
                        name=None)

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
     - 283.15, express 296.15
     - Ambient temperature in K the pipe loses heat to.
   * - ``roughness_m``
     - 4.5e-5, express 1e-3
     - Absolute wall roughness in m. Note the two APIs apply different
       defaults.
   * - ``lambda_insulation_w_per_m_k``
     - 0.025
     - Thermal conductivity of the insulation in W/(m K).
   * - ``insulation_thickness_m``
     - 0.12, express 0.2
     - Insulation thickness in m. Together with the conductivity it sets the
       heat transfer coefficient of the pipe.
   * - ``on_off``
     - 1
     - In service flag.
   * - ``friction``
     - ``None``
     - Class only. Accepted but not read by any built-in formulation; the
       friction factor comes from the formulation's ``friction_model``, by
       default a constant fully rough value from ``diameter_m`` and
       ``roughness_m``.
   * - ``unidirectional``
     - ``False``
     - ``True`` forbids reverse flow in optimisation solves: ``mass_flow_kgs
       <= 0`` under the smooth NLP, ``direction == 0`` under the bilinear
       MIQCQP, where it removes the binary. A plain energy-flow simulation
       does not enforce it.

.. results: WaterPipe

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - solved
     - Signed flow in kg/s, negative for flow from the from-node to the to-node.
   * - ``mass_flow_pos_kgs``
     - solved
     - Flow from the to-node to the from-node in kg/s, never negative.
   * - ``mass_flow_neg_kgs``
     - solved
     - Flow from the from-node to the to-node in kg/s, never negative.
   * - ``velocity_mps``
     - report
     - Mean velocity in m/s with the sign of the flow, ``mass_flow_kgs / (rho * A)``. Optimisation solves cap the mass flow at the grid ``v_max_mps``; a simulation does not.
   * - ``t_from_pu``
     - solved
     - Water temperature at the from-end in per unit of ``t_ref_k``.
   * - ``t_to_pu``
     - solved
     - Water temperature at the to-end. Along the flow direction it is lower than the inlet temperature by the insulation loss, a fraction of a percent on a short well-insulated pipe.
   * - ``friction``
     - derived
     - Darcy friction factor set by the formulation: the fully rough asymptote from diameter and roughness under the default constant model, a solved variable only with ``friction_model="nonlinear"``.
   * - ``reynolds_scaled``
     - internal
     - Reynolds number divided by 1e6 when the formulation uses ``friction_model="nonlinear"``; 0 under the default constant friction model.
   * - ``direction``
     - internal
     - Direction binary of the MIQCQP formulations; pinned at 1 and meaningless under the default smooth formulation, read the sign of ``mass_flow_kgs`` instead.
   * - ``mass_flow_pos_kgs_squared``
     - internal
     - Squared flow half used by the pressure-drop equation.
   * - ``mass_flow_neg_kgs_squared``
     - internal
     - Squared flow half used by the pressure-drop equation.
   * - ``q_mw``
     - internal
     - Unused by the default formulation, which expresses the insulation loss through ``alpha``; a simulation pins it at its 1e-6 seed, an optimisation leaves it a free, meaningless variable.
   * - ``diameter_m``
     - input
     - Inner diameter in m.
   * - ``length_m``
     - input
     - Length in m.
   * - ``temperature_ext_k``
     - input
     - Ambient temperature in K.
   * - ``roughness_m``
     - input
     - Wall roughness in m.
   * - ``lambda_insulation_w_per_m_k``
     - input
     - Insulation conductivity.
   * - ``insulation_thickness_m``
     - input
     - Insulation thickness in m.
   * - ``unidirectional``
     - input
     - Whether reverse flow is forbidden.

After a solve with the default formulation the table also carries
``t_in_pu`` and ``t_out_pu`` (inlet and outlet temperature in flow
direction), ``mass_flow_mag_kgs`` and ``alpha``, the attenuation factor with
``t_out = t_ambient + alpha * (t_in - t_ambient)``.

HeatExchanger
-------------

Active heat exchanger between a supply and a return junction that drives its
own design mass flow, ``|q_mw| * 1e6 / (c_p * 30 K)`` unless
``mass_flow_design_kgs`` is given. Water enters at the from-node and leaves
at the to-node. Two of them on one closed loop over-determine it; use the
passive variant for in-line consumers.

The base class :class:`~monee.model.branch.HeatExchanger` takes the raw
setpoint where a negative ``q_mw`` builds a load. Prefer the aliases, which
take magnitudes: :class:`~monee.model.branch.HeatExchangerLoad` consumes and
:class:`~monee.model.branch.HeatExchangerGenerator` injects. The express
function picks the alias from the sign: ``q_mw > 0`` is a load, ``q_mw < 0``
a generator.

.. component: HeatExchangerLoad, HeatExchangerGenerator, HeatExchanger | create_heat_exchanger

.. code-block:: python

   mm.HeatExchangerLoad(q_mw, mass_flow_design_kgs=None, regulation=1)
   mm.HeatExchangerGenerator(q_mw, mass_flow_design_kgs=None, regulation=1, cost=None)
   mm.HeatExchanger(q_mw, mass_flow_design_kgs=None, T_delta_design_K=30, regulation=1)
   mx.create_heat_exchanger(network, from_node_id, to_node_id, q_mw, regulation=1,
                            mass_flow_design_kgs=None, constraints=None, grid=None, name=None,
                            cost=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``q_mw``
     - required
     - Heat duty in MW. Aliases take the magnitude; the base class and the
       express function read the sign (express: positive consumes).
   * - ``mass_flow_design_kgs``
     - ``None``
     - Driven mass flow in kg/s. ``None`` derives it from the duty at the
       design temperature spread.
   * - ``T_delta_design_K``
     - 30
     - Base class only: design temperature spread in K used to derive the
       flow.
   * - ``regulation``
     - 1
     - Scales duty and mass flow together, so 0 is hydraulically an absent
       branch.
   * - ``cost``
     - ``None``
     - Generator only: price per MW heat for
       :doc:`../problems/economic_dispatch`. The express function rejects it
       for a consuming exchanger. The result table gains a ``cost`` column
       only when it is set.

.. results: HeatExchangerLoad, HeatExchangerGenerator

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw``
     - solved
     - Heat transferred in MW, positive for a load and negative for a generator: a 0.5 MW consumer reports 0.5, a 0.5 MW plant -0.5.
   * - ``q_mw_set``
     - input
     - Stored setpoint with the same sign as ``q_mw``; ``q_mw`` is ``regulation`` times it.
   * - ``mass_flow_design_kgs``
     - derived
     - The driven mass flow in kg/s, ``|q_mw| * 1e6 / (c_p * 30 K)`` unless given: about 3.98 for 0.5 MW.
   * - ``mass_flow_kgs``
     - solved
     - Signed flow in kg/s, negative for flow from the from-node to the to-node, so a consumer between supply and return reports ``-mass_flow_design_kgs``.
   * - ``mass_flow_pos_kgs``
     - solved
     - Reverse flow half in kg/s, 0 for an exchanger.
   * - ``mass_flow_neg_kgs``
     - solved
     - Forward flow half in kg/s, the driven magnitude.
   * - ``t_from_pu``
     - solved
     - Inlet temperature in per unit of ``t_ref_k``.
   * - ``t_to_pu``
     - solved
     - Outlet temperature; for a load the design spread below the inlet, 30 K at full duty, for a generator above it.
   * - ``direction``
     - internal
     - Direction indicator, 0 for the forward flow.

The table is named after the alias, ``HeatExchangerLoad`` or
``HeatExchangerGenerator``. After a solve it also carries
``q_mw_delivered``, ``t_in_pu``, ``t_out_pu`` and ``mass_flow_mag_kgs``.

PassiveHeatExchanger
--------------------

In-line heat exchanger whose mass flow follows the surrounding hydraulics;
the fixed duty then sets the outlet temperature. Hydraulically it is a short
pipe, or a zero-length minor loss when ``loss_coefficient`` is given. The
aliases :class:`~monee.model.branch.PassiveHeatExchangerLoad` and
:class:`~monee.model.branch.PassiveHeatExchangerGenerator` take magnitudes;
the express function picks the alias from the sign of ``q_mw``.

.. component: PassiveHeatExchangerLoad, PassiveHeatExchangerGenerator, PassiveHeatExchanger | create_passive_heat_exchanger

.. code-block:: python

   mm.PassiveHeatExchangerLoad(q_mw, diameter_m, temperature_ext_k=293, loss_coefficient=None)
   mm.PassiveHeatExchangerGenerator(q_mw, diameter_m, temperature_ext_k=293, loss_coefficient=None,
                                    cost=None)
   mm.PassiveHeatExchanger(q_mw, diameter_m, roughness_m=0.0001, length_m=2.5,
                           temperature_ext_k=293, regulation=1, friction=None,
                           loss_coefficient=None)
   mx.create_passive_heat_exchanger(network, from_node_id, to_node_id, q_mw, diameter_m,
                                    temperature_ext_k=293, constraints=None, grid=None,
                                    name=None, cost=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``q_mw``
     - required
     - Heat duty in MW; aliases take the magnitude, express reads the sign.
   * - ``diameter_m``
     - required
     - Flow diameter in m for the pressure drop.
   * - ``temperature_ext_k``
     - 293
     - Stored only; the built-in passive formulations model no loss to the
       surroundings and do not read it.
   * - ``loss_coefficient``
     - ``None``
     - Minor-loss coefficient zeta. Given, the pressure drop becomes ``zeta *
       rho / 2 * v ** 2`` and ``length_m`` and ``friction`` are ignored; 0 is
       a lossless heat injector.
   * - ``cost``
     - ``None``
     - Generator only: price per MW heat for
       :doc:`../problems/economic_dispatch`. The express function rejects it
       for a consuming exchanger. The result table gains a ``cost`` column
       only when it is set.
   * - ``roughness_m``
     - 1e-4
     - Base class only: wall roughness in m.
   * - ``length_m``
     - 2.5
     - Base class only: equivalent length in m for the friction drop.
   * - ``regulation``
     - 1
     - Base class only: scales the duty.
   * - ``friction``
     - ``None``
     - Base class only. Accepted but not read by any built-in formulation;
       the friction factor comes from the formulation's ``friction_model``.

.. results: PassiveHeatExchangerLoad, PassiveHeatExchangerGenerator

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw``
     - solved
     - Heat transferred in MW, positive for a load and negative for a generator.
   * - ``q_mw_set``
     - input
     - Stored setpoint with the same sign; ``q_mw`` is ``regulation`` times it.
   * - ``mass_flow_kgs``
     - solved
     - Signed flow in kg/s as the surrounding hydraulics set it, negative for flow from the from-node to the to-node.
   * - ``mass_flow_pos_kgs``
     - solved
     - Reverse flow half in kg/s.
   * - ``mass_flow_neg_kgs``
     - solved
     - Forward flow half in kg/s.
   * - ``velocity_mps``
     - report
     - Mean velocity in m/s, ``mass_flow_kgs / (rho * A)``.
   * - ``t_from_pu``
     - solved
     - Temperature at the from-end in per unit.
   * - ``t_to_pu``
     - solved
     - Temperature at the to-end; the spread is ``q_mw * 1e6 / (c_p * mass flow)``, so a 0.1 MW load on 1.7 kg/s cools the stream by about 14 K.
   * - ``friction``
     - derived
     - Darcy friction factor set by the formulation: the fully rough asymptote from diameter and roughness under the default constant model, a solved variable only with ``friction_model="nonlinear"``.
   * - ``reynolds_scaled``
     - internal
     - Reynolds number divided by 1e6 when the formulation uses ``friction_model="nonlinear"``; 0 under the default constant friction model.
   * - ``direction``
     - internal
     - Direction binary of the MIQCQP formulations; pinned at 1 and meaningless under the default smooth formulation, read the sign of ``mass_flow_kgs`` instead.
   * - ``mass_flow_pos_kgs_squared``
     - internal
     - Squared flow half used by the pressure-drop equation.
   * - ``mass_flow_neg_kgs_squared``
     - internal
     - Squared flow half used by the pressure-drop equation.
   * - ``limit``
     - internal
     - Unused constant 0.1.
   * - ``diameter_m``
     - input
     - Flow diameter in m.
   * - ``temperature_ext_k``
     - input
     - Stored only, not read.
   * - ``roughness_m``
     - input
     - Wall roughness in m.
   * - ``length_m``
     - input
     - Equivalent length in m.
   * - ``loss_coefficient``
     - input
     - Minor-loss coefficient, ``None`` for the friction model.

The table is named after the alias. After a solve it also carries
``t_inc_pu`` (the temperature change across the branch in per unit, negative
for a load), ``t_in_pu``, ``t_out_pu`` and ``mass_flow_mag_kgs``.

ConsumeHydrGrid
---------------

Demand point with a free mass flow that absorbs the imbalance of its island,
without pinning pressure or temperature. ``pressure_pu`` and ``t_k`` are
descriptive inputs echoed back unchanged; read the solved node state from the
junction table.

.. component: ConsumeHydrGrid | create_consume_hydr_grid

.. code-block:: python

   mm.ConsumeHydrGrid(mass_flow_kgs=0.1, pressure_pu=1, t_k=293)
   mx.create_consume_hydr_grid(network, node_id, mass_flow_kgs=1, pressure_pu=1, t_k=293,
                               grid_key=mm.WATER_KEY, constraints=None, overwrite_id=None,
                               name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``mass_flow_kgs``
     - 0.1, express 1
     - Initial guess of the free withdrawal in kg/s. Bounded to the grid's
       ``max_mass_flow_kgs`` unless bounds are set explicitly.
   * - ``pressure_pu``
     - 1
     - Descriptive setpoint, not enforced.
   * - ``t_k``
     - 293
     - Descriptive setpoint, not enforced.
   * - ``grid_key``
     - ``mm.WATER_KEY``
     - Express only: carrier of the auto-created junction.

.. results: ConsumeHydrGrid

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``mass_flow_kgs``
     - solved
     - Withdrawal in kg/s that balances the island, positive for consumption and negative when the island needs supply; bounded to plus or minus the grid ``max_mass_flow_kgs``.
   * - ``pressure_pu``
     - input
     - Descriptive setpoint, echoed unchanged; the solved pressure is on the junction.
   * - ``t_k``
     - input
     - Descriptive setpoint, echoed unchanged; the solved temperature is on the junction.

HeatGenerator and HeatLoad
--------------------------

Node-based heat injection and withdrawal without a mass flow of their own.
The junction heat balance picks up ``q_mw_heat`` directly, converted to an
enthalpy flow with the grid ``t_ref_k``. The generator takes a positive
magnitude and stores the generation sign.

.. component: HeatGenerator, HeatLoad | create_heat_generator, create_heat_load

.. code-block:: python

   mm.HeatGenerator(q_mw, cost=None)
   mm.HeatLoad(q_mw)
   mx.create_heat_generator(network, node_id, q_mw, constraints=None, overwrite_id=None,
                            name=None)
   mx.create_heat_load(network, node_id, q_mw, constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``q_mw``
     - required
     - Heat in MW as a positive magnitude.
   * - ``cost``
     - ``None``
     - Generator only: price per MW heat for
       :doc:`../problems/economic_dispatch`, forwarded by the express
       function. The result table gains a ``cost`` column only when it is
       set.

.. results: HeatGenerator, HeatLoad

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``q_mw_heat``
     - input
     - Heat setpoint in MW, negative on the generator and positive on the load. The junction sees ``regulation`` times it.

ThermalStorage
--------------

Hot-water tank at a water junction with a tracked stored mass and an optional
standing loss. The state update integrates ``mass_flow_kgs`` over ``dt_h *
3600`` seconds; positive dispatch is charging.

.. component: ThermalStorage | create_water_child

.. code-block:: python

   mm.ThermalStorage(m_stored_kg_initial, m_stored_kg_max, flow_max_kgs,
                     mass_flow_initial_kgs=0.0, loss_factor_per_h=0.0, regulation=1)
   mx.create_water_child(network, model, node_id, constraints=None, overwrite_id=None, name=None)

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
   * - ``loss_factor_per_h``
     - 0.0
     - Fraction of the stored mass lost per hour.
   * - ``regulation``
     - 1
     - Scales the dispatch seen by the junction and by the state update.
   * - ``model``
     - required
     - Express only: the child model instance to attach.

.. results: ThermalStorage

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
   * - ``loss_factor_per_h``
     - input
     - Standing loss per hour.
