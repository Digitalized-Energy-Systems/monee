.. _components-electricity:

===========
Electricity
===========

Physics: the AC power flow and MISOCP branch-flow sections of
:doc:`../concepts/domains`. Per-unit bases and the ``base_kv`` rule:
:ref:`concepts/conventions:Electricity`.

Bus
---

Electrical node. Holds the voltage magnitude and angle, and sums the child
injections into ``p_mw`` and ``q_mvar``. Buses joined by a line must share
their ``base_kv``; a transformer is where the base changes.

.. component: Bus | create_bus

.. code-block:: python

   mm.Bus(base_kv)
   mx.create_bus(network, base_kv=1, *, constraints=None, grid=mm.EL,
                 overwrite_id=None, name=None, position=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``base_kv``
     - required, express 1
     - Nominal voltage in kV. ``vm_pu`` is relative to it and the branch
       impedance base is ``base_kv ** 2 / sn_mva``. Set it to the real
       voltage level whenever ohmic line data is used.

.. results: Bus

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``vm_pu``
     - solved
     - Voltage magnitude in per unit of ``base_kv``. Held at the slack's setpoint on the slack bus, otherwise inside the bus variable's own bounds, 0.5 to 1.5; optimisation problems tighten them with ``bounds_vm``.
   * - ``va_degree``
     - report
     - Voltage angle in degrees, 0 at the slack, small negative values downstream of it.
   * - ``va_radians``
     - solved
     - The angle decision variable behind ``va_degree``.
   * - ``vm_pu_squared``
     - report
     - ``vm_pu ** 2``, post-processed under the AC energy flow. Under the MISOCP formulation it is the decision variable and ``vm_pu`` is derived from it; in an AC NLP optimisation the column is untied and should be ignored.
   * - ``p_mw``
     - solved
     - Net child injection at the bus in MW, load convention: loads count positive, generators and the slack negative. Branch flows are not part of it, so a bus with only a 0.2 MW load reports 0.2 and the slack bus of that network about -0.2.
   * - ``q_mvar``
     - solved
     - Net reactive child injection in Mvar, same convention.
   * - ``base_kv``
     - input
     - Voltage base in kV.

PowerLine
---------

Overhead line or cable between two buses, given as ohmic parameters per metre.
The per-unit impedance is ``r_ohm_per_m * length_m / (base_kv ** 2 / sn_mva)
/ parallel`` with the from-bus ``base_kv``.

.. component: PowerLine | create_line

.. code-block:: python

   mm.PowerLine(length_m, r_ohm_per_m, x_ohm_per_m, parallel, backup=False, on_off=1,
                *, max_i_ka=3.19, max_s_mva=None)
   mx.create_line(network, from_node_id, to_node_id, length_m, r_ohm_per_m, x_ohm_per_m,
                  parallel=1, constraints=None, grid=None, name=None, on_off=1,
                  max_i_ka=None, max_s_mva=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``length_m``
     - required
     - Line length in m.
   * - ``r_ohm_per_m``
     - required
     - Resistance per metre in ohm/m.
   * - ``x_ohm_per_m``
     - required
     - Reactance per metre in ohm/m.
   * - ``parallel``
     - required, express 1
     - Number of parallel systems; divides the impedance.
   * - ``backup``
     - ``False``
     - Flags a reserve line for the optimisation problem's
       ``controllable_backup_lines``, which makes ``on_off`` of such lines a
       binary decision. The flag itself changes nothing, so pass ``on_off=0``
       as well to build the line open.
   * - ``on_off``
     - 1
     - In service (1) or out of service (0). Multiplies every flow term of the
       branch in the nodal balances.
   * - ``max_i_ka``
     - 3.19, express ``None``
     - Thermal current rating in kA, the base of ``loading_from_pu`` and
       ``loading_to_pu``. The express function passes it through only when
       given, so the class default applies otherwise.
   * - ``max_s_mva``
     - ``None``
     - Optional apparent-power rating in MVA used by loading constraints.

.. results: PowerLine

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_from_mw``
     - solved
     - Active power entering the branch at the from-end in MW, positive for flow from the from-node to the to-node. A 0.2 MW load fed over one line reports about 0.2.
   * - ``p_to_mw``
     - solved
     - Active power entering at the to-end in MW; the same case reports about -0.2, and ``p_from_mw + p_to_mw`` is the loss.
   * - ``q_from_mvar``
     - solved
     - Reactive power entering at the from-end in Mvar.
   * - ``q_to_mvar``
     - solved
     - Reactive power entering at the to-end in Mvar.
   * - ``i_from_ka``
     - solved
     - Current magnitude at the from-end in kA.
   * - ``i_to_ka``
     - solved
     - Current magnitude at the to-end in kA.
   * - ``loading_from_pu``
     - solved
     - ``i_from_ka / max_i_ka``; 1.0 is the thermal limit.
   * - ``loading_to_pu``
     - solved
     - ``i_to_ka * base_kv_to / (base_kv_from * max_i_ka)``: the to-end current referred to the from-side base, so both ends of a transformer report a comparable loading. The model property ``loading_pu`` is the larger of the two ends.
   * - ``br_r_pu``
     - derived
     - Series resistance in per unit, computed from the ohmic inputs, the from-bus ``base_kv`` and the grid ``sn_mva``.
   * - ``br_x_pu``
     - derived
     - Series reactance in per unit.
   * - ``length_m``
     - input
     - Line length in m.
   * - ``r_ohm_per_m``
     - input
     - Resistance per metre.
   * - ``x_ohm_per_m``
     - input
     - Reactance per metre.
   * - ``parallel``
     - input
     - Number of parallel systems.
   * - ``max_i_ka``
     - input
     - Current rating in kA.
   * - ``max_s_mva``
     - input
     - Apparent-power rating in MVA, ``None`` when not given.
   * - ``backup``
     - input
     - Reserve-line flag.
   * - ``tap``
     - internal
     - Always 1 for a line.
   * - ``shift``
     - internal
     - Always 0 for a line.
   * - ``g_fr_pu``
     - internal
     - Shunt conductance at the from-end, 0 for a line.
   * - ``b_fr_pu``
     - internal
     - Shunt susceptance at the from-end, 0 for a line.
   * - ``g_to_pu``
     - internal
     - Shunt conductance at the to-end, 0 for a line.
   * - ``b_to_pu``
     - internal
     - Shunt susceptance at the to-end, 0 for a line.

Read the table with ``result.get(mm.PowerLine)``. Losses are
``p_from_mw + p_to_mw`` per row.

Trafo
-----

Two-winding transformer. The from-node is the low-voltage bus and the
to-node the high-voltage bus. The per-unit impedance the model computes is
``r = vkr_percent / 100 / sn_trafo_mva * (base_kv_lv / base_kv_hv) ** 2 * sn_mva``
and ``x = sqrt(z ** 2 - r ** 2)`` with ``z`` from ``vk_percent`` the same way.
There is no tap changer model: ``tap`` is fixed at 1.

.. component: Trafo | create_trafo

.. code-block:: python

   mm.Trafo(vk_percent=12.2, vkr_percent=0.25, sn_trafo_mva=160, shift=0)
   mx.create_trafo(network, from_node_id, to_node_id, vk_percent=12.2, vkr_percent=0.25,
                   sn_trafo_mva=160, shift=0, constraints=None, grid=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``vk_percent``
     - 12.2
     - Short-circuit voltage in percent of the rating.
   * - ``vkr_percent``
     - 0.25
     - Real part of the short-circuit voltage in percent. Must not exceed
       ``vk_percent``, otherwise the constructor raises ``ValueError``.
   * - ``sn_trafo_mva``
     - 160
     - Rated apparent power in MVA. The default describes a large HV unit;
       a distribution transformer needs something like 0.4 to 63.
   * - ``shift``
     - 0
     - Phase shift in radians (the MATPOWER importer converts degrees).
       Vector-group shifts of 150 degrees make the flat start of the
       nonlinear solvers unusable; keep them at 0 unless the study needs them.

.. results: Trafo

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_from_mw``
     - solved
     - Active power entering the branch at the low-voltage end in MW, positive for flow towards the high-voltage bus. A transformer feeding 0.05 MW down to its low-voltage bus reports about -0.05.
   * - ``p_to_mw``
     - solved
     - Active power entering at the high-voltage end in MW; the same case reports about 0.05, and ``p_from_mw + p_to_mw`` is the loss.
   * - ``q_from_mvar``
     - solved
     - Reactive power entering at the from-end in Mvar.
   * - ``q_to_mvar``
     - solved
     - Reactive power entering at the to-end in Mvar.
   * - ``i_from_ka``
     - solved
     - Current magnitude at the from-end in kA.
   * - ``i_to_ka``
     - solved
     - Current magnitude at the to-end in kA.
   * - ``loading_from_pu``
     - solved
     - ``i_from_ka / max_i_ka``; 1.0 is the thermal limit.
   * - ``loading_to_pu``
     - solved
     - ``i_to_ka * base_kv_to / (base_kv_from * max_i_ka)``: the to-end current referred to the from-side base, so both ends of a transformer report a comparable loading. The model property ``loading_pu`` is the larger of the two ends.
   * - ``br_r_pu``
     - derived
     - Short-circuit resistance in per unit from the formula above.
   * - ``br_x_pu``
     - derived
     - Short-circuit reactance in per unit from the formula above.
   * - ``vk_percent``
     - input
     - Short-circuit voltage in percent.
   * - ``vkr_percent``
     - input
     - Real part of the short-circuit voltage in percent.
   * - ``sn_trafo_mva``
     - input
     - Rated apparent power in MVA.
   * - ``shift``
     - input
     - Phase shift in radians.
   * - ``max_i_ka``
     - internal
     - Fixed at the GenericPowerBranch default of 3.19 kA; the constructor does not expose it, so set ``model.max_i_ka`` yourself before reading the loading columns.
   * - ``max_s_mva``
     - internal
     - Fixed at ``None``.
   * - ``backup``
     - internal
     - Fixed at ``False``.
   * - ``tap``
     - internal
     - Fixed at 1; there is no tap changer.
   * - ``vn_trafo_lv``
     - internal
     - Fixed at 1.
   * - ``g_fr_pu``
     - internal
     - Shunt conductance at the from-end, 0.
   * - ``b_fr_pu``
     - internal
     - Shunt susceptance at the from-end, 0.
   * - ``g_to_pu``
     - internal
     - Shunt conductance at the to-end, 0.
   * - ``b_to_pu``
     - internal
     - Shunt susceptance at the to-end, 0.


GenericPowerBranch
------------------

The MATPOWER-style branch that ``PowerLine`` and ``Trafo`` derive from: a
pi-equivalent with per-unit series impedance and shunt admittances at both
ends. The importers (:doc:`../how-to/matpower_io`,
:doc:`../how-to/convert_from_pandapower`) build these directly. There is no
dedicated express function; ``mx.create_el_branch`` attaches any electrical
branch model between two existing buses and, unlike ``create_line``, does not
auto-create missing ones.

.. component: GenericPowerBranch | create_el_branch

.. code-block:: python

   mm.GenericPowerBranch(tap, shift, br_r_pu, br_x_pu, g_fr_pu, b_fr_pu, g_to_pu, b_to_pu,
                         max_i_ka=3.19, max_s_mva=None, backup=False, on_off=1)
   mx.create_el_branch(network, from_node_id, to_node_id, model, constraints=None,
                       grid=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``tap``
     - required
     - Off-nominal tap ratio (1 for none).
   * - ``shift``
     - required
     - Phase shift in radians.
   * - ``br_r_pu``, ``br_x_pu``
     - required
     - Series resistance and reactance in per unit of ``base_kv ** 2 /
       sn_mva``.
   * - ``g_fr_pu``, ``b_fr_pu``
     - required
     - Shunt conductance and susceptance at the from-end in per unit.
   * - ``g_to_pu``, ``b_to_pu``
     - required
     - Shunt conductance and susceptance at the to-end in per unit.
   * - ``max_i_ka``
     - 3.19
     - Thermal current rating in kA.
   * - ``max_s_mva``
     - ``None``
     - Optional apparent-power rating in MVA.
   * - ``backup``
     - ``False``
     - Reserve-branch flag, see ``PowerLine``.
   * - ``on_off``
     - 1
     - In service flag.
   * - ``model``
     - required
     - Express only: the branch model instance to attach.

.. results: GenericPowerBranch

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_from_mw``
     - solved
     - Active power entering the branch at the from-end in MW, positive for flow from the from-node to the to-node. A 0.2 MW load fed over one line reports about 0.2.
   * - ``p_to_mw``
     - solved
     - Active power entering at the to-end in MW; the same case reports about -0.2, and ``p_from_mw + p_to_mw`` is the loss.
   * - ``q_from_mvar``
     - solved
     - Reactive power entering at the from-end in Mvar.
   * - ``q_to_mvar``
     - solved
     - Reactive power entering at the to-end in Mvar.
   * - ``i_from_ka``
     - solved
     - Current magnitude at the from-end in kA.
   * - ``i_to_ka``
     - solved
     - Current magnitude at the to-end in kA.
   * - ``loading_from_pu``
     - solved
     - ``i_from_ka / max_i_ka``; 1.0 is the thermal limit.
   * - ``loading_to_pu``
     - solved
     - ``i_to_ka * base_kv_to / (base_kv_from * max_i_ka)``: the to-end current referred to the from-side base, so both ends of a transformer report a comparable loading. The model property ``loading_pu`` is the larger of the two ends.
   * - ``tap``
     - input
     - Off-nominal tap ratio.
   * - ``shift``
     - input
     - Phase shift in radians.
   * - ``br_r_pu``
     - input
     - Series resistance in per unit.
   * - ``br_x_pu``
     - input
     - Series reactance in per unit.
   * - ``g_fr_pu``
     - input
     - Shunt conductance at the from-end in per unit.
   * - ``b_fr_pu``
     - input
     - Shunt susceptance at the from-end in per unit.
   * - ``g_to_pu``
     - input
     - Shunt conductance at the to-end in per unit.
   * - ``b_to_pu``
     - input
     - Shunt susceptance at the to-end in per unit.
   * - ``max_i_ka``
     - input
     - Current rating in kA.
   * - ``max_s_mva``
     - input
     - Apparent-power rating in MVA, ``None`` when not given.
   * - ``backup``
     - input
     - Reserve-line flag.

ExtPowerGrid
------------

The slack: pins the bus voltage magnitude and angle and absorbs whatever
active and reactive power the network needs. Seen from the network it is a
child, so ``p_mw`` is positive when the network exports into the external
grid and negative when it imports (:ref:`data-model-slack-sign`).

.. component: ExtPowerGrid | create_ext_power_grid

.. code-block:: python

   mm.ExtPowerGrid(p_mw, q_mvar, vm_pu=1, va_degree=0, max_import_mw=None,
                   max_export_mw=None, regulate_vm=True, cost=None)
   mx.create_ext_power_grid(network, node_id, p_mw=0, q_mvar=0, vm_pu=1, va_degree=0,
                            max_import_mw=None, max_export_mw=None, regulate_vm=True,
                            cost=None, constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``p_mw``
     - required, express 0
     - Initial guess for the free exchange variable in MW, not a setpoint.
   * - ``q_mvar``
     - required, express 0
     - Initial guess for the free reactive exchange in Mvar.
   * - ``vm_pu``
     - 1
     - Voltage magnitude the slack holds the bus at.
   * - ``va_degree``
     - 0
     - Reference angle in degrees.
   * - ``max_import_mw``
     - ``None``
     - Import cap as a positive magnitude; bounds ``p_mw`` from below at
       ``-max_import_mw``.
   * - ``max_export_mw``
     - ``None``
     - Export cap as a positive magnitude; bounds ``p_mw`` from above.
   * - ``regulate_vm``
     - ``True``
     - ``True`` pins the magnitude (power-flow semantics). ``False`` pins only
       the angle and leaves the magnitude a bounded decision variable, the
       optimal-power-flow convention.
   * - ``cost``
     - ``None``
     - Price of the exchange in currency per MW for the economic dispatch
       objective. It counts only when the problem sets
       ``ext_grid_cost_default``; otherwise the slack stays out of the
       objective and this value is ignored.

.. results: ExtPowerGrid

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - solved
     - Exchange with the external grid in MW, load convention: a network that draws 0.2 MW plus losses reports about -0.2; a net exporting network reports a positive value.
   * - ``q_mvar``
     - solved
     - Reactive exchange in Mvar, same convention.
   * - ``vm_pu``
     - input
     - Voltage magnitude the slack holds, or the setpoint it ignores when ``regulate_vm`` is ``False``.
   * - ``va_degree``
     - input
     - Reference angle in degrees.
   * - ``regulate_vm``
     - input
     - Whether the magnitude is pinned.
   * - ``cost``
     - input
     - Exchange price; the column exists only when ``cost`` was given.

PowerLoad
---------

Fixed active and reactive demand at a bus.

.. component: PowerLoad | create_power_load

.. code-block:: python

   mm.PowerLoad(p_mw, q_mvar)
   mx.create_power_load(network, node_id, p_mw, q_mvar, constraints=None,
                        overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``p_mw``
     - required
     - Active demand in MW, positive.
   * - ``q_mvar``
     - required
     - Reactive demand in Mvar, positive for inductive.

.. results: PowerLoad

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - input
     - Active setpoint in MW. The bus sees ``regulation * p_mw``, so under load shedding the setpoint stays and ``regulation`` drops below 1.
   * - ``q_mvar``
     - input
     - Reactive setpoint in Mvar.

PowerGenerator
--------------

Fixed active and reactive injection (a PQ generator). The constructor takes
positive magnitudes and stores them with the generation sign, so the result
table shows a 0.5 MW unit as ``p_mw = -0.5``.

.. component: PowerGenerator | create_power_generator

.. code-block:: python

   mm.PowerGenerator(p_mw, q_mvar, cost=None)
   mx.create_power_generator(network, node_id, p_mw, q_mvar, cost=None, constraints=None,
                             overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``p_mw``
     - required
     - Active generation in MW as a positive magnitude; a negative value
       raises ``ValueError``.
   * - ``q_mvar``
     - required
     - Reactive generation in Mvar, positive for capacitive; a negative value
       is accepted and models absorption.
   * - ``cost``
     - ``None``
     - Marginal cost in currency per MW for :doc:`../problems/economic_dispatch`.

.. results: PowerGenerator

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - input
     - Active setpoint in MW with the generation sign: a 0.5 MW unit reports -0.5. Becomes a solved dispatch when the problem's ``controllable_generators`` promotes it.
   * - ``q_mvar``
     - input
     - Reactive setpoint in Mvar, generation sign.
   * - ``cost``
     - input
     - Marginal cost; the column exists only when ``cost`` was given.

VoltageControlledGenerator
--------------------------

A PV bus: fixed active injection, bus voltage magnitude held at ``vm_pu``,
reactive power free. Not grid-forming; a slack is still needed elsewhere.

.. component: VoltageControlledGenerator | create_el_child

.. code-block:: python

   mm.VoltageControlledGenerator(p_mw, vm_pu=1.0, q_mvar=0.0)
   mx.create_el_child(network, model, node_id, constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``p_mw``
     - required
     - Active generation in MW as a positive magnitude.
   * - ``vm_pu``
     - 1.0
     - Voltage magnitude setpoint the bus is pinned to.
   * - ``q_mvar``
     - 0.0
     - Seed of the free reactive variable in Mvar as a positive magnitude,
       stored with the generation sign like the result column.
   * - ``model``
     - required
     - Express only: the child model instance to attach.

.. results: VoltageControlledGenerator

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - input
     - Active setpoint in MW, generation sign.
   * - ``q_mvar``
     - solved
     - Reactive power in Mvar the unit needs to hold ``vm_pu``, generation sign: a unit supplying 0.18 Mvar reports -0.18.
   * - ``vm_pu``
     - input
     - Voltage setpoint the bus is held at.

PowerShunt
----------

Constant admittance at a bus: capacitor banks, reactors and lumped line
charging. Unlike a load its draw scales with the squared bus voltage,
``p_mw = gs_mw * v ** 2`` and ``q_mvar = -bs_mvar * v ** 2``.

.. component: PowerShunt | create_el_child

.. code-block:: python

   mm.PowerShunt(gs_mw, bs_mvar)
   mx.create_el_child(network, model, node_id, constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``gs_mw``
     - required
     - Real power drawn at 1.0 pu voltage in MW (MATPOWER ``GS``).
   * - ``bs_mvar``
     - required
     - Reactive power injected at 1.0 pu voltage in Mvar (MATPOWER ``BS``);
       positive is a capacitor, negative a reactor.
   * - ``model``
     - required
     - Express only: the child model instance to attach.

.. results: PowerShunt

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - solved
     - ``gs_mw * vm_pu ** 2`` in MW.
   * - ``q_mvar``
     - solved
     - ``-bs_mvar * vm_pu ** 2`` in Mvar: a 0.05 Mvar capacitor at 1.0 pu reports -0.05.
   * - ``gs_mw``
     - input
     - Conductance as MW at 1.0 pu.
   * - ``bs_mvar``
     - input
     - Susceptance as Mvar at 1.0 pu.

ElectricStorage
---------------

Battery at a bus with a tracked state of charge. Dispatch ``p_mw`` is a
fixed float until :meth:`~monee.model.storage.ElectricStorage.make_controllable`
or the problem's ``controllable_storages`` promotes it; positive is charging.
:doc:`../how-to/storage` covers prescribed and optimised dispatch.

.. component: ElectricStorage | create_el_child

.. code-block:: python

   mm.ElectricStorage(e_mwh_initial, e_mwh_max, p_max_mw, p_mw_initial=0.0,
                      efficiency_charge=1.0, efficiency_discharge=1.0, regulation=1)
   mx.create_el_child(network, model, node_id, constraints=None, overwrite_id=None, name=None)

.. list-table::
   :header-rows: 1
   :widths: 28 16 56

   * - Parameter
     - Default
     - Meaning
   * - ``e_mwh_initial``
     - required
     - State of charge at the first step in MWh.
   * - ``e_mwh_max``
     - required
     - Capacity in MWh; becomes the upper bound of the ``e_mwh`` variable.
   * - ``p_max_mw``
     - required
     - Symmetric charge and discharge power limit in MW, enforced once the
       dispatch is controllable.
   * - ``p_mw_initial``
     - 0.0
     - Dispatch in MW used while the storage is not controllable.
   * - ``efficiency_charge``
     - 1.0
     - Charging efficiency; the stored energy grows by ``eta * p * dt``.
   * - ``efficiency_discharge``
     - 1.0
     - Discharging efficiency; the stored energy drops by ``p / eta * dt``.
   * - ``regulation``
     - 1
     - Scales the dispatch seen by the bus and by the state update.
   * - ``model``
     - required
     - Express only: the child model instance to attach.

.. results: ElectricStorage

.. list-table::
   :header-rows: 1
   :widths: 26 11 63

   * - Column
     - Kind
     - Meaning
   * - ``p_mw``
     - input
     - Dispatch in MW, positive charging. Solved instead once the storage is controllable.
   * - ``e_mwh``
     - solved
     - State of charge in MWh at the end of the step. Meaningful in a timeseries or multi-period run, where the inter-temporal equation ties it to the previous step; a single solve leaves it a free variable, so ignore the value there.
   * - ``efficiency_charge``
     - input
     - Charging efficiency.
   * - ``efficiency_discharge``
     - input
     - Discharging efficiency.
   * - ``q_mvar``
     - internal
     - Always 0; the storage exchanges no reactive power.
