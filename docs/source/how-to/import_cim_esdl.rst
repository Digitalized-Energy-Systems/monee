====================================
Import CIM/CGMES and ESDL models
====================================

monee ships two **experimental** importers for common European energy-system
exchange formats: :mod:`monee.io.from_cim` reads CGMES (CIM) profile files
into an electrical :class:`~monee.model.Network`, and
:mod:`monee.io.from_esdl` reads ESDL energy-system descriptions into a
multi-energy network. Both are deliberately *not* re-exported from any
package ``__init__`` - import the modules explicitly.

.. warning::

   Both importers are **experimental**. They emit a ``UserWarning`` on every
   call and return a transparency **report** alongside the network so you can
   see exactly what was mapped and what was skipped - nothing is dropped
   silently. The mappings are incomplete (see the tables below); always
   verify the imported network before relying on it.

----

Import a CGMES model (CIM)
==========================

The CIM importer covers the CGMES **bus-branch** model (TopologicalNodes,
not ConnectivityNodes) and is **electricity only**. It needs the optional
`cimpy <https://github.com/sogno-platform/cimpy>`_ package:

.. code-block:: bash

    pip install cimpy

Pass the EQ/TP/SSH (and optionally SV) profile files of one CGMES model -
a single path or a list of XML/zip files:

.. code-block:: python

    from monee.io.from_cim import import_cim_files

    net, report = import_cim_files(
        [
            "case_EQ.xml",
            "case_TP.xml",
            "case_SSH.xml",
        ],
        cgmes_version="cgmes_v2_4_15",  # or "cgmes_v3_0_0"
    )

    report.log()  # log counters and skip reasons via logging
    print(report.buses, report.lines, report.skipped)

If you already loaded the model with cimpy yourself, convert the merged
``{mRID: object}`` topology dict directly:

.. code-block:: python

    import cimpy
    from monee.io.from_cim import cim_objects_to_network

    imported = cimpy.cim_import(file_list, "cgmes_v2_4_15")
    net, report = cim_objects_to_network(imported["topology"])

Generator sign convention
-------------------------

CGMES SSH stores machine setpoints in the **load reference** convention at
the terminal, so generation usually appears as negative ``p``. The
``gen_sign`` parameter (default ``-1.0``) flips that sign before the value
is handed to :class:`~monee.model.child.PowerGenerator` as a positive
generation magnitude. Pass ``gen_sign=1.0`` if your data set already uses
the generation reference.

What is mapped from CGMES
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - CGMES class
     - monee equivalent
   * - ``TopologicalNode``
     - :class:`~monee.model.node.Bus` node; ``base_kv`` from the
       ``BaseVoltage`` reference (default ``1.0`` when missing)
   * - ``ACLineSegment``
     - :class:`~monee.model.branch.GenericPowerBranch`; r/x/bch/gch are
       converted to p.u. via ``base_kv**2 / sn_mva``, the total shunt
       admittance is split evenly across both ends (pi model), and
       ``max_i_ka=9999`` is a placeholder (operational limits TODO)
   * - 2-winding ``PowerTransformer``
     - :class:`~monee.model.branch.GenericPowerBranch`; both winding
       impedances are lumped onto the FROM side via
       ``(U_from / U_e)**2``. Tap changers are TODO (``tap=1``,
       ``shift=0``); 3-winding transformers are skipped
   * - ``EnergyConsumer`` / ``ConformLoad`` / ``NonConformLoad`` /
       ``StationSupply``
     - :class:`~monee.model.child.PowerLoad` child
   * - ``SynchronousMachine``
     - :class:`~monee.model.child.PowerGenerator` child (sign handled by
       ``gen_sign``, see above)
   * - ``ExternalNetworkInjection``
     - :class:`~monee.model.child.ExtPowerGrid` slack; ``vm_pu`` is derived
       from the ``RegulatingControl`` target value when present, else ``1.0``

If the model contains no ``ExternalNetworkInjection``, the importer logs a
warning - the network then has **no slack**, and you must designate one with
:func:`~monee.express.create_ext_power_grid` before solving.

The import report
-----------------

``CimImportReport`` is a small dataclass with counters
``buses``, ``lines``, ``transformers``, ``loads``, ``generators`` and
``ext_grids``, plus a ``skipped`` dict mapping each skip *reason* to a
count (e.g. ``"3-winding PowerTransformer (only 2-winding supported)"``).
Call ``report.log()`` to emit the summary through the standard
:mod:`logging` machinery. Every object the importer cannot map ends up in
``skipped`` - nothing is silently dropped.

.. tip::

   After importing, save the network to the native OMEF format with
   :func:`monee.io.native.write_omef_network` so future sessions do not need
   cimpy (or the original CGMES files) at all.

----

Import an ESDL energy system
============================

The ESDL importer is a **multi-energy sketch**: it reconstructs the
topology and the basic transport/demand/producer assets across electricity,
gas and heat, but skips conversion assets. It needs the optional
`pyESDL <https://pypi.org/project/pyESDL/>`_ package:

.. code-block:: bash

    pip install pyESDL

.. code-block:: python

    from monee.io.from_esdl import import_esdl_file

    net, report = import_esdl_file("my_system.esdl")
    report.log()

For an ``EnergySystem`` you already loaded with pyESDL, use
``esdl_system_to_network``:

.. code-block:: python

    from esdl.esdl_handler import EnergySystemHandler
    from monee.io.from_esdl import esdl_system_to_network

    es = EnergySystemHandler().load_file("my_system.esdl")
    net, report = esdl_system_to_network(es)

How the topology is reconstructed
---------------------------------

ESDL has no explicit bus/junction objects, so the importer runs a
**union-find over the port** ``connectedTo`` **relations**: every group of
mutually connected ports becomes one monee node, typed by the group's
carrier - electricity ports become a :class:`~monee.model.node.Bus`
(``base_kv=1``) on the EL grid, gas ports a
:class:`~monee.model.node.Junction` on the GAS grid, and heat ports a
:class:`~monee.model.node.Junction` on the WATER grid.

What is mapped from ESDL
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - ESDL asset
     - monee equivalent
   * - ``ElectricityCable`` / ``Transformer``
     - :class:`~monee.model.branch.PowerLine` with **placeholder** impedance
       (``1e-4`` ohm/m for both r and x) - ESDL cables rarely carry r/x
   * - ``Pipe``
     - :class:`~monee.model.branch.GasPipe` or
       :class:`~monee.model.branch.WaterPipe`, depending on the port carrier;
       default diameter ``0.1`` m when ``innerDiameter`` is missing
   * - ``ElectricityDemand`` / ``EConnection``
     - :class:`~monee.model.child.PowerLoad`
   * - ``ElectricityProducer`` / ``PVInstallation`` / ``PVPanel`` /
       ``WindTurbine``
     - :class:`~monee.model.child.PowerGenerator`
   * - ``HeatingDemand`` / ``HeatDemand``
     - :class:`~monee.model.child.HeatLoad`
   * - ``ResidualHeatSource`` / ``GeothermalSource`` / ``HeatProducer``
     - :class:`~monee.model.child.HeatGenerator`

ESDL ``power`` values are given in **W** and converted to **MW**.

.. warning::

   The electrical line impedance is a flat placeholder - **override it with
   real cable data** before drawing any conclusions from power-flow results.

What is skipped
---------------

The following asset families are skipped on purpose and counted in the
report, rather than imported with invented parameters:

- **Gas demands and producers** (``GasDemand``, ``GasProducer``) - mapping
  ESDL power to a monee mass flow needs the gas carrier's heating value;
  this conversion is not implemented yet.
- **Conversion assets** (``CHP``, ``HeatPump``, ``PowerToGas``,
  ``Electrolyzer``, ``GasHeater``) - these would become monee coupling
  compounds (see :doc:`../concepts/multi_energy`) and are a TODO.
- Any other unmapped asset type, recorded per class name.

``EsdlImportReport`` mirrors the CIM report: counters ``nodes``,
``el_branches``, ``gas_pipes``, ``heat_pipes``, ``loads`` and ``sources``,
plus the ``skipped`` reason counts and a ``log()`` method.

----

See also
========

.. grid:: 1 2 2 2
   :gutter: 3

   .. grid-item-card:: Import MATPOWER case files
      :link: matpower_io
      :link-type: doc
      :shadow: sm

      Read standard ``.mat`` test cases and persist networks in the native
      OMEF JSON format.

   .. grid-item-card:: Convert from pandapower
      :link: convert_from_pandapower
      :link-type: doc
      :shadow: sm

      Import an existing pandapower network via the MATPOWER round-trip.
