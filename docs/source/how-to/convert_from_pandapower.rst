============================
Convert from pandapower
============================

If you already have an electrical network defined in
`pandapower <https://www.pandapower.org>`_, monee can import it directly.
The conversion works by first serialising the pandapower network to an
intermediate MATPOWER ``.mat`` file and then reading it back into monee -
all handled automatically by the helper function.

.. warning::

   This feature is **experimental**. Complex pandapower networks with
   special elements (three-winding transformers, DC lines, switches) may
   not convert correctly. Always verify results against the original network.

----

Prerequisites
=============

pandapower (and simbench, for the section below) are **optional
dependencies**. Install them via the ``simbench`` extra:

.. code-block:: bash

    pip install monee[simbench]

or install pandapower alone if you only need the converter:

.. code-block:: bash

    pip install pandapower

----

Converting a network
====================

.. code-block:: python

    import pandapower as pp
    import pandapower.networks as pn
    from monee.io.from_pandapower import from_pandapower_net

    # Load a built-in IEEE test case from pandapower
    pp_net = pn.case9()

    # Convert to a monee Network
    net = from_pandapower_net(pp_net)

The returned :class:`~monee.model.Network` is a standard monee electricity
network. Run an energy flow immediately:

.. code-block:: python

    from monee import run_energy_flow

    result = run_energy_flow(net)

----

What is preserved
=================

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - pandapower element
     - monee equivalent
   * - ``bus``
     - :class:`~monee.model.node.Bus` node (id, name, geodata)
   * - ``line``
     - :class:`~monee.model.branch.GenericPowerBranch` (r, x, b parameters,
       current limit)
   * - ``ext_grid``
     - :class:`~monee.model.child.ExtPowerGrid` child
   * - ``gen`` / ``sgen``
     - Generator child
   * - ``load``
     - :class:`~monee.model.child.PowerLoad` child (aggregated per bus,
       see below)

Node ids, names and positions
-----------------------------

The MATPOWER format numbers buses from 1, so the **monee node id is the
pandapower bus index + 1**. Bus names and geographic coordinates
(``bus_geodata``) are copied onto the monee nodes when present: after
conversion ``node.name`` holds the pandapower bus name and
``node.position`` the ``(x, y)`` coordinates.

Line current limits
-------------------

The intermediate MATPOWER round-trip drops current limits, so
:func:`~monee.io.from_pandapower.from_pandapower_net` recovers them
afterwards from the original pandapower tables. Each line's limit is
rebuilt as ``max_i_ka * parallel * df`` and written back to the matching
monee branch.

.. note::

   **Transformers deliberately keep a placeholder limit.** monee's single
   ``max_i_ka`` per branch cannot represent the different rated currents of
   the HV and LV sides at once, so any tight bound would make one side
   infeasible. Check transformer limits manually before running
   loading-constrained studies on a converted network.

Load aggregation and naming
---------------------------

All pandapower loads connected to the same bus are **aggregated into a
single** :class:`~monee.model.child.PowerLoad`. The aggregated child is
named deterministically by joining the original pandapower load names with
``+`` (see :func:`~monee.io.from_pandapower.aggregated_pp_load_name`), e.g.
``"LV1.101 Load 1+LV1.101 Load 2"``. This deterministic naming is what
allows simbench timeseries to be matched back to the aggregated loads by
name.

.. tip::

   After converting, you can save the result to the native OMEF format using
   :func:`monee.io.native.write_omef_network` so you do not need pandapower
   installed in future sessions. See :doc:`matpower_io` for details.

----

SimBench
========

`SimBench <https://simbench.de>`_ provides standardised benchmark grids
with full-year load profiles. :mod:`monee.io.from_simbench` wraps the
pandapower converter and turns the simbench profiles into a
:class:`~monee.simulation.TimeseriesData` ready for
:func:`~monee.run_timeseries`:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Function
     - Returns
   * - ``obtain_simbench_net(sb_code)``
     - :class:`~monee.model.Network`
   * - ``obtain_simbench_profile(sb_code)``
     - :class:`~monee.simulation.TimeseriesData`
   * - ``obtain_simbench_net_with_td(sb_code)``
     - ``(Network, TimeseriesData)`` from a single simbench fetch
   * - ``obtain_simbench_profile_by_pp_net(pp_net)``
     - :class:`~monee.simulation.TimeseriesData` from an already loaded
       simbench pandapower net

Load profiles are scaled per load (``base_p_mw * profile``) and **summed
per bus** so they match the per-bus PowerLoad aggregation described above.
Each summed series is registered via
:meth:`~monee.simulation.TimeseriesData.add_child_series_by_name` under the
aggregated load name, for both the ``p_mw`` and ``q_mvar`` attributes.
Profiles of other element categories pass through under their raw simbench
column names as ``p_mw`` series.

Feeding the result into a timeseries simulation takes four lines:

.. code-block:: python

    from monee import run_timeseries
    from monee.io.from_simbench import obtain_simbench_net_with_td

    net, td = obtain_simbench_net_with_td("1-LV-rural1--0-no_sw")
    ts_result = run_timeseries(net, td, steps=96)

.. note::

   SimBench profiles cover a full year at 15-minute resolution. Pass
   ``steps`` to simulate only the first part of the horizon, as above.

See :doc:`timeseries` for everything you can do with the resulting
:class:`~monee.simulation.TimeseriesResult`.
