============================
Convert from pandapower
============================

If you already have an electrical network defined in
`pandapower <https://www.pandapower.org>`_, monee can import it directly.
The conversion works by first serialising the pandapower network to an
intermediate MATPOWER ``.mat`` file and then reading it back into monee —
all handled automatically by the helper function.

.. warning::

   This feature is **experimental**. Complex pandapower networks with
   special elements (three-winding transformers, DC lines, switches) may
   not convert correctly. Always verify results against the original network.

----

Prerequisites
=============

Install pandapower:

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
     - :class:`~monee.model.branch.GenericPowerBranch` (r, x, b parameters)
   * - ``ext_grid``
     - :class:`~monee.model.child.ExtPowerGrid` child
   * - ``gen`` / ``sgen``
     - Generator child
   * - ``load``
     - :class:`~monee.model.child.PowerLoad` child

Bus names and geographic coordinates (``bus_geodata``) are carried over when
present in the pandapower network.

.. tip::

   After converting, you can save the result to the native OMEF format using
   :func:`monee.io.native.write_omef_network` so you do not need pandapower
   installed in future sessions. See :doc:`matpower_io` for details.
