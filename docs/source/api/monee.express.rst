monee express
===================

Create functions
----------------

Flat ``create_*`` convenience functions for adding individual nodes,
branches, children and coupling points. Conventionally imported as
``import monee.express as mx``.

.. automodule:: monee.express
   :members:
   :undoc-members:
   :show-inheritance:

Topology structures
-------------------

Bulk builders for common topologies (line, ring, star, and paired
district-heating supply/return structures). The factories
:func:`~monee.express.structures.gas_structure`,
:func:`~monee.express.structures.water_structure`,
:func:`~monee.express.structures.el_structure` and
:func:`~monee.express.structures.dhs_structure` return builders that
remember carrier-level defaults; their shape methods return
:class:`~monee.express.structures.Segment`,
:class:`~monee.express.structures.StarSegment` or
:class:`~monee.express.structures.DhsSegment` handles that can be
composed via ``start_from=``. All names are also re-exported from
:mod:`monee.express`.

.. automodule:: monee.express.structures
   :members:
   :undoc-members:
   :show-inheritance:
