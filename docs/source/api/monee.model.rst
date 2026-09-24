monee models
===================

Package-level re-exports
------------------------

Most model classes are re-exported directly from :mod:`monee.model`, so
``import monee.model as mm`` gives access to everything listed below. The
per-module sections that follow document each submodule under its canonical
path (e.g. :class:`~monee.model.core.Var`, :class:`~monee.model.node.Bus`).

.. automodule:: monee.model
   :members:
   :undoc-members:
   :show-inheritance:
   :imported-members:
   :no-index:

Core protocol
-------------

Attribute types (``Var``, ``Const``, ``Intermediate``, ``PostProcess``),
abstract model bases and the component containers.

.. automodule:: monee.model.core
   :members:
   :undoc-members:
   :show-inheritance:

Network container
-----------------

.. automodule:: monee.model.network
   :members:
   :undoc-members:
   :show-inheritance:

Grids
-----

Per-carrier physical constants and bounds.

.. automodule:: monee.model.grid
   :members:
   :undoc-members:
   :show-inheritance:

Node models
-----------

.. automodule:: monee.model.node
   :members:
   :undoc-members:
   :show-inheritance:

Branch models
-------------

.. automodule:: monee.model.branch
   :members:
   :undoc-members:
   :show-inheritance:

Child models
------------

.. automodule:: monee.model.child
   :members:
   :undoc-members:
   :show-inheritance:

Storage models
--------------

.. automodule:: monee.model.storage
   :members:
   :undoc-members:
   :show-inheritance:

Multi-energy couplers
---------------------

.. automodule:: monee.model.multi
   :members:
   :undoc-members:
   :show-inheritance:
