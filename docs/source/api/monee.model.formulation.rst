monee.model.formulation
=======================

The package is organised by **optimization class** first, sector second:

* :mod:`~monee.model.formulation.nlp` — smooth non-convex NLPs
  (polar AC, smooth Weymouth, smooth Darcy–Weisbach) for IPOPT/APOPT.
* :mod:`~monee.model.formulation.milp` — LP/MILP models (PWL Weymouth,
  McCormick district heating, fixed-flow heat exchanger).
* :mod:`~monee.model.formulation.miqcqp.convex` — certifiable relaxations
  (branch-flow MISOCP, epigraph-relaxed Weymouth).
* :mod:`~monee.model.formulation.miqcqp.nonconvex` — exact quadratic models
  for global solvers (exact branch flow, exact Weymouth, bilinear
  Darcy–Weisbach).

Core
----

.. automodule:: monee.model.formulation.core
   :members:
   :undoc-members:
   :show-inheritance:

Bundles & sector constants
--------------------------

Sector constants (``EL_*``, ``GAS_*``, ``HEAT_*``), the sector-complete
bundles (:data:`~monee.model.formulation.bundles.SMOOTH_NLP_FORMULATION`,
:data:`~monee.model.formulation.bundles.CONVEX_MIQCQP_FORMULATION`,
:data:`~monee.model.formulation.bundles.NONCONVEX_MIQCQP_FORMULATION`) and
the :func:`~monee.model.formulation.bundles.combine` helper. All of them are
importable directly from ``monee.model.formulation``.

.. automodule:: monee.model.formulation.bundles
   :members:
   :undoc-members:

Shared node formulations
------------------------

.. automodule:: monee.model.formulation.common
   :members:
   :undoc-members:
   :show-inheritance:

Smooth NLP (``nlp``)
--------------------

Polar AC electricity
~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.nlp.el
   :members:
   :undoc-members:
   :show-inheritance:

Smooth Weymouth gas
~~~~~~~~~~~~~~~~~~~

Binary-free smooth Weymouth physics for pure-NLP solves (IPOPT/APOPT).

.. automodule:: monee.model.formulation.nlp.gas
   :members:
   :undoc-members:
   :show-inheritance:

Smooth Darcy–Weisbach heat
~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary-free smooth pipe and heat-exchanger formulations for pure-NLP solves.

.. automodule:: monee.model.formulation.nlp.heat
   :members:
   :undoc-members:
   :show-inheritance:

LP / MILP (``milp``)
--------------------

PWL Weymouth gas
~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.milp.gas
   :members:
   :undoc-members:
   :show-inheritance:

McCormick district heating & fixed-flow heat exchanger
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LP/MILP relaxation of district heating including the gap-bound diagnostics
:func:`~monee.model.formulation.milp.heat.mccormick_dhs_gap_bound_mw`
and
:func:`~monee.model.formulation.milp.heat.mccormick_dhs_gap_bound_k`.

.. automodule:: monee.model.formulation.milp.heat
   :members:
   :undoc-members:
   :show-inheritance:

Convex MIQCQP (``miqcqp.convex``)
---------------------------------

Branch-flow MISOCP electricity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.miqcqp.convex.el
   :members:
   :undoc-members:
   :show-inheritance:

Epigraph-relaxed Weymouth gas
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.miqcqp.convex.gas
   :members:
   :undoc-members:
   :show-inheritance:

Non-convex MIQCQP (``miqcqp.nonconvex``)
----------------------------------------

Exact branch-flow electricity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.miqcqp.nonconvex.el
   :members:
   :undoc-members:
   :show-inheritance:

Exact Weymouth gas
~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.miqcqp.nonconvex.gas
   :members:
   :undoc-members:
   :show-inheritance:

Bilinear Darcy–Weisbach heat
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.miqcqp.nonconvex.heat
   :members:
   :undoc-members:
   :show-inheritance:
