monee.model.formulation
=======================

Core
----

.. automodule:: monee.model.formulation.core
   :members:
   :undoc-members:
   :show-inheritance:

Ready-to-use network formulations
----------------------------------

The following constants are importable directly from ``monee.model.formulation``:

.. automodule:: monee.model.formulation
   :members:
   :undoc-members:
   :no-index:

Electricity
-----------

Nonlinear AC
~~~~~~~~~~~~

.. automodule:: monee.model.formulation.nonlinear.ac
   :members:
   :undoc-members:
   :show-inheritance:

MISOCP relaxation
~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.misoc.el
   :members:
   :undoc-members:
   :show-inheritance:

Electricity wiring
~~~~~~~~~~~~~~~~~~

Prebuilt electricity bundles
(:data:`~monee.model.formulation.el.AC_NETWORK_FORMULATION`,
:data:`~monee.model.formulation.el.MISOCP_NETWORK_FORMULATION`).

.. automodule:: monee.model.formulation.el
   :members:
   :undoc-members:
   :show-inheritance:

Gas
---

Nonlinear Weymouth
~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.nonlinear.gas
   :members:
   :undoc-members:
   :show-inheritance:

Smooth Weymouth (pure NLP)
~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary-free smooth Weymouth physics for pure-NLP solves (IPOPT/APOPT).

.. automodule:: monee.model.formulation.nonlinear.gas_smooth
   :members:
   :undoc-members:
   :show-inheritance:

Gas wiring & factories
~~~~~~~~~~~~~~~~~~~~~~

Prebuilt gas bundles and the factories
:func:`~monee.model.formulation.gas.make_nl_weymouth_pwl_network_formulation`
and
:func:`~monee.model.formulation.gas.make_smooth_weymouth_network_formulation`.

.. automodule:: monee.model.formulation.gas
   :members:
   :undoc-members:
   :show-inheritance:

Water / district heating
------------------------

Nonlinear Darcy–Weisbach
~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.nonlinear.water
   :members:
   :undoc-members:
   :show-inheritance:

Smooth Darcy–Weisbach & heat (pure NLP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Binary-free smooth pipe and heat-exchanger formulations for pure-NLP solves.

.. automodule:: monee.model.formulation.nonlinear.heat_smooth
   :members:
   :undoc-members:
   :show-inheritance:

Linear heat exchanger
~~~~~~~~~~~~~~~~~~~~~

.. automodule:: monee.model.formulation.linear.water
   :members:
   :undoc-members:
   :show-inheritance:

McCormick district heating relaxation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LP/MILP relaxation of district heating including the gap-bound diagnostics
:func:`~monee.model.formulation.mccormick.water.mccormick_dhs_gap_bound_mw`
and
:func:`~monee.model.formulation.mccormick.water.mccormick_dhs_gap_bound_k`.

.. automodule:: monee.model.formulation.mccormick.water
   :members:
   :undoc-members:
   :show-inheritance:

Water wiring & factories
~~~~~~~~~~~~~~~~~~~~~~~~

Prebuilt water/heat bundles and the factories
:func:`~monee.model.formulation.water.make_nl_darcy_weisbach_pwl_network_formulation`
and
:func:`~monee.model.formulation.water.make_smooth_darcy_weisbach_network_formulation`.

.. automodule:: monee.model.formulation.water
   :members:
   :undoc-members:
   :show-inheritance:
