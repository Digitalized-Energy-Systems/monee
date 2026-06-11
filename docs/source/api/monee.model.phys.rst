monee.model.phys
================

:mod:`monee.model.phys` is the **pure-math layer** of monee: stateless,
solver-agnostic functions that take plain variables or expressions and return
the relational expressions assembled by the formulation classes in
:mod:`monee.model.formulation`. The modules contain no classes, no solver
imports and no model state; backend-specific operations are injected as
``*_impl`` callables (e.g. ``cos_impl``, ``log_impl``, ``sqrt_impl``,
``pwl_impl``), so the same physics runs on GEKKO, Pyomo or native monee
variables.

Constants
---------

.. automodule:: monee.model.phys.constant
   :members:
   :undoc-members:

Pipe hydraulics
---------------

Carrier-independent pipe hydraulics shared by the gas, water and heat models:
Reynolds number, friction correlations, piecewise-linear breakpoint generation
and sizing helpers.

.. automodule:: monee.model.phys.core.hydraulics
   :members:
   :undoc-members:

AC power flow
-------------

.. automodule:: monee.model.phys.nonlinear.ac
   :members:
   :undoc-members:

Gas flow (Weymouth)
-------------------

.. automodule:: monee.model.phys.nonlinear.gf
   :members:
   :undoc-members:

Water flow (Darcy–Weisbach)
---------------------------

.. automodule:: monee.model.phys.nonlinear.wf
   :members:
   :undoc-members:

Heat flow
---------

.. automodule:: monee.model.phys.nonlinear.hf
   :members:
   :undoc-members:

Smooth binary-free hydraulics
-----------------------------

Smooth hydraulic primitives for pure-NLP (IPOPT/APOPT) solves: a single signed
mass flow with ``|m| ≈ √(m² + ε²)`` replaces the direction binary and the
positive/negative flow split.

.. automodule:: monee.model.phys.nonlinear.smooth
   :members:
   :undoc-members:

MISOCP branch flow
------------------

.. automodule:: monee.model.phys.misoc.pf
   :members:
   :undoc-members:

Islanding connectivity
----------------------

Single-commodity connectivity-flow constraints used for island detection and
energization.

.. automodule:: monee.model.phys.islanding
   :members:
   :undoc-members:
