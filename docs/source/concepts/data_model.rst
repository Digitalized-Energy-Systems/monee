==========
Data model
==========

monee represents multi-energy grids as a directed graph. There are four
fundamental model types:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Type
     - Description
   * - Node
     - A connection point where energy flows meet: a bus in an electricity
       grid, or a junction in a pipe network.
   * - Branch
     - An energy transfer element connecting two nodes: an electric line, a
       gas pipe, or a water pipe.
   * - Child
     - A unit that feeds power into or draws power from a node: generators,
       loads, external grids.
   * - Grid
     - Grid-level parameters shared by all components in the same carrier
       (e.g. reference pressure, temperature, base MVA).

Each type is represented by a Component container class
(:class:`~monee.model.Node`, :class:`~monee.model.Branch`,
:class:`~monee.model.Child`). The physical behaviour lives in a *model object*
attached to the component. This separation makes it easy to swap models
without touching the topology.

.. tip::

   The :mod:`monee.express` API creates all of these objects for you. You only
   need to work with the low-level classes when implementing custom components.

----

Variables and parameters
========================

Inside a model object, decision variables are declared with
:class:`~monee.model.core.Var` and fixed parameters with
:class:`~monee.model.core.Const`. The solver replaces ``Var`` instances with
its own variable type at solve time.

.. testcode::

    from monee.model.core import Var, Const

    class MyPipeModel:
        def __init__(self):
            self.mass_flow_kgs = Var(0.0, min=-5.0, max=5.0, name="mass_flow_kgs")
            self.diameter_m = Const(0.1)

Variables in neighbouring models are accessible via the ``vars`` dictionary:

.. code-block:: python

    # Inside equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
    p_from = from_node_model.vars["pressure_pu"]

The attribute protocol
----------------------

Every public attribute of a model object, any attribute whose name does
not start with an underscore, is solver-visible state. Underscore-prefixed
attributes are private bookkeeping the solver never sees. The attribute's
*type* determines its role:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Type
     - Role
   * - :class:`~monee.model.core.Var`
     - A decision variable. Supports ``min``/``max`` bounds, an
       ``integer`` flag and an optional ``name``. The solver replaces it with
       a backend variable at solve time and writes the optimal value back to
       ``.value``.
   * - :class:`~monee.model.core.Const`
     - A pinned setpoint: a fixed value that participates in equations
       but is never optimised. :meth:`~monee.model.core.ChildModel.overwrite`
       uses ``Const`` to pin node variables for grid-forming slacks (e.g.
       :class:`~monee.model.child.ExtPowerGrid` pins ``vm_pu`` and the bus
       angle).
   * - :class:`~monee.model.core.Intermediate`
     - A solver-computed derived value. Declare how to compute it by
       returning an :class:`~monee.model.core.IntermediateEq` from
       ``equations()``; the solver evaluates the expression and stores the
       result on the named attribute (e.g. ``Bus.p_mw`` is the sum of child
       injections).
   * - :class:`~monee.model.core.PostProcess`
     - A report-only quantity computed *after* the solve, outside the
       solver (see :ref:`post-processed-values` below).
   * - plain scalar (``float``/``int``)
     - A fixed parameter (e.g. a pipe length or an efficiency). Visible
       to equations and exported to dataframes, but never a solver variable.

The ``vars`` property returns the public attributes as a dictionary; the
``values`` property additionally unwraps ``Var``/``Const``/``Intermediate``/
``PostProcess`` instances to their current numeric values.

.. warning::

   The comparison operators on :class:`~monee.model.core.Var` (``<``, ``<=``,
   ``>``, ``>=``) compare against the variable's bounds, not its value,
   and they return ``False`` when the relevant bound is ``None``. For example
   ``Var(0.5, max=1.0) < 2`` is ``True`` because ``max`` is compared, but
   ``Var(0.5) < 2`` is ``False`` because there is no upper bound. Use
   :func:`~monee.model.core.value` (or ``.value``) when you want to compare
   solved values.

.. _post-processed-values:

Post-processed report values
----------------------------

A :class:`~monee.model.core.PostProcess` attribute is a derived quantity that
exists purely for reporting. It is never injected as a solver variable and
no equation may reference it, so it carries no degrees of freedom and has zero
effect on convergence or on the squareness of a simulation solve. After the
solve, monee evaluates ``fn(values)`` with a namespace of the model's solved
fields and stores the result in ``.value``:

.. testcode::

    import monee.model as mm

    class MyJunction(mm.NodeModel):
        def __init__(self):
            self.t_pu = mm.Var(1.0, min=0.3, max=2.0, name="t_pu")
            # Report-only: evaluated after the solve from the solved values.
            self.t_k = mm.PostProcess(lambda v: v.t_pu * 356)

        def equations(self, grid, from_branch_models, to_branch_models,
                      connected_child_models, **kwargs):
            return []

Built-in models use this for quantities that previously cluttered the solver:
``Bus.va_degree`` (degrees from the ``va_radians`` decision variable),
``Junction.t_k`` (Kelvin from the per-unit temperature), and the ``pressure_pa``
report on hydraulic nodes and coupling control nodes. Prefer ``PostProcess``
over ``Intermediate`` whenever the quantity is not needed *inside* an
equation: physics stays in the solver, reporting stays outside.

----

Sign and unit conventions
=========================

monee uses the load convention everywhere: positive = consumption,
negative = generation. This holds for child setpoints, nodal balances and
result dataframes alike.

Generator-type constructors take *positive* magnitudes and apply the sign
internally; passing a negative number raises a ``ValueError``:

.. testcode::

    import monee.model as mm

    gen = mm.PowerGenerator(p_mw=0.5, q_mvar=0.0)  # pass the magnitude ...
    print(gen.p_mw)                                # ... stored with generation sign

.. testoutput::

    -0.5

The same applies to :class:`~monee.model.child.Source` (``mass_flow_kgs``) and
:class:`~monee.model.child.HeatGenerator` (``q_mw``). Storage models follow
the same convention: positive = charging (consuming from the network),
negative = discharging.

Units are encoded in attribute-name suffixes:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Suffix
     - Unit / meaning
   * - ``_mw``, ``_mvar``
     - Megawatt (active power), megavolt-ampere reactive.
   * - ``mass_flow_kgs``
     - kg/s (water and gas).
   * - ``_pu``
     - Per-unit, relative to the grid's reference values (``t_ref_k``,
       ``pressure_ref_pa`` for hydraulic grids; nominal voltage for ``vm_pu``).
   * - ``_k``
     - Kelvin.
   * - ``_m``, ``_kv``, ``_ka``
     - Metre, kilovolt, kiloampere.
   * - ``_mwh``, ``_kg``
     - Megawatt-hour and kilogram (storage state of charge).

For gas, the grid's ``higher_heating_value_kwh_per_kg`` is given in kWh/kg, so
thermal power converts as ``MW = mass_flow_kgs [kg/s] * 3.6 * hhv``.

----

Nodes
=====

Subclass :class:`~monee.model.core.NodeModel` and implement
:meth:`~monee.model.core.NodeModel.equations`. The method receives:

- ``grid``: the grid the node belongs to.
- ``from_branch_models``: models of branches *leaving* this node (this node is their from-end).
- ``to_branch_models``: models of branches *arriving at* this node (this node is their to-end).
- ``connected_child_models``: models of child components at this node.

.. testcode::

    import monee.model as mm

    @mm.model
    class MyBus(mm.NodeModel):
        def __init__(self):
            self.pressure_pu = mm.Var(1, min=0, max=2, name="pressure_pu")

        def equations(self, grid, from_branch_models, to_branch_models,
                      connected_child_models, **kwargs):
            flow_out = sum(b.vars.get("mass_flow_kgs", 0) for b in from_branch_models)
            flow_in  = sum(b.vars.get("mass_flow_kgs", 0) for b in to_branch_models)
            injected = sum(c.vars.get("mass_flow_kgs", 0) for c in connected_child_models)
            return [flow_in - flow_out + injected == 0]

.. note::

   Nodes that participate in more than one grid (for example a power-to-heat
   coupling junction) should use :class:`~monee.model.core.MultiGridNodeModel`
   as the base class instead of ``NodeModel``.

Branches
========

Subclass :class:`~monee.model.core.BranchModel` and implement
:meth:`~monee.model.core.BranchModel.equations`. Parameters:

- ``grid``: the grid the branch belongs to.
- ``from_node_model``: model of the upstream node.
- ``to_node_model``: model of the downstream node.

For multi-carrier branches (e.g. a gas-to-power unit), use
:class:`~monee.model.core.MultiGridBranchModel`.

Children
========

Subclass :class:`~monee.model.core.ChildModel` and implement
:meth:`~monee.model.core.ChildModel.equations`. Parameters:

- ``grid``: the grid the child is attached to.
- ``node``: the node the child is connected to.

Every child carries a ``regulation`` attribute in ``[0.0, 1.0]`` that scales
its setpoint (heat exchangers and coupling components have it too). It is the
hook used for partial dispatch and by the load-shedding problems, which
promote ``regulation`` to a decision variable.

Children may also implement
:meth:`~monee.model.core.ChildModel.overwrite`, which pins variables of the
attached node model to :class:`~monee.model.core.Const` setpoints. This is
how grid-forming slacks work: :class:`~monee.model.child.ExtPowerGrid` pins
bus voltage and angle, :class:`~monee.model.child.ExtHydrGrid` pins pressure
(and, unless ``pin_temperature=False``, temperature).

.. note::

   The storage models :class:`~monee.model.ElectricStorage`,
   :class:`~monee.model.GasStorage` and :class:`~monee.model.ThermalStorage`
   are child models whose state of charge couples consecutive timesteps. See
   :doc:`../how-to/storage` for their full usage.

Compounds
=========

A :class:`~monee.model.core.CompoundModel` is a composite component spanning
several nodes: the multi-energy couplers (CHP, power-to-heat, gas-to-heat)
are compounds with a hidden internal control node. Subclasses implement
``create(network, ...)``, which adds the sub-nodes, sub-branches and
sub-children to the network, and may override ``equations()`` for additional
coupling constraints.

Add a compound with :meth:`~monee.model.Network.compound`. Keyword arguments
ending in ``_id`` are resolved to :class:`~monee.model.Node` objects and
passed to ``create()`` with the suffix stripped:

.. testcode::

    import monee.model as mm

    net = mm.Network()

    power_node = net.node(mm.Bus(base_kv=1), grid=mm.EL)
    heat_node = net.node(mm.Junction(), grid=mm.WATER)
    heat_return_node = net.node(mm.Junction(), grid=mm.WATER)

    net.compound(
        mm.PowerToHeat(
            heat_energy_mw=0.1,
            diameter_m=0.1,
            temperature_ext_k=293,
            efficiency=0.9,
        ),
        power_node_id=power_node,
        heat_node_id=heat_node,
        heat_return_node_id=heat_return_node,
    )

Everything ``create()`` adds is recorded as a *subcomponent* with
``independent=False``, so subcomponents do not show up in
:meth:`~monee.model.Network.statistics` and are managed through their parent.
The resulting :class:`~monee.model.Compound` container exposes
``component_of_type(cls)`` to retrieve subcomponents by container type, and
:meth:`~monee.model.Network.deactivate` propagates recursively through all
subcomponents, calling the model's ``set_active()`` hook if present.

For compounds coupling several carriers, subclass
:class:`~monee.model.core.MultiGridCompoundModel`; like a plain
:class:`~monee.model.core.CompoundModel`, its ``equations`` still receive the
whole ``network``. Coupling can also be expressed at the branch or node level:
:class:`~monee.model.core.MultiGridBranchModel` overrides ``equations`` to
receive a ``grids`` dictionary keyed by grid type (instead of a single
``grid``), while :class:`~monee.model.core.MultiGridNodeModel` keeps the
single ``grid`` parameter, which for couplers holds a list of the
participating grids.

Model registration
==================

Decorate every custom model class with the :func:`~monee.model.core.model`
decorator (``@mm.model`` in the examples above). It registers the class in
monee's global component registry, which is required for the model to
round-trip through the native OMEF serialisation
(:func:`~monee.io.native.write_omef_network` /
:func:`~monee.io.native.load_to_network`). Models that are never serialised
work without it, but registering costs nothing, so always decorate.

----

Grids
=====

Grid objects are plain dataclasses carrying per-carrier constants. A
:class:`~monee.model.Network` registers one default grid per carrier, keyed by
``mm.EL``, ``mm.WATER`` and ``mm.GAS``. Override a default by passing a custom
instance to the constructor (``Network(el_model=...)``,
``water_model=...``, ``gas_model=...``).

- :class:`~monee.model.PowerGrid` ``(name, sn_mva=1, vm_pu_max=1.5,
  vm_pu_min=0.5)``: electrical domain. ``Network.node()`` raises a bus's
  ``vm_pu.min`` to ``grid.vm_pu_min`` whenever that bound is still ``0`` or
  unset. An exact zero lets the :math:`1/v_m` terms of the AC current equations
  blow up, so the floor keeps the NLP well-conditioned while still allowing
  stressed operating points.
- :class:`~monee.model.WaterGrid` ``(name, ...)``: water/heat domain with
  reference temperature ``t_ref_k`` and pressure ``pressure_ref_pa`` (the bases
  for ``_pu`` quantities). ``v_max_mps`` (default 5 m/s) caps each pipe's
  mass flow via the pipe cross-section; it can only tighten the generic
  ``max_mass_flow_kgs`` bound.
- :class:`~monee.model.GasGrid`: gas domain; construct via
  :func:`~monee.model.create_gas_grid` ``(name, type="lgas")``, which fills
  in the physical constants for the chosen gas type (including
  ``higher_heating_value_kwh_per_kg`` in kWh/kg).

----

Assembling a network
====================

Use :class:`~monee.model.Network` to assemble the graph:

.. testcode::

    import monee.model as mm

    net = mm.Network(el_model=mm.PowerGrid(name="power", sn_mva=1))

    child_id  = net.child(mm.PowerLoad(p_mw=mm.Const(0.1), q_mvar=mm.Const(0.0)))
    node_id   = net.node(mm.Bus(base_kv=1), child_ids=[child_id], grid=mm.EL)
    node_id_2 = net.node(mm.Bus(base_kv=1), grid=mm.EL)
    net.branch(mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
               from_node_id=node_id, to_node_id=node_id_2, grid=mm.EL)

.. tip::

   For most use cases, prefer the :mod:`monee.express` API over this low-level
   interface. The express functions set sensible defaults and handle multi-energy
   bookkeeping automatically.

Network conveniences
--------------------

Beyond the builder methods, :class:`~monee.model.Network` offers:

- ``activate(component)`` / ``deactivate(component)`` (and the ``*_by_id``
  variants): switch components in and out of the solve; compound-aware, i.e.
  deactivating a compound deactivates all of its subcomponents.
- Per-component ``constraints=`` keyword on ``node()``/``branch()``/
  ``child()``/``compound()``: a list of callables receiving the model plus
  the same context as ``equations()``; plus network-level
  ``constraint(fn)`` and ``objective(fn)`` for global constraints and
  objective terms.
- ``auto_node_creator`` / ``auto_grid_key`` on ``child()`` and ``branch()``:
  create missing nodes on the fly (used heavily by the IO importers).
- ``as_dataframe_dict()`` / ``as_result_dataframe_dict()``: one
  :class:`pandas.DataFrame` per model class with the input parameters and the
  solved values, respectively.
- ``copy()``: a fast custom deep copy. Formulations, constraints and
  objectives are shared *by reference*, so they must be stateless.
- :func:`~monee.model.transform_network` and
  :func:`~monee.model.to_spanning_tree`: apply a networkx graph transform to
  a copy of the network, pruning orphaned children and compounds.
