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

The wrapping persists on a solved network: on the ``result.network``
returned by :func:`monee.run_energy_flow` and
:func:`monee.run_energy_flow_optimization`, every attribute that entered the
solve as a decision variable is still a ``Var``, with the optimum written to
its ``.value``; an attribute the problem never promoted stays a plain
number, so the same attribute name can be a float on one component and a
``Var`` on another. A ``Var`` neither formats nor casts as a number
(``f"{v:.2f}"`` and ``float(v)`` both raise ``TypeError``), so read
``v.value``, or call :func:`monee.model.value`, which unwraps
``Var``/``Const``/``Intermediate`` and passes plain numbers through
unchanged. The result dataframes (``result.dataframes`` and
``result.get(ModelType)``) contain the unwrapped floats already, so prefer
them for reporting.

.. note::

   A model constructor only understands the keywords its own signature
   declares. Any other keyword is not model state: it is neither solved nor
   exported, so passing one (a typo, or an option the model does not have)
   emits a warning naming the ignored keys and will become an error in a
   future release. To attach your own quantity to a component, declare it as
   an attribute on a model subclass rather than passing it as a keyword.

.. _settable-attributes:

Which attributes are settable after construction
------------------------------------------------

A model's public attributes stay writable after it is built, and the solver
reads whatever value is on the object when the solve starts. Three groups
behave differently, so the group decides what a write actually does.

Settable, and read by the equations
   Every attribute the model declares in its ``__init__``, plus any parameter
   its constructor accepts even when the default left it unset (a
   :class:`~monee.model.child.PowerGenerator` created without ``cost`` still
   takes ``model.cost = 12.0`` for the economic dispatch objective). Writing a
   plain number here is the normal way to change a setpoint or a parameter
   between solves::

       gen = net.child_by_id(gen_id).model
       gen.p_mw = -0.8          # generation, load convention
       gen.cost = 12.0

   On a model whose attribute is already a :class:`~monee.model.core.Var`
   (a slack, a promoted control variable), assign to ``model.attr.value``
   instead: replacing the ``Var`` with a float removes the decision variable
   from the model.

Baked into ``Var`` bounds at build time
   A constructor argument that only sizes a variable is copied into that
   variable's bounds when the model is built, and the bounds are what the
   solver enforces. Rewriting the argument later moves nothing, because
   nothing reads it again. An
   :class:`~monee.model.storage.ElectricStorage` built with ``e_mwh_max=2``
   carries that limit as ``e_mwh.max``, so a bigger battery is::

       storage.e_mwh.max = 4.0

   Where the sizing argument is a documented capacity, the model exposes a
   property that rewrites the bounds for you, for example
   :attr:`~monee.model.extension.islanding.el.GridFormingGenerator.p_mw_max`,
   which is settable and re-bounds ``p_mw`` symmetrically. Everything else
   needs the ``Var`` itself. :doc:`../how-to/diagnose_infeasibility` explains
   how to see the bounds a solve actually used.

Private metadata, invisible to the solver
   Any name starting with an underscore is yours. It is skipped by ``vars``
   and ``values``, never injected as a variable and never exported to a result
   frame, which makes it the place for study bookkeeping::

       load._priority_weight = 3
       load._feeder = "F12"

Assigning a plain value to any other public name creates an inert field: it is
stored on the object, it shows up in ``vars``, and no equation ever reads it,
so the solve returns a perfectly clean result computed without it. monee warns
on that write with
:class:`~monee.model.core.UnknownAttributeWarning`, naming the attribute, the
model class and the attributes that model does have. ``gen.p_mw_maximum = 5.0``
reports:

.. code-block:: text

   UnknownAttributeWarning: PowerGenerator does not declare the attribute
   'p_mw_maximum': the assignment stores an inert field that no equation reads,
   so the solve silently ignores it. Settable attributes of this model:
   ['cost', 'p_mw', 'q_mvar', 'regulation']. Use a leading underscore
   (_p_mw_maximum) for your own metadata.

Assigning a :class:`~monee.model.core.Var`, :class:`~monee.model.core.Const`,
:class:`~monee.model.core.Intermediate` or
:class:`~monee.model.core.PostProcess` to a new name is how a formulation
declares extra solver state, so it is always accepted.

:func:`~monee.model.core.set_attribute_guard` selects what happens on such a
write, and the environment variable ``MONEE_ATTRIBUTE_GUARD`` sets the same
mode without touching the code:

.. list-table::
   :header-rows: 1
   :widths: 16 84

   * - Mode
     - Effect
   * - ``warn``
     - Default. Emits :class:`~monee.model.core.UnknownAttributeWarning`, which
       the standard :mod:`warnings` filters silence or promote per module.
   * - ``error``
     - Strict mode: raises ``AttributeError`` at the assignment, so a study or
       a CI job fails on the typo instead of publishing the number computed
       without it.
   * - ``off``
     - Removes the hook from the model base class, restoring plain attribute
       writes. Use it only for a hot loop that assigns millions of attributes.

If your own subclass or tooling writes a public attribute the class cannot
declare in ``__init__``, register it once with
:func:`~monee.model.core.register_optional_attributes` and the guard stays
quiet about that name on that class. A :class:`~monee.model.core.CompoundModel`
may record the ids and sizing it derives during
:meth:`~monee.model.core.CompoundModel.create` without registering anything:
``create`` is a second construction phase and is exempt for its duration.

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

.. _data-model-slack-sign:

Sign of the slack
-----------------

External grids are children too, so they follow the same rule seen from the
network: :attr:`~monee.model.child.ExtPowerGrid.p_mw` is positive when the
network exports into the external grid and negative when it imports. A
network that draws 1 MW from its substation reports ``p_mw = -1.0`` (plus
losses). Anything that bounds the slack therefore reads "import" as the lower
bound: ``max_import_mw=X`` becomes ``p_mw >= -X``, a hand-written constraint
capping the import at 0.6 MW is ``model.p_mw >= -0.6``, and the
``bounds_ext_el`` tuple of
:func:`~monee.problem.min_load_shedding.create_min_load_shedding_problem` caps
the import at ``X`` MW with ``bounds_ext_el=(-X, 0)``. The same holds for
:class:`~monee.model.child.ExtHydrGrid` and its ``mass_flow_kgs``. Writing the
bound the other way round does not fail, it silently forbids importing at all.

Sign on branches
----------------

Branches are pairwise, so their sign says which way the flow runs rather than
whether it is consumption, and the two branch families do not agree.

Hydraulic branches (:class:`~monee.model.branch.GasPipe`,
:class:`~monee.model.branch.WaterPipe`, pumps, valves and heat exchangers)
split the flow into two non-negative variables and report
``mass_flow_kgs = mass_flow_pos_kgs - mass_flow_neg_kgs``. Flow from the
from-node to the to-node lives in ``mass_flow_neg_kgs``, so ``mass_flow_kgs``
(and ``velocity_mps``, which shares its sign) is negative for from-node to
to-node flow and positive for the reverse. A pipe feeding a sink downstream of
the external grid therefore reports a negative mass flow. This is what the
outflow-positive nodal balance expects; take the magnitude with ``abs()`` when
reporting throughput, and read the direction off the sign only.

Power branches use the opposite convention. ``p_from_mw`` and ``p_to_mw`` are
both the power flowing into the branch at their end, so ``p_from_mw`` is
positive for from-node to to-node flow while ``p_to_mw`` is negative, and the
two add up to the branch losses. A 1 MW load fed over one line reports
``p_from_mw = 1.0825``, ``p_to_mw = -1.0``, hence 0.0825 MW of loss.

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

Branch author contract
----------------------

Formulations and the nodal balances look up branch variables by name, so a
custom branch has to declare a small set of expected attributes:

- Call ``super().__init__()`` in your ``__init__``. It provides ``on_off``
  (default 1, in service), which the nodal balances multiply into every flow
  term of the branch.
- A branch between electrical buses declares the terminal Vars ``p_from_mw``,
  ``q_from_mvar``, ``p_to_mw`` and ``q_to_mvar``; the bus power balance reads
  them unconditionally.
- A branch between hydraulic junctions (water or gas) declares the
  non-negative flow split ``mass_flow_pos_kgs`` and ``mass_flow_neg_kgs``.
  Flow from the from node to the to node is carried by ``mass_flow_neg_kgs``
  and ``mass_flow_pos_kgs`` carries the reverse direction, so the signed
  report ``mass_flow_kgs = mass_flow_pos_kgs - mass_flow_neg_kgs`` is
  negative for forward flow.
- Heat participation is opt in: a hydraulic branch that also declares
  ``t_from_pu`` and ``t_to_pu`` enters the nodal heat balance.
- Every other name is optional; the balances skip keys a branch does not
  declare.
- When a :class:`~monee.model.formulation.core.BranchFormulation` is attached
  to the branch type, the model's own ``equations()`` is still called and both
  sets of equations are collected. A formulated model should return ``[]``
  from ``equations()`` (or only equations the formulation does not emit).

A minimal working gas branch, a linear valve with a fixed forward direction:

.. testcode::

    import monee
    import monee.model as mm
    from monee import mx

    @mm.model
    class LinearValve(mm.BranchModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.mass_flow_pos_kgs = mm.Var(0.0, min=0, name="mass_flow_pos_kgs")
            self.mass_flow_neg_kgs = mm.Var(0.1, min=0, name="mass_flow_neg_kgs")

        def equations(self, grid, from_node_model, to_node_model, **kwargs):
            return [
                self.mass_flow_pos_kgs == 0,
                from_node_model.pressure_squared_pu
                - to_node_model.pressure_squared_pu
                == 0.01 * self.mass_flow_neg_kgs,
            ]

    net = mm.Network()
    gas = mx.gas_structure(net, diameter_m=0.3, length_m=500)
    seg = gas.line(3, sink_mass_flow=0.05)
    gas.attach_ext_grid(seg.first)
    net.remove_branch_between(seg.nodes[1], seg.nodes[2])
    net.branch(LinearValve(), seg.nodes[1], seg.nodes[2])

    result = monee.run_energy_flow(net, formulation="smooth_nlp")
    row = result.get(LinearValve).iloc[0]
    print(f"valve flow: {row.mass_flow_neg_kgs:.3f} kg/s")

.. testoutput::

    valve flow: 0.050 kg/s

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
    # Water flows heat_node -> heat_return_node through the internal heat
    # exchanger and is heated on the way, so heat_return_node is the hot outlet.
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
  in the physical constants for the chosen gas type. Both types share the
  operating point ``t_k=300`` K, ``t_ref_k=356`` K and ``pressure_ref_pa=1e6``,
  that is 10 bar absolute, which is the base ``pressure_pu`` is relative to.
  They differ in the fluid: ``lgas`` (the default) has
  ``molar_mass=0.0181138902`` kg/mol and
  ``higher_heating_value_kwh_per_kg=11.79011`` kWh/kg, ``methane`` has
  ``0.0165`` kg/mol and ``15.3`` kWh/kg; the compressibility is derived from
  the two and is 0.9736 for lgas. To run a network at a different operating
  pressure, build the grid with that pressure rather than editing the field
  afterwards, so the derived compressibility follows:
  ``mm.create_gas_grid("mp", pressure_ref_pa=1.5e5)``. Pass
  ``pressure_ambient_pa=mm.STANDARD_ATMOSPHERE_PA`` to state node pressures as
  gauge instead of absolute. ``pressure_squared_pu_min`` /
  ``pressure_squared_pu_max`` (0.7 to 1.3 by default) are the operational
  pressure limits of the junctions on the grid. They are stated in SQUARED
  per unit, the state the gas formulations solve in, so a 0.9 to 1.1 pressure
  band is ``0.81`` to ``1.21``. An optimization run applies them to every
  junction ``pressure_squared_pu`` variable whose bound the model has not
  already overridden; a plain simulation keeps the wide 0 to 3 range, so a
  square solve is never cut off by an operational limit. To tighten the limits
  for one problem only, use
  ``prob.bounds((p_min ** 2, p_max ** 2), attributes=["pressure_squared_pu"])``
  or :func:`~monee.problem.create_min_load_shedding_problem`
  ``(bounds_pressure=...)``.

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

Component ids
-------------

``node()``, ``child()`` and ``compound()`` return integer ids drawn from
independent counters per container kind, so the same integer routinely names
both a node and a child: above, the load is child ``0`` while the first bus
is node ``0``. Keep external references such as co-simulation wiring tables
as ``(kind, id)`` pairs rather than bare integers, and use the kind argument
of the id-addressed APIs (the first argument of
``activate_by_id``/``deactivate_by_id``, or ``component_type=`` on the
stepper) to disambiguate; :doc:`../how-to/stepper` documents the resolution
rules for the stepper's id forms.

A branch has no counter of its own: its id is the tuple
``(from_node_id, to_node_id, key)``, where ``key`` counts parallel branches
between the same node pair starting at ``0``. The ``id`` column of a branch
result dataframe holds exactly this tuple, so a row's endpoints are recovered
by unpacking it (``from_id, to_id, key = row["id"]``). Child result frames
carry their attachment point in a separate ``node_id`` column.

Result queries are id-addressed too, and they see the same collision: an
external grid attached to the first bus is child ``0`` while that bus is node
``0``, so ``get_result_for_id(0, "p_mw")`` matches both the ``Bus`` and the
``ExtPowerGrid`` table. The lookup then raises a ``ValueError`` naming the
candidate tables instead of guessing. Pass the model class to pick one
(``get_result_for_id`` is the timeseries and multi period form, ``result[id,
ModelType]`` the snapshot one):

.. code-block:: python

    p_ext = result.get_result_for_id(ext_id, "p_mw", mm.ExtPowerGrid)
    row   = result[ext_id, mm.ExtPowerGrid]

The gas side works the same way, with :class:`~monee.model.child.ExtHydrGrid`
against :class:`~monee.model.node.Junction`:

.. code-block:: python

    m_ext = result.get_result_for_id(gas_ext_id, "mass_flow_kgs", mm.ExtHydrGrid)

Passing ``model_type`` is always safe, so the how-to examples use it wherever
the queried id belongs to a child.

.. _result-frame-index:

Where the id lives in a result frame
------------------------------------

A snapshot result frame (``result.dataframes[...]`` and
``result.get(ModelType)``, from :func:`monee.run_energy_flow` and
:func:`monee.run_energy_flow_optimization`) is indexed positionally: its index
is a plain ``RangeIndex`` counting rows, and the component id sits in the ``id``
column. Node ids skip numbers as soon as a network mixes node types (buses and
junctions draw from one counter but land in separate frames), so the row number
and the node id part company on the first multi energy network, and
``df.loc[node_id]`` then reads a different component without any error. Select
rows by id in one of two ways:

.. code-block:: python

    buses = result.get(mm.Bus)
    row   = buses[buses["id"] == node_id].iloc[0]   # mask on the id column

    by_id = result.get(mm.Bus, index="id")          # rows indexed by id
    row   = by_id.loc[node_id]

``index="id"`` returns a copy of the same frame with the same columns (the
``id`` column is kept) under an index built from the ids, so it joins directly
against an id map such as ``net.pp_bus_to_node``; see
:doc:`../how-to/convert_from_pandapower`. Only the view is new: the stored
frame and the default ``get(ModelType)`` keep the positional index they always
had. For a branch frame, whose ids are tuples, the id-indexed view is a
``MultiIndex`` with the levels ``from_node_id``, ``to_node_id`` and ``key``, so
``by_id.loc[(4, 7, 0)]`` returns the row and ``by_id.loc[4]`` returns every
branch leaving node 4. A tuple id cannot be used as a label on a plain index at
all: ``buses.set_index("id").loc[(4, 7, 0)]`` raises an ``IndexingError``,
because pandas reads the tuple as one key per axis.

Timeseries and multi period results turn this around. There a frame spans the
run, so rows are steps or periods and the columns are the component ids:
``result.get_result_for(mm.Bus, "vm_pu")[node_id]`` is a column lookup, which
is why ``[node_id]`` works there and ``loc[node_id]`` misleads on a snapshot.
:doc:`../how-to/timeseries` covers the timeseries side.

Names and other container attributes
------------------------------------

The ``name=`` keyword of the ``create_*`` functions, and of
``node()``/``branch()``/``child()``, lands on the Component container, not on
the model object. Bookkeeping lives on the container (``id``, ``name``,
``grid``, ``active``, ``node_id`` for a child, ``from_node_id`` and
``to_node_id`` for a branch, ``position`` for a node); the physics lives on
``.model`` (``p_mw``, ``vm_pu``, ``mass_flow_kgs``, and the other ``Var`` and
``Const`` attributes). ``component.model.name`` therefore raises an
``AttributeError``: a model object has no name of its own.

.. testcode::

    import monee.express as mx

    net_named = mx.create_multi_energy_network()
    bus = mx.create_bus(net_named)
    load_id = mx.create_power_load(net_named, bus, p_mw=0.1, q_mvar=0.0,
                                   name="hp_feeder")

    load = net_named.child_by_id(load_id)
    print(load.name, type(load.model).__name__)

.. testoutput::

    hp_feeder PowerLoad

A result frame carries the container ``name`` in a ``name`` column next to
``id`` as soon as one component of that type was named, so a solved run can be
read back by name as well. Names are also the key used by
``TimeseriesData.add_child_series_by_name`` and its branch and compound
siblings (see :doc:`../how-to/timeseries`).

Network conveniences
--------------------

Beyond the builder methods, :class:`~monee.model.Network` offers:

- ``activate(component)`` / ``deactivate(component)`` (and the ``*_by_id``
  variants): switch components in and out of the solve; compound-aware, i.e.
  deactivating a compound deactivates all of its subcomponents. The first
  argument of ``activate_by_id`` / ``deactivate_by_id`` is the kind of the
  component, either a container class (``mm.Node``, ``mm.Branch``,
  ``mm.Child``, ``mm.Compound``) or a model class carried by one of them
  (``mm.PowerLine``, ``mm.PowerLoad``); anything else raises a ``ValueError``.
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
