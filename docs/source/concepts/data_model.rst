==========
Data model
==========

monee represents multi-energy grids as a **directed graph**. There are four
fundamental model types:

.. list-table::
   :header-rows: 1
   :widths: 15 85

   * - Type
     - Description
   * - **Node**
     - A connection point where energy flows meet — a bus in an electricity
       grid, or a junction in a pipe network.
   * - **Branch**
     - An energy transfer element connecting two nodes — an electric line, a
       gas pipe, or a water pipe.
   * - **Child**
     - A unit that feeds power into or draws power from a node — generators,
       loads, external grids.
   * - **Grid**
     - Grid-level parameters shared by all components in the same carrier
       (e.g. reference pressure, temperature, base MVA).

Each type is represented by a **Component** container class
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
            self.mass_flow = Var(0.0, min=-5.0, max=5.0, name="mass_flow")
            self.diameter_m = Const(0.1)

Variables in neighbouring models are accessible via the ``vars`` dictionary:

.. code-block:: python

    # Inside equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
    p_from = from_node_model.vars["pressure_pu"]

----

Nodes
=====

Subclass :class:`~monee.model.core.NodeModel` and implement
:meth:`~monee.model.core.NodeModel.equations`. The method receives:

- ``grid`` — the grid the node belongs to.
- ``from_branch_models`` — models of branches *arriving at* this node.
- ``to_branch_models`` — models of branches *leaving* this node.
- ``connected_child_models`` — models of child components at this node.

.. testcode::

    import monee.model as mm

    @mm.model
    class MyBus(mm.NodeModel):
        def __init__(self):
            self.pressure_pu = mm.Var(1, min=0, max=2, name="pressure_pu")

        def equations(self, grid, from_branch_models, to_branch_models,
                      connected_child_models, **kwargs):
            flow_in  = sum(b.vars.get("mass_flow", 0) for b in from_branch_models)
            flow_out = sum(b.vars.get("mass_flow", 0) for b in to_branch_models)
            injected = sum(c.vars.get("mass_flow", 0) for c in connected_child_models)
            return [flow_in - flow_out + injected == 0]

.. note::

   Nodes that participate in more than one grid — for example a power-to-heat
   coupling junction — should use :class:`~monee.model.core.MultiGridNodeModel`
   as the base class instead of ``NodeModel``.

Branches
========

Subclass :class:`~monee.model.core.BranchModel` and implement
:meth:`~monee.model.core.BranchModel.equations`. Parameters:

- ``grid`` — the grid the branch belongs to.
- ``from_node_model`` — model of the upstream node.
- ``to_node_model`` — model of the downstream node.

For multi-carrier branches (e.g. a gas-to-power unit), use
:class:`~monee.model.core.MultiGridBranchModel`.

Children
========

Subclass :class:`~monee.model.core.ChildModel` and implement
:meth:`~monee.model.core.ChildModel.equations`. Parameters:

- ``grid`` — the grid the child is attached to.
- ``node`` — the node the child is connected to.

----

Assembling a network
====================

Use :class:`~monee.model.Network` to assemble the graph:

.. testcode::

    import monee.model as mm

    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))

    child_id  = net.child(mm.PowerLoad(p_mw=mm.Const(0.1), q_mvar=mm.Const(0.0)))
    node_id   = net.node(mm.Bus(base_kv=1), child_ids=[child_id], grid=mm.EL)
    node_id_2 = net.node(mm.Bus(base_kv=1), grid=mm.EL)
    net.branch(mm.PowerLine(length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4, parallel=1),
               from_node_id=node_id, to_node_id=node_id_2, grid=mm.EL)

.. tip::

   For most use cases, prefer the :mod:`monee.express` API over this low-level
   interface. The express functions set sensible defaults and handle multi-energy
   bookkeeping automatically.
