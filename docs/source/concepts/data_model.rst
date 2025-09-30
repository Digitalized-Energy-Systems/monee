==========
Data model
==========

In monee there is a graph-based data model to represent the multi-energy grid. There are four different types of models

- **Nodes** represent the buses and junctions. Every time energy flows cross it can be represented as a node.
- **Branches** represent energy transfer lines, like electric lines or pipelines.
- **Childs** represent units feeding power in/out.
- **Grids** represent grid types and characteristics to parameterize different energy grids (like electric or hydraulic grids)

In monee there these model types are implemented as grid **Component**. For each type there is one container/component class :class:`monee.model.Branch`, :class:`monee.model.Node`, :class:`monee.model.Child`, which represent the underlying component of the grid. There are also grid models (:class:`monee.model.Grid`), which can be assigned per component.

To implement the physicial properties of a component each component class consists, besides the meta data (such as id, name, location) of a specific model object (ChildModel, NodeModel, ...). These model objects can easily be replaced and reimplemented to account for individual requirements.

In the following the base classes to implement custom component models

Nodes
------

To implement custom nodes, the :class:`monee.model.NodeModel` must be used. To implement the models' behavior the method :meth:`monee.model.NodeModel.equations` must be implemented.

The equations method has several important parameters:

- grid: the grid to which the node has been assigned
- in_branch_models: the branch models going in that node
- out_branch_models: the branch models going out from that node
- childs: the childs connected to the node.

To the variables in neighbor models (child or branch models) can be accessed using `model.vars["variable_name"]`.

Note that, nodes which are part of multiple grids should use :class:`monee.model.MultiGridNodeModel` as base class.

Branches
---------

For branches the classes :class:`monee.model.BranchModel` and for multi-energy branches, :class:`monee.model.MultiGridBranchModel` can be used.

Branch models work similar to node models, the most important difference lies in the `equations` method, which has (besides the grid) the following parameters

- from_node_model: node model from which the branch starts
- to_node_model: node model to which the branch ends

Childs
------

For child models the class :class:`monee.model.ChildModel` can be used.
