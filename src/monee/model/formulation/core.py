class Formulation:
    def ensure_var(self, model, simulation=False, grid=None):
        """Declare/adjust the model's Vars for this formulation.

        ``grid`` is the component's grid (None for grid-less components); it
        lets formulations express grid-derived bounds on the Var abstraction
        itself instead of mutating backend variables in ``equations()``.
        """


class BranchFormulation(Formulation):
    def minimize(
        self, branch, grid, from_node_model, to_node_model, **kwargs
    ):  # NOSONAR
        return []

    def equations(
        self, branch, grid, from_node_model, to_node_model, **kwargs
    ):  # NOSONAR
        return []


class NodeFormulation(Formulation):
    def minimize(
        self,
        node,  # NOSONAR
        grid,  # NOSONAR
        from_branch_models,  # NOSONAR
        to_branch_models,  # NOSONAR
        connected_child_models,  # NOSONAR
        **kwargs,
    ):
        return []

    def equations(
        self,
        node,  # NOSONAR
        grid,  # NOSONAR
        from_branch_models,  # NOSONAR
        to_branch_models,  # NOSONAR
        connected_child_models,  # NOSONAR
        **kwargs,
    ):
        return []


class CompoundFormulation(Formulation):
    def minimize(self, compound, network, **kwargs):  # NOSONAR
        return []

    def equations(self, compound, network, **kwargs):  # NOSONAR
        return []


class ChildFormulation(Formulation):
    """``child`` is the child model and ``node`` its parent node *model* (e.g. a
    Bus), so ``node.vars[...]`` reads the node's solver variables directly -
    consistent with the node-model arguments handed to Branch/Node formulations."""

    def minimize(self, child, grid, node, **kwargs):  # NOSONAR
        return []

    def equations(self, child, grid, node, **kwargs):  # NOSONAR
        return []

    def overwrite(self, child, node_model, grid):
        """Hook for subclasses to overwrite child model state; no-op by default."""


def _or_dict(d: dict):
    return {} if d is None else d


class NetworkFormulation:
    branch_type_to_formulations: dict[tuple[type, type], BranchFormulation]
    node_type_to_formulations: dict[tuple[type, type], NodeFormulation]
    child_type_to_formulations: dict[tuple[type, type], ChildFormulation]
    compound_type_to_formulations: dict[tuple[type, type], CompoundFormulation]

    def __init__(
        self,
        branch_type_to_formulations=None,
        node_type_to_formulations=None,
        child_type_to_formulations=None,
        compound_type_to_formulations=None,
    ):
        self.branch_type_to_formulations = _or_dict(branch_type_to_formulations)
        self.node_type_to_formulations = _or_dict(node_type_to_formulations)
        self.child_type_to_formulations = _or_dict(child_type_to_formulations)
        self.compound_type_to_formulations = _or_dict(compound_type_to_formulations)

    def items(self):
        """All ``(model_type | (model_type, grid_type), formulation)`` pairs."""
        return (
            list(self.branch_type_to_formulations.items())
            + list(self.node_type_to_formulations.items())
            + list(self.child_type_to_formulations.items())
            + list(self.compound_type_to_formulations.items())
        )

    def lookup(self, model, grid) -> Formulation | None:
        """The registered formulation matching *model* (and *grid*), or None.

        Mirrors :meth:`monee.model.network.Network.apply_formulation` matching:
        ``isinstance`` on the model type, exact type match on the grid when the
        key is a ``(model_type, grid_type)`` tuple; on multiple matches the
        last registration wins.
        """
        found = None
        for type_or_tuple, formulation in self.items():
            if isinstance(type_or_tuple, tuple):
                tc, tg = type_or_tuple
            else:
                tc, tg = type_or_tuple, None
            if isinstance(model, tc) and (tg is None or type(grid) is tg):
                found = formulation
        return found
