class NetworkConstraint:
    """
    Solver-agnostic network-level constraint extension.

    Analogous to ``BranchFormulation`` / ``NodeFormulation`` but spanning the
    entire network.  Register with ``network.add_extension(constraint)``.

    Phase 1 — ``prepare(network)``: called *before* variable injection; add
    ``Var`` placeholders to model objects so the injection loop picks them up.

    Phase 2 — ``equations(network, ignored_nodes) → list``: called *after*
    variable injection; return relational expressions (``==``, ``<=``,
    ``>=``) built from injected model attributes.  The solver registers them
    with ``m.Equations(eqs)`` / ``pm.cons.add`` without inspecting their
    content — exactly like branch/node equations.
    """

    def prepare(self, network) -> None:
        """Add Var placeholders before variable injection (no-op by default)."""

    def equations(self, network, ignored_nodes: set) -> list:
        """Return solver-agnostic relational expressions (empty by default)."""
        return []


class Formulation:
    def ensure_var(self, model):
        pass


class BranchFormulation(Formulation):
    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return []

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return []


class NodeFormulation(Formulation):
    def minimize(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        return []

    def equations(
        self,
        node,
        grid,
        from_branch_models,
        to_branch_models,
        connected_child_models,
        **kwargs,
    ):
        return []


class CompoundFormulation(Formulation):
    def minimize(self, compound, network, **kwargs):
        return []

    def equations(self, compound, network, **kwargs):
        return []


class ChildFormulation(Formulation):
    def minimize(self, child, grid, node, **kwargs):
        return []

    def equations(self, child, grid, node, **kwargs):
        return []

    def overwrite(self, child, node_model, grid):
        pass


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
