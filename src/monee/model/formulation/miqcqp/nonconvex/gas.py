"""Exact Weymouth gas formulation: the epigraph pinned to equality
``m·m == m_sq`` - a non-convex MIQCQP for global solvers."""

from ..convex.gas import RelaxedWeymouthBranchFormulation


class ExactWeymouthBranchFormulation(RelaxedWeymouthBranchFormulation):
    """Weymouth with ``mass_flow_*_squared`` pinned exactly - no relaxation
    gap and no ε tightening term in the objective."""

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return []

    def _epigraph_eqs(self, branch):
        return [
            branch.mass_flow_pos * branch.mass_flow_pos
            == branch.mass_flow_pos_squared,
            branch.mass_flow_neg * branch.mass_flow_neg
            == branch.mass_flow_neg_squared,
        ]
