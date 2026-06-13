r"""Exact Weymouth gas formulation: the epigraph pinned to equality
:math:`m \cdot m = m_{sq}` - a non-convex MIQCQP for global solvers."""

from ..convex.gas import RelaxedWeymouthBranchFormulation


class ExactWeymouthBranchFormulation(RelaxedWeymouthBranchFormulation):
    r"""Weymouth with ``mass_flow_*_squared`` pinned exactly - no relaxation
    gap and no :math:`\varepsilon` tightening term in the objective."""

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return []

    def _epigraph_eqs(self, branch):
        return [
            branch.mass_flow_pos_kgs * branch.mass_flow_pos_kgs
            == branch.mass_flow_pos_kgs_squared,
            branch.mass_flow_neg_kgs * branch.mass_flow_neg_kgs
            == branch.mass_flow_neg_kgs_squared,
        ]
