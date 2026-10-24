r"""Exact branch-flow electricity formulation: the MISOCP with the SOC pinned
to equality :math:`P^2 + Q^2 = (W/tap^2) \cdot ell` - a non-convex MIQCQP for global
solvers (SCIP, Gurobi). It is the angle-relaxed branch-flow model: exact AC on
radial networks, while on a meshed network the cycle consistency of the
voltage angles is still dropped."""

from monee.model.phys.misoc.pf import soc_eq

from ..convex.el import (
    MISOCPElectricityBranchFormulation,
    MISOCPElectricityNodeFormulation,
)

# The node side is identical to the convex variant (W decision var + vm report).
ExactBranchFlowNodeFormulation = MISOCPElectricityNodeFormulation


class ExactBranchFlowBranchFormulation(MISOCPElectricityBranchFormulation):
    r"""Branch-flow model with the SOC constraint as an equality - exact AC
    power flow in the :math:`(P, Q, W, ell)` variables on a radial network; a
    meshed one keeps the angle relaxation (no cycle constraint)."""

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # The equality pins ell; no loss-based tightening incentive needed.
        return []

    def _soc_constraints(self, branch, from_node_model, p_series_pu, q_series_pu, tap):
        return [
            soc_eq(
                from_node_model.vars["vm_pu_squared"],
                p_series_pu,
                q_series_pu,
                branch.current_pu_squared,
                tap=tap,
            ),
        ]
