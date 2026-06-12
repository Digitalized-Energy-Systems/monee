"""Exact branch-flow electricity formulation: the MISOCP with the SOC pinned
to equality ``P² + Q² == (W/tap²)·ell`` - a non-convex MIQCQP for global
solvers (SCIP, Gurobi)."""

from monee.model.phys.misoc.pf import soc_eq

from ..convex.el import (
    MISOCPElectricityBranchFormulation,
    MISOCPElectricityNodeFormulation,
)

# The node side is identical to the convex variant (W decision var + vm report).
ExactBranchFlowNodeFormulation = MISOCPElectricityNodeFormulation


class ExactBranchFlowBranchFormulation(MISOCPElectricityBranchFormulation):
    """Branch-flow model with the SOC constraint as an equality - exact AC
    power flow in the (P, Q, W, ell) variables, no relaxation gap."""

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        # The equality pins ell; no loss-based tightening incentive needed.
        return []

    def _soc_constraints(self, branch, grid, from_node_model, tap):
        return [
            soc_eq(
                from_node_model.vars["vm_pu_squared"],
                branch.vars["p_from_mw"] / grid.sn_mva,
                branch.vars["q_from_mvar"] / grid.sn_mva,
                branch.current_pu,
                tap=tap,
            ),
        ]
