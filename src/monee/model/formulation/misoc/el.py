import math

from monee.model.phys.misoc.pf import voltage_drop, soc_rel, active_power_loss, reactive_power_loss, gap_expr
from monee.model.core import Var, Intermediate, IntermediateEq
from monee.model.formulation.core import BranchFormulation, NodeFormulation

class MISOCPElectricityNodeFormulation(NodeFormulation):
    
    def ensure_var(self, node):
        node.vm_pu_squared = Var(1, min=0)
        node.vm_pu = Intermediate(1)

    def equations(self, node, grid, from_branch_models, to_branch_models, connected_node_models, **kwargs):
        if type(node.vm_pu) != Intermediate:
            return [node.vm_pu ** 2 == node.vm_pu_squared]
        return [
            IntermediateEq("vm_pu", kwargs["sqrt_impl"](node.vm_pu_squared)),
        ]


class MISOCPElectricityBranchFormulation(BranchFormulation):
    
    def ensure_var(self, branch):
        branch.big_M = 2
        branch.current_pu = Var(1, min=0)
        branch.gap = Var(1, min=0)

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [
            branch.current_pu*branch.br_r
        ]

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return [
            voltage_drop(from_node_model.vars["vm_pu_squared"],
                         to_node_model.vars["vm_pu_squared"],
                         branch.vars["p_from_mw"],
                         branch.vars["q_from_mvar"],
                         branch.current_pu,
                         branch.br_r,
                         branch.br_x) <= branch.big_M * (1 - branch.on_off),
            voltage_drop(from_node_model.vars["vm_pu_squared"],
                         to_node_model.vars["vm_pu_squared"],
                         branch.vars["p_from_mw"],
                         branch.vars["q_from_mvar"],
                         branch.current_pu,
                         branch.br_r,
                         branch.br_x) >= -branch.big_M * (1 - branch.on_off),
            soc_rel(from_node_model.vars["vm_pu_squared"], 
                    branch.vars["p_from_mw"], 
                    branch.vars["q_from_mvar"], 
                    branch.current_pu),
            active_power_loss(branch.vars["p_from_mw"], branch.vars["p_to_mw"], 
                              branch.current_pu, branch.br_r),
            reactive_power_loss(branch.vars["q_from_mvar"], branch.vars["q_to_mvar"], 
                              branch.current_pu, branch.br_r),
            # Calculate the voltage/current error, However using this make the Problem non-convex
            # branch.gap == gap_expr(from_node_model.vars["vm_pu_squared"], 
            #         branch.vars["p_from_mw"], 
            #         branch.vars["q_from_mvar"], 
            #         branch.current_pu),
        ]