
class Formulation:
    
    def ensure_var(self, model):
        pass

class BranchFormulation(Formulation):

    def minimize(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return []

    def equations(self, branch, grid, from_node_model, to_node_model, **kwargs):
        return []

class NodeFormulation(Formulation):
        
    def minimize(self, node, grid, from_branch_models, to_branch_models, connected_child_models, **kwargs):
        return []

    def equations(self, node, grid, from_branch_models, to_branch_models, connected_child_models, **kwargs):
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

    def overwrite(self, child, node_model):
        pass

def _or_dict(d: dict):
    return {} if d is None else d

class NetworkFormulation:
    branch_type_to_formulations: dict[type, BranchFormulation]
    node_type_to_formulations: dict[type, NodeFormulation]
    child_type_to_formulations: dict[type, ChildFormulation]
    compound_type_to_formulations: dict[type, CompoundFormulation]

    def __init__(self, branch_type_to_formulations=None, 
                    node_type_to_formulations=None, 
                    child_type_to_formulations=None,
                    compound_type_to_formulations=None):
        self.branch_type_to_formulations = _or_dict(branch_type_to_formulations)
        self.node_type_to_formulations = _or_dict(node_type_to_formulations)
        self.child_type_to_formulations = _or_dict(child_type_to_formulations)
        self.compound_type_to_formulations = _or_dict(compound_type_to_formulations)