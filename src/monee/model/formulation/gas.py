from monee.model.branch import GasPipe
from monee.model.grid import GasGrid
from monee.model.node import Junction

from .core import NetworkFormulation
from .nonlinear.gas import NLWeymouthBranchFormulation, NLWeymouthNodeFormulation

NL_WEYMOUTH_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={GasPipe: NLWeymouthBranchFormulation()},
    node_type_to_formulations={(Junction, GasGrid): NLWeymouthNodeFormulation()},
)
