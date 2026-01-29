from monee.model.branch import GenericPowerBranch
from monee.model.node import Bus

from .core import NetworkFormulation
from .misoc.el import (
    MISOCPElectricityBranchFormulation,
    MISOCPElectricityNodeFormulation,
)
from .nonlinear.ac import ACElectricityBranchFormulation, ACElectricityNodeFormulation

AC_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={GenericPowerBranch: ACElectricityBranchFormulation()},
    node_type_to_formulations={Bus: ACElectricityNodeFormulation()},
)

MISOCP_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GenericPowerBranch: MISOCPElectricityBranchFormulation()
    },
    node_type_to_formulations={Bus: MISOCPElectricityNodeFormulation()},
)
