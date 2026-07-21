from monee.model.branch import GenericPowerBranch
from monee.model.node import Bus

from .core import NetworkFormulation
from .misoc.el import (
    MISOCPElectricityBranchFormulation,
    MISOCPElectricityNodeFormulation,
)
from .nonlinear.ac import ACElectricityBranchFormulation, ACElectricityNodeFormulation
from .quadratic_convex.cq_with_switch import QCElectricityNodeFormulation, QCElectricityBranchFormulation
from .quadratic_convex.qc_mixed_int import MIQCBranchFormulation, MIQCNodeFormulation

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
QC_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GenericPowerBranch: QCElectricityBranchFormulation()
    },
    node_type_to_formulations={Bus: QCElectricityNodeFormulation()},
)

MIQC_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GenericPowerBranch: MIQCBranchFormulation(),
    },
    node_type_to_formulations={Bus: MIQCNodeFormulation()},
)
