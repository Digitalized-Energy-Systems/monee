from monee.model.branch import GenericPowerBranch
from monee.model.node import Bus

from .core import NetworkFormulation
from .misoc.el import (
    MISOCPElectricityBranchFormulation,
    MISOCPElectricityNodeFormulation,
)
from .nonlinear.ac import (
    ACElectricityBranchFormulation,
    ACElectricityNodeFormulation,
)

# Single AC formulation for both modes. Under a simulation solve the node
# formulation's ``ensure_var(simulation=True)`` demotes vm_pu_squared to a
# PostProcess report (no solver variable), removing the phantom DOF so an
# IMODE=1 square solve is feasible; the optimization solve keeps it.
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
