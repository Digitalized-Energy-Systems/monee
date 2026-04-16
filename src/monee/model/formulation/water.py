from monee.model.branch import HeatExchanger, PassiveHeatExchanger, WaterPipe
from monee.model.grid import WaterGrid
from monee.model.node import Junction

from .core import NetworkFormulation
from .linear.water import LinearHeatExchangerFormulation
from .nonlinear.water import (
    NLDarcyWeisbachBranchFormulation,
    NLDarcyWeisbachHeatExchangerFormulation,
    NLDarcyWeisbachNodeFormulation,
)

NL_DARCY_WEISBACH_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        WaterPipe: NLDarcyWeisbachBranchFormulation(),
        HeatExchanger: LinearHeatExchangerFormulation(),
        PassiveHeatExchanger: NLDarcyWeisbachHeatExchangerFormulation(),
    },
    node_type_to_formulations={(Junction, WaterGrid): NLDarcyWeisbachNodeFormulation()},
)
