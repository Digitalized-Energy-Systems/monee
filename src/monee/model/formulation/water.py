from monee.model.branch import HeatExchanger, PassiveHeatExchanger, WaterPipe
from monee.model.grid import WaterGrid
from monee.model.node import Junction

from .core import NetworkFormulation
from .linear.water import LinearHeatExchangerFormulation
from .nonlinear.heat_smooth import (
    SmoothDarcyWeisbachBranchFormulation,
    SmoothHeatExchangerFormulation,
    SmoothPassiveHeatExchangerFormulation,
)
from .nonlinear.water import (
    NLDarcyWeisbachBranchFormulation,
    NLDarcyWeisbachHeatExchangerFormulation,
    NLDarcyWeisbachNodeFormulation,
    NLDarcyWeisbachPWLBranchFormulation,
)

NL_DARCY_WEISBACH_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        WaterPipe: NLDarcyWeisbachBranchFormulation(),
        HeatExchanger: LinearHeatExchangerFormulation(),
        PassiveHeatExchanger: NLDarcyWeisbachHeatExchangerFormulation(),
    },
    node_type_to_formulations={(Junction, WaterGrid): NLDarcyWeisbachNodeFormulation()},
)


def make_nl_darcy_weisbach_pwl_network_formulation(
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Variable-friction Darcy-Weisbach via per-pipe PWL of ``φ(m)``.

    Opt-in alternative to :data:`NL_DARCY_WEISBACH_NETWORK_FORMULATION` for
    laminar-heavy networks (Re < 2300); the default's asymptotic shortcut
    under-estimates pressure drop there. HeatExchanger formulations are
    unaffected.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            WaterPipe: NLDarcyWeisbachPWLBranchFormulation(n_breakpoints=n_breakpoints),
            HeatExchanger: LinearHeatExchangerFormulation(),
            PassiveHeatExchanger: NLDarcyWeisbachHeatExchangerFormulation(),
        },
        node_type_to_formulations={
            (Junction, WaterGrid): NLDarcyWeisbachNodeFormulation()
        },
    )


def make_smooth_darcy_weisbach_network_formulation(
    friction_model: str = "constant",
    smoothing_eps: float = 1e-3,
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Pure-NLP Darcy-Weisbach water/heat formulation for GEKKO IPOPT/APOPT.

    Binary-free smooth signed flow, smooth temperature upwinding and an active /
    passive heat-exchanger pair with their ``direction`` binaries removed.
    ``friction_model`` selects ``"constant"`` / ``"pwl"`` / ``"nonlinear"``.
    Opt-in alternative to :data:`NL_DARCY_WEISBACH_NETWORK_FORMULATION`.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            WaterPipe: SmoothDarcyWeisbachBranchFormulation(
                friction_model=friction_model,
                smoothing_eps=smoothing_eps,
                n_breakpoints=n_breakpoints,
            ),
            HeatExchanger: SmoothHeatExchangerFormulation(),
            PassiveHeatExchanger: SmoothPassiveHeatExchangerFormulation(
                friction_model=friction_model,
                smoothing_eps=smoothing_eps,
                n_breakpoints=n_breakpoints,
            ),
        },
        node_type_to_formulations={
            (Junction, WaterGrid): NLDarcyWeisbachNodeFormulation()
        },
    )


SMOOTH_DARCY_WEISBACH_NETWORK_FORMULATION = (
    make_smooth_darcy_weisbach_network_formulation()
)
