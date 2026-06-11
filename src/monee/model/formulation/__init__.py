from .core import NetworkFormulation, Formulation

from .el import (
    AC_NETWORK_FORMULATION,
    MISOCP_NETWORK_FORMULATION,
)
from .gas import (
    NL_WEYMOUTH_NETWORK_FORMULATION,
    SMOOTH_WEYMOUTH_NETWORK_FORMULATION,
    make_nl_weymouth_pwl_network_formulation,
    make_smooth_weymouth_network_formulation,
)
from .water import (
    NL_DARCY_WEISBACH_NETWORK_FORMULATION,
    SMOOTH_DARCY_WEISBACH_NETWORK_FORMULATION,
    make_nl_darcy_weisbach_pwl_network_formulation,
    make_smooth_darcy_weisbach_network_formulation,
)
from .mccormick.water import (
    MCCORMICK_DHS_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
)


def make_smooth_network_formulation(friction_model: str = "constant"):
    """Combined AC + smooth gas + smooth heat formulation in one apply.

    Pure-NLP across all three carriers. Solve with ``run_energy_flow`` for a
    fast square steady-state simulation (GEKKO IMODE=1, falls back to IMODE=3 if
    not square), or pass an optimization problem for an IMODE=3 optimize. The
    simulation squaring (phantom-var pinning, flow-limit drop, vm_pu_squared
    demotion) is applied by the solver from its ``simulation`` flag - there is
    no separate simulation formulation."""
    el = AC_NETWORK_FORMULATION
    gas = make_smooth_weymouth_network_formulation(friction_model)
    heat = make_smooth_darcy_weisbach_network_formulation(friction_model)
    return NetworkFormulation(
        branch_type_to_formulations={
            **el.branch_type_to_formulations,
            **gas.branch_type_to_formulations,
            **heat.branch_type_to_formulations,
        },
        node_type_to_formulations={
            **el.node_type_to_formulations,
            **gas.node_type_to_formulations,
            **heat.node_type_to_formulations,
        },
    )


SMOOTH_NETWORK_FORMULATION = make_smooth_network_formulation()
