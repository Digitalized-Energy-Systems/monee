from .core import NetworkFormulation, Formulation

from .el import (
    AC_NETWORK_FORMULATION,
    AC_SIM_NETWORK_FORMULATION,
    MISOCP_NETWORK_FORMULATION,
)
from .gas import (
    NL_WEYMOUTH_NETWORK_FORMULATION,
    SMOOTH_WEYMOUTH_NETWORK_FORMULATION,
    make_nl_weymouth_pwl_network_formulation,
    make_simulation_weymouth_network_formulation,
    make_smooth_weymouth_network_formulation,
)
from .water import (
    NL_DARCY_WEISBACH_NETWORK_FORMULATION,
    SMOOTH_DARCY_WEISBACH_NETWORK_FORMULATION,
    make_nl_darcy_weisbach_pwl_network_formulation,
    make_simulation_darcy_weisbach_network_formulation,
    make_smooth_darcy_weisbach_network_formulation,
)
from .mccormick.water import (
    MCCORMICK_DHS_NETWORK_FORMULATION,
    make_mccormick_dhs_formulation,
)


def make_simulation_network_formulation(friction_model: str = "constant"):
    """Combined AC + gas + heat formulation squared for a steady-state IMODE=1
    simulation. Apply to a network and solve with ``GEKKOSolver(simulation=True)``
    for a fast, unique plain energy flow (falls back to IMODE=3 if not square)."""
    el = AC_SIM_NETWORK_FORMULATION
    gas = make_simulation_weymouth_network_formulation(friction_model)
    heat = make_simulation_darcy_weisbach_network_formulation(friction_model)
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
