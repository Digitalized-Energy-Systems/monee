from monee.model.branch import GasPipe
from monee.model.grid import GasGrid
from monee.model.node import Junction

from .core import NetworkFormulation
from .nonlinear.gas import (
    NLWeymouthBranchFormulation,
    NLWeymouthNodeFormulation,
    NLWeymouthPWLBranchFormulation,
)

NL_WEYMOUTH_NETWORK_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GasPipe: NLWeymouthBranchFormulation(),
    },
    node_type_to_formulations={(Junction, GasGrid): NLWeymouthNodeFormulation()},
)


def make_nl_weymouth_pwl_network_formulation(
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Variable-friction Weymouth via per-pipe PWL of ``φ(m) = friction(Re(m))·m²``.

    Opt-in alternative to :data:`NL_WEYMOUTH_NETWORK_FORMULATION` for
    networks where laminar-regime accuracy matters (``Re < 2300`` on
    lightly-loaded pipes).  See
    :class:`monee.model.formulation.nonlinear.gas.NLWeymouthPWLBranchFormulation`
    for details and trade-offs.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            GasPipe: NLWeymouthPWLBranchFormulation(n_breakpoints=n_breakpoints),
        },
        node_type_to_formulations={(Junction, GasGrid): NLWeymouthNodeFormulation()},
    )
