from monee.model.branch import GasPipe
from monee.model.grid import GasGrid
from monee.model.node import Junction

from .core import NetworkFormulation
from .nonlinear.gas import (
    NLWeymouthBranchFormulation,
    NLWeymouthNodeFormulation,
    NLWeymouthPWLBranchFormulation,
)
from .nonlinear.gas_smooth import (
    SmoothWeymouthBranchFormulation,
    SmoothWeymouthSimNodeFormulation,
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


def make_smooth_weymouth_network_formulation(
    friction_model: str = "constant",
    smoothing_eps: float = 1e-3,
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Pure-NLP Weymouth gas formulation for GEKKO IPOPT/APOPT.

    Binary-free (no ``direction`` switch), numerically smooth signed pressure
    drop. ``friction_model`` selects ``"constant"`` / ``"pwl"`` / ``"nonlinear"``
    friction. Opt-in alternative to :data:`NL_WEYMOUTH_NETWORK_FORMULATION` for
    full-MES solves where the MISOCP-shaped default stalls IPOPT.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            GasPipe: SmoothWeymouthBranchFormulation(
                friction_model=friction_model,
                smoothing_eps=smoothing_eps,
                n_breakpoints=n_breakpoints,
            ),
        },
        node_type_to_formulations={(Junction, GasGrid): NLWeymouthNodeFormulation()},
    )


SMOOTH_WEYMOUTH_NETWORK_FORMULATION = make_smooth_weymouth_network_formulation()


def make_simulation_weymouth_network_formulation(
    friction_model: str = "constant",
    smoothing_eps: float = 1e-3,
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Square (IMODE=1-ready) Weymouth gas formulation: the smooth formulation
    with phantom vars pinned and operational flow limits dropped. Use with
    :class:`~monee.solver.gekko.GEKKOSolver` in ``simulation=True`` mode."""
    return NetworkFormulation(
        branch_type_to_formulations={
            GasPipe: SmoothWeymouthBranchFormulation(
                friction_model=friction_model,
                smoothing_eps=smoothing_eps,
                n_breakpoints=n_breakpoints,
                simulation=True,
            ),
        },
        node_type_to_formulations={
            (Junction, GasGrid): SmoothWeymouthSimNodeFormulation()
        },
    )
