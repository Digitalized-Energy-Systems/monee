"""Sector formulation constants and sector-complete network bundles.

Per-sector building blocks follow ``<SECTOR>_<CLASS>_FORMULATION``; the
network-wide bundles cover all three sectors with one consistent
optimization class:

* :data:`SMOOTH_NLP_FORMULATION` - smooth non-convex NLP (IPOPT/APOPT):
  polar AC + smooth Weymouth + smooth Darcy-Weisbach.
* :data:`CONVEX_MIQCQP_FORMULATION` - convex relaxations a MISOCP/MIQCQP
  solver can certify: branch-flow MISOCP + epigraph-relaxed Weymouth +
  McCormick district heating (incl. convex heat-exchanger variants).
* :data:`NONCONVEX_MIQCQP_FORMULATION` - the exact quadratic models for
  global solvers (SCIP, Gurobi): exact branch flow + exact Weymouth +
  bilinear Darcy-Weisbach.
* :data:`DEFAULT_SIMULATION_FORMULATION` - the deliberate hybrid
  :class:`~monee.model.network.Network` applies by default (polar-AC NLP +
  relaxed Weymouth + bilinear Darcy-Weisbach).
"""

from monee.model.branch import (
    GasPipe,
    GenericPowerBranch,
    HeatExchanger,
    PassiveHeatExchanger,
    WaterPipe,
)
from monee.model.grid import GasGrid, WaterGrid
from monee.model.node import Bus, Junction

from .common import GasNodeFormulation, WaterNodeFormulation
from .core import NetworkFormulation
from .milp.gas import PwlWeymouthBranchFormulation
from .milp.heat import (
    FixedFlowHeatExchangerFormulation,
    McCormickHeatBranchFormulation,
    McCormickHeatExchangerFormulation,
    McCormickHeatNodeFormulation,
    McCormickPassiveHeatExchangerFormulation,
)
from .miqcqp.convex.el import (
    MISOCPElectricityBranchFormulation,
    MISOCPElectricityNodeFormulation,
)
from .miqcqp.convex.gas import RelaxedWeymouthBranchFormulation
from .miqcqp.nonconvex.el import (
    ExactBranchFlowBranchFormulation,
    ExactBranchFlowNodeFormulation,
)
from .miqcqp.nonconvex.gas import ExactWeymouthBranchFormulation
from .miqcqp.nonconvex.heat import (
    BilinearDarcyWeisbachBranchFormulation,
    BilinearPassiveHeatExchangerFormulation,
    PwlDarcyWeisbachBranchFormulation,
)
from .nlp.el import AcPolarNlpBranchFormulation, AcPolarNlpNodeFormulation
from .nlp.gas import SmoothWeymouthBranchFormulation
from .nlp.heat import (
    SmoothDarcyWeisbachBranchFormulation,
    SmoothHeatExchangerFormulation,
    SmoothPassiveHeatExchangerFormulation,
)


def combine(*network_formulations: NetworkFormulation) -> NetworkFormulation:
    """Merge per-sector :class:`NetworkFormulation` objects into one apply.
    Later arguments win on type collisions."""
    branch, node, child, compound = {}, {}, {}, {}
    for nf in network_formulations:
        branch.update(nf.branch_type_to_formulations)
        node.update(nf.node_type_to_formulations)
        child.update(nf.child_type_to_formulations)
        compound.update(nf.compound_type_to_formulations)
    return NetworkFormulation(
        branch_type_to_formulations=branch,
        node_type_to_formulations=node,
        child_type_to_formulations=child,
        compound_type_to_formulations=compound,
    )


# ---------------------------------------------------------------------------
# Electricity
# ---------------------------------------------------------------------------

# Single AC formulation for both modes. Under a simulation solve the node
# formulation's ``ensure_var(simulation=True)`` demotes vm_pu_squared to a
# PostProcess report (no solver variable), removing the phantom DOF so an
# IMODE=1 square solve is feasible; the optimization solve keeps it.
EL_NLP_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={GenericPowerBranch: AcPolarNlpBranchFormulation()},
    node_type_to_formulations={Bus: AcPolarNlpNodeFormulation()},
)

EL_MISOCP_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GenericPowerBranch: MISOCPElectricityBranchFormulation()
    },
    node_type_to_formulations={Bus: MISOCPElectricityNodeFormulation()},
)

EL_NONCONVEX_MIQCQP_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GenericPowerBranch: ExactBranchFlowBranchFormulation()
    },
    node_type_to_formulations={Bus: ExactBranchFlowNodeFormulation()},
)


# ---------------------------------------------------------------------------
# Gas
# ---------------------------------------------------------------------------

GAS_CONVEX_MIQCQP_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GasPipe: RelaxedWeymouthBranchFormulation(),
    },
    node_type_to_formulations={(Junction, GasGrid): GasNodeFormulation()},
)

GAS_NONCONVEX_MIQCQP_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        GasPipe: ExactWeymouthBranchFormulation(),
    },
    node_type_to_formulations={(Junction, GasGrid): GasNodeFormulation()},
)


def make_gas_milp_pwl_formulation(n_breakpoints: int = 12) -> NetworkFormulation:
    """Variable-friction Weymouth via per-pipe PWL of ``φ(m) = friction(Re(m))·m²``.

    Opt-in alternative to :data:`GAS_CONVEX_MIQCQP_FORMULATION` for networks
    where laminar-regime accuracy matters (``Re < 2300`` on lightly-loaded
    pipes). See
    :class:`monee.model.formulation.milp.gas.PwlWeymouthBranchFormulation`
    for details and trade-offs.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            GasPipe: PwlWeymouthBranchFormulation(n_breakpoints=n_breakpoints),
        },
        node_type_to_formulations={(Junction, GasGrid): GasNodeFormulation()},
    )


def make_gas_nlp_formulation(
    friction_model: str = "constant",
    smoothing_eps: float = 1e-3,
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Pure-NLP Weymouth gas formulation for GEKKO IPOPT/APOPT.

    Binary-free (no ``direction`` switch), numerically smooth signed pressure
    drop. ``friction_model`` selects ``"constant"`` / ``"pwl"`` / ``"nonlinear"``
    friction. Opt-in alternative to :data:`GAS_CONVEX_MIQCQP_FORMULATION` for
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
        node_type_to_formulations={(Junction, GasGrid): GasNodeFormulation()},
    )


GAS_NLP_FORMULATION = make_gas_nlp_formulation()


# ---------------------------------------------------------------------------
# Water / heat
# ---------------------------------------------------------------------------

HEAT_NONCONVEX_MIQCQP_FORMULATION = NetworkFormulation(
    branch_type_to_formulations={
        WaterPipe: BilinearDarcyWeisbachBranchFormulation(),
        HeatExchanger: FixedFlowHeatExchangerFormulation(),
        PassiveHeatExchanger: BilinearPassiveHeatExchangerFormulation(),
    },
    node_type_to_formulations={(Junction, WaterGrid): WaterNodeFormulation()},
)


def make_heat_nonconvex_pwl_formulation(
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Variable-friction Darcy-Weisbach via per-pipe PWL of ``φ(m)``.

    Opt-in alternative to :data:`HEAT_NONCONVEX_MIQCQP_FORMULATION` for
    laminar-heavy networks (Re < 2300); the default's asymptotic shortcut
    under-estimates pressure drop there. HeatExchanger formulations are
    unaffected.
    """
    return NetworkFormulation(
        branch_type_to_formulations={
            WaterPipe: PwlDarcyWeisbachBranchFormulation(n_breakpoints=n_breakpoints),
            HeatExchanger: FixedFlowHeatExchangerFormulation(),
            PassiveHeatExchanger: BilinearPassiveHeatExchangerFormulation(),
        },
        node_type_to_formulations={(Junction, WaterGrid): WaterNodeFormulation()},
    )


def make_heat_convex_milp_formulation(
    num_partitions: int = 1,
    include_heat_exchangers: bool = True,
) -> NetworkFormulation:
    """McCormick district-heating relaxation: LP for ``num_partitions == 1``,
    piecewise MILP above. Raise the partition count only when
    :func:`~monee.model.formulation.milp.heat.mccormick_dhs_gap_bound_k` shows
    the LP corner is non-physical on the network at hand.

    ``include_heat_exchangers`` also maps the active/passive heat exchangers
    to their convex variants so the sector has no non-convex fallback; disable
    it to reproduce the legacy pipes-only McCormick apply.
    """
    branch_map = {
        WaterPipe: McCormickHeatBranchFormulation(num_partitions=num_partitions),
    }
    if include_heat_exchangers:
        branch_map[HeatExchanger] = McCormickHeatExchangerFormulation()
        branch_map[PassiveHeatExchanger] = McCormickPassiveHeatExchangerFormulation(
            num_partitions=num_partitions
        )
    return NetworkFormulation(
        branch_type_to_formulations=branch_map,
        node_type_to_formulations={
            (Junction, WaterGrid): McCormickHeatNodeFormulation(
                num_partitions=num_partitions
            ),
        },
    )


HEAT_CONVEX_MILP_FORMULATION = make_heat_convex_milp_formulation(num_partitions=1)


def make_heat_nlp_formulation(
    friction_model: str = "constant",
    smoothing_eps: float = 1e-3,
    n_breakpoints: int = 12,
) -> NetworkFormulation:
    """Pure-NLP Darcy-Weisbach water/heat formulation for GEKKO IPOPT/APOPT.

    Binary-free smooth signed flow, smooth temperature upwinding and an active /
    passive heat-exchanger pair with their ``direction`` binaries removed.
    ``friction_model`` selects ``"constant"`` / ``"pwl"`` / ``"nonlinear"``.
    Opt-in alternative to :data:`HEAT_NONCONVEX_MIQCQP_FORMULATION`.
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
        node_type_to_formulations={(Junction, WaterGrid): WaterNodeFormulation()},
    )


HEAT_NLP_FORMULATION = make_heat_nlp_formulation()


# ---------------------------------------------------------------------------
# Sector-complete bundles
# ---------------------------------------------------------------------------


def make_smooth_nlp_formulation(friction_model: str = "constant"):
    """Combined polar-AC + smooth gas + smooth heat formulation in one apply.

    Pure-NLP across all three carriers. Solve with ``run_energy_flow`` for a
    fast square steady-state simulation (GEKKO IMODE=1, falls back to IMODE=3 if
    not square), or pass an optimization problem for an IMODE=3 optimize. The
    simulation squaring (phantom-var pinning, flow-limit drop, vm_pu_squared
    demotion) is applied by the solver from its ``simulation`` flag - there is
    no separate simulation formulation."""
    return combine(
        EL_NLP_FORMULATION,
        make_gas_nlp_formulation(friction_model),
        make_heat_nlp_formulation(friction_model),
    )


SMOOTH_NLP_FORMULATION = make_smooth_nlp_formulation()


def make_convex_miqcqp_formulation(num_partitions: int = 1) -> NetworkFormulation:
    """Convex MIQCQP across all three carriers: branch-flow MISOCP electricity,
    epigraph-relaxed Weymouth gas and McCormick district heating (incl. the
    convex heat-exchanger variants). Every constraint is certifiable by a
    convex MIQCQP/MISOCP solver; solutions are relaxation optima - check the
    documented gap bounds where exactness matters."""
    return combine(
        EL_MISOCP_FORMULATION,
        GAS_CONVEX_MIQCQP_FORMULATION,
        make_heat_convex_milp_formulation(num_partitions=num_partitions),
    )


CONVEX_MIQCQP_FORMULATION = make_convex_miqcqp_formulation()

# Exact quadratic models across all three carriers - for global MIQCQP
# solvers (SCIP, Gurobi). No relaxation gap, no trigonometric terms.
NONCONVEX_MIQCQP_FORMULATION = combine(
    EL_NONCONVEX_MIQCQP_FORMULATION,
    GAS_NONCONVEX_MIQCQP_FORMULATION,
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
)

# The deliberate hybrid Network() applies by default: exact polar AC for
# electricity, the MISOCP-shaped relaxed Weymouth for gas and the bilinear
# Darcy-Weisbach for heat. Documented here so its mixed nature is explicit.
DEFAULT_SIMULATION_FORMULATION = combine(
    EL_NLP_FORMULATION,
    GAS_CONVEX_MIQCQP_FORMULATION,
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
)
