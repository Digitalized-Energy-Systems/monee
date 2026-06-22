"""Formulation registry.

Packages are organised by optimization class first, sector second:

* ``nlp`` - smooth non-convex NLPs (polar AC, smooth Weymouth, smooth
  Darcy-Weisbach) for IPOPT/APOPT.
* ``milp`` - LP/MILP models (PWL Weymouth, McCormick district heating,
  fixed-flow heat exchanger).
* ``miqcqp.convex`` - certifiable relaxations (branch-flow MISOCP,
  epigraph-relaxed Weymouth).
* ``miqcqp.nonconvex`` - exact quadratic models for global solvers
  (exact branch flow, exact Weymouth, bilinear Darcy-Weisbach).

Sector constants and the sector-complete bundles
(:data:`SMOOTH_NLP_FORMULATION`, :data:`CONVEX_MIQCQP_FORMULATION`,
:data:`NONCONVEX_MIQCQP_FORMULATION`) live in :mod:`.bundles`.

The pre-restructure names (``AC_NETWORK_FORMULATION``,
``NL_WEYMOUTH_NETWORK_FORMULATION``, ``make_mccormick_dhs_formulation``, …)
remain importable from here but emit a :class:`DeprecationWarning`.
"""

import warnings

from .bundles import (
    CONVEX_MIQCQP_FORMULATION,
    DEFAULT_SIMULATION_FORMULATION,
    EL_MISOCP_FORMULATION,
    EL_QC_FORMULATION,
    EL_NLP_FORMULATION,
    EL_NONCONVEX_MIQCQP_FORMULATION,
    GAS_CONVEX_MIQCQP_FORMULATION,
    GAS_NLP_FORMULATION,
    GAS_NONCONVEX_MIQCQP_FORMULATION,
    HEAT_CONVEX_MILP_FORMULATION,
    HEAT_NLP_FORMULATION,
    HEAT_NONCONVEX_MIQCQP_FORMULATION,
    NONCONVEX_MIQCQP_FORMULATION,
    SMOOTH_NLP_FORMULATION,
    combine,
    make_convex_miqcqp_formulation,
    make_gas_milp_pwl_formulation,
    make_gas_nlp_formulation,
    make_heat_convex_milp_formulation,
    make_heat_nlp_formulation,
    make_heat_nonconvex_pwl_formulation,
    make_smooth_nlp_formulation,
)
from .core import Formulation, NetworkFormulation
from .milp.heat import mccormick_dhs_gap_bound_k, mccormick_dhs_gap_bound_mw
from .registry import (
    FORMULATIONS,
    attach_formulations,
    register_formulation,
    resolve_formulation,
)

__all__ = [
    "Formulation",
    "NetworkFormulation",
    "combine",
    # registry / solve-time attachment
    "FORMULATIONS",
    "register_formulation",
    "resolve_formulation",
    "attach_formulations",
    # electricity
    "EL_NLP_FORMULATION",
    "EL_MISOCP_FORMULATION",
    "EL_QC_FORMULATION",
    "EL_NONCONVEX_MIQCQP_FORMULATION",
    # gas
    "GAS_NLP_FORMULATION",
    "GAS_CONVEX_MIQCQP_FORMULATION",
    "GAS_NONCONVEX_MIQCQP_FORMULATION",
    "make_gas_nlp_formulation",
    "make_gas_milp_pwl_formulation",
    # water / heat
    "HEAT_NLP_FORMULATION",
    "HEAT_CONVEX_MILP_FORMULATION",
    "HEAT_NONCONVEX_MIQCQP_FORMULATION",
    "make_heat_nlp_formulation",
    "make_heat_convex_milp_formulation",
    "make_heat_nonconvex_pwl_formulation",
    # bundles
    "SMOOTH_NLP_FORMULATION",
    "CONVEX_MIQCQP_FORMULATION",
    "NONCONVEX_MIQCQP_FORMULATION",
    "DEFAULT_SIMULATION_FORMULATION",
    "make_smooth_nlp_formulation",
    "make_convex_miqcqp_formulation",
    # gap diagnostics
    "mccormick_dhs_gap_bound_mw",
    "mccormick_dhs_gap_bound_k",
]


def _legacy_mccormick_dhs_formulation(num_partitions: int = 1) -> NetworkFormulation:
    """Pipes-only McCormick apply (the pre-restructure behaviour: heat
    exchangers keep whatever formulation was applied before)."""
    return make_heat_convex_milp_formulation(
        num_partitions=num_partitions, include_heat_exchangers=False
    )


def _deprecated_constants():
    # Lazily built to avoid constructing legacy bundles on import.
    return {
        "AC_NETWORK_FORMULATION": lambda: EL_NLP_FORMULATION,
        "MISOCP_NETWORK_FORMULATION": lambda: EL_MISOCP_FORMULATION,
        "QC_NETWORK_FORMULATION": lambda: EL_QC_FORMULATION,
        "NL_WEYMOUTH_NETWORK_FORMULATION": lambda: GAS_CONVEX_MIQCQP_FORMULATION,
        "SMOOTH_WEYMOUTH_NETWORK_FORMULATION": lambda: GAS_NLP_FORMULATION,
        "NL_DARCY_WEISBACH_NETWORK_FORMULATION": (
            lambda: HEAT_NONCONVEX_MIQCQP_FORMULATION
        ),
        "SMOOTH_DARCY_WEISBACH_NETWORK_FORMULATION": lambda: HEAT_NLP_FORMULATION,
        "MCCORMICK_DHS_NETWORK_FORMULATION": _legacy_mccormick_dhs_formulation,
        "SMOOTH_NETWORK_FORMULATION": lambda: SMOOTH_NLP_FORMULATION,
    }


def _deprecated_factories():
    return {
        "make_smooth_network_formulation": make_smooth_nlp_formulation,
        "make_smooth_weymouth_network_formulation": make_gas_nlp_formulation,
        "make_smooth_darcy_weisbach_network_formulation": make_heat_nlp_formulation,
        "make_nl_weymouth_pwl_network_formulation": make_gas_milp_pwl_formulation,
        "make_nl_darcy_weisbach_pwl_network_formulation": (
            make_heat_nonconvex_pwl_formulation
        ),
        "make_mccormick_dhs_formulation": _legacy_mccormick_dhs_formulation,
    }


def _deprecated_classes():
    from .milp.gas import PwlWeymouthBranchFormulation
    from .milp.heat import (
        FixedFlowHeatExchangerFormulation,
        McCormickHeatBranchFormulation,
        McCormickHeatNodeFormulation,
    )
    from .common import GasNodeFormulation, WaterNodeFormulation
    from .miqcqp.convex.gas import RelaxedWeymouthBranchFormulation
    from .miqcqp.nonconvex.heat import (
        BilinearDarcyWeisbachBranchFormulation,
        BilinearPassiveHeatExchangerFormulation,
        PwlDarcyWeisbachBranchFormulation,
    )
    from .nlp.el import AcPolarNlpBranchFormulation, AcPolarNlpNodeFormulation

    return {
        "ACElectricityBranchFormulation": AcPolarNlpBranchFormulation,
        "ACElectricityNodeFormulation": AcPolarNlpNodeFormulation,
        "NLWeymouthBranchFormulation": RelaxedWeymouthBranchFormulation,
        "NLWeymouthNodeFormulation": GasNodeFormulation,
        "NLWeymouthPWLBranchFormulation": PwlWeymouthBranchFormulation,
        "NLDarcyWeisbachBranchFormulation": BilinearDarcyWeisbachBranchFormulation,
        "NLDarcyWeisbachNodeFormulation": WaterNodeFormulation,
        "NLDarcyWeisbachPWLBranchFormulation": PwlDarcyWeisbachBranchFormulation,
        "NLDarcyWeisbachHeatExchangerFormulation": (
            BilinearPassiveHeatExchangerFormulation
        ),
        "LinearHeatExchangerFormulation": FixedFlowHeatExchangerFormulation,
        "MccDHSBranchFormulation": McCormickHeatBranchFormulation,
        "MccDHSNodeFormulation": McCormickHeatNodeFormulation,
    }


def __getattr__(name):
    constants = _deprecated_constants()
    if name in constants:
        warnings.warn(
            f"monee.model.formulation.{name} is deprecated; use the "
            "optimization-class names from monee.model.formulation.bundles "
            "(e.g. EL_MISOCP_FORMULATION, GAS_CONVEX_MIQCQP_FORMULATION, "
            "SMOOTH_NLP_FORMULATION) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return constants[name]()
    factories = _deprecated_factories()
    if name in factories:
        replacement = (
            "make_heat_convex_milp_formulation(include_heat_exchangers=False)"
            if name == "make_mccormick_dhs_formulation"
            else factories[name].__name__
        )
        warnings.warn(
            f"monee.model.formulation.{name} is deprecated; use {replacement} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return factories[name]
    classes = _deprecated_classes()
    if name in classes:
        warnings.warn(
            f"monee.model.formulation.{name} is deprecated; use "
            f"{classes[name].__name__} from its optimization-class package "
            "(monee.model.formulation.{nlp,milp,miqcqp}) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
