import monee.model as md
from .economic_dispatch import (
    create_economic_dispatch_problem,
    create_multi_period_economic_dispatch_problem,
)
from .min_load_shedding import (
    WEIGHT_DEMAND,
    WEIGHT_GENERATOR,
    create_min_load_shedding_problem,
)
from .metric import GeneralResiliencePerformanceMetric
from monee.problem.core import (
    AttributeParameter,
    Constraints,
    Objectives,
    OptimizationProblem,
)


def calc_general_resilience_performance(network: md.Network, **kwargs):
    """Curtailed demand of a solved *network* as ``(power, heat, gas)`` in MW.

    Thin wrapper around :meth:`GeneralResiliencePerformanceMetric.calc`; see
    there for ``inv``, ``include_ext_grid`` (off by default: it accounts
    external import as unserved demand, which is the islanding view) and
    ``include_coupling_points``.
    """
    return GeneralResiliencePerformanceMetric().calc(network, **kwargs)
