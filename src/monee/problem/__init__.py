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
    return GeneralResiliencePerformanceMetric().calc(network, **kwargs)
