import monee.model as md
from .load_shedding import (
    create_load_shedding_optimization_problem,
)
from .metric import GeneralResiliencePerformanceMetric
from monee.problem.core import (
    AttributeParameter,
    Constraints,
    Objectives,
    OptimizationProblem,
)


def calc_general_resilience_performance(network: md.Network, **kwargs):
    """
    No docstring provided.
    """
    return GeneralResiliencePerformanceMetric().calc(network, **kwargs)
