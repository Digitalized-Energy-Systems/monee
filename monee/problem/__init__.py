import monee.model as md
from .core import OptimizationProblem
from .core import Constraints
from .core import Objectives
from .load_shedding import (
    create_load_shedding_optimization_problem,
    create_ls_init_optimization_problem,
)
from .metric import GeneralResiliencePerformanceMetric


def calc_general_resilience_performance(network: md.Network, **kwargs):
    """
    No docstring provided.
    """
    return GeneralResiliencePerformanceMetric().calc(network, **kwargs)
