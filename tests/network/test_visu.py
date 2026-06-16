import os

import pytest

from monee import PyomoSolver, run_energy_flow
from monee.network import create_monee_benchmark_net, create_mv_multi_cigre
from monee.visualization import plot_network


@pytest.mark.pptest
def test_visu_with_monee_bench_net():
    # GIVEN
    net = create_monee_benchmark_net()
    result = run_energy_flow(net, solver=PyomoSolver())

    # WHEN
    plot_network(result.network, write_to="net.pdf")

    # THEN
    assert result.success

    os.remove("net.pdf")


@pytest.mark.pptest
def test_visu_with_cigre_bench_net():
    # GIVEN
    net = create_mv_multi_cigre()
    result = run_energy_flow(net, solver=PyomoSolver())

    # WHEN
    plot_network(result.network, write_to="net.pdf")

    # THEN
    assert result.success

    os.remove("net.pdf")
