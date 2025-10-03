import os

import pytest

from monee import run_energy_flow
from monee.network import create_monee_benchmark_net, create_mv_multi_cigre
from monee.visualization import plot_network


@pytest.mark.pptest
def test_visu_with_monee_bench_net():
    net = create_monee_benchmark_net()
    net = run_energy_flow(net).network

    plot_network(net, write_to="net.pdf")
    os.remove("net.pdf")


@pytest.mark.pptest
def test_visu_with_cigre_bench_net():
    net = create_mv_multi_cigre()
    result = run_energy_flow(net)

    plot_network(result.network, plot_node_characteristics=False, write_to="net.pdf")
    os.remove("net.pdf")
