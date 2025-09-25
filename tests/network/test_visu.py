import os
import random

import pandapower.networks as pn

import monee.model as mm
import monee.network.mes as mes
from monee import run_energy_flow
from monee.io.from_pandapower import from_pandapower_net
from monee.network import create_monee_benchmark_net
from monee.visualization import plot_network


def test_visu_with_monee_bench_net():
    net = create_monee_benchmark_net()
    net = run_energy_flow(net).network

    plot_network(net, write_to="net.pdf")
    os.remove("net.pdf")


def create_cigre_multi():
    random.seed(9002)
    pnet = pn.create_cigre_network_mv(with_der="pv_wind")

    monee_net = from_pandapower_net(pnet)
    new_mes = monee_net.copy()
    bus_to_gas_junc = mes.create_gas_net_for_power(
        monee_net, new_mes, 1, scaling=2, length_scale=0.001, default_length=100000
    )
    ne = mm.Network()
    bus_index_to_junction_index, bus_index_to_end_junction_index = (
        mes.create_heat_net_for_power(
            monee_net,
            new_mes,
            0.3,
            mass_flow_rate=30,
            default_diameter_m=0.4,
            length_scale=0.001,
            default_length=100000,
        )
    )
    return new_mes


def test_visu_with_cigre_bench_net():
    net = create_cigre_multi()
    result = run_energy_flow(net)

    plot_network(result.network, plot_node_characteristics=False, write_to="net.pdf")
    os.remove("net.pdf")
