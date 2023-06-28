import os

from peext.scenario.network import create_small_test_multinet
import pandapower.converter as pc


from monee.io.matpower import read_matpower_case
from monee.solver.gekko import GEKKOSolver


def test_read_network_data_matpower():
    multinet = create_small_test_multinet()
    power = multinet["nets"]["power"]
    pc.to_mpc(power, filename="a.mat", init="flat")

    network = read_matpower_case("a.mat")
    assert network is not None
    assert len(network.nodes) == 4
    assert len(network.branches) == 3
    assert len(network.childs) == 5
    assert network.childs[0].model is not None
    os.remove("a.mat")


def test_solve_read_network_data_matpower():
    multinet = create_small_test_multinet()
    power = multinet["nets"]["power"]
    pc.to_mpc(power, filename="a.mat", init="flat")
    network = read_matpower_case("a.mat")
    solver = GEKKOSolver()
    result = solver.solve(network)

    assert len(result.dataframes) == 5
    os.remove("a.mat")


def test_import_simbench_net():
    network = read_matpower_case("1-LV-rural3--1-no_sw.mat")
    assert network is not None
    solver = GEKKOSolver()
    result = solver.solve(network)

    assert len(result.dataframes["Bus"]) == 129
