from monee.io.matpower import read_matpower_case
from monee.solver.gekko import GEKKOSolver


def test_import_simbench_net():
    network = read_matpower_case("tests/data/1-LV-rural3--1-no_sw.mat")
    assert network is not None
    solver = GEKKOSolver()
    result = solver.solve(network)

    assert len(result.dataframes["Bus"]) == 129
