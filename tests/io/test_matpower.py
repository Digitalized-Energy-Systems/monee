from monee.io.matpower import read_matpower_case
from monee.solver.gekko import GEKKOSolver


def test_import_simbench_net():
    # GIVEN
    network = read_matpower_case("tests/data/1-LV-rural3--1-no_sw.mat")

    # WHEN
    solver = GEKKOSolver()
    result = solver.solve(network)

    # THEN
    assert result.success

    assert network is not None
    assert len(result.dataframes["Bus"]) == 129
