import monee.express as mx
import monee.model as mm
from monee.model.formulation.registry import attach_formulations
from monee.model.grid import NO_GRID, NoGrid


def test_no_grid_marker():
    assert isinstance(NO_GRID, NoGrid)


def _junction_psq_var(net, node_id, simulation):
    copy = net.copy()
    attach_formulations(copy, None, simulation=simulation)
    return copy.node_by_id(node_id).model.pressure_squared_pu


def test_gas_grid_pressure_bounds_reach_the_junction_var():
    grid = mm.create_gas_grid("site", type="lgas")
    grid.pressure_squared_pu_min = 0.81
    grid.pressure_squared_pu_max = 1.21
    net = mx.create_multi_energy_network()
    junction = mx.create_gas_junction(net, grid=grid)

    optimization = _junction_psq_var(net, junction, simulation=False)
    assert (optimization.min, optimization.max) == (0.81, 1.21)

    # A square simulation must never be cut off by an operational limit.
    simulation = _junction_psq_var(net, junction, simulation=True)
    assert (simulation.min, simulation.max) == (0, 3)


def test_user_pressure_bounds_win_over_the_grid():
    grid = mm.create_gas_grid("site", type="lgas")
    net = mx.create_multi_energy_network()
    junction = mx.create_gas_junction(net, grid=grid)
    net.node_by_id(junction).model.pressure_squared_pu.min = 0.9
    net.node_by_id(junction).model.pressure_squared_pu.max = 1.05

    var = _junction_psq_var(net, junction, simulation=False)
    assert (var.min, var.max) == (0.9, 1.05)
