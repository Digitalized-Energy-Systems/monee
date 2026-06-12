import math

from monee import mm, mx, run_energy_flow
from monee.model import Var


def test_on_off_el():
    # GIVEN
    net = mm.Network()

    bus_0 = mx.create_bus(net)
    bus_1 = mx.create_bus(net)
    bus_2 = mx.create_bus(net)

    mx.create_ext_power_grid(net, bus_1)
    mx.create_power_generator(net, bus_0, 1, 0)
    mx.create_power_load(net, bus_2, Var(1), 0)

    mx.create_line(net, bus_0, bus_1, 100, r_ohm_per_m=0.00007, x_ohm_per_m=0.00007)

    def my_constraint(line, grid, fn, tn, **kwargs):
        return line.on_off == 0

    mx.create_line(
        net,
        bus_0,
        bus_2,
        200,
        r_ohm_per_m=0.00007,
        x_ohm_per_m=0.00007,
        on_off=Var(1, integer=True),
        constraints=[my_constraint],
    )

    # WHEN
    result = run_energy_flow(net, solver="apopt")
    print(result)

    # THEN
    assert result.success

    assert math.isclose(result.dataframes["PowerLine"]["p_from_mw"][1], 0, abs_tol=1e-9)
    assert result.dataframes["PowerLine"]["on_off"][1] == 0
    assert result.dataframes["PowerLine"]["on_off"][0] == 1


def test_on_off_water():
    # GIVEN
    net = mm.Network()

    j_0 = mx.create_water_junction(net)
    j_1 = mx.create_water_junction(net)
    j_2 = mx.create_water_junction(net)

    mx.create_water_ext_grid(net, j_1)
    mx.create_water_source(net, j_0, 0.1)
    mx.create_water_sink(net, j_2, Var(0))

    mx.create_water_pipe(net, j_0, j_1, diameter_m=0.1, length_m=100)

    def my_constraint(line, grid, fn, tn, **kwargs):
        return line.on_off == 0

    mx.create_water_pipe(
        net,
        j_0,
        j_2,
        diameter_m=0.1,
        length_m=100,
        on_off=Var(1, max=1, min=0, integer=True, name="on_off"),
        constraints=[my_constraint],
    )

    # WHEN
    result = run_energy_flow(net, solver="scip")
    print(result)

    # THEN
    assert result.success

    assert result.dataframes["Sink"]["mass_flow"][0] < 0.000001
    assert (
        result.dataframes["WaterPipe"]["mass_flow"][1] < 0.01
        and result.dataframes["WaterPipe"]["mass_flow"][1] > -0.0009
    )


def test_on_off_gas():
    # GIVEN
    net = mm.Network()

    j_0 = mx.create_gas_junction(net)
    j_1 = mx.create_gas_junction(net)
    j_2 = mx.create_gas_junction(net)

    mx.create_ext_hydr_grid(net, j_1, pressure_pu=1)
    mx.create_source(net, j_0, 0.1)
    mx.create_sink(net, j_2, Var(1))

    def my_constraint(line, grid, fn, tn, **kwargs):
        return line.on_off == 0

    mx.create_gas_pipe(net, j_0, j_1, diameter_m=0.1, length_m=1000, roughness=0.001)

    mx.create_gas_pipe(
        net,
        j_0,
        j_2,
        diameter_m=0.1,
        length_m=1000,
        on_off=Var(1, min=0, max=1, integer=True),
        constraints=[my_constraint],
    )

    # WHEN
    result = run_energy_flow(net)
    print(result)

    # THEN
    assert result.success

    assert math.isclose(result.dataframes["Sink"]["mass_flow"][0], 0, abs_tol=1e-9)
    assert (
        result.dataframes["GasPipe"]["mass_flow"][1] < 0.001
        and result.dataframes["GasPipe"]["mass_flow"][1] > -0.0009
    )
    assert result.dataframes["GasPipe"]["on_off"][1] == 0
