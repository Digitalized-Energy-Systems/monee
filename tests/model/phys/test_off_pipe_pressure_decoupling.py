"""Regression tests: a deactivated (on_off=0) water pipe must not couple the
pressures of its endpoints (mirrors the gas-side Weymouth on_off gating)."""

import math

import monee.model as mm
import monee.solver as ms
from monee.model.formulation import make_heat_nlp_formulation
from monee.model.phys.nonlinear.smooth import darcy_pressure
from monee.model.phys.nonlinear.wf import darcy_weisbach_equation


def test_darcy_weisbach_equation_gated_by_on_off():
    common = dict(
        m_pos_sq=0.0,
        m_neg_sq=0.0,
        pipe_length=100.0,
        diameter_m=0.1,
        fluid_density_kg_per_m3=970.0,
        friction=0.02,
    )
    assert darcy_weisbach_equation(2.0, 1.0, on_off=0, **common)
    assert not darcy_weisbach_equation(2.0, 1.0, on_off=1, **common)


def test_smooth_darcy_pressure_gated_by_on_off():
    assert darcy_pressure(2.0, 1.0, 0.0, 100.0, 0.1, 970.0, on_off=0)
    assert not darcy_pressure(2.0, 1.0, 0.0, 100.0, 0.1, 970.0, on_off=1)


def _pressure_separated_net(on_off):
    """Two independently slacked water islands joined only by an *off* pipe."""
    net = mm.Network(mm.create_water_grid("water"))

    n0 = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.ExtHydrGrid(pressure_pu=1.0, t_k=356))],
    )
    n1 = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.Sink(mass_flow_kgs=0.1))],
    )
    n2 = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.ExtHydrGrid(pressure_pu=0.6, t_k=340))],
    )
    n3 = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.Sink(mass_flow_kgs=0.1))],
    )

    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=100), n0, n1)
    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=100), n2, n3)
    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=100, on_off=on_off), n1, n3)
    return net


def test_smooth_nlp_off_pipe_between_pressure_separated_islands():
    net = _pressure_separated_net(on_off=0)
    net.apply_formulation(make_heat_nlp_formulation())

    result = ms.GEKKOSolver().solve(net)

    _assert_islands_decoupled(result)


def test_bilinear_miqcqp_off_pipe_between_pressure_separated_islands():
    net = _pressure_separated_net(on_off=0)

    result = ms.PyomoSolver().solve(net)

    _assert_islands_decoupled(result)


def _assert_islands_decoupled(result):
    assert result.success
    junctions = result.dataframes["Junction"]
    assert junctions["pressure_pa"][1] - junctions["pressure_pa"][3] > 3e5
    pipes = result.dataframes["WaterPipe"]
    off_pipe = pipes[pipes["on_off"] == 0]
    assert len(off_pipe) == 1
    assert abs(off_pipe["mass_flow_kgs"].iloc[0]) < 1e-3


def test_water_pipe_velocity_follows_flow_rate_relation():
    grid = mm.create_water_grid("water")
    net = mm.Network(grid)
    n0 = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.ExtHydrGrid(t_k=356))],
    )
    n1 = net.node(
        mm.Junction(),
        child_ids=[net.child(mm.Sink(mass_flow_kgs=1.0))],
    )
    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=100), n0, n1)
    net.apply_formulation(make_heat_nlp_formulation())

    result = ms.GEKKOSolver().solve(net)

    assert result.success
    pipes = result.dataframes["WaterPipe"]
    rho = grid.fluid_density_kg_per_m3
    area = math.pi / 4 * 0.1**2
    assert math.isclose(
        pipes["velocity_mps"][0],
        pipes["mass_flow_kgs"][0] / (rho * area),
        rel_tol=1e-3,
    )
