import math

import monee.model.phys.core.hydraulics as ml


def test_calc_pipe_area():
    # GIVEN
    diameter = 2

    # WHEN
    area = ml.calc_pipe_area(diameter)

    # THEN
    assert area == math.pi


def test_calc_pipe_area_sub_1():
    # GIVEN
    diameter = 0.1

    # WHEN
    area = ml.calc_pipe_area(diameter)

    # THEN
    assert math.isclose(area, 0.007853981633974483)


def test_calc_nikurdse_friction_factor():
    # GIVEN
    diameter = 2
    roughness = 0.7

    # WHEN
    nikurdse_friction = ml.calc_nikurdse(diameter, roughness)

    # THEN
    assert math.isclose(nikurdse_friction, 0.23781164943674166, rel_tol=1e-3)


def test_balance_equation():
    # GIVEN
    mass_flows = [1, -1, 2, -2]

    # WHEN
    balance = ml.junction_mass_flow_balance(mass_flows)

    # THEN
    assert balance


def test_reynolds_equation():
    # GIVEN
    # rey_var is scaled by REYNOLDS_SCALE = 1e6, so m*D/(mu*A*1e6) = rey_var
    rey_var = 3.21e-4
    mass_flow = 321
    diameter = 2
    dynamic_visc = 0.1
    area = 20

    # WHEN
    reynolds_correct = ml.reynolds_equation(
        rey_var, mass_flow, diameter, dynamic_visc, area
    )

    # THEN
    assert reynolds_correct


def test_pipe_mass_flow_constraint():
    # GIVEN
    max_v = 10
    min_v = 1

    # WHEN
    mass_flow_bound = ml.pipe_mass_flow(max_v, min_v, 10)
    mass_flow_bound_2 = ml.pipe_mass_flow(max_v, min_v, 0)

    # THEN
    assert mass_flow_bound
    assert not mass_flow_bound_2
