import math

import monee.model.phys.core.hydraulics as ml


def test_calc_pipe_area():
    area = ml.calc_pipe_area(2)

    assert area == math.pi


def test_calc_pipe_area_sub_1():
    area = ml.calc_pipe_area(0.1)

    assert math.isclose(area, 0.007853981633974483)


def test_calc_nikurdse_friction_factor():
    nikurdse_friction = ml.calc_nikurdse(2, 0.7)

    assert math.isclose(nikurdse_friction, 0.23781164943674166, rel_tol=1e-3)


def test_balance_equation():
    balance = ml.junction_mass_flow_balance([1, -1, 2, -2])

    assert balance


def test_reynolds_equation():
    reynolds_correct = ml.reynolds_equation(321, 321, 2, 0.1, 20)

    assert reynolds_correct


def test_pipe_mass_flow_constraint():
    mass_flow_bound = ml.pipe_mass_flow(10, 1, 10)
    mass_flow_bound_2 = ml.pipe_mass_flow(10, 1, 0)

    assert mass_flow_bound
    assert not mass_flow_bound_2
