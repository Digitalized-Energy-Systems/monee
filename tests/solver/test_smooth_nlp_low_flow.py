"""smooth_nlp must stay solvable when a gas branch carries only a fraction of its capacity."""

import pytest

import monee
import monee.model as mm
from monee.model.phys.nonlinear.smooth import scaled_smoothing_eps
from monee.network import create_urban_district_net


def test_scaled_smoothing_eps_tracks_the_branch_flow_scale():
    assert scaled_smoothing_eps(1e-3, 1.14) == pytest.approx(1.14e-4)
    assert scaled_smoothing_eps(1e-3, 1e6) == pytest.approx(1e-3)
    assert scaled_smoothing_eps(1e-3, 0.0) == pytest.approx(1e-3)


def test_smooth_nlp_solves_at_a_low_gas_sink():
    net = create_urban_district_net()
    net.child_by_id(6).model.mass_flow_kgs = 0.01

    result = monee.run_energy_flow(net, formulation="smooth_nlp")

    assert result.success


def test_smooth_nlp_solves_with_pinned_gas_linepack():
    net = create_urban_district_net()
    net.add_extension(mm.GasLinepack())

    result = monee.run_energy_flow(net, formulation="smooth_nlp")

    assert result.success
