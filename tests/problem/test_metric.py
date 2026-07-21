"""Unit tests for GeneralResiliencePerformanceMetric heat-exchanger accounting."""

import pytest

import monee.express as mx
import monee.model as mm
from monee.problem.metric import GeneralResiliencePerformanceMetric, ResilienceMetric

Q_SET_MW = 0.2


def _hx_net(q_mw):
    """Water net with a single heat-exchanger branch (load for q_mw > 0)."""
    net = mx.create_multi_energy_network()
    s = mx.create_water_junction(net)
    r = mx.create_water_junction(net)
    bid = mx.create_heat_exchanger(net, s, r, q_mw)
    return net, net.branch_by_id(bid)


def _heat_curtailed(net):
    _, heat, _ = GeneralResiliencePerformanceMetric().calc(net)
    return heat


def test_destroyed_hx_load_counts_setpoint():
    net, branch = _hx_net(Q_SET_MW)
    branch.ignored = True
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW)


def test_inactive_hx_load_counts_setpoint():
    net, branch = _hx_net(Q_SET_MW)
    branch.active = False
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW)


def test_hx_load_fully_regulated_down_counts_full_shed():
    net, branch = _hx_net(Q_SET_MW)
    branch.model.regulation = 0.0
    branch.model.q_mw.value = 0.0  # q_mw = q_mw_set * regulation
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW)


def test_hx_load_unregulated_counts_zero():
    net, branch = _hx_net(Q_SET_MW)
    branch.model.regulation = 1.0
    branch.model.q_mw.value = Q_SET_MW
    assert _heat_curtailed(net) == pytest.approx(0.0)


def test_hx_load_half_regulated_counts_half():
    net, branch = _hx_net(Q_SET_MW)
    branch.model.regulation = 0.5
    branch.model.q_mw.value = Q_SET_MW * 0.5
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW * 0.5)


def test_hx_generator_never_counts_as_curtailed_load():
    net, branch = _hx_net(-Q_SET_MW)
    assert _heat_curtailed(net) == pytest.approx(0.0)
    branch.ignored = True
    assert _heat_curtailed(net) == pytest.approx(0.0)


def _bare_hx_net(q_mw_set, model_cls=mm.HeatExchanger, **kwargs):
    """Bare (Passive)HeatExchanger, no Load/Generator subclass; the model
    stores q_mw_set = -q_mw, so q_mw_set > 0 = consuming."""
    net = mx.create_multi_energy_network()
    s = mx.create_water_junction(net)
    r = mx.create_water_junction(net)
    bid = net.branch(model_cls(q_mw=-q_mw_set, **kwargs), s, r)
    return net, net.branch_by_id(bid)


def test_bare_consuming_hx_destroyed_counts_setpoint():
    net, branch = _bare_hx_net(Q_SET_MW)
    branch.ignored = True
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW)


def test_bare_consuming_hx_half_regulated_counts_half():
    net, branch = _bare_hx_net(Q_SET_MW)
    branch.model.regulation = 0.5
    branch.model.q_mw.value = Q_SET_MW * 0.5
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW * 0.5)


def test_bare_generating_hx_never_counts_as_curtailed_load():
    net, branch = _bare_hx_net(-Q_SET_MW)
    branch.model.q_mw.value = -Q_SET_MW
    assert _heat_curtailed(net) == pytest.approx(0.0)
    branch.ignored = True
    assert _heat_curtailed(net) == pytest.approx(0.0)


def test_bare_consuming_passive_hx_destroyed_counts_setpoint():
    net, branch = _bare_hx_net(
        Q_SET_MW, model_cls=mm.PassiveHeatExchanger, diameter_m=0.1
    )
    branch.ignored = True
    assert _heat_curtailed(net) == pytest.approx(Q_SET_MW)


def test_resilience_metric_is_abstract():
    with pytest.raises(TypeError):
        ResilienceMetric()

    class _Metric(ResilienceMetric):
        def gather(self, network, step, **kwargs):
            return step

        def calc(self):
            return 42

    m = _Metric()
    assert m.gather(None, 3) == 3
    assert m.calc() == 42


def test_calc_inv_negates_the_curtailment_tuple():
    net, branch = _hx_net(Q_SET_MW)
    branch.ignored = True  # curtailed heat load registers a non-zero metric
    metric = GeneralResiliencePerformanceMetric()
    normal = metric.calc(net, inv=False)
    inverted = metric.calc(net, inv=True)
    assert inverted == tuple(-x for x in normal)
    assert normal[1] > 0  # (power, heat, gas) -> heat curtailment registered
