"""Input-mutation contract of run_energy_flow: in-place write-back by default
(warm starts), copy=True solving a copy and leaving the input untouched."""

import monee
import monee.express as mx
import monee.model as mm


def _net():
    net = mx.create_multi_energy_network()
    b0 = mx.create_bus(net)
    b1 = mx.create_bus(net)
    mx.create_ext_power_grid(net, b0)
    mx.create_power_load(net, b1, p_mw=0.4, q_mvar=0.0)
    mx.create_line(net, b0, b1, 200, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5)
    return net


def _ext_p(net):
    for child in net.childs:
        if isinstance(child.model, mm.ExtPowerGrid):
            value = child.model.p_mw
            return value.value if hasattr(value, "value") else value
    raise AssertionError("no ext grid")


def test_run_energy_flow_writes_solved_values_into_the_input():
    net = _net()
    before = _ext_p(net)
    monee.run_energy_flow(net)
    assert _ext_p(net) != before
    assert _ext_p(net) < -0.3


def test_run_energy_flow_copy_true_leaves_the_input_untouched():
    net = _net()
    before = _ext_p(net)
    result = monee.run_energy_flow(net, copy=True)
    assert _ext_p(net) == before
    assert result.network is not net
    assert result.get(mm.ExtPowerGrid)["p_mw"].iloc[0] < -0.3
