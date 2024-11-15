import pytest


@pytest.mark.pptest
def test_from_pandapower_net():
    import simbench

    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN
    assert len(monee_net.nodes) == 129
    assert monee_net.nodes[0].name == "MV1.101 Bus 12"
    assert monee_net.nodes[0].position == (11.4096, 53.6531)
    assert monee_net.nodes[101].name == "LV3.101 Bus 11"
    assert monee_net.nodes[101].position == (11.4045, 53.6538)
