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


@pytest.mark.pptest
def test_from_pandapower_max_i_ka_overrides_matpower_placeholder():
    import simbench

    import monee.model as mm
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    net_mv = simbench.get_simbench_net("1-MV-rural--0-no_sw")

    # WHEN
    monee_net = from_pandapower_net(net)
    monee_mv = from_pandapower_net(net_mv)

    # THEN
    # line max_i_ka must come from net.line, not the 0.319 kA matpower placeholder
    branches = [
        b for b in monee_net.branches if isinstance(b.model, mm.GenericPowerBranch)
    ]
    nb_at_line_rating = sum(1 for b in branches if abs(b.model.max_i_ka - 0.27) < 1e-6)
    assert nb_at_line_rating == len(net.line), (
        f"expected all {len(net.line)} line branches at 0.27 kA, got "
        f"{nb_at_line_rating}"
    )

    # every distinct MV line rating must appear on at least one imported branch
    mv_branches = [
        b for b in monee_mv.branches if isinstance(b.model, mm.GenericPowerBranch)
    ]
    mv_max_i_kas = {round(b.model.max_i_ka, 6) for b in mv_branches}
    for v in net_mv.line["max_i_ka"].unique():
        assert round(float(v), 6) in mv_max_i_kas, (
            f"MV line rating {v} not present after import"
        )
