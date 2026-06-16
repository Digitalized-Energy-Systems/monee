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
def test_sgens_import_as_distinct_generators_without_positive_reactive():
    import simbench

    import monee.model as mm
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN a grid whose buses carry both loads and static generators
    net = simbench.get_simbench_net("1-MV-rural--0-sw")
    n_sgen = int(net.sgen.in_service.sum())
    assert n_sgen > 0

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN every in-service sgen becomes its own PowerGenerator ...
    gens = [c for c in monee_net.childs if isinstance(c.model, mm.PowerGenerator)]
    assert len(gens) == n_sgen

    # ... and none carries positive (consuming) reactive power. monee stores
    # generation in load convention, so an injecting generator's q_mvar is <= 0;
    # the load's reactive power now stays on its own PowerLoad child.
    assert all(g.model.q_mvar <= 1e-9 for g in gens)

    # The net nodal injection is unchanged vs the raw pandapower elements
    # (load - sgen, both in generation convention for the sgen contribution).
    childs = [
        c
        for c in monee_net.childs
        if isinstance(c.model, (mm.PowerGenerator, mm.PowerLoad))
    ]
    assert abs(sum(c.model.p_mw for c in childs) - (net.load.p_mw.sum() - net.sgen.p_mw.sum())) < 1e-6
    assert abs(sum(c.model.q_mvar for c in childs) - (net.load.q_mvar.sum() - net.sgen.q_mvar.sum())) < 1e-6

    # The caller's net is not mutated.
    assert len(net.sgen) == n_sgen


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
