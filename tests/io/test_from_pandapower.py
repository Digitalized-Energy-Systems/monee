import math

import pytest


def _switch_fusion_net():
    """4-bus net whose buses 1 and 2 are fused by a closed bus-bus switch."""
    import pandapower as pp

    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20.0, name="A", geodata=(0.0, 0.0))
    b1 = pp.create_bus(net, vn_kv=20.0, name="B", geodata=(1.0, 1.0))
    b2 = pp.create_bus(net, vn_kv=20.0, name="C", geodata=(2.0, 2.0))
    b3 = pp.create_bus(net, vn_kv=20.0, name="D", geodata=(3.0, 3.0))
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(
        net,
        b0,
        b1,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=0.4,
    )
    pp.create_switch(net, b1, b2, et="b", closed=True)
    pp.create_line_from_parameters(
        net,
        b2,
        b3,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=0.25,
    )
    pp.create_load(net, b3, p_mw=0.1, q_mvar=0.02)
    return net, (b0, b1, b2, b3)


@pytest.mark.pptest
def test_bus_bus_switch_fusion_maps_names_and_geodata():
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN buses B and C fused by a closed bus-bus switch (3 mpc buses from
    # 4 pp buses), which shifts the positional bus numbering
    net, (b0, b1, b2, b3) = _switch_fusion_net()

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN there are 3 nodes; names/positions land on the fusion-aware nodes
    assert len(monee_net.nodes) == 3
    by_name = {n.name: n for n in monee_net.nodes}
    assert set(by_name) == {"A", "B", "D"}  # first fused bus wins for B/C
    assert by_name["A"].position == (0.0, 0.0)
    assert by_name["B"].position == (1.0, 1.0)
    assert by_name["D"].position == (3.0, 3.0)


@pytest.mark.pptest
def test_bus_bus_switch_fusion_maps_max_i_ka_overrides():
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN
    net, _ = _switch_fusion_net()

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN both line ratings survive on the right branches
    by_name = {n.name: n for n in monee_net.nodes}
    ratings = {
        frozenset((b.from_node_id, b.to_node_id)): b.model.max_i_ka
        for b in monee_net.branches
    }
    assert ratings[frozenset((by_name["A"].id, by_name["B"].id))] == pytest.approx(0.4)
    assert ratings[frozenset((by_name["B"].id, by_name["D"].id))] == pytest.approx(0.25)


def _line(net, from_bus, to_bus, max_i_ka, r=0.1, **kwargs):
    import pandapower as pp

    return pp.create_line_from_parameters(
        net,
        from_bus,
        to_bus,
        length_km=1.0,
        r_ohm_per_km=r,
        x_ohm_per_km=r,
        c_nf_per_km=0.0,
        max_i_ka=max_i_ka,
        **kwargs,
    )


def _fused_parallel_net():
    """Buses B and B' fused by a closed switch; two parallel lines A->B, A->B'."""
    import pandapower as pp

    net = pp.create_empty_network()
    a = pp.create_bus(net, vn_kv=20.0, name="A")
    b = pp.create_bus(net, vn_kv=20.0, name="B")
    b2 = pp.create_bus(net, vn_kv=20.0, name="B'")
    pp.create_ext_grid(net, a)
    pp.create_switch(net, b, b2, et="b", closed=True)
    pp.create_load(net, b, p_mw=0.1)
    return net, (a, b, b2)


@pytest.mark.pptest
def test_fused_parallel_lines_and_coupler_keep_distinct_ratings():
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN a coupler line across the closed switch (both endpoints fuse into a
    # self-loop mpc branch) created BEFORE two parallel lines onto the fused bus
    net, (a, b, b2) = _fused_parallel_net()
    _line(net, b, b2, max_i_ka=0.1, r=0.05)
    _line(net, a, b, max_i_ka=0.4, r=0.1)
    _line(net, a, b2, max_i_ka=0.25, r=0.2)

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN each of the three branches carries its own line rating
    by_name = {n.name: n.id for n in monee_net.nodes}
    ratings = {b.id: b.model.max_i_ka for b in monee_net.branches}
    a_id, b_id = by_name["A"], by_name["B"]
    assert ratings == {
        (b_id, b_id, 0): pytest.approx(0.1),
        (a_id, b_id, 0): pytest.approx(0.4),
        (a_id, b_id, 1): pytest.approx(0.25),
    }


@pytest.mark.pptest
def test_out_of_service_line_does_not_shift_parallel_ratings():
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN an out-of-service line (dropped by to_mpc) between two in-service
    # parallel lines onto a fused bus
    net, (a, b, b2) = _fused_parallel_net()
    _line(net, a, b, max_i_ka=0.4, r=0.1)
    _line(net, a, b2, max_i_ka=0.11, r=0.2, in_service=False)
    _line(net, a, b2, max_i_ka=0.25, r=0.3)

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN the dropped line consumes no parallel slot and its rating appears
    # nowhere
    ratings = sorted(b.model.max_i_ka for b in monee_net.branches)
    assert ratings == [pytest.approx(0.25), pytest.approx(0.4)]


@pytest.mark.pptest
def test_reversed_direction_fused_parallel_lines_keep_ratings():
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN fused parallel lines defined in opposite directions; monee keys
    # parallels per undirected node pair
    net, (a, b, b2) = _fused_parallel_net()
    _line(net, a, b, max_i_ka=0.4, r=0.1)
    _line(net, b2, a, max_i_ka=0.25, r=0.2)

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN
    ratings = sorted(b.model.max_i_ka for b in monee_net.branches)
    assert ratings == [pytest.approx(0.25), pytest.approx(0.4)]


@pytest.mark.pptest
def test_line_to_out_of_service_bus_warns_instead_of_mismapping():
    import pandapower as pp

    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN a line to an out-of-service bus, which to_mpc renumbers with a
    # stale bus lookup
    net = pp.create_empty_network()
    a = pp.create_bus(net, vn_kv=20.0, name="A")
    b = pp.create_bus(net, vn_kv=20.0, name="B")
    d = pp.create_bus(net, vn_kv=10.0, name="D", in_service=False)
    pp.create_ext_grid(net, a)
    _line(net, a, b, max_i_ka=0.4)
    _line(net, b, d, max_i_ka=0.15)
    pp.create_load(net, b, p_mw=0.1)

    # WHEN
    with pytest.warns(UserWarning, match="max_i_ka"):
        monee_net = from_pandapower_net(net)

    # THEN the reliable rating is attached and the unmappable one is not
    # attached anywhere (the affected branch keeps the MATPOWER-derived rating)
    ratings = {round(b.model.max_i_ka, 6) for b in monee_net.branches}
    assert 0.4 in ratings
    assert 0.15 not in ratings


def _two_bus_net_with_sgen(p_mw, q_mvar, scaling=1.0):
    import pandapower as pp

    net = pp.create_empty_network()
    b0 = pp.create_bus(net, vn_kv=20.0, name="slack")
    b1 = pp.create_bus(net, vn_kv=20.0, name="feed")
    pp.create_ext_grid(net, b0)
    pp.create_line_from_parameters(
        net,
        b0,
        b1,
        length_km=1.0,
        r_ohm_per_km=0.1,
        x_ohm_per_km=0.1,
        c_nf_per_km=0.0,
        max_i_ka=0.4,
    )
    pp.create_sgen(net, b1, p_mw=p_mw, q_mvar=q_mvar, scaling=scaling, name="sg")
    return net


@pytest.mark.pptest
def test_negative_p_sgen_imports_as_power_load():
    import monee.model as mm
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN an sgen consuming power (negative p in generation convention)
    net = _two_bus_net_with_sgen(p_mw=-0.5, q_mvar=-0.1)

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN it becomes a PowerLoad with positive (consuming) setpoints
    loads = [
        c
        for c in monee_net.childs
        if isinstance(c.model, mm.PowerLoad) and c.name == "sg"
    ]
    assert len(loads) == 1
    assert loads[0].model.p_mw == pytest.approx(0.5)
    assert loads[0].model.q_mvar == pytest.approx(0.1)
    assert not any(isinstance(c.model, mm.PowerGenerator) for c in monee_net.childs)


@pytest.mark.pptest
def test_scaling_zero_sgen_imports_zero_output():
    import monee.model as mm
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN a switched-off sgen (scaling=0), which must not fall back to 1.0
    net = _two_bus_net_with_sgen(p_mw=0.5, q_mvar=0.1, scaling=0.0)

    # WHEN
    monee_net = from_pandapower_net(net)

    # THEN
    gens = [
        c
        for c in monee_net.childs
        if isinstance(c.model, mm.PowerGenerator) and c.name == "sg"
    ]
    assert len(gens) == 1
    assert gens[0].model.p_mw == 0.0
    assert gens[0].model.q_mvar == 0.0


@pytest.mark.pptest
def test_caller_net_not_mutated_without_sgens():
    from monee.io.from_pandapower import from_pandapower_net

    # GIVEN a net without sgens
    net, _ = _switch_fusion_net()
    columns_before = {
        name: list(df.columns) for name, df in net.items() if hasattr(df, "columns")
    }
    assert not hasattr(net, "_pd2ppc_lookups") or net._pd2ppc_lookups.get("bus") is None

    # WHEN
    from_pandapower_net(net)

    # THEN no to_mpc conversion artifacts leak into the caller's net
    assert net._pd2ppc_lookups.get("bus") is None
    for name, cols in columns_before.items():
        assert list(net[name].columns) == cols


def test_strip_shifts_only_multiples_of_30_degrees():
    import monee.model as mm
    from monee.io.from_pandapower import _strip_transformer_vector_group_shifts
    from monee.model.branch import GenericPowerBranch

    pn = mm.Network(mm.create_power_grid("power"))
    n0 = pn.node(mm.Bus(base_kv=1))
    n1 = pn.node(mm.Bus(base_kv=1))
    n2 = pn.node(mm.Bus(base_kv=1))
    kwargs = dict(
        tap=1.0,
        br_r_pu=0.01,
        br_x_pu=0.05,
        g_fr_pu=0,
        b_fr_pu=0,
        g_to_pu=0,
        b_to_pu=0,
    )
    pn.branch(GenericPowerBranch(shift=math.radians(150.0), **kwargs), n0, n1)
    pn.branch(GenericPowerBranch(shift=math.radians(7.5), **kwargs), n1, n2)

    with pytest.warns(UserWarning, match="vector-group"):
        _strip_transformer_vector_group_shifts(pn)

    shift_by_id = {b.id[:2]: b.model.shift for b in pn.branches}
    assert shift_by_id[(n0, n1)] == 0.0  # 150 deg vector-group shift stripped
    assert shift_by_id[(n1, n2)] == pytest.approx(math.radians(7.5))  # kept


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
    assert (
        abs(
            sum(c.model.p_mw for c in childs)
            - (net.load.p_mw.sum() - net.sgen.p_mw.sum())
        )
        < 1e-6
    )
    assert (
        abs(
            sum(c.model.q_mvar for c in childs)
            - (net.load.q_mvar.sum() - net.sgen.q_mvar.sum())
        )
        < 1e-6
    )

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
