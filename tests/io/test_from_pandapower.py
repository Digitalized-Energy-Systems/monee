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
    """``max_i_ka`` for lines must come from ``net.line``, not from
    matpower's apparent-power ``rateA`` (which the converter drops to a
    250 MVA placeholder) or the legacy ``0.319 kA`` fallback in
    :func:`monee.io.matpower.fill_branch_dict`.

    The bug previously made every imported branch — LV cable, MV cable, HV
    line, or distribution transformer — share the same 0.319 kA limit,
    silently breaking any line-loading constraint downstream.  The fix
    overrides each *line* branch's ``max_i_ka`` from
    ``net.line.max_i_ka × parallel × df``.

    Transformers are intentionally not covered (the single-``max_i_ka``
    branch model cannot represent the HV / LV side asymmetry — see the
    docstring of ``_pp_branch_max_i_ka_overrides``); they retain the
    legacy placeholder so the test asserts presence of line ratings only.
    """
    import simbench

    import monee.model as mm
    from monee.io.from_pandapower import from_pandapower_net

    # LV-rural3: 127 cables at 0.27 kA each + one transformer.  Every line
    # branch must now carry 0.27, and the legacy 0.319 must appear at most
    # as the trafo / switch-aux placeholder — not on real lines.
    net = simbench.get_simbench_net("1-LV-rural3--1-no_sw")
    monee_net = from_pandapower_net(net)
    branches = [
        b for b in monee_net.branches if isinstance(b.model, mm.GenericPowerBranch)
    ]
    nb_at_line_rating = sum(1 for b in branches if abs(b.model.max_i_ka - 0.27) < 1e-6)
    assert nb_at_line_rating == len(net.line), (
        f"expected all {len(net.line)} line branches at 0.27 kA, got "
        f"{nb_at_line_rating}"
    )

    # MV-rural: every distinct value in ``net.line.max_i_ka`` must show up
    # on at least one imported branch (modulo switch-auxiliary branches
    # documented in ``_pp_branch_max_i_ka_overrides``).
    net_mv = simbench.get_simbench_net("1-MV-rural--0-no_sw")
    monee_mv = from_pandapower_net(net_mv)
    mv_branches = [
        b for b in monee_mv.branches if isinstance(b.model, mm.GenericPowerBranch)
    ]
    mv_max_i_kas = {round(b.model.max_i_ka, 6) for b in mv_branches}
    for v in net_mv.line["max_i_ka"].unique():
        assert round(float(v), 6) in mv_max_i_kas, (
            f"MV line rating {v} not present after import"
        )
