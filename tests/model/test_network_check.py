import pytest

import monee.express as mx
import monee.model as mm
from monee.model import Var
from monee.model.network import CheckFinding


def _clean_el_net():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    mx.create_ext_power_grid(net, b0)
    mx.create_line(net, b0, b1, length_m=500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.1)
    return net


def test_clean_net_returns_empty_and_prints_clean(capsys):
    findings = _clean_el_net().check()

    assert findings == []
    assert "no findings" in capsys.readouterr().out


def test_missing_slack_fires_per_carrier():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    mx.create_line(net, b0, b1, length_m=500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.1)

    findings = net.check(dof_preview=False)

    assert len(findings) == 1
    assert findings[0].category == "missing_slack"
    assert "electrical" in findings[0].message
    assert "diagnose_infeasibility" in findings[0].message


def test_heat_only_dead_end_fires():
    net = mm.Network(mm.create_water_grid("water"))
    j0 = mx.create_water_junction(net)
    j1 = mx.create_water_junction(net)
    j2 = mx.create_water_junction(net)
    mx.create_ext_hydr_grid(net, j0)
    mx.create_water_pipe(net, j0, j1, diameter_m=0.2, length_m=100)
    mx.create_water_pipe(net, j1, j2, diameter_m=0.2, length_m=100)
    mx.create_sink(net, j1, mass_flow_kgs=1)
    mx.create_heat_load(net, j2, q_mw=0.1, name="my heat load")

    findings = net.check(dof_preview=False)

    dead_ends = [f for f in findings if f.category == "heat_only_dead_end"]
    assert len(dead_ends) == 1
    assert "my heat load" in dead_ends[0].message
    assert f"Junction({j2})" == dead_ends[0].component


def test_mixed_base_kv_fires_on_line_not_on_trafo():
    net = _clean_el_net()
    b2 = mx.create_bus(net, base_kv=0.4)
    mx.create_line(net, 1, b2, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b2, p_mw=0.1, q_mvar=0.0)

    findings = net.check(dof_preview=False)
    mixed = [f for f in findings if f.category == "mixed_base_kv"]
    assert len(mixed) == 1
    assert "20" in mixed[0].message and "0.4" in mixed[0].message

    trafo_net = _clean_el_net()
    b2 = mx.create_bus(trafo_net, base_kv=0.4)
    trafo_net.branch(mm.Trafo(), 1, b2, grid=mm.EL)
    mx.create_power_load(trafo_net, b2, p_mw=0.1, q_mvar=0.0)
    assert trafo_net.check(dof_preview=False) == []


def test_dof_preview_reports_unpinned_var():
    @mm.model
    class _FreeVarChild(mm.ChildModel):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.some_setting = Var(1.0, min=0.5, max=3.0, name="some_setting")
            self.p_mw = 0.0
            self.q_mvar = 0.0

        def equations(self, grid, node, **kwargs):
            return []

    net = _clean_el_net()
    mx.create_el_child(net, _FreeVarChild(), node_id=1)

    findings = net.check()

    dof = [f for f in findings if f.category == "dof"]
    assert len(dof) == 1
    assert "not square" in dof[0].message
    assert "_FreeVarChild.some_setting" in dof[0].message


def test_check_mode_optimization_skips_dof_preview():
    net = _clean_el_net()
    net.child_to(mm.ElectricStorage(e_mwh_initial=0.5, e_mwh_max=1.0, p_max_mw=0.2), 1)

    dof = [f for f in net.check() if f.category == "dof"]
    assert len(dof) == 1
    assert "mode='optimization'" in dof[0].message

    assert net.check(mode="optimization") == []

    with pytest.raises(ValueError, match="mode"):
        net.check(mode="opt")


def test_check_mode_optimization_keeps_graph_checks():
    net = mm.Network(mm.create_power_grid("power"))
    b0 = mx.create_bus(net, base_kv=20)
    b1 = mx.create_bus(net, base_kv=20)
    mx.create_line(net, b0, b1, length_m=500, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4)
    mx.create_power_load(net, b1, p_mw=0.5, q_mvar=0.1)

    findings = net.check(mode="optimization")
    assert [f.category for f in findings] == ["missing_slack"]


def test_mixed_base_kv_exempts_trafo_kind_marker():
    net = _clean_el_net()
    b2 = mx.create_bus(net, base_kv=0.4)
    line_id = mx.create_line(
        net, 1, b2, length_m=100, r_ohm_per_m=1e-4, x_ohm_per_m=1e-4
    )
    mx.create_power_load(net, b2, p_mw=0.1, q_mvar=0.0)

    branch = net.branch_by_id(line_id)
    branch.model.kind = "trafo"
    assert net.check(dof_preview=False) == []

    branch.model.kind = "line"
    mixed = [f for f in net.check(dof_preview=False) if f.category == "mixed_base_kv"]
    assert len(mixed) == 1


def test_deactivate_docstrings_cover_two_arg_form():
    import inspect

    doc = inspect.getdoc(mm.Network.deactivate)
    assert doc is not None and "deactivate_by_id" in doc
    assert inspect.getdoc(mm.Network.activate) is not None
    for name in (
        "remove_child",
        "remove_node",
        "remove_branch",
        "remove_branch_between",
        "remove_compound",
    ):
        assert inspect.getdoc(getattr(mm.Network, name)) is not None


def test_check_does_not_mutate_network():
    net = _clean_el_net()
    before = {type(c.model).__name__ for c in net.iter_all_components()}
    net.check()
    after = {type(c.model).__name__ for c in net.iter_all_components()}
    assert before == after


def test_check_finding_str_includes_category_and_component():
    finding = CheckFinding("mixed_base_kv", "message text", component="PowerLine(x)")
    assert str(finding) == "[mixed_base_kv] [PowerLine(x)] message text"
