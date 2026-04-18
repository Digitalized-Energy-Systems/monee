"""Tests for the bulk-construction structure builders in monee.express."""

import pytest

import monee.express as mx
import monee.model as mm

# ----------------------- Gas -----------------------


def test_gas_line_counts():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)

    seg = g.line(5)

    assert len(seg.nodes) == 5
    assert len(seg.branches) == 4
    assert seg.first == seg.nodes[0]
    assert seg.last == seg.nodes[-1]
    assert len(net.nodes_by_type(mm.Junction)) == 5


def test_gas_line_with_sinks_creates_children():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)

    seg = g.line(4, sink_mass_flow=0.05)

    assert len(seg.children) == 4
    assert len(net.childs_by_type(mm.Sink)) == 4


def test_gas_ring_closes_topology():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)

    ring = g.ring(6)

    assert len(ring.nodes) == 6
    assert len(ring.branches) == 6
    # The last branch closes back to the first node.
    last_branch = ring.branches[-1]
    assert last_branch[0] == ring.last
    assert last_branch[1] == ring.first


def test_gas_star_counts():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)

    star = g.star([2, 3, 1])

    # 1 hub + 2 + 3 + 1 arm nodes
    assert len(star.nodes) == 1 + 2 + 3 + 1
    # branches = sum of arm lengths
    assert len(star.branches) == 2 + 3 + 1
    # all arms anchor at the shared hub
    assert all(arm.first == star.hub for arm in star.arms)


def test_gas_line_extends_from_existing_node():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)

    main = g.line(5)
    branch = g.line(3, start_from=main.nodes[2], sink_mass_flow=0.01)

    assert branch.first == main.nodes[2]
    assert len(branch.nodes) == 3
    assert len(branch.branches) == 2
    # Only the 2 new nodes get the sink, not the reused start_from.
    assert len(branch.children) == 2


def test_gas_structure_attach_ext_grid():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)

    seg = g.line(3, sink_mass_flow=0.01)
    ext_id = g.attach_ext_grid(seg.first)

    assert isinstance(ext_id, int)
    assert len(net.childs_by_type(mm.ExtHydrGrid)) == 1


# ----------------------- Water -----------------------


def test_water_line_with_heat_loads():
    net = mm.Network()
    w = mx.water_structure(net, diameter_m=0.15, length_m=100)

    seg = w.line(4, heat_load_q_w=1000.0)

    assert len(seg.nodes) == 4
    assert len(seg.branches) == 3
    assert len(net.childs_by_type(mm.HeatLoad)) == 4


# ----------------------- Electricity -----------------------


def test_el_line_with_loads_and_ext_grid():
    net = mm.Network()
    e = mx.el_structure(
        net, length_m=500, r_ohm_per_m=7e-5, x_ohm_per_m=7e-5, base_kv=20
    )

    seg = e.line(3, load_p_mw=1.0, load_q_mvar=0.2)
    ext = e.attach_ext_grid(seg.first)

    assert len(seg.nodes) == 3
    assert len(seg.branches) == 2
    assert len(net.childs_by_type(mm.PowerLoad)) == 3
    assert isinstance(ext, int)


# ----------------------- DHS -----------------------


def test_dhs_line_with_heat_exchangers():
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    seg = dhs.line(5, heat_exchanger_q_mw=0.03)

    assert len(seg.supply.nodes) == 5
    assert len(seg.return_.nodes) == 5
    # 4 supply + 4 return pipes + 5 heat exchangers
    assert len(seg.branches) == 4 + 4 + 5
    assert len(seg.heat_exchangers) == 5


def test_dhs_line_per_node_heat_exchanger_list():
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    seg = dhs.line(4, heat_exchanger_q_mw=[None, 0.02, 0, 0.03])

    # Only the entries with non-zero, non-None q become heat exchangers.
    assert len(seg.heat_exchangers) == 2


def test_dhs_line_heat_exchanger_list_length_mismatch():
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    with pytest.raises(ValueError):
        dhs.line(4, heat_exchanger_q_mw=[0.02, 0.02])


def test_dhs_attach_heat_plant():
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    seg = dhs.line(3, heat_exchanger_q_mw=0.02)
    sid, rid = dhs.attach_heat_plant(
        seg.supply.first, seg.return_.first, t_k=358.0, name="Plant1"
    )

    assert isinstance(sid, int)
    assert isinstance(rid, int)
    assert len(net.childs_by_type(mm.ExtHydrGrid)) == 1
    assert len(net.childs_by_type(mm.ConsumeHydrGrid)) == 1


def test_dhs_ring_counts():
    net = mm.Network()
    dhs = mx.dhs_structure(net, diameter_m=0.15, length_m=100)

    seg = dhs.ring(6, heat_exchanger_q_mw=0.02)

    assert len(seg.supply.nodes) == 6
    assert len(seg.return_.nodes) == 6
    # Each ring has 6 pipes + 6 heat exchangers bridging.
    assert len(seg.supply.branches) == 6
    assert len(seg.return_.branches) == 6
    assert len(seg.heat_exchangers) == 6


# ----------------------- Validation -----------------------


def test_line_rejects_zero_length():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)
    with pytest.raises(ValueError):
        g.line(0)


def test_ring_rejects_less_than_three():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)
    with pytest.raises(ValueError):
        g.ring(2)


def test_star_rejects_empty_arms():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)
    with pytest.raises(ValueError):
        g.star([])


def test_star_rejects_zero_arm():
    net = mm.Network()
    g = mx.gas_structure(net, diameter_m=0.3, length_m=100)
    with pytest.raises(ValueError):
        g.star([2, 0, 3])
