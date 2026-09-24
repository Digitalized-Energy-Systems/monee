"""Backbone selection for MES generation: weighted spanning tree and the
Steiner-tree option. Pure topology tests, no solver."""

import pytest

import monee.model as mm
from monee.network import generate_supply_return_mes_based_on_power_net


def _line(length_m):
    return mm.PowerLine(
        length_m=length_m,
        r_ohm_per_m=1e-4,
        x_ohm_per_m=1e-4,
        max_i_ka=1.0,
        parallel=1,
    )


def _edges(net):
    return {frozenset((b.from_node_id, b.to_node_id)) for b in net.branches}


def _total_length(net):
    return sum(b.model.length_m for b in net.branches)


def _triangle():
    """0-1-2 triangle: short legs (len 1) and a long hypotenuse (len 10) added
    first, so the unit-weight MST keeps the long edge but a length-weighted MST
    drops it."""
    pn = mm.Network(el_model=mm.PowerGrid(name="power", sn_mva=100))
    n0 = pn.node(mm.Bus(base_kv=20), mm.EL)
    n1 = pn.node(mm.Bus(base_kv=20), mm.EL)
    n2 = pn.node(mm.Bus(base_kv=20), mm.EL)
    pn.branch(_line(10), n0, n2)
    pn.branch(_line(1), n0, n1)
    pn.branch(_line(1), n1, n2)
    return pn


def _path_with_leaf():
    """Path 0-1-2 with a leaf 3 hanging off node 1."""
    pn = mm.Network(el_model=mm.PowerGrid(name="power", sn_mva=100))
    ids = [pn.node(mm.Bus(base_kv=20), mm.EL) for _ in range(4)]
    pn.branch(_line(1), ids[0], ids[1])
    pn.branch(_line(1), ids[1], ids[2])
    pn.branch(_line(1), ids[1], ids[3])
    return pn


def _power_path_with_terminals():
    """Path 0-1-2-3-4: slack at 0, load at 2, generator at 3; 1 and 4 are
    transit buses (4 is a droppable leaf)."""
    pn = mm.Network(el_model=mm.PowerGrid(name="power", sn_mva=100))
    b0 = pn.node(
        mm.Bus(base_kv=20),
        mm.EL,
        child_ids=[pn.child(mm.ExtPowerGrid(p_mw=0, q_mvar=0, vm_pu=1, va_degree=0))],
    )
    b1 = pn.node(mm.Bus(base_kv=20), mm.EL)
    b2 = pn.node(
        mm.Bus(base_kv=20), mm.EL, child_ids=[pn.child(mm.PowerLoad(p_mw=1, q_mvar=0))]
    )
    b3 = pn.node(
        mm.Bus(base_kv=20),
        mm.EL,
        child_ids=[pn.child(mm.PowerGenerator(p_mw=1, q_mvar=0, regulation=0.5))],
    )
    b4 = pn.node(mm.Bus(base_kv=20), mm.EL)
    for a, b in ((b0, b1), (b1, b2), (b2, b3), (b3, b4)):
        pn.branch(_line(100), a, b)
    return pn


def _gas_junctions(mes):
    return [
        n
        for n in mes.nodes
        if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.GasGrid)
    ]


def _water_junctions(mes):
    return [
        n
        for n in mes.nodes
        if isinstance(n.model, mm.Junction) and isinstance(n.grid, mm.WaterGrid)
    ]


def test_spanning_tree_default_is_unit_weighted():
    pn = _triangle()
    tree = mm.to_spanning_tree(pn)
    assert len(list(tree.branches)) == 2  # n - 1
    # Unit weights + insertion order keep the long hypotenuse.
    assert frozenset((0, 2)) in _edges(tree)
    assert _total_length(tree) == 11


def test_spanning_tree_length_weighted_minimises_length():
    pn = _triangle()
    weighted = mm.to_spanning_tree(
        pn, weight=lambda branch, a, b: branch.model.length_m
    )
    assert len(list(weighted.branches)) == 2
    # Length weighting drops the long edge for the two short legs.
    assert frozenset((0, 2)) not in _edges(weighted)
    assert _total_length(weighted) == 2
    assert _total_length(weighted) < _total_length(mm.to_spanning_tree(pn))


def test_backbone_span_equivalent_to_spanning_tree():
    pn = _triangle()
    assert _edges(mm.to_backbone(pn, method="span")) == _edges(mm.to_spanning_tree(pn))


def test_backbone_steiner_keeps_transit_drops_leaf():
    pn = _path_with_leaf()
    bb = mm.to_backbone(pn, method="steiner", terminals={0, 2})
    node_ids = {n.id for n in bb.nodes}
    assert node_ids == {0, 1, 2}  # transit 1 kept, leaf 3 dropped
    assert _edges(bb) == {frozenset((0, 1)), frozenset((1, 2))}


def test_backbone_steiner_is_deterministic():
    pn = _path_with_leaf()
    a = mm.to_backbone(pn, method="steiner", terminals={0, 2})
    b = mm.to_backbone(pn, method="steiner", terminals={0, 2})
    assert _edges(a) == _edges(b)
    assert {n.id for n in a.nodes} == {n.id for n in b.nodes}


def test_backbone_invalid_args():
    pn = _triangle()
    with pytest.raises(ValueError):
        mm.to_backbone(pn, method="steiner")  # terminals required
    with pytest.raises(ValueError):
        mm.to_backbone(pn, method="bogus")


def test_generate_mes_default_spans_all_buses():
    pn = _power_path_with_terminals()
    mes = generate_supply_return_mes_based_on_power_net(pn, coupling_density=0)
    # span backbone keeps every bus: one gas junction per bus, one water supply
    # junction per bus plus the shared return junction.
    assert len(_gas_junctions(mes)) == 5
    assert len(_water_junctions(mes)) == 5 + 1


def test_generate_mes_steiner_drops_transit_leaf():
    pn = _power_path_with_terminals()
    span = generate_supply_return_mes_based_on_power_net(pn, coupling_density=0)
    steiner = generate_supply_return_mes_based_on_power_net(
        pn, coupling_density=0, backbone_method="steiner"
    )
    # Terminals {0, 2, 3} need transit 1 but not the leaf 4 -> one bus dropped.
    assert len(_gas_junctions(steiner)) == 4
    assert len(_gas_junctions(steiner)) < len(_gas_junctions(span))
    assert len(_water_junctions(steiner)) == 4 + 1


def test_generate_mes_length_weight_runs():
    pn = _power_path_with_terminals()
    mes = generate_supply_return_mes_based_on_power_net(
        pn, coupling_density=0, backbone_weight="length"
    )
    assert len(_gas_junctions(mes)) == 5  # still a full span, just length-weighted
