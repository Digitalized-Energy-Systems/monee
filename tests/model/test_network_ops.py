"""Network structural operations: compound lifecycle across graph transforms,
compound() reentrancy and exception safety, grid flattening, and error paths
of the builder/removal API."""

import pytest

import monee.model as mm
from monee.model.core import CompoundModel


class _AppendixCompound(CompoundModel):
    """Creates one internal junction (with a Sink child) attached to the
    connect node via a pipe - acyclic, so a spanning tree keeps all parts."""

    def create(self, network, connect_node):
        self.internal_node_id = network.node(mm.Junction(), grid=mm.WATER_KEY)
        self.child_id = network.child_to(
            mm.Sink(mass_flow_kgs=0.1), self.internal_node_id
        )
        self.branch_id = network.branch(
            mm.WaterPipe(diameter_m=0.1, length_m=10),
            connect_node.id,
            self.internal_node_id,
        )

    def equations(self, network, **kwargs):
        return []


class _NestingCompound(CompoundModel):
    def create(self, network, connect_node):
        self.own_node_id = network.node(mm.Junction(), grid=mm.WATER_KEY)
        network.branch(
            mm.WaterPipe(diameter_m=0.1, length_m=10),
            connect_node.id,
            self.own_node_id,
        )
        self.inner = _AppendixCompound()
        self.inner_id = network.compound(self.inner, connect_node_id=connect_node.id)

    def equations(self, network, **kwargs):
        return []


class _FailingCompound(CompoundModel):
    def create(self, network, connect_node):
        network.node(mm.Junction(), grid=mm.WATER_KEY)
        raise RuntimeError("boom")

    def equations(self, network, **kwargs):
        return []


def _water_net_with_two_nodes():
    net = mm.Network()
    n0 = net.node(
        mm.Junction(), grid=mm.WATER_KEY, child_ids=[net.child(mm.ExtHydrGrid())]
    )
    n1 = net.node(
        mm.Junction(),
        grid=mm.WATER_KEY,
        child_ids=[net.child(mm.Sink(mass_flow_kgs=0.1))],
    )
    net.branch(mm.WaterPipe(diameter_m=0.1, length_m=10), n0, n1)
    return net, n0, n1


def test_spanning_tree_keeps_fully_intact_compound():
    # GIVEN
    net, n0, _ = _water_net_with_two_nodes()
    model = _AppendixCompound()
    net.compound(model, connect_node_id=n0)

    # WHEN a transform that keeps every compound part (net is acyclic)
    tree = mm.to_spanning_tree(net)

    # THEN the compound survives intact
    assert len(tree.compounds) == 1
    assert tree.has_node(model.internal_node_id)
    assert tree.has_child(model.child_id)
    assert tree.has_branch(model.branch_id)


def test_transform_removes_broken_compound_cleanly():
    # GIVEN
    net, n0, _ = _water_net_with_two_nodes()
    model = _AppendixCompound()
    net.compound(model, connect_node_id=n0)
    branch_id = model.branch_id

    # WHEN a transform drops the compound-internal branch
    def drop_compound_branch(g):
        g = g.copy()
        g.remove_edge(branch_id[0], branch_id[1], branch_id[2])
        return g

    reduced = mm.transform_network(net, drop_compound_branch)

    # THEN the compound and all its parts are gone, without orphans
    assert len(reduced.compounds) == 0
    assert not reduced.has_node(model.internal_node_id)
    assert not reduced.has_child(model.child_id)
    for child in reduced.childs:
        assert any(child.id in node.child_ids for node in reduced.nodes)
    # the non-compound part of the network is untouched
    assert len(reduced.nodes) == 2
    assert len(reduced.childs) == 2


def test_nested_compound_build():
    # GIVEN
    net, n0, _ = _water_net_with_two_nodes()
    model = _NestingCompound()

    # WHEN
    outer_id = net.compound(model, connect_node_id=n0)

    # THEN both compounds are registered and the inner one is a subcomponent
    # of the outer (not one of its leaked parts)
    assert net.has_compound(outer_id)
    assert net.has_compound(model.inner_id)
    outer = net.compound_by_id(outer_id)
    inner = net.compound_by_id(model.inner_id)
    assert inner in outer.subcomponents
    # the inner compound's parts belong to the inner compound only
    inner_part_ids = {id(c) for c in inner.subcomponents}
    outer_part_ids = {id(c) for c in outer.subcomponents}
    assert not inner_part_ids & outer_part_ids
    assert any(c.id == model.own_node_id for c in outer.subcomponents)
    assert any(c.id == inner.model.internal_node_id for c in inner.subcomponents)


def test_compound_exception_does_not_pollute_next_compound():
    # GIVEN
    net, n0, _ = _water_net_with_two_nodes()

    # WHEN a compound build fails mid-create
    with pytest.raises(RuntimeError, match="boom"):
        net.compound(_FailingCompound(), connect_node_id=n0)

    # THEN the next compound only collects its own parts
    model = _AppendixCompound()
    compound_id = net.compound(model, connect_node_id=n0)
    subcomponent_ids = {
        c.id for c in net.compound_by_id(compound_id).component_of_type(mm.Node)
    }
    assert subcomponent_ids == {model.internal_node_id}
    # and components created afterwards are independent again
    extra = net.node(mm.Junction(), grid=mm.WATER_KEY)
    assert net.node_by_id(extra).independent


def test_grids_flattens_multi_grid_nodes():
    # GIVEN
    net, _, _ = _water_net_with_two_nodes()
    water = net.node_by_id(0).grid
    gas = mm.create_gas_grid("gas")
    net.node(mm.Junction(), grid=[water, gas])

    # WHEN / THEN
    grids = net.grids
    assert water in grids
    assert gas in grids
    assert len(grids) == 2


def test_compound_internal_components_are_blacklisted():
    # GIVEN
    net, n0, _ = _water_net_with_two_nodes()
    model = _AppendixCompound()
    net.compound(model, connect_node_id=n0)

    # THEN
    assert net.is_blacklisted(net.node_by_id(model.internal_node_id))
    assert net.is_blacklisted(net.child_by_id(model.child_id))
    assert not net.is_blacklisted(net.node_by_id(n0))


def test_remove_node_removes_attached_childs():
    # GIVEN
    net, _, n1 = _water_net_with_two_nodes()
    child_ids = list(net.node_by_id(n1).child_ids)
    assert child_ids

    # WHEN
    net.remove_node(n1)

    # THEN
    for child_id in child_ids:
        assert not net.has_child(child_id)


def test_get_branch_between_missing_edge_raises_value_error():
    net, n0, n1 = _water_net_with_two_nodes()
    with pytest.raises(ValueError):
        net.get_branch_between(n0, 999)
    with pytest.raises(ValueError):
        net.get_branch_between(n0, n1, bid=5)


def test_child_to_missing_node_without_creator_raises_value_error():
    net, _, _ = _water_net_with_two_nodes()
    with pytest.raises(ValueError):
        net.child_to(mm.Sink(mass_flow_kgs=0.1), 999)


def test_overwrite_id_collision_raises():
    net, n0, _ = _water_net_with_two_nodes()
    with pytest.raises(ValueError):
        net.node(mm.Junction(), grid=mm.WATER_KEY, overwrite_id=n0)
    with pytest.raises(ValueError):
        net.child(mm.Sink(mass_flow_kgs=0.1), overwrite_id=0)


def test_move_branch_preserves_component_state():
    # GIVEN
    net, n0, n1 = _water_net_with_two_nodes()
    n2 = net.node(mm.Junction(), grid=mm.WATER_KEY)
    branch = net.get_branch_between(n0, n1)
    branch.active = False

    # WHEN
    new_id = net.move_branch(branch.id, n0, n2)

    # THEN
    moved = net.branch_by_id(new_id)
    assert moved.active is False
    assert moved.independent is True
    assert moved.model is branch.model


def test_parallel_branches_have_distinct_tids():
    net, n0, n1 = _water_net_with_two_nodes()
    second = net.branch(mm.WaterPipe(diameter_m=0.1, length_m=10), n0, n1)
    first_branch = net.get_branch_between(n0, n1, bid=0)
    second_branch = net.branch_by_id(second)
    assert first_branch.tid != second_branch.tid


def _el_net_with_load(p_mw=5.0, q_mvar=1.0):
    net = mm.Network(mm.PowerGrid(name="power", sn_mva=1))
    net.node(
        mm.Bus(base_kv=1),
        child_ids=[net.child(mm.PowerLoad(p_mw, q_mvar))],
        grid=mm.EL,
    )
    return net


def test_child_type_queries():
    net = _el_net_with_load()
    node = net.nodes[0]
    assert net.has_any_child_of_type(node, mm.PowerLoad) is True
    assert net.has_any_child_of_type(node, mm.PowerGenerator) is False
    childs = net.get_childs_by_type(node, mm.PowerLoad)
    assert len(childs) == 1
    assert isinstance(childs[0].model, mm.PowerLoad)


def test_compound_of_returns_none_without_compounds():
    net = _el_net_with_load()
    assert net.compound_of(net.childs[0].id) is None


def test_clear_childs():
    net = _el_net_with_load()
    assert len(net.childs) == 1
    net.clear_childs()
    assert len(net.childs) == 0
    assert net.nodes[0].child_ids == []


def test_statistics_counts_independent_models():
    net = _el_net_with_load()
    stats = net.statistics()
    assert isinstance(stats, dict)
    assert stats.get(mm.PowerLoad, 0) >= 1
